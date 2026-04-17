"""
EpiBERT — PyTorch implementation for GPU inference.

Weights are loaded from the TensorFlow checkpoint by extracting tensors
and transposing/reshaping to match PyTorch Conv1D/Linear conventions.

TF checkpoint weight conventions:
  Conv1D kernel: [kernel_size, in_channels, out_channels]  (TF)
              → [out_channels, in_channels, kernel_size]  (PyTorch)
  Dense kernel:  [in_features, out_features]              (TF)
              → [out_features, in_features]               (PyTorch)
  Q/K/V attention kernels: [1024, 8, 128] → flatten → [1024, 1024]
  Output attention kernel:  [8, 128, 1024] → [1024, 1024] → transpose

GPU: uses PyTorch CUDA (torch.cuda.is_available() = True).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, cfg

log = get_logger("model_torch")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Layers
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPool1D(nn.Module):
    """Learned soft-attention pooling (stride 2), equivalent to TF version."""
    def __init__(self, channels, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.score = nn.Linear(channels, 1, bias=False)

    def forward(self, x):
        # x: (B, C, L) — PyTorch convention
        B, C, L = x.shape
        L2 = L // self.pool_size
        x = x[:, :, :L2 * self.pool_size]
        x = x.view(B, C, L2, self.pool_size)   # (B, C, L2, pool)
        x_t = x.permute(0, 2, 3, 1)             # (B, L2, pool, C)
        w = torch.softmax(self.score(x_t), dim=2)  # (B, L2, pool, 1)
        out = (x_t * w).sum(dim=2)               # (B, L2, C)
        return out.permute(0, 2, 1)              # (B, C, L2)


class ResConvBlock(nn.Module):
    """BN → GELU → Conv1D → residual add."""
    def __init__(self, channels, kernel_size=1):
        super().__init__()
        self.bn   = nn.BatchNorm1d(channels, momentum=0.1, eps=1e-5)
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding='same', bias=True)

    def forward(self, x):
        # x: (B, C, L)
        y = self.bn(x)
        y = F.gelu(y)
        y = self.conv(y)
        return x + y


class ConvTowerBlock(nn.Module):
    """BN → GELU → Conv1D(5bp) → pool (attention or max)."""
    def __init__(self, in_channels, out_channels, kernel_size=5, use_attn_pool=True):
        super().__init__()
        self.bn   = nn.BatchNorm1d(in_channels, momentum=0.1, eps=1e-5)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same', bias=True)
        if use_attn_pool:
            self.pool = AttentionPool1D(out_channels, pool_size=2)
        else:
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.use_attn_pool = use_attn_pool

    def forward(self, x):
        y = self.bn(x)
        y = F.gelu(y)
        y = self.conv(y)
        return self.pool(y)


class PerformerLayer(nn.Module):
    """
    One Performer/BERT block (pre-norm).
    Matches TF checkpoint structure exactly:
      - layer_norm → pre-attention LN
      - Q/K/V/O dense
      - FFN: wide(1024→2048) → GELU → narrow(2048→1024) → FFN_layer_norm → residual
    """
    def __init__(self, d_model=1024, num_heads=8, head_dim=128, ffn_dim=2048):
        super().__init__()
        self.H  = num_heads
        self.Dh = head_dim
        self.D  = d_model

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.q_dense = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_dense = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.v_dense = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.o_dense = nn.Linear(num_heads * head_dim, d_model, bias=False)

        self.FFN_dense_wide   = nn.Linear(d_model, ffn_dim,  bias=True)
        self.FFN_layer_norm   = nn.LayerNorm(d_model, eps=1e-6)
        self.FFN_dense_narrow = nn.Linear(ffn_dim, d_model, bias=True)

    def forward(self, x):
        # x: (B, S, D)
        r = x
        x = self.layer_norm(x)

        B, S, D = x.shape
        # Q/K/V
        q = self.q_dense(x).view(B, S, self.H, self.Dh).transpose(1, 2)  # (B,H,S,Dh)
        k = self.k_dense(x).view(B, S, self.H, self.Dh).transpose(1, 2)
        v = self.v_dense(x).view(B, S, self.H, self.Dh).transpose(1, 2)

        scale = self.Dh ** 0.5
        attn  = torch.softmax((q @ k.transpose(-2, -1)) / scale, dim=-1)  # (B,H,S,S)
        ctx   = (attn @ v).transpose(1, 2).reshape(B, S, self.H * self.Dh)
        x = r + self.o_dense(ctx)

        r = x
        y = self.FFN_dense_wide(x)
        y = F.gelu(y)
        y = self.FFN_dense_narrow(y)
        y = self.FFN_layer_norm(y)
        return r + y


class EpiBERTTorch(nn.Module):
    """
    EpiBERT in PyTorch — GPU-native.

    Inputs (all float32 tensors):
      seq    (B, L, 4)   one-hot DNA
      atac   (B, L, 1)   ATAC signal
      motifs (B, 693)    motif scores

    Outputs dict:
      tracks  (B, L//128, 170)
      profile (B, L//128, 1)
    """

    def __init__(self, num_tracks=170, capture_layers=None):
        super().__init__()
        self.num_tracks    = num_tracks
        self.capture_layers = capture_layers or {}  # {name: idx}

        # DNA stem
        self.stem_conv     = nn.Conv1d(4, 512, 15, padding=7, bias=True)
        self.stem_res_conv = ResConvBlock(512, kernel_size=1)
        self.stem_pool     = AttentionPool1D(512, pool_size=2)

        # DNA conv tower [512, 512, 640, 640, 768, 896, 1024]
        tower_ch = [(512, 512), (512, 640), (640, 640), (640, 768), (768, 896), (896, 1024)]
        self.conv_tower = nn.ModuleList([
            ConvTowerBlock(ic, oc, use_attn_pool=True) for ic, oc in tower_ch
        ])

        # ATAC stem
        self.stem_conv_atac     = nn.Conv1d(1, 32, 50, padding=25, bias=True)
        self.stem_res_conv_atac = ResConvBlock(32, kernel_size=1)
        self.conv_tower_atac    = nn.ModuleList([
            ConvTowerBlock(32, 32, use_attn_pool=False),
            ConvTowerBlock(32, 64, use_attn_pool=False),
        ])
        self.atac_extra_pool = nn.AvgPool1d(kernel_size=32, stride=32)

        # Motif
        self.motif_fc1 = nn.Linear(693, 32, bias=True)
        self.motif_fc2 = nn.Linear(32,  8,  bias=True)

        # Projection (1096 = 1024 + 64 + 8) → 1024, no bias
        self.projection = nn.Linear(1096, 1024, bias=False)

        # 8 Performer layers
        self.performer = nn.ModuleList([
            PerformerLayer(d_model=1024, num_heads=8, head_dim=128, ffn_dim=2048)
            for _ in range(8)
        ])
        self.performer_ln = nn.LayerNorm(1024, eps=1e-6)

        # Output head
        self.out_bn   = nn.BatchNorm1d(1024, momentum=0.1, eps=1e-5)
        self.out_conv  = nn.Conv1d(1024, num_tracks, 1, bias=True)
        self.out_dense = nn.Linear(num_tracks, 1, bias=True)

        # Activation cache (filled during forward)
        self.activation_cache = {}   # name → (B, S, 1024) tensor
        self.pooled_cache     = {}   # name → (B, 1024) tensor

    def _encode(self, seq, atac, motifs):
        """Encode inputs up to pre-transformer projection."""
        # seq: (B, L, 4) → (B, 4, L) for Conv1d
        x = seq.permute(0, 2, 1)
        x = self.stem_conv(x)           # (B, 512, L)
        x = self.stem_res_conv(x)
        x = self.stem_pool(x)           # (B, 512, L/2)
        for blk in self.conv_tower:
            x = blk(x)                 # → (B, 1024, L/128)

        # ATAC
        a = atac.permute(0, 2, 1)      # (B, 1, L)
        a = self.stem_conv_atac(a)      # (B, 32, L)  padding=25 → truncate to L
        a = a[:, :, :seq.shape[1]]     # trim to seq length (padding might overshoot)
        a = self.stem_res_conv_atac(a)
        for blk in self.conv_tower_atac:
            a = blk(a)                 # (B, 64, L/4)
        a = self.atac_extra_pool(a)    # (B, 64, L/128)
        a = a[:, :, :x.shape[2]]       # align

        # Motif
        m = F.gelu(self.motif_fc1(motifs))  # (B, 32)
        m = self.motif_fc2(m)               # (B, 8)
        m = m.unsqueeze(2).expand(-1, -1, x.shape[2])  # (B, 8, L/128)

        # Concat [DNA(1024), ATAC(64), motif(8)] → (B, 1096, L/128)
        combined = torch.cat([x, a, m], dim=1)
        # Projection: (B, 1096, L/128) → (B, 1024, L/128)
        # Linear expects (..., in) so permute
        combined = self.projection(combined.permute(0, 2, 1))  # (B, L/128, 1024)
        return combined

    def _transformer(self, z, capture=True):
        """Run 8 Performer layers with optional activation capture."""
        self.activation_cache.clear()
        self.pooled_cache.clear()
        idx_to_name = {v: k for k, v in self.capture_layers.items()}

        for i, layer in enumerate(self.performer):
            z = layer(z)
            if capture and i in idx_to_name:
                name = idx_to_name[i]
                self.activation_cache[name] = z                    # (B, S, 1024)
                self.pooled_cache[name]     = z.mean(dim=1)       # (B, 1024)

        z = self.performer_ln(z)
        return z

    def _output_head(self, z):
        """Chromatin output head."""
        # z: (B, S, 1024) → (B, 1024, S) for BN/Conv1d
        h = z.permute(0, 2, 1)           # (B, 1024, S)
        h = self.out_bn(h)
        h = F.gelu(h)
        tracks = F.softplus(self.out_conv(h))   # (B, 170, S)
        tracks = tracks.permute(0, 2, 1)         # (B, S, 170)
        profile = self.out_dense(tracks)          # (B, S, 1)
        return {'tracks': tracks, 'profile': profile}

    def forward(self, inputs, capture=True):
        seq, atac, motifs = inputs
        z = self._encode(seq, atac, motifs)
        z = self._transformer(z, capture=capture)
        return self._output_head(z)

    def forward_from_layer(self, hidden_state, start_layer_idx):
        """
        Run layers [start_layer_idx+1 .. 7] + output head.
        hidden_state: (B, S, 1024) torch.Tensor
        """
        z = hidden_state
        for i in range(start_layer_idx + 1, len(self.performer)):
            z = self.performer[i](z)
        z = self.performer_ln(z)
        return self._output_head(z)

    def get_pooled_activations(self):
        return {k: v.detach().cpu().numpy() for k, v in self.pooled_cache.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Weight loading from TF checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _g(reader, path):
    """Read tensor from TF checkpoint."""
    return reader.get_tensor(path + "/.ATTRIBUTES/VARIABLE_VALUE")


def load_weights_from_tf_checkpoint(model, ckpt_path):
    """
    Load TF checkpoint weights into PyTorch EpiBERTTorch model.

    Handles dimension transpositions:
      TF Conv1D kernel [k, in, out] → PyTorch [out, in, k]
      TF Dense kernel  [in, out]    → PyTorch [out, in]
    """
    import tensorflow as tf
    r = tf.train.load_checkpoint(ckpt_path)
    loaded = 0
    skipped = []

    def assign_conv(param, key):
        """Load TF Conv1D kernel [k, in, out] → PyTorch [out, in, k]."""
        nonlocal loaded
        try:
            t = _g(r, key)  # [k, in_ch, out_ch]
            pt = torch.from_numpy(t.transpose(2, 1, 0).copy())  # [out, in, k]
            param.data.copy_(pt)
            loaded += 1
        except Exception as e:
            skipped.append((key, str(e)))

    def assign_linear(param, key):
        """Load TF Dense kernel [in, out] → PyTorch [out, in]."""
        nonlocal loaded
        try:
            t = _g(r, key)  # [in, out]
            pt = torch.from_numpy(t.T.copy())  # [out, in]
            param.data.copy_(pt)
            loaded += 1
        except Exception as e:
            skipped.append((key, str(e)))

    def assign_linear_raw(param, key):
        """Load TF Dense kernel [in, out] without transposing (already correct shape)."""
        nonlocal loaded
        try:
            t = _g(r, key)
            param.data.copy_(torch.from_numpy(t.T.copy()))
            loaded += 1
        except Exception as e:
            skipped.append((key, str(e)))

    def assign_bias(param, key):
        nonlocal loaded
        try:
            t = _g(r, key)
            param.data.copy_(torch.from_numpy(t.copy()))
            loaded += 1
        except Exception as e:
            skipped.append((key, str(e)))

    def assign_bn(module, prefix):
        """Load TF BatchNorm into PyTorch BatchNorm1d."""
        assign_bias(module.weight,        f"{prefix}/gamma")  # gamma → weight
        assign_bias(module.bias,          f"{prefix}/beta")
        assign_bias(module.running_mean,  f"{prefix}/moving_mean")
        assign_bias(module.running_var,   f"{prefix}/moving_variance")

    def assign_ln(module, prefix):
        """Load TF LayerNorm into PyTorch LayerNorm."""
        assign_bias(module.weight, f"{prefix}/gamma")
        assign_bias(module.bias,   f"{prefix}/beta")

    # ── DNA stem ──────────────────────────────────────────────────────────────
    assign_conv(model.stem_conv.weight,  "model/stem_conv/kernel")
    assign_bias(model.stem_conv.bias,    "model/stem_conv/bias")

    assign_bn(model.stem_res_conv.bn, "model/stem_res_conv/_layer/layer_with_weights-0")
    assign_conv(model.stem_res_conv.conv.weight, "model/stem_res_conv/_layer/layer_with_weights-1/kernel")
    assign_bias(model.stem_res_conv.conv.bias,   "model/stem_res_conv/_layer/layer_with_weights-1/bias")

    assign_linear(model.stem_pool.score.weight, "model/stem_pool/dense/kernel")

    # ── DNA conv tower ─────────────────────────────────────────────────────────
    for i, blk in enumerate(model.conv_tower):
        p = f"model/conv_tower/layer_with_weights-{i}"
        assign_bn(blk.bn,    f"{p}/layer_with_weights-0/layer_with_weights-0")
        assign_conv(blk.conv.weight, f"{p}/layer_with_weights-0/layer_with_weights-1/kernel")
        assign_bias(blk.conv.bias,   f"{p}/layer_with_weights-0/layer_with_weights-1/bias")
        if blk.use_attn_pool:
            assign_linear(blk.pool.score.weight, f"{p}/layer_with_weights-1/dense/kernel")

    # ── ATAC stem ──────────────────────────────────────────────────────────────
    assign_conv(model.stem_conv_atac.weight, "model/stem_conv_atac/kernel")
    assign_bias(model.stem_conv_atac.bias,   "model/stem_conv_atac/bias")
    assign_bn(model.stem_res_conv_atac.bn, "model/stem_res_conv_atac/_layer/layer_with_weights-0")
    assign_conv(model.stem_res_conv_atac.conv.weight, "model/stem_res_conv_atac/_layer/layer_with_weights-1/kernel")
    assign_bias(model.stem_res_conv_atac.conv.bias,   "model/stem_res_conv_atac/_layer/layer_with_weights-1/bias")

    # ── ATAC conv tower ─────────────────────────────────────────────────────────
    for i, blk in enumerate(model.conv_tower_atac):
        p = f"model/conv_tower_atac/layer_with_weights-{i}/layer_with_weights-0"
        assign_bn(blk.bn,    f"{p}/layer_with_weights-0")
        assign_conv(blk.conv.weight, f"{p}/layer_with_weights-1/kernel")
        assign_bias(blk.conv.bias,   f"{p}/layer_with_weights-1/bias")

    # ── Motif activity ──────────────────────────────────────────────────────────
    assign_linear(model.motif_fc1.weight, "model/motif_activity_fc1/kernel")
    assign_bias(model.motif_fc1.bias,     "model/motif_activity_fc1/bias")
    assign_linear(model.motif_fc2.weight, "model/motif_activity_fc2/kernel")
    assign_bias(model.motif_fc2.bias,     "model/motif_activity_fc2/bias")

    # ── Pre-transformer projection (no bias) ────────────────────────────────────
    assign_linear(model.projection.weight, "model/pre_transformer_projection/kernel")

    # ── Performer layers ──────────────────────────────────────────────────────
    for i, layer in enumerate(model.performer):
        p = f"model/performer/layers/{i}"
        assign_ln(layer.layer_norm, f"{p}/layer_norm")

        # Q/K/V: TF stores [d_model, heads, head_dim] → reshape to [d_model, heads*head_dim] → transpose → [heads*head_dim, d_model]
        for dense, key in [
            (layer.q_dense, f"{p}/self_attention/query_dense_layer/kernel"),
            (layer.k_dense, f"{p}/self_attention/key_dense_layer/kernel"),
            (layer.v_dense, f"{p}/self_attention/value_dense_layer/kernel"),
        ]:
            try:
                t = _g(r, key)   # [1024, 8, 128]
                t2d = t.reshape(t.shape[0], -1)   # [1024, 1024]
                # TF Dense: [in, out], PyTorch Linear: [out, in]
                dense.weight.data.copy_(torch.from_numpy(t2d.T.copy()))
                loaded += 1
            except Exception as e:
                skipped.append((key, str(e)))

        # Output: TF [heads, head_dim, d_model] → [1024, 1024]
        try:
            t = _g(r, f"{p}/self_attention/output_dense_layer/kernel")   # [8, 128, 1024]
            t2d = t.reshape(-1, t.shape[-1])  # [1024, 1024]
            # TF Dense: [in=1024, out=1024], PyTorch Linear: [out, in]
            layer.o_dense.weight.data.copy_(torch.from_numpy(t2d.T.copy()))
            loaded += 1
        except Exception as e:
            skipped.append((f"{p}/self_attention/output_dense_layer/kernel", str(e)))

        # FFN
        assign_linear(layer.FFN_dense_wide.weight,   f"{p}/FFN/FFN_dense_wide/kernel")
        assign_bias(layer.FFN_dense_wide.bias,        f"{p}/FFN/FFN_dense_wide/bias")
        assign_ln(layer.FFN_layer_norm, f"{p}/FFN/FFN_layer_norm")
        assign_linear(layer.FFN_dense_narrow.weight, f"{p}/FFN/FFN_dense_narrow/kernel")
        assign_bias(layer.FFN_dense_narrow.bias,      f"{p}/FFN/FFN_dense_narrow/bias")

    # ── Performer final LayerNorm ───────────────────────────────────────────
    assign_ln(model.performer_ln, "model/performer/layer_norm")

    # ── Output head ────────────────────────────────────────────────────────
    assign_bn(model.out_bn, "model/final_pointwise_conv/layer_with_weights-0")
    assign_conv(model.out_conv.weight, "model/final_pointwise_conv/layer_with_weights-1/kernel")
    assign_bias(model.out_conv.bias,   "model/final_pointwise_conv/layer_with_weights-1/bias")
    assign_linear(model.out_dense.weight, "model/final_dense_profile/kernel")
    assign_bias(model.out_dense.bias,     "model/final_dense_profile/bias")

    return loaded, skipped


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_CACHE = {}

def get_model(ckpt_path=None, force_reload=False):
    global _MODEL_CACHE
    if 'default' in _MODEL_CACHE and not force_reload:
        return _MODEL_CACHE['default']

    if ckpt_path is None:
        ckpt_path = cfg('paths', 'checkpoint')

    hook_cfg = cfg('model', 'hook_layers')
    capture_layers = {name: idx for name, idx in hook_cfg.items()}

    log.info(f"Building EpiBERTTorch (device={DEVICE}) ...")
    model = EpiBERTTorch(
        num_tracks=cfg('model', 'num_tracks'),
        capture_layers=capture_layers,
    )

    log.info(f"Loading weights from TF checkpoint: {ckpt_path}")
    loaded, skipped = load_weights_from_tf_checkpoint(model, ckpt_path)
    log.info(f"  Loaded: {loaded} tensors, skipped: {len(skipped)}")
    if skipped:
        log.warning(f"  Skipped keys: {[k for k, _ in skipped[:5]]}")

    model = model.to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Parameters: {total_params:,}")

    _MODEL_CACHE['default'] = model
    return model


def verify_model_sanity(model, seq_len=4096):
    """Sanity check: shape, no NaN, forward_from_layer identity."""
    log.info("Running PyTorch model sanity check ...")
    torch.manual_seed(42)

    B, L = 1, seq_len
    seq    = torch.zeros(B, L, 4, device=DEVICE); seq[:, :, 0] = 1.0
    atac   = torch.randn(B, L, 1, device=DEVICE).abs() * 0.5
    motifs = torch.zeros(B, 693, device=DEVICE)
    inputs = [seq, atac, motifs]

    with torch.no_grad():
        out = model(inputs, capture=True)

    tracks = out['tracks']
    n_bins = L // cfg('model', 'downsample_factor')
    ok = True

    if tracks.shape != (B, n_bins, cfg('model', 'num_tracks')):
        log.error(f"  [FAIL] tracks shape {tracks.shape} != (1, {n_bins}, 170)")
        ok = False
    else:
        log.info(f"  [PASS] tracks shape: {tracks.shape}")

    if torch.isnan(tracks).any():
        log.error("  [FAIL] NaN in output")
        ok = False
    else:
        log.info(f"  [PASS] No NaN")

    if (tracks < 0).any():
        log.error("  [FAIL] Negative in softplus output")
        ok = False
    else:
        log.info(f"  [PASS] All tracks ≥ 0")

    log.info(f"  tracks: min={tracks.min():.4f} max={tracks.max():.4f} mean={tracks.mean():.4f}")

    # Check activation capture
    for name in cfg('model', 'hook_layers'):
        if name in model.pooled_cache:
            p = model.pooled_cache[name]
            log.info(f"  [PASS] Layer '{name}': pooled shape={p.shape}")
        else:
            log.error(f"  [FAIL] Layer '{name}' not captured")
            ok = False

    # Verify forward_from_layer identity
    hook_cfg = cfg('model', 'hook_layers')
    for layer_name, layer_idx in hook_cfg.items():
        if layer_name not in model.activation_cache:
            continue
        hidden = model.activation_cache[layer_name]  # (B, S, 1024)
        with torch.no_grad():
            partial_out = model.forward_from_layer(hidden, layer_idx)
        diff = (partial_out['tracks'] - tracks).abs().max().item()
        if diff < 1e-3:
            log.info(f"  [PASS] forward_from_layer({layer_name}): max_diff={diff:.2e}")
        else:
            log.warning(f"  [WARN] forward_from_layer({layer_name}): max_diff={diff:.4f}")
            if diff > 0.5:
                log.error(f"  [FAIL] max_diff={diff:.4f} too large!")
                ok = False

    log.info(f"  Sanity check {'PASSED' if ok else 'FAILED'}")
    return ok
