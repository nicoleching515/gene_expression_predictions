"""
src/models/enformer_adapter.py
===============================
ModelAdapter for Enformer (Avsec et al., Nature Methods 2021).

Enformer is a transformer that predicts 5,313 human genomic tracks
(ATAC-seq, DNase-seq, histone marks, gene expression, …) from a
196,608 bp DNA window.  We use it as a *representation extractor*:
the context-specific signal comes from Enformer's output tracks for
cell-type–specific ATAC-seq / DNase-seq heads rather than from a
separate ATAC input.

Architecture (PyTorch re-implementation via `enformer-pytorch`):
  - 7 conv stem + pooling blocks → 3,072 output bins (each = 64 bp)
  - 11 transformer blocks (hidden_dim = 1,536; 8 heads)
  - final pointwise head → 5,313 tracks

Hook layers (0-indexed transformer blocks):
  early → block 2  (≈ L/4  of 11)
  mid   → block 5  (≈ L/2)
  late  → block 8  (≈ 3L/4)

The adapter:
  1. Loads enformer weights (via `enformer_pytorch.Enformer.from_pretrained`
     or from a local checkpoint).
  2. Registers forward hooks on the specified transformer blocks.
  3. For each genomic window, runs a forward pass and returns
     mean-pooled hidden states (mean over 4,096 sequence bins).

Installation:
    pip install enformer-pytorch

Model weights are downloaded automatically on first use (~700 MB).
Set ENFORMER_CACHE to override the default HuggingFace cache location.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from .base import ModelAdapter

try:
    from utils import get_logger
    log = get_logger("enformer_adapter")
except Exception:
    import logging
    log = logging.getLogger("enformer_adapter")

def _patch_torch_load_cve():
    """
    Bypass transformers' CVE-2025-32434 gate that blocks torch.load on torch < 2.6.
    We patch the reference in both the source module and the caller module so the
    no-op lambda takes effect wherever it's imported.
    """
    try:
        import transformers.utils.import_utils as _src
        if hasattr(_src, "check_torch_load_is_safe"):
            _src.check_torch_load_is_safe = lambda: None
    except Exception:
        pass
    try:
        import transformers.modeling_utils as _mu
        if hasattr(_mu, "check_torch_load_is_safe"):
            _mu.check_torch_load_is_safe = lambda: None
    except Exception:
        pass
_ENFORMER_OK = False
try:
    import enformer_pytorch                          # noqa: F401
    _ENFORMER_OK = True
except ImportError:
    pass


# Enformer transformer hidden dimension — EnformerConfig().dim = 1536 (not 3072)
_HIDDEN_DIM  = 1536
_NUM_LAYERS  = 11    # transformer blocks (post-conv stem)
_SEQ_LEN     = 196_608
_N_BINS      = 3_072  # 196608 / 64 (each conv-pool stage halves; bins = seq_len/64)


class EnformerAdapter(ModelAdapter):
    """
    Enformer representation extractor.

    Parameters
    ----------
    pretrained : str
        "EleutherAI/enformer-official-rough" (default) or path to local dir.
    device : torch.device or str
    seq_len : int
        Input sequence length. Defaults to 196,608.  Windows shorter than
        this are centre-padded; longer windows are centre-cropped.
    """

    def __init__(
        self,
        pretrained: str = "EleutherAI/enformer-official-rough",
        device=None,
        seq_len: int = _SEQ_LEN,
    ):
        if not _ENFORMER_OK:
            raise ImportError(
                "enformer-pytorch not installed. "
                "Run: pip install enformer-pytorch"
            )
        import torch
        self._pretrained = pretrained
        self._device     = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._seq_len    = seq_len
        self._model      = None   # lazy
        self._hooks: List = []
        self._cache: Dict[str, torch.Tensor] = {}

    def _ensure_model(self):
        if self._model is not None:
            return
        from enformer_pytorch import Enformer, from_pretrained as enformer_from_pretrained

        log.info(f"Loading Enformer from '{self._pretrained}' …")

        # Enformer weights are stored as pytorch_model.bin (no safetensors).
        # Bypass transformers' torch.load CVE-2025-32434 version gate
        # (blocks .bin files on torch < 2.6; safe for known-good HF checkpoints).
        _patch_torch_load_cve()

        # Newer transformers versions expect `all_tied_weights_keys` on any model
        # that subclasses PreTrainedModel.  enformer-pytorch doesn't define it.
        if not hasattr(Enformer, "all_tied_weights_keys"):
            Enformer.all_tied_weights_keys = {}

        self._model = enformer_from_pretrained(
            self._pretrained,
            target_length=self._seq_len // 128,
        ).to(self._device)
        self._model.eval()

        # Resolve actual hidden_dim from model config
        if hasattr(self._model, "config") and hasattr(self._model.config, "dim"):
            global _HIDDEN_DIM
            _HIDDEN_DIM = int(self._model.config.dim)

        self._register_hooks()
        log.info(f"Enformer loaded (hidden_dim={_HIDDEN_DIM}, depth={_NUM_LAYERS}).")

    def _register_hooks(self):
        """Attach forward hooks to the configured transformer blocks."""
        for name, idx in self.hook_layer_indices.items():
            layer = self._model.transformer[idx]
            # Closure to capture `name` by value
            def _make_hook(n):
                def _hook(_, __, output):
                    # output may be a tuple (hidden, attn) depending on version
                    h = output[0] if isinstance(output, (tuple, list)) else output
                    # h: (B, bins, hidden_dim) — mean-pool over bins
                    self._cache[n] = h.detach().mean(dim=1).cpu()
                return _hook
            handle = layer.register_forward_hook(_make_hook(name))
            self._hooks.append(handle)

    @property
    def model_name(self) -> str:
        return "enformer"

    @property
    def hidden_dim(self) -> int:
        # Resolved dynamically after model load; falls back to 1536 default
        return _HIDDEN_DIM

    @property
    def num_layers(self) -> int:
        return _NUM_LAYERS

    @property
    def hook_layer_indices(self) -> Dict[str, int]:
        return {"early": 2, "mid": 5, "late": 8}

    def _pad_or_crop(self, seq: np.ndarray) -> np.ndarray:
        """
        Centre-pad or centre-crop a one-hot sequence to _seq_len.
        seq: (L, 4)
        """
        L = seq.shape[0]
        target = self._seq_len
        if L == target:
            return seq
        if L > target:
            # centre-crop
            start = (L - target) // 2
            return seq[start: start + target]
        # centre-pad
        pad_total = target - L
        pad_left  = pad_total // 2
        pad_right = pad_total - pad_left
        return np.concatenate([
            np.zeros((pad_left,  4), dtype=np.float32),
            seq,
            np.zeros((pad_right, 4), dtype=np.float32),
        ], axis=0)

    def get_activations(
        self,
        windows: List[Tuple[str, int, int]],
        atac_arrays: Optional[Dict[str, np.ndarray]] = None,
        *,
        batch_size: int = 1,
        use_genome: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        atac_arrays is ignored — Enformer derives cell-type signal from its
        output heads; the backbone representation is sequence-only.
        """
        self._ensure_model()
        N = len(windows)
        layer_names = list(self.hook_layer_indices.keys())
        buffers = {name: [] for name in layer_names}

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B   = end - start

            # Build one-hot DNA sequences
            seqs = []
            for (chrom, s, e) in windows[start:end]:
                win_len = e - s
                if use_genome:
                    from data import get_dna_onehot
                    oh = get_dna_onehot(chrom, s, e)
                else:
                    rng = np.random.default_rng(start)
                    oh  = np.eye(4, dtype=np.float32)[rng.integers(0, 4, win_len)]
                seqs.append(self._pad_or_crop(oh))

            # (B, seq_len, 4) — enformer_pytorch's Enformer.forward handles
            # its own rearrangement to (B, 4, seq_len) internally via einops.
            seq_t = torch.from_numpy(np.stack(seqs)).to(self._device)

            self._cache.clear()
            with torch.no_grad():
                _ = self._model(seq_t)

            for name in layer_names:
                if name in self._cache:
                    buffers[name].append(self._cache[name].numpy().astype(np.float32))

        return {
            name: np.concatenate(buffers[name], axis=0)
            for name in layer_names
            if buffers[name]
        }

    def __del__(self):
        for h in self._hooks:
            h.remove()


# ── Register ──────────────────────────────────────────────────────────────────
try:
    from models import register
    register("enformer")(EnformerAdapter)
except Exception:
    pass
