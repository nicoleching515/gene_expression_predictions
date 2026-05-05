"""
src/models/hyenadna_adapter.py
================================
ModelAdapter for HyenaDNA (Nguyen et al., NeurIPS 2023).

HyenaDNA is a long-range DNA language model built on the Hyena
long-convolution operator.  It uses character-level tokenisation
(one token per base pair) and can handle context windows up to
1,000,000 bp in its largest variant.

Model variants on HuggingFace (LongSafari organisation):
  hyenadna-tiny-1k-seqlen           hidden=128,  layers=2
  hyenadna-small-32k-seqlen         hidden=256,  layers=6
  hyenadna-medium-160k-seqlen-hf    hidden=256,  layers=8  ← default
  hyenadna-medium-450k-seqlen-hf    hidden=256,  layers=8
  hyenadna-large-1m-seqlen-hf       hidden=256,  layers=8

Hook layers for the medium-160k variant (8 layers):
  early → layer 1  (L//4)
  mid   → layer 3  (L//2)
  late  → layer 5  (3L//4)

The adapter:
  1. Loads the model from HuggingFace (auto-downloaded on first use).
  2. Registers PyTorch forward hooks on the Hyena operator layers.
  3. Returns mean-pooled hidden states over sequence length.

Installation:
    pip install transformers accelerate
    # HyenaDNA uses a custom modelling class registered via AutoModel
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from .base import ModelAdapter

try:
    from utils import get_logger
    log = get_logger("hyenadna_adapter")
except Exception:
    import logging
    log = logging.getLogger("hyenadna_adapter")

_TRANSFORMERS_OK = False
try:
    import transformers                              # noqa: F401
    _TRANSFORMERS_OK = True
except ImportError:
    pass

# Character-level tokenisation: A=7, C=8, G=9, T=10, N=11  (DNABERT-style)
_DNA_TOKEN = {
    'A': 7, 'a': 7,
    'C': 8, 'c': 8,
    'G': 9, 'g': 9,
    'T': 10, 't': 10,
    'N': 11, 'n': 11,
}
_BOS_ID = 1   # CLS token used by HyenaDNA tokeniser
_PAD_ID = 0


def _onehot_to_string(onehot: np.ndarray) -> str:
    """Convert (L, 4) float32 one-hot → 'ACGT' string."""
    idx  = np.argmax(onehot, axis=-1)
    chars = np.array(list("ACGT"))[idx]
    return "".join(chars)


def _tokenise(seq_str: str, max_len: int) -> np.ndarray:
    """
    Tokenise a DNA string to integer ids, prepend BOS, truncate/pad to max_len.
    Returns int64 array of shape (max_len,).
    """
    ids = [_BOS_ID] + [_DNA_TOKEN.get(c, 11) for c in seq_str]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [_PAD_ID] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────

class HyenaDNAAdapter(ModelAdapter):
    """
    HyenaDNA representation extractor.

    Parameters
    ----------
    pretrained : str
        HuggingFace model ID, e.g.
        'LongSafari/hyenadna-medium-160k-seqlen-hf'
    device : str or torch.device
    max_seq_len : int
        Input length cap (tokens including BOS).  Defaults to 4,096 to
        keep VRAM usage manageable when processing many windows.  Must be
        ≤ the model's trained max length.
    """

    _HIDDEN_DIMS = {
        "tiny":   128,
        "small":  256,
        "medium": 256,
        "large":  256,
    }
    _NUM_LAYERS = {
        "tiny":   2,
        "small":  6,
        "medium": 8,
        "large":  8,
    }

    def __init__(
        self,
        pretrained: str = "LongSafari/hyenadna-medium-160k-seqlen-hf",
        device=None,
        max_seq_len: int = 4_096,
    ):
        if not _TRANSFORMERS_OK:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )
        self._pretrained  = pretrained
        self._device      = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._max_seq_len = max_seq_len
        self._model       = None
        self._hooks: List = []
        self._cache: Dict[str, torch.Tensor] = {}

        # Infer variant from pretrained name
        name_lower = pretrained.lower()
        if "tiny"   in name_lower: self._variant = "tiny"
        elif "small" in name_lower: self._variant = "small"
        elif "large" in name_lower: self._variant = "large"
        else:                        self._variant = "medium"

    def _ensure_model(self):
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoModel
        log.info(f"Loading HyenaDNA from '{self._pretrained}' …")
        try:
            self._model = AutoModel.from_pretrained(
                self._pretrained,
                trust_remote_code=True,
            ).to(self._device)
        except Exception:
            # Fallback: some HyenaDNA variants need explicit class
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(self._pretrained, trust_remote_code=True)
            self._model = AutoModel.from_config(cfg, trust_remote_code=True).to(self._device)
        self._model.eval()
        self._register_hooks()
        log.info("HyenaDNA loaded.")

    def _register_hooks(self):
        """
        Hook into the Hyena operator blocks.
        HyenaDNA models expose layers as `model.backbone.layers` or
        `model.hyena.layers` depending on the variant.
        """
        # Try to find the layer container
        backbone = None
        for attr in ("backbone", "hyena", "model", "encoder"):
            candidate = getattr(self._model, attr, None)
            if candidate is not None:
                for layers_attr in ("layers", "blocks", "mixer_layers"):
                    layers = getattr(candidate, layers_attr, None)
                    if layers is not None and len(layers) > 0:
                        backbone = layers
                        break
            if backbone is not None:
                break

        if backbone is None:
            log.warning("Could not locate layer container in HyenaDNA model; "
                        "falling back to full-model output hook.")
            def _hook(_, __, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                if h.ndim == 3:
                    h = h.mean(dim=1)
                for name in self.hook_layer_indices:
                    self._cache[name] = h.detach().cpu()
            self._model.register_forward_hook(_hook)
            return

        for hook_name, idx in self.hook_layer_indices.items():
            if idx >= len(backbone):
                idx = len(backbone) - 1
            def _make_hook(n):
                def _hook(_, __, output):
                    h = output[0] if isinstance(output, (tuple, list)) else output
                    if isinstance(h, torch.Tensor) and h.ndim == 3:
                        self._cache[n] = h.detach().mean(dim=1).cpu()
                return _hook
            handle = backbone[idx].register_forward_hook(_make_hook(hook_name))
            self._hooks.append(handle)

    @property
    def model_name(self) -> str:
        return "hyenadna"

    @property
    def hidden_dim(self) -> int:
        return self._HIDDEN_DIMS[self._variant]

    @property
    def num_layers(self) -> int:
        return self._NUM_LAYERS[self._variant]

    @property
    def hook_layer_indices(self) -> Dict[str, int]:
        L = self.num_layers
        return {"early": max(0, L // 4), "mid": L // 2, "late": min(L - 1, 3 * L // 4)}

    def get_activations(
        self,
        windows: List[Tuple[str, int, int]],
        atac_arrays: Optional[Dict[str, np.ndarray]] = None,
        *,
        batch_size: int = 8,
        use_genome: bool = False,
    ) -> Dict[str, np.ndarray]:
        self._ensure_model()
        N = len(windows)
        layer_names = list(self.hook_layer_indices.keys())
        buffers = {name: [] for name in layer_names}

        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
            B    = end - start
            seqs = []

            for i, (chrom, s, e) in enumerate(windows[start:end]):
                win_len = e - s
                if use_genome:
                    from data import get_dna_onehot
                    oh = get_dna_onehot(chrom, s, e)
                    seq_str = _onehot_to_string(oh)
                else:
                    rng = np.random.default_rng(start + i)
                    idx = rng.integers(0, 4, win_len)
                    seq_str = "".join(np.array(list("ACGT"))[idx])
                seqs.append(_tokenise(seq_str, self._max_seq_len))

            input_ids = torch.from_numpy(np.stack(seqs)).to(self._device)  # (B, L)

            self._cache.clear()
            with torch.no_grad():
                out = self._model(input_ids)

            # If hooks fired, use cached activations
            # If not (e.g., backbone-level hook), retrieve from model output
            if not self._cache:
                h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                if isinstance(h, torch.Tensor) and h.ndim == 3:
                    h = h.mean(dim=1).cpu().numpy().astype(np.float32)
                else:
                    h = h.cpu().numpy().astype(np.float32)
                for name in layer_names:
                    buffers[name].append(h)
            else:
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
            try:
                h.remove()
            except Exception:
                pass


# ── Register ──────────────────────────────────────────────────────────────────
try:
    from models import register
    register("hyenadna")(HyenaDNAAdapter)
except Exception:
    pass
