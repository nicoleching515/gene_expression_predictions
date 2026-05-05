"""
src/models/epibert_adapter.py
==============================
ModelAdapter wrapping the existing EpiBERT PyTorch implementation.

EpiBERT takes three inputs per genomic window:
    seq   : (B, window_bp, 4)  one-hot DNA
    atac  : (B, window_bp, 1)  RPM-normalised, smoothed, log1p ATAC coverage
    motif : (B, 693)           JASPAR motif scores (zeros when DB unavailable)

It outputs chromatin-accessibility predictions and exposes intermediate
transformer block activations through model.pooled_cache.

Hook layers follow the existing config convention (early=L//4, mid=L//2,
late=3L//4) with L=8 performer layers, hidden_dim=1024.
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
    from utils import get_logger, cfg
    log = get_logger("epibert_adapter")
except Exception:
    import logging
    log = logging.getLogger("epibert_adapter")

try:
    from model_torch import get_model, DEVICE as _EPIBERT_DEVICE
    _EPIBERT_OK = True
except ImportError:
    _EPIBERT_DEVICE = torch.device("cpu")
    _EPIBERT_OK = False

try:
    import pysam
    _PYSAM_OK = True
except ImportError:
    _PYSAM_OK = False


class EpiBERTAdapter(ModelAdapter):
    """Adapter for EpiBERT (existing PyTorch implementation)."""

    def __init__(self, device=None):
        self._device = torch.device(device or (_EPIBERT_DEVICE if _EPIBERT_OK else "cpu"))
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            if not _EPIBERT_OK:
                raise ImportError("EpiBERT model_torch not available.")
            self._model = get_model().to(self._device)
            self._model.eval()

    @property
    def model_name(self) -> str:
        return "epibert"

    @property
    def hidden_dim(self) -> int:
        try:
            return int(cfg("model", "hidden_dim"))
        except Exception:
            return 1024

    @property
    def num_layers(self) -> int:
        try:
            return int(cfg("model", "num_performer_layers"))
        except Exception:
            return 8

    @property
    def hook_layer_indices(self) -> Dict[str, int]:
        try:
            return dict(cfg("model", "hook_layers"))
        except Exception:
            return {"early": 2, "mid": 4, "late": 6}

    def get_activations(
        self,
        windows: List[Tuple[str, int, int]],
        atac_arrays: Optional[Dict[str, np.ndarray]] = None,
        *,
        batch_size: int = 4,
        use_genome: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        atac_arrays: dict mapping any key to (N, window_bp) float32 array.
            Exactly one key is expected (the condition being processed).
            If None or empty, zeros are used.
        """
        self._ensure_model()

        N         = len(windows)
        window_bp = windows[0][2] - windows[0][1]
        layer_names = list(self.hook_layer_indices.keys())
        try:
            num_motifs = int(cfg("model", "num_motifs"))
        except Exception:
            num_motifs = 693

        # Build ATAC array (N, window_bp, 1)
        if atac_arrays:
            atac_key  = next(iter(atac_arrays))
            atac_full = atac_arrays[atac_key].astype(np.float32)  # (N, window_bp)
        else:
            atac_full = np.zeros((N, window_bp), dtype=np.float32)

        buffers = {name: [] for name in layer_names}

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B   = end - start

            # DNA
            if use_genome:
                from data import get_dna_onehot
                seq_np = np.stack([
                    get_dna_onehot(c, s, e) for c, s, e in windows[start:end]
                ])
            else:
                seq_np = self.make_random_dna_onehot(B, window_bp, seed=start)

            atac_np  = atac_full[start:end, :, np.newaxis]   # (B, L, 1)
            motif_np = np.zeros((B, num_motifs), dtype=np.float32)

            seq_t   = torch.from_numpy(seq_np).to(self._device)
            atac_t  = torch.from_numpy(atac_np).to(self._device)
            motif_t = torch.from_numpy(motif_np).to(self._device)

            with torch.no_grad():
                _ = self._model([seq_t, atac_t, motif_t], capture=True)

            for name in layer_names:
                v = self._model.pooled_cache.get(name)
                if v is not None:
                    buffers[name].append(v.cpu().numpy().astype(np.float32))

        return {
            name: np.concatenate(buffers[name], axis=0)
            for name in layer_names
            if buffers[name]
        }


# ── Register ──────────────────────────────────────────────────────────────────
try:
    from models import register
    register("epibert")(EpiBERTAdapter)
except Exception:
    pass  # standalone import
