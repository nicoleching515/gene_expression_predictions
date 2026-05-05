"""
src/models/base.py
==================
Abstract base class for all sequence-model adapters in the
multi-model contrastive SAE pipeline.

Every adapter must implement:
    model_name     → str
    hidden_dim     → int  (dimension of hidden states returned)
    num_layers     → int  (total transformer/operator depth)
    hook_layer_indices → dict[str, int]  {"early": i, "mid": j, "late": k}
    get_activations(windows, atac_arrays, **kw) → dict[str, np.ndarray]

The get_activations() method runs a forward pass over the supplied genomic
windows and returns mean-pooled hidden states for each hook layer, shaped
(N_windows, hidden_dim).
"""

from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple

import numpy as np


class ModelAdapter(abc.ABC):
    """
    Abstract base class.

    Sub-classes must decorate themselves with @models.register("name") to
    appear in the adapter registry.
    """

    # ── Properties every adapter must expose ──────────────────────────────────

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Canonical short name, e.g. 'epibert', 'enformer'."""
        ...

    @property
    @abc.abstractmethod
    def hidden_dim(self) -> int:
        """Dimension of hidden vectors returned by this adapter."""
        ...

    @property
    @abc.abstractmethod
    def num_layers(self) -> int:
        """Total number of transformer / operator layers in the model."""
        ...

    @property
    def hook_layer_indices(self) -> Dict[str, int]:
        """
        Default early/mid/late hook positions (0-indexed).
        Override if non-standard positions are needed.
        """
        L = self.num_layers
        return {
            "early": L // 4,
            "mid":   L // 2,
            "late":  3 * L // 4,
        }

    @property
    def layer_names(self) -> List[str]:
        return list(self.hook_layer_indices.keys())

    # ── Core interface ─────────────────────────────────────────────────────────

    @abc.abstractmethod
    def get_activations(
        self,
        windows: List[Tuple[str, int, int]],
        atac_arrays: Optional[Dict[str, np.ndarray]] = None,
        *,
        batch_size: int = 4,
        use_genome: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Forward pass over genomic windows; return pooled hidden states.

        Parameters
        ----------
        windows : list of (chrom, start, end)
            Genomic coordinates. Length = N.
        atac_arrays : dict[condition_name -> np.ndarray (N, window_bp)]
            Cell-type-specific ATAC signal. Required for models that take
            epigenomic input (e.g. EpiBERT). Pass None for sequence-only models.
        batch_size : int
        use_genome : bool
            If True, load real DNA from hg38.fa; otherwise use synthetic random
            one-hot sequences (same for all conditions → cancels in CDS).

        Returns
        -------
        dict[layer_name -> np.ndarray (N, hidden_dim)]
        """
        ...

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name}, "
            f"hidden_dim={self.hidden_dim}, "
            f"layers={self.num_layers}, "
            f"hooks={self.hook_layer_indices})"
        )

    @staticmethod
    def mean_pool(tensor: np.ndarray) -> np.ndarray:
        """
        Mean-pool a (batch, seq_len, hidden) tensor over the sequence axis.
        Also handles (batch, hidden) tensors (no-op).
        """
        if tensor.ndim == 3:
            return tensor.mean(axis=1)
        return tensor

    @staticmethod
    def make_random_dna_onehot(
        batch_size: int, seq_len: int, seed: int = 0
    ) -> np.ndarray:
        """Random one-hot DNA (batch, seq_len, 4)."""
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, 4, (batch_size, seq_len))
        return np.eye(4, dtype=np.float32)[idx]
