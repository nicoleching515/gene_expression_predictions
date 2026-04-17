"""
Activation capture utilities.

In TensorFlow/Keras, we capture activations by storing them in the model's
activation_cache during each forward call (see model.py EpiBERTModel._transformer).

This module provides helpers for collecting, stacking, and saving activations
across many forward passes (the main activation collection loop).
"""

import os
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, cfg, activation_path, save_activations, load_activations

log = get_logger("hooks")


class ActivationCollector:
    """
    Collects mean-pooled activations across batches for one (pair, condition).

    Usage:
        collector = ActivationCollector(pair='blood', condition='K562')
        for batch in dataset:
            model(batch)  # fills model.pooled_cache
            collector.add(model.pooled_cache)
        collector.save()
    """

    def __init__(self, pair, condition, layer_names=None):
        self.pair      = pair
        self.condition = condition
        self.layer_names = layer_names or list(cfg('model', 'hook_layers').keys())
        self.buffers   = {name: [] for name in self.layer_names}
        self.n_added   = 0

    def add(self, pooled_cache):
        """
        Add activations from one batch.
        pooled_cache: dict of {layer_name: tf.Tensor (batch, 1024)} or numpy.
        """
        for name in self.layer_names:
            if name not in pooled_cache:
                log.warning(f"Layer '{name}' not in pooled_cache")
                continue
            t = pooled_cache[name]
            if hasattr(t, 'numpy'):
                t = t.numpy()
            self.buffers[name].append(t.astype(np.float32))
        self.n_added += 1

    def save(self, checkpoint=False):
        """
        Concatenate and save activations to .pt files.
        Shape: (n_windows, 1024).
        """
        for name in self.layer_names:
            if not self.buffers[name]:
                log.warning(f"No data for layer '{name}' in {self.pair}/{self.condition}")
                continue
            arr = np.concatenate(self.buffers[name], axis=0)  # (N, 1024)
            path = activation_path(self.pair, self.condition, name)
            suffix = "_partial" if checkpoint else ""
            path_to_save = path.replace('.pt', f'{suffix}.pt')
            save_activations(arr, path_to_save)
            log.info(f"  Saved activations [{name}]: {arr.shape} → {path_to_save}")

    def save_checkpoint(self):
        """Save partial activations (for crash recovery)."""
        self.save(checkpoint=True)

    def n_windows(self):
        if self.buffers[self.layer_names[0]]:
            return sum(b.shape[0] for b in self.buffers[self.layer_names[0]])
        return 0

    def finalize(self):
        """Stack all buffers into numpy arrays and clear raw buffers."""
        result = {}
        for name in self.layer_names:
            if self.buffers[name]:
                result[name] = np.concatenate(self.buffers[name], axis=0)
        self.buffers = {name: [] for name in self.layer_names}
        return result


def load_all_activations(layer_name):
    """
    Load activations for all 6 conditions × 1 layer.
    Returns dict: {pair: {condition: np.array (n_windows, d)}}
    """
    pairs_cfg = cfg('pairs')
    result = {}
    for pair_name, pair_conds in pairs_cfg.items():
        result[pair_name] = {}
        for regime, condition in pair_conds.items():
            path = activation_path(pair_name, condition, layer_name)
            if os.path.isfile(path):
                arr = load_activations(path).numpy()
                result[pair_name][regime] = arr
                log.info(f"  Loaded {pair_name}/{condition}/{layer_name}: {arr.shape}")
            else:
                log.warning(f"  Missing: {path}")
                result[pair_name][regime] = None
    return result


def get_activation_stats(activations_dict):
    """Compute basic statistics for a set of activations."""
    stats = {}
    for key, arr in activations_dict.items():
        if arr is None:
            continue
        stats[key] = {
            'shape': arr.shape,
            'mean': float(np.mean(arr)),
            'std':  float(np.std(arr)),
            'min':  float(np.min(arr)),
            'max':  float(np.max(arr)),
            'frac_zero': float(np.mean(arr == 0)),
        }
    return stats
