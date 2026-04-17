"""Shared utilities: config loading, logging, paths, seeding."""

import os
import sys
import logging
import random
import numpy as np
import yaml
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG = None

def load_config(path=None):
    global _CONFIG
    if _CONFIG is not None and path is None:
        return _CONFIG
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "main.yaml"
    with open(path) as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


def cfg(*keys):
    """Nested config access: cfg('sae', 'lr') → config['sae']['lr']."""
    c = load_config()
    for k in keys:
        c = c[k]
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed=None):
    if seed is None:
        seed = cfg('seed')
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name, log_file=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def activation_path(pair, condition, layer_name):
    """Path for a cached activation tensor."""
    p = Path(cfg('paths', 'activations')) / pair / condition
    p.mkdir(parents=True, exist_ok=True)
    return str(p / f"{layer_name}.pt")


def sae_path(layer_name, regime):
    """Path for a trained SAE checkpoint."""
    p = Path(cfg('paths', 'saes')) / layer_name
    p.mkdir(parents=True, exist_ok=True)
    return str(p / f"{regime}.pt")


def atac_processed_path(condition):
    p = Path(cfg('paths', 'atac_processed'))
    p.mkdir(parents=True, exist_ok=True)
    return str(p / f"{condition}.npy")


# ─────────────────────────────────────────────────────────────────────────────
# Layer name helpers
# ─────────────────────────────────────────────────────────────────────────────

LAYER_NAMES = ['early', 'mid', 'late']

def layer_idx(layer_name):
    L = cfg('model', 'num_performer_layers')
    mapping = {
        'early': L // 4,
        'mid':   L // 2,
        'late':  3 * L // 4,
    }
    return mapping[layer_name]


# ─────────────────────────────────────────────────────────────────────────────
# Numpy/torch save/load helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_activations(tensor, path):
    """Save activation tensor (numpy array) as torch .pt file."""
    import torch
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    torch.save(tensor.cpu(), path)


def load_activations(path):
    """Load activation tensor as torch tensor."""
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def load_activations_np(path):
    """Load activations as numpy array."""
    return load_activations(path).numpy()
