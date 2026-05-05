"""
src/models/__init__.py
======================
Model adapter registry for the multi-model contrastive SAE pipeline.

Usage:
    from models import get_adapter

    adapter = get_adapter("enformer")
    acts    = adapter.get_activations(windows, atac_arrays)  # dict[layer_name -> np.ndarray]
"""

from .base import ModelAdapter  # noqa: F401

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator: register a ModelAdapter subclass under a canonical name."""
    def _inner(cls):
        _REGISTRY[name.lower()] = cls
        return cls
    return _inner


def get_adapter(name: str, **kwargs) -> "ModelAdapter":
    """
    Instantiate and return a registered model adapter.

    Parameters
    ----------
    name : str
        One of: "epibert", "enformer", "hyenadna", "nucleotide_transformer"
    **kwargs
        Passed to the adapter constructor (e.g. model_path, device).
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _REGISTRY[key](**kwargs)


def list_adapters() -> list[str]:
    """Return all registered adapter names."""
    return sorted(_REGISTRY)


# Trigger registration by importing adapters
def _load_all():
    from . import epibert_adapter          # noqa: F401
    from . import enformer_adapter         # noqa: F401
    from . import hyenadna_adapter         # noqa: F401
    from . import nucleotide_transformer_adapter  # noqa: F401


_load_all()
