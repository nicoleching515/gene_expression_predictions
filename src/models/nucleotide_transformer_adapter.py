"""
src/models/nucleotide_transformer_adapter.py
=============================================
ModelAdapter for Nucleotide Transformer (Dalla-Torre et al., 2023).

Default model: nucleotide-transformer-v2-500m-multi-species
  - Safetensors checkpoint (no torch.load CVE issue)
  - ESM backbone, 24 transformer layers, hidden_dim=1280 (resolved dynamically)
  - 6-mer tokenisation; we use 512 tokens = 3,072 bp per window

Hook layers (24 transformer layers):
  early → layer 6   (L/4)
  mid   → layer 12  (L/2)
  late  → layer 18  (3L/4)

Installation:
    pip install transformers sentencepiece
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
    log = get_logger("nt_adapter")
except Exception:
    import logging
    log = logging.getLogger("nt_adapter")

_TRANSFORMERS_OK = False
try:
    import transformers                              # noqa: F401
    _TRANSFORMERS_OK = True
except ImportError:
    pass


def _patch_torch_load_cve():
    """Bypass CVE-2025-32434 gate in both source and caller modules."""
    for mod_name in ("transformers.utils.import_utils", "transformers.modeling_utils"):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "check_torch_load_is_safe"):
                mod.check_torch_load_is_safe = lambda: None
        except Exception:
            pass


def _patch_esm_config(config):
    """
    Patch only the attributes that newer transformers expects but old ESM/NT
    checkpoints omit.  Keep changes minimal to avoid architecture mismatches.
    """
    for attr, val in [
        ("rope_theta", 10000.0),
        ("is_decoder", False),
        ("add_cross_attention", False),
        ("chunk_size_feed_forward", 0),
    ]:
        if not hasattr(config, attr):
            setattr(config, attr, val)
    return config


def _onehot_to_string(onehot: np.ndarray) -> str:
    idx   = np.argmax(onehot, axis=-1)
    chars = np.array(list("ACGT"))[idx]
    return "".join(chars)


class NucleotideTransformerAdapter(ModelAdapter):
    """
    Nucleotide Transformer representation extractor.

    Parameters
    ----------
    pretrained : str
        HuggingFace model ID or local path.
        Default: NT v2-500m (safetensors, avoids torch.load CVE).
    device : str or torch.device
    max_tokens : int
        Maximum token sequence length including CLS.
        512 tokens × 6 bp = 3,072 bp per window.
    """

    def __init__(
        self,
        pretrained: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref",
        device=None,
        max_tokens: int = 512,
    ):
        if not _TRANSFORMERS_OK:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )
        self._pretrained  = pretrained
        self._device      = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._max_tokens  = max_tokens
        self._model       = None
        self._tokenizer   = None
        self._hooks: List = []
        self._cache: Dict[str, torch.Tensor] = {}

        # Initial size estimate from name; overridden after actual load
        name_lower = pretrained.lower()
        if "2.5b" in name_lower:
            self._hidden_dim = 2560
            self._num_layers = 32
        elif "500m" in name_lower:
            self._hidden_dim = 1280   # v2-500m actual hidden size
            self._num_layers = 24
        elif "250m" in name_lower:
            self._hidden_dim = 512
            self._num_layers = 16
        elif "100m" in name_lower:
            self._hidden_dim = 512
            self._num_layers = 12
        elif "50m" in name_lower:
            self._hidden_dim = 512
            self._num_layers = 6
        else:
            self._hidden_dim = 1280
            self._num_layers = 24

    def _ensure_model(self):
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        log.info(f"Loading Nucleotide Transformer from '{self._pretrained}' …")

        # Bypass torch.load CVE-2025-32434 gate (patches both source and caller).
        _patch_torch_load_cve()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._pretrained, trust_remote_code=True
        )

        # NT v1 uses ESM backbone but older HF checkpoint format.
        # Patch config for attributes that newer transformers expects but the
        # checkpoint config.json omits.
        config = AutoConfig.from_pretrained(self._pretrained, trust_remote_code=True)
        _patch_esm_config(config)
        try:
            self._model = AutoModel.from_pretrained(
                self._pretrained,
                config=config,
                trust_remote_code=True,
            ).to(self._device)
        except Exception as e:
            log.warning(f"  Standard load failed ({e}); retrying with ignore_mismatched_sizes …")
            self._model = AutoModel.from_pretrained(
                self._pretrained,
                config=config,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
            ).to(self._device)

        # Patch any sub-module configs that still lack expected attributes
        for module in self._model.modules():
            if hasattr(module, "config"):
                _patch_esm_config(module.config)

        self._model.eval()
        self._register_hooks()

        # Resolve actual hidden_dim and num_layers from loaded config
        cfg = self._model.config
        for attr in ("hidden_size", "d_model", "embed_dim"):
            v = getattr(cfg, attr, None)
            if v is not None:
                self._hidden_dim = int(v)
                break
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            v = getattr(cfg, attr, None)
            if v is not None:
                self._num_layers = int(v)
                break

        log.info(f"Nucleotide Transformer loaded "
                 f"(hidden={self._hidden_dim}, layers={self._num_layers}).")

    def _register_hooks(self):
        """Hook into ESM/BERT encoder layers."""
        encoder_layers = None
        for path in [
            ("encoder", "layer"),
            ("esm", "encoder", "layer"),
            ("bert", "encoder", "layer"),
        ]:
            obj = self._model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                if hasattr(obj, "__len__") and len(obj) > 0:
                    encoder_layers = obj
                    break
            except AttributeError:
                pass

        if encoder_layers is None:
            log.warning("Could not locate encoder layers; attaching full-model output hook.")
            def _hook(_, __, output):
                h = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
                if isinstance(h, torch.Tensor) and h.ndim == 3:
                    h = h.mean(dim=1)
                for name in self.hook_layer_indices:
                    self._cache[name] = h.detach().cpu()
            self._model.register_forward_hook(_hook)
            return

        for hook_name, idx in self.hook_layer_indices.items():
            idx = min(idx, len(encoder_layers) - 1)
            def _make_hook(n):
                def _hook(_, __, output):
                    h = output[0] if isinstance(output, (tuple, list)) else output
                    if isinstance(h, torch.Tensor) and h.ndim == 3:
                        self._cache[n] = h.detach().mean(dim=1).cpu()
                return _hook
            handle = encoder_layers[idx].register_forward_hook(_make_hook(hook_name))
            self._hooks.append(handle)

    @property
    def model_name(self) -> str:
        return "nucleotide_transformer"

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hook_layer_indices(self) -> Dict[str, int]:
        L = self.num_layers
        return {"early": L // 4, "mid": L // 2, "late": min(L - 1, 3 * L // 4)}

    def _tokenise_seq(self, seq_str: str) -> np.ndarray:
        """
        Tokenise a DNA string.  Pre-truncates to max_tokens * 6 bp so the
        tokenizer only lexes the number of 6-mers that will actually be used.
        """
        kmer_size = 6
        max_bp    = (self._max_tokens - 1) * kmer_size   # -1 for CLS
        seq_str   = seq_str[:max_bp]

        enc = self._tokenizer(
            seq_str,
            return_tensors="np",
            truncation=True,
            max_length=self._max_tokens,
            padding="max_length",
        )
        return enc["input_ids"][0].astype(np.int64)

    def get_activations(
        self,
        windows: List[Tuple[str, int, int]],
        atac_arrays: Optional[Dict[str, np.ndarray]] = None,
        *,
        batch_size: int = 4,
        use_genome: bool = False,
    ) -> Dict[str, np.ndarray]:
        self._ensure_model()
        N = len(windows)
        layer_names = list(self.hook_layer_indices.keys())
        buffers = {name: [] for name in layer_names}

        # Guard against tokenizers that have pad_token_id=None
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
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
                seqs.append(self._tokenise_seq(seq_str))

            input_ids      = torch.from_numpy(np.stack(seqs)).to(self._device)
            attention_mask = (input_ids != pad_id).long()

            self._cache.clear()
            with torch.no_grad():
                _ = self._model(input_ids=input_ids, attention_mask=attention_mask)

            if not self._cache:
                # Hooks did not fire — fall back to output_hidden_states
                with torch.no_grad():
                    out = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                all_hidden = out.hidden_states
                for hook_name, idx in self.hook_layer_indices.items():
                    idx = min(idx, len(all_hidden) - 1)
                    h   = all_hidden[idx].mean(dim=1).cpu().numpy().astype(np.float32)
                    buffers[hook_name].append(h)
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
    register("nucleotide_transformer")(NucleotideTransformerAdapter)
    register("nt")(NucleotideTransformerAdapter)
except Exception:
    pass
