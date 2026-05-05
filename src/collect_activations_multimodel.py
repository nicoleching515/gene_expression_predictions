#!/usr/bin/env python3
"""
src/collect_activations_multimodel.py
======================================
Phase 12 — Collect hidden-state activations for multiple sequence models
(EpiBERT, Enformer, HyenaDNA, Nucleotide Transformer) over the 10,000
chr8/chr9 genomic windows.

Each model × condition produces three .pt files (early/mid/late layer),
written to:

    activations/{model_name}/{pair}/{condition}/{layer}.pt

The file format is identical to the original EpiBERT activations so the
existing SAE training and analysis pipeline is fully reusable.

Usage:
    # All models, all conditions
    python src/collect_activations_multimodel.py

    # Single model, single condition
    python src/collect_activations_multimodel.py --model enformer --condition K562

    # Quick smoke-test (100 windows)
    python src/collect_activations_multimodel.py --n-windows 100 --model hyenadna

    # Skip already-completed output files
    python src/collect_activations_multimodel.py --skip-existing

Supported --model values:
    epibert  enformer  hyenadna  nucleotide_transformer (nt)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

from utils import get_logger, cfg, seed_everything
from data import load_windows, get_total_mapped_reads

log = get_logger("collect_multimodel")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

def multimodel_activation_path(model_name: str, pair: str, condition: str, layer: str) -> Path:
    base = REPO / "activations" / model_name / pair / condition
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{layer}.pt"


# ─────────────────────────────────────────────────────────────────────────────
# ATAC array loader (for EpiBERT / future multi-modal models)
# ─────────────────────────────────────────────────────────────────────────────

def load_atac_for_condition(condition: str, windows, bam_map: dict) -> Optional[np.ndarray]:
    """
    Stream ATAC coverage from BAM for a condition.
    Returns float32 (N, window_bp) or None if BAM unavailable.
    """
    bam_path = bam_map.get(condition, "")
    if not os.path.isfile(bam_path):
        log.warning(f"  BAM not found for {condition}: {bam_path} — using zeros")
        return None

    try:
        import pysam
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        log.warning("pysam/scipy not available — ATAC = zeros")
        return None

    # Import compute_atac_batch from sibling module
    sys.path.insert(0, str(REPO / "src"))
    try:
        from collect_activations import compute_atac_batch
    except ImportError:
        log.warning("collect_activations not importable — ATAC = zeros")
        return None

    window_bp = windows[0][2] - windows[0][1]
    N         = len(windows)
    smoothing = int(cfg("atac", "smoothing_bp"))
    do_log1p  = bool(cfg("atac", "log1p"))

    total_reads = get_total_mapped_reads(bam_path)
    log.info(f"  ATAC BAM: {bam_path}  ({total_reads:,} mapped reads)")

    batch_size = 64
    all_atac   = []

    for start in range(0, N, batch_size):
        end        = min(start + batch_size, N)
        win_batch  = windows[start:end]
        arr        = compute_atac_batch(
            bam_path, win_batch, total_reads,
            smoothing_bp=smoothing, do_log1p=do_log1p,
        )                                   # (B, window_bp, 1)
        all_atac.append(arr[:, :, 0])       # → (B, window_bp)

    return np.concatenate(all_atac, axis=0)   # (N, window_bp)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_for_model(
    model_name: str,
    conditions: List[str],
    windows,
    bam_map: dict,
    pairs_cfg: dict,
    pair_for_condition: dict,
    batch_size: int,
    use_genome: bool,
    skip_existing: bool,
) -> None:
    """Collect activations for one model over all requested conditions."""

    from models import get_adapter

    log.info(f"\n{'='*70}")
    log.info(f"Model: {model_name.upper()}")
    log.info(f"{'='*70}")

    adapter = get_adapter(model_name)
    log.info(f"  {adapter}")

    for cond in conditions:
        if cond not in pair_for_condition:
            log.warning(f"  Condition {cond} not in any pair, skipping")
            continue

        pair_name, side = pair_for_condition[cond]
        layer_names     = adapter.layer_names

        # Check if already done
        if skip_existing:
            all_done = all(
                multimodel_activation_path(model_name, pair_name, cond, ln).exists()
                for ln in layer_names
            )
            if all_done:
                log.info(f"  [SKIP] {cond}: files already exist.")
                continue

        log.info(f"\n  Condition: {cond}  ({pair_name}/{side})")
        t0 = time.time()

        # Load ATAC only for EpiBERT (sequence-only models ignore it)
        atac_arrays = None
        if model_name == "epibert":
            atac_raw = load_atac_for_condition(cond, windows, bam_map)
            if atac_raw is not None:
                atac_arrays = {cond: atac_raw}

        acts = adapter.get_activations(
            windows,
            atac_arrays=atac_arrays,
            batch_size=batch_size,
            use_genome=use_genome,
        )

        for layer_name, arr in acts.items():
            out_path = multimodel_activation_path(model_name, pair_name, cond, layer_name)
            torch.save(torch.from_numpy(arr), str(out_path))
            log.info(f"    Saved {layer_name}: {arr.shape} → {out_path}")

        elapsed = time.time() - t0
        log.info(f"  {cond}: done in {elapsed/60:.1f} min")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model activation collection for contrastive SAE analysis."
    )
    parser.add_argument(
        "--model", "-m",
        nargs="+",
        default=["epibert", "enformer", "hyenadna", "nucleotide_transformer"],
        help="Model(s) to run (space-separated). Default: all four.",
    )
    parser.add_argument(
        "--condition", "-c",
        nargs="+",
        default=None,
        help="Condition(s) to process (e.g. K562 HepG2). Default: all 6.",
    )
    parser.add_argument("--n-windows",     type=int, default=None,
                        help="Limit windows (default: all 10K)")
    parser.add_argument("--batch-size",    type=int, default=None,
                        help="Windows per GPU batch (per-model defaults if not set)")
    parser.add_argument("--use-genome",    action="store_true",
                        help="Load real DNA from hg38.fa")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip conditions with complete activation files (default: on)")
    parser.add_argument("--force",         action="store_true",
                        help="Recompute even if outputs exist (overrides --skip-existing)")
    args = parser.parse_args()

    seed_everything()

    # Faster GPU matmul + cuDNN autotune (safe for fixed-shape EpiBERT batches).
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    pairs_cfg      = cfg("pairs")
    all_conditions = cfg("all_conditions")
    bam_map        = cfg("bam_files")

    # Map condition → (pair_name, side)
    pair_for_condition = {}
    for pair_name, pair_conds in pairs_cfg.items():
        for side, cond in pair_conds.items():
            pair_for_condition[cond] = (pair_name, side)

    conditions = args.condition or all_conditions
    windows    = load_windows()
    if args.n_windows:
        windows = windows[: args.n_windows]

    skip = args.skip_existing and not args.force

    log.info(f"Windows  : {len(windows)}")
    log.info(f"Models   : {args.model}")
    log.info(f"Conditions: {conditions}")
    log.info(f"Skip existing: {skip}")

    per_model_batch_size = {
        # Default EpiBERT batch 8 — tuned for ~80GB GPUs; override with --batch-size.
        "epibert":              args.batch_size or 8,
        "enformer":             args.batch_size or 1,   # 196k bp, high VRAM
        "hyenadna":             args.batch_size or 8,
        "nucleotide_transformer": args.batch_size or 4,
    }

    for model_name in args.model:
        bs = per_model_batch_size.get(model_name.lower(), args.batch_size or 4)
        try:
            collect_for_model(
                model_name        = model_name,
                conditions        = conditions,
                windows           = windows,
                bam_map           = bam_map,
                pairs_cfg         = pairs_cfg,
                pair_for_condition= pair_for_condition,
                batch_size        = bs,
                use_genome        = args.use_genome,
                skip_existing     = skip,
            )
        except Exception as exc:
            log.error(f"  FAILED for model '{model_name}': {exc}")
            import traceback; traceback.print_exc()
            continue

    log.info("\nAll models done.")


if __name__ == "__main__":
    main()
