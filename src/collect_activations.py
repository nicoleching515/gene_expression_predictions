"""
Phase 2: Activation collection — EpiBERT (PyTorch/GPU) with real cell-line ATAC.

Each condition's BAM file is streamed on-the-fly; no pre-saved ATAC arrays needed.
ATAC signal = RPM-normalized, smoothed, log1p-transformed coverage per base pair.

Usage:
    python collect_activations.py                     # all 6 conditions, all windows
    python collect_activations.py --condition K562    # single condition
    python collect_activations.py --n-windows 100     # quick test
    python collect_activations.py --batch-size 8      # override batch size
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import pysam
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, cfg, activation_path, LAYER_NAMES, seed_everything
from data import load_windows, get_total_mapped_reads
from model_torch import get_model, verify_model_sanity, DEVICE
from hooks import ActivationCollector

log = get_logger("collect_activations")


# ─────────────────────────────────────────────────────────────────────────────
# On-the-fly ATAC coverage from BAM
# ─────────────────────────────────────────────────────────────────────────────

def compute_atac_batch(bam_path, windows_batch, total_reads,
                        smoothing_bp=150, do_log1p=True):
    """
    Compute ATAC coverage for a batch of (chrom, start, end) windows.
    Returns float32 array (B, window_bp, 1).
    """
    window_bp = windows_batch[0][2] - windows_batch[0][1]
    B = len(windows_batch)
    out = np.zeros((B, window_bp, 1), dtype=np.float32)
    rpm_scale = 1_000_000.0 / total_reads if total_reads > 0 else 1.0
    sigma = smoothing_bp / 2.355

    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        for i, (chrom, start, end) in enumerate(windows_batch):
            try:
                cov = bam.count_coverage(chrom, start, end,
                                          quality_threshold=0, read_callback='all')
                sig = (np.array(cov[0]) + np.array(cov[1]) +
                       np.array(cov[2]) + np.array(cov[3])).astype(np.float32)
            except (ValueError, KeyError):
                sig = np.zeros(window_bp, dtype=np.float32)

            sig *= rpm_scale
            sig = gaussian_filter1d(sig, sigma=sigma).astype(np.float32)
            if do_log1p:
                sig = np.log1p(sig)

            out[i, :len(sig), 0] = sig[:window_bp]

    return out   # (B, window_bp, 1)


def make_random_dna_batch(B, window_bp, seed=0):
    """Random one-hot DNA (B, window_bp, 4). Used when hg38 not available."""
    rng = np.random.default_rng(seed)
    bases = rng.integers(0, 4, (B, window_bp))
    return np.eye(4, dtype=np.float32)[bases]   # (B, L, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────────────────────────────────────

def run_collection(conditions=None, n_windows=None, batch_size=None,
                   checkpoint_every=None, use_genome=False, force=False):
    seed_everything()

    pairs_cfg      = cfg('pairs')
    all_conditions = cfg('all_conditions')
    bam_map        = cfg('bam_files')
    hook_layers    = cfg('model', 'hook_layers')
    layer_names    = list(hook_layers.keys())

    smoothing_bp = cfg('atac', 'smoothing_bp')
    do_log1p     = cfg('atac', 'log1p')

    if conditions is None:
        conditions = all_conditions
    if batch_size is None:
        batch_size = cfg('activation', 'batch_size')
    if checkpoint_every is None:
        checkpoint_every = cfg('activation', 'checkpoint_every')

    # Map condition → (pair_name, side)
    pair_for_condition = {}
    for pair_name, pair_conds in pairs_cfg.items():
        for side, cond in pair_conds.items():
            pair_for_condition[cond] = (pair_name, side)

    # Load windows
    windows = load_windows()
    if n_windows is not None:
        windows = windows[:n_windows]
    window_bp = windows[0][2] - windows[0][1]
    n_total   = len(windows)
    log.info(f"Windows: {n_total}  window_bp: {window_bp}  batch_size: {batch_size}")
    log.info(f"Device: {DEVICE}")

    # Load model (GPU)
    model = get_model()
    model.eval()
    ok = verify_model_sanity(model, seq_len=min(window_bp, 4096))
    if not ok:
        log.error("Model sanity check FAILED. Aborting.")
        sys.exit(1)

    # Tune batch size for GPU memory
    batch_size = _tune_batch_size(model, window_bp, batch_size)

    for cond in conditions:
        if cond not in pair_for_condition:
            log.warning(f"Condition {cond} not in any pair, skipping")
            continue

        pair_name, side = pair_for_condition[cond]
        bam_path = bam_map.get(cond, '')
        if not os.path.isfile(bam_path):
            log.error(f"BAM not found for {cond}: {bam_path}")
            continue

        # Check if already done
        all_done = all(
            os.path.isfile(activation_path(pair_name, cond, ln))
            for ln in layer_names
        )
        if all_done and not force:
            log.info(f"[SKIP] {cond}: activation files already exist. Use --force to redo.")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"Collecting: {cond}  ({pair_name}/{side})")
        log.info(f"  BAM: {bam_path}")
        log.info(f"{'='*60}")

        total_reads = get_total_mapped_reads(bam_path)
        log.info(f"  Library size: {total_reads:,} mapped reads")

        collector = ActivationCollector(pair_name, cond, layer_names)
        ckpt_int  = max(1, checkpoint_every // batch_size)
        t0        = time.time()

        n_batches = (n_total + batch_size - 1) // batch_size

        with torch.no_grad():
            for bi in tqdm(range(n_batches), desc=cond, unit='batch'):
                batch_start = bi * batch_size
                batch_end   = min(batch_start + batch_size, n_total)
                win_batch   = windows[batch_start:batch_end]
                B           = len(win_batch)

                # ATAC from BAM (cell-line specific)
                atac_np = compute_atac_batch(
                    bam_path, win_batch, total_reads,
                    smoothing_bp=smoothing_bp, do_log1p=do_log1p
                )                                    # (B, window_bp, 1)

                # DNA one-hot (random if no genome; same for all conditions)
                if use_genome:
                    from data import get_dna_onehot
                    seq_np = np.stack([
                        get_dna_onehot(c, s, e) for c, s, e in win_batch
                    ])                               # (B, window_bp, 4)
                else:
                    seq_np = make_random_dna_batch(B, window_bp, seed=batch_start)

                # Motifs: zeros (same for all conditions; cancels in CDS)
                motif_np = np.zeros((B, cfg('model', 'num_motifs')), dtype=np.float32)

                # GPU inference
                seq_t   = torch.from_numpy(seq_np).to(DEVICE)
                atac_t  = torch.from_numpy(atac_np).to(DEVICE)
                motif_t = torch.from_numpy(motif_np).to(DEVICE)

                _ = model([seq_t, atac_t, motif_t], capture=True)

                # Collect mean-pooled activations → CPU
                cpu_cache = {k: v.cpu().numpy() for k, v in model.pooled_cache.items()}
                collector.add(cpu_cache)

                # Incremental save
                if (bi + 1) % ckpt_int == 0:
                    collector.save_checkpoint()
                    elapsed = time.time() - t0
                    rate    = collector.n_windows() / elapsed
                    eta     = (n_total - collector.n_windows()) / max(rate, 1)
                    log.info(f"  {cond}: {collector.n_windows()}/{n_total} windows | "
                             f"{rate:.1f} win/s | ETA {eta/60:.1f} min")

        collector.save()
        elapsed = time.time() - t0
        log.info(f"  {cond}: done — {n_total} windows in {elapsed/60:.1f} min "
                 f"({n_total/elapsed:.1f} win/s)")

    # Storage audit
    log.info("\nStorage audit:")
    total_bytes = 0
    for cond in conditions:
        if cond not in pair_for_condition:
            continue
        pair_name, side = pair_for_condition[cond]
        for ln in layer_names:
            path = activation_path(pair_name, cond, ln)
            if os.path.isfile(path):
                size = os.path.getsize(path)
                total_bytes += size
    log.info(f"  Total activation storage: {total_bytes/1e6:.1f} MB")
    log.info("Done.")


def _tune_batch_size(model, window_bp, initial_bs):
    """
    Try increasing batch size until OOM, then step back one.
    Returns safe batch size.
    """
    if not torch.cuda.is_available():
        return initial_bs

    bs = initial_bs
    log.info(f"Tuning batch size (start={bs}) ...")
    while True:
        try:
            seq_np   = np.zeros((bs, window_bp, 4), dtype=np.float32)
            atac_np  = np.zeros((bs, window_bp, 1), dtype=np.float32)
            motif_np = np.zeros((bs, cfg('model', 'num_motifs')), dtype=np.float32)
            seq_t   = torch.from_numpy(seq_np).to(DEVICE)
            atac_t  = torch.from_numpy(atac_np).to(DEVICE)
            motif_t = torch.from_numpy(motif_np).to(DEVICE)
            with torch.no_grad():
                _ = model([seq_t, atac_t, motif_t], capture=True)
            torch.cuda.empty_cache()
            if bs >= 64:
                break
            next_bs = bs * 2
            bs = next_bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            log.info(f"  OOM at bs={bs*2}, using bs={bs}")
            break

    log.info(f"  Using batch_size={bs}")
    return bs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Collect EpiBERT activations using real cell-line ATAC (PyTorch/GPU)"
    )
    parser.add_argument('--n-windows',   type=int,  default=None,
                        help='Limit windows (default: all 10K)')
    parser.add_argument('--batch-size',  type=int,  default=None,
                        help='Windows per batch (auto-tuned if not set)')
    parser.add_argument('--condition',   type=str,  default=None,
                        help='Single condition to process (e.g. K562)')
    parser.add_argument('--use-genome',  action='store_true',
                        help='Load real DNA from hg38.fa (requires genome download)')
    parser.add_argument('--force',       action='store_true',
                        help='Overwrite existing activation files')
    args = parser.parse_args()

    conditions = [args.condition] if args.condition else None
    run_collection(
        conditions=conditions,
        n_windows=args.n_windows,
        batch_size=args.batch_size,
        use_genome=args.use_genome,
        force=args.force,
    )
