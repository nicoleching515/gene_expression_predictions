"""
SAE training script.

Trains 9 SAEs: 3 layers × 3 regimes (pooled, vitro, vivo).
SAE_pooled for all 3 layers is trained first (load-bearing).
Vitro/vivo variants are trained if time permits.

Usage:
    python train_sae.py --layer early --regime pooled
    python train_sae.py --layer mid --regime pooled
    python train_sae.py --layer late --regime pooled
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, cfg, activation_path, sae_path, load_activations, seed_everything
from sae import BatchTopKSAE, ActivationDataset, sae_loss, DEVICE

log = get_logger("train_sae")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_sae(layer_name, regime, use_wandb=False):
    """
    Train a single SAE for the given layer and regime.

    Parameters
    ----------
    layer_name : str  'early' | 'mid' | 'late'
    regime     : str  'pooled' | 'vitro' | 'vivo'
    use_wandb  : bool

    Returns
    -------
    sae, qc_metrics dict
    """
    seed_everything()
    sae_cfg = cfg('sae')
    pairs_cfg = cfg('pairs')
    all_conds = cfg('all_conditions')

    # Determine which conditions to include
    if regime == 'pooled':
        conditions = all_conds
    elif regime == 'vitro':
        conditions = cfg('conditions')['vitro']
    elif regime == 'vivo':
        conditions = cfg('conditions')['vivo']
    else:
        raise ValueError(f"Unknown regime: {regime}")

    # Collect activation paths
    act_paths = []
    for pair_name, pair_conds in pairs_cfg.items():
        for side, cond in pair_conds.items():
            if cond not in conditions:
                continue
            p = activation_path(pair_name, cond, layer_name)
            if os.path.isfile(p):
                act_paths.append(p)
            else:
                log.warning(f"Missing activation file: {p}")

    if not act_paths:
        raise FileNotFoundError(f"No activation files found for layer={layer_name}, regime={regime}")

    log.info(f"Training SAE [{layer_name}/{regime}] on {len(act_paths)} condition files")

    # Load all activations onto GPU
    dataset = ActivationDataset.from_paths(act_paths)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=sae_cfg['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    d_input   = cfg('model', 'hidden_dim')
    expansion = sae_cfg['expansion']
    k         = sae_cfg['k']

    sae = BatchTopKSAE(d_input=d_input, expansion=expansion, k=k).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=sae_cfg['lr'], eps=1e-8)

    # LR warmup scheduler
    warmup = sae_cfg['warmup_steps']
    total_steps = sae_cfg['steps']
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="epibert-sae",
            name=f"SAE_{layer_name}_{regime}",
            config={**sae_cfg, 'layer': layer_name, 'regime': regime},
            mode=os.environ.get('WANDB_MODE', 'online'),
        )

    log.info(f"  SAE: d_input={d_input}, d_latent={d_input*expansion}, k={k}")
    log.info(f"  Dataset: {len(dataset)} samples, batch={sae_cfg['batch_size']}")
    log.info(f"  Steps: {total_steps} (warmup={warmup})")
    log.info(f"  Device: {DEVICE}")

    step       = 0
    epoch      = 0
    best_loss  = float('inf')
    out_path   = sae_path(layer_name, regime)

    # Load all data to GPU if it fits (< 2GB)
    data_size_gb = dataset.data.nbytes / 1e9
    log.info(f"  Activation data size: {data_size_gb:.2f} GB")
    if data_size_gb < 2.0:
        gpu_data = dataset.data.to(DEVICE)
        log.info(f"  Loaded all activations to GPU")
        use_gpu_data = True
    else:
        gpu_data = None
        use_gpu_data = False

    t_start = time.time()

    while step < total_steps:
        epoch += 1

        if use_gpu_data:
            # Sample batches directly from GPU tensor
            perm = torch.randperm(len(gpu_data), device=DEVICE)
            for batch_start in range(0, len(perm), sae_cfg['batch_size']):
                if step >= total_steps:
                    break
                idx = perm[batch_start: batch_start + sae_cfg['batch_size']]
                if len(idx) < sae_cfg['batch_size']:
                    continue
                x = gpu_data[idx]
                step = _train_step(sae, optimizer, scheduler, x, step,
                                   total_steps, sae_cfg, use_wandb, t_start, out_path)
        else:
            for x in loader:
                if step >= total_steps:
                    break
                x = x.to(DEVICE)
                step = _train_step(sae, optimizer, scheduler, x, step,
                                   total_steps, sae_cfg, use_wandb, t_start, out_path)

    # Final save
    sae.save(out_path)
    log.info(f"  Training complete in {(time.time()-t_start)/60:.1f} min")

    # QC evaluation
    qc = evaluate_sae_qc(sae, dataset)
    log.info(f"  QC: norm_mse={qc['norm_mse']:.4f}, l0={qc['l0']:.1f}, "
             f"dead_frac={qc['dead_frac']:.3f}")

    qc_gates = cfg('sae', 'qc')
    qc['pass'] = (
        qc['norm_mse'] < qc_gates['max_norm_mse'] and
        qc_gates['l0_min'] <= qc['l0'] <= qc_gates['l0_max'] and
        qc['dead_frac'] < qc_gates['max_dead_frac']
    )
    if not qc['pass']:
        log.warning(f"  [QC FAIL] {layer_name}/{regime}: {qc}")
    else:
        log.info(f"  [QC PASS] {layer_name}/{regime}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({f'qc/{k}': v for k, v in qc.items()})
        wandb.finish()

    return sae, qc


def _train_step(sae, optimizer, scheduler, x, step, total_steps, sae_cfg,
                use_wandb, t_start, out_path):
    """Single training step. Returns updated step count."""
    sae.train()
    optimizer.zero_grad()

    x_hat, z, pre_act = sae(x)
    loss, mse, l1 = sae_loss(x, x_hat, z)
    loss.backward()

    # Gradient clipping
    nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # Renormalize decoder columns
    with torch.no_grad():
        sae._normalize_decoder()

    # Update dead feature tracking
    with torch.no_grad():
        sae.update_dead_features(z)

    # Dead feature resampling
    dead_thresh = sae_cfg['dead_feature_threshold_steps']
    if sae_cfg.get('dead_feature_resample', True) and step % dead_thresh == 0 and step > 0:
        dead_mask = sae.get_dead_features(dead_thresh)
        n_resampled = sae.resample_dead_features(x, dead_mask)
        if n_resampled > 0:
            log.info(f"    Step {step}: resampled {n_resampled} dead features")

    step += 1

    # Logging
    if step % 500 == 0:
        sae.eval()
        with torch.no_grad():
            metrics = sae.compute_metrics(x, x_hat, z)
            dead_frac = (sae.steps_since_active > dead_thresh).float().mean().item()
            elapsed = time.time() - t_start
            log.info(f"  step={step}/{total_steps} | "
                     f"mse={metrics['l2_loss']:.4f} | "
                     f"norm_mse={metrics['norm_mse']:.4f} | "
                     f"l0={metrics['l0']:.1f} | "
                     f"dead={dead_frac:.3f} | "
                     f"lr={scheduler.get_last_lr()[0]:.2e} | "
                     f"t={elapsed:.0f}s")
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({**metrics, 'dead_frac': dead_frac, 'step': step})
        sae.train()

    # Save checkpoint every 10K steps
    if step % 10000 == 0:
        sae.save(out_path)

    return step


def evaluate_sae_qc(sae, dataset, n_samples=10000):
    """
    Evaluate SAE QC metrics on a random sample of the dataset.
    Returns dict of metrics.
    """
    sae.eval()
    sae = sae.to(DEVICE)

    # Sample
    idx = torch.randperm(len(dataset))[:n_samples]
    x = dataset.data[idx].to(DEVICE)

    with torch.no_grad():
        x_hat, z, _ = sae(x)
        metrics = sae.compute_metrics(x, x_hat, z)
        dead_thresh = cfg('sae', 'dead_feature_threshold_steps')
        dead_frac = (sae.steps_since_active > dead_thresh).float().mean().item()

    return {**metrics, 'dead_frac': dead_frac}


# ─────────────────────────────────────────────────────────────────────────────
# QC table
# ─────────────────────────────────────────────────────────────────────────────

def save_qc_table(qc_results):
    """
    Save QC results to results/sae_qc.tsv.
    qc_results: dict of {(layer, regime): metrics_dict}
    """
    import pandas as pd
    rows = []
    for (layer, regime), metrics in qc_results.items():
        rows.append({
            'layer': layer,
            'regime': regime,
            **metrics,
        })
    df = pd.DataFrame(rows)
    out = f"{cfg('paths', 'results')}/sae_qc.tsv"
    df.to_csv(out, sep='\t', index=False)
    log.info(f"QC table saved: {out}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train EpiBERT SAEs")
    parser.add_argument('--layer', choices=['early', 'mid', 'late', 'all'],
                        default='all', help='Which layer SAE to train')
    parser.add_argument('--regime', choices=['pooled', 'vitro', 'vivo', 'all'],
                        default='pooled', help='Training regime')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    args = parser.parse_args()

    layers  = ['early', 'mid', 'late'] if args.layer == 'all' else [args.layer]
    regimes = ['pooled', 'vitro', 'vivo'] if args.regime == 'all' else [args.regime]

    qc_results = {}
    for layer in layers:
        for regime in regimes:
            log.info(f"\n{'='*60}")
            log.info(f"Training SAE: layer={layer}, regime={regime}")
            log.info(f"{'='*60}")
            try:
                sae, qc = train_sae(layer, regime, use_wandb=args.wandb)
                qc_results[(layer, regime)] = qc
            except FileNotFoundError as e:
                log.error(f"Skipping {layer}/{regime}: {e}")

    if qc_results:
        save_qc_table(qc_results)

        # Summary
        log.info("\n" + "="*60)
        log.info("SAE Training Summary")
        log.info("="*60)
        all_pass = True
        for (layer, regime), qc in qc_results.items():
            status = "PASS" if qc.get('pass') else "FAIL"
            log.info(f"  [{status}] {layer}/{regime}: "
                     f"norm_mse={qc['norm_mse']:.4f}, "
                     f"l0={qc['l0']:.1f}, "
                     f"dead={qc['dead_frac']:.3f}")
            if not qc.get('pass'):
                all_pass = False

        if not all_pass:
            log.warning("Some SAEs failed QC! Check logs and consider retraining with adjusted k.")
            sys.exit(1)


if __name__ == '__main__':
    main()
