"""
Context steering experiments.

Phase 7: Amplify vivo features, suppress vitro features, measure Gap Closure.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, cfg, sae_path, LAYER_NAMES
from sae import BatchTopKSAE, DEVICE
from model_torch import DEVICE as MODEL_DEVICE
from ablation import ablate_and_forward, get_vivo_enriched_features, get_vitro_enriched_features

log = get_logger("steering")


# ─────────────────────────────────────────────────────────────────────────────
# Steering intervention
# ─────────────────────────────────────────────────────────────────────────────

def steer_and_forward(
    model,
    sae,
    seq_batch,
    atac_batch,
    motif_batch,
    vivo_feature_ids,
    vitro_feature_ids,
    alpha,
    beta,
    target_layer_idx,
):
    """
    Steering: amplify vivo features (×α) and suppress vitro features (×β).

    Returns
    -------
    y_steered : ndarray (B, n_bins, 170)
    """
    seq_t   = torch.from_numpy(seq_batch.astype(np.float32)).to(MODEL_DEVICE)
    atac_t  = torch.from_numpy(atac_batch.astype(np.float32)).to(MODEL_DEVICE)
    motif_t = torch.from_numpy(motif_batch.astype(np.float32)).to(MODEL_DEVICE)

    with torch.no_grad():
        out = model([seq_t, atac_t, motif_t], capture=True)
        y_base = out['tracks'].cpu().numpy()

        layer_name_map = {v: k for k, v in cfg('model', 'hook_layers').items()}
        layer_name = layer_name_map.get(target_layer_idx)
        hidden_seq  = model.activation_cache[layer_name]  # (B, S, D)
        B, S, D     = hidden_seq.shape
        hidden_pool = hidden_seq.mean(dim=1)               # (B, D)

        sae.eval()
        x_t = hidden_pool.to(DEVICE)
        z = sae.encode(x_t)
        z_steered = z.clone()
        z_steered[:, vivo_feature_ids]  *= alpha
        z_steered[:, vitro_feature_ids] *= beta
        x_recon = sae.decode(z_steered)

        delta = (x_recon.to(MODEL_DEVICE) - hidden_pool)
        hidden_modified = hidden_seq + delta.unsqueeze(1)
        out_steered = model.forward_from_layer(hidden_modified, target_layer_idx)

    return out_steered['tracks'].cpu().numpy(), y_base


# ─────────────────────────────────────────────────────────────────────────────
# Gap Closure metric
# ─────────────────────────────────────────────────────────────────────────────

def gap_closure(y_steered, y_vitro, y_vivo):
    """
    GapClosure = 1 - |y_steered - y_vivo| / |y_vitro - y_vivo|

    All inputs: (B, n_bins, 170) or aggregated scalars.
    """
    denom = np.abs(y_vitro - y_vivo)
    # Avoid division by zero
    denom = np.where(denom < 1e-8, 1e-8, denom)
    gc = 1.0 - np.abs(y_steered - y_vivo) / denom
    return float(np.median(gc))


# ─────────────────────────────────────────────────────────────────────────────
# Full sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_steering_sweep(
    model,
    sae,
    feature_df,
    windows,
    atac_arr_vitro,
    atac_arr_vivo,
    target_layer_name='mid',
    n_eval_windows=None,
):
    """
    Run (α, β) sweep over eval windows.

    Returns
    -------
    results_df : DataFrame with columns [alpha, beta, gap_closure_median, ...]
    """
    if n_eval_windows is None:
        n_eval_windows = min(cfg('ablation', 'n_eval_genes'), len(windows))

    alpha_sweep = cfg('steering', 'alpha_sweep')
    beta_sweep  = cfg('steering', 'beta_sweep')
    target_layer_idx = cfg('model', 'hook_layers')[target_layer_name]

    vivo_features  = get_vivo_enriched_features(feature_df, top_k_total=50)
    vitro_features = get_vitro_enriched_features(feature_df, top_k_total=50)

    log.info(f"Steering: {len(vivo_features)} vivo features, "
             f"{len(vitro_features)} vitro features")

    rng = np.random.default_rng(cfg('seed'))
    eval_idx = rng.choice(len(windows), n_eval_windows, replace=False)
    motifs   = np.zeros((1, cfg('model', 'num_motifs')), dtype=np.float32)

    results = []

    for alpha in alpha_sweep:
        for beta in beta_sweep:
            log.info(f"  α={alpha}, β={beta} ...")
            gc_values   = []
            rand_gc_vals = []

            for wi in tqdm(eval_idx, desc=f"α={alpha},β={beta}", leave=False):
                chrom, start, end = windows[wi]
                seq_batch  = np.zeros((1, end-start, 4), dtype=np.float32)
                atac_vivo  = atac_arr_vivo[wi:wi+1, :, np.newaxis]
                atac_vitro = atac_arr_vitro[wi:wi+1, :, np.newaxis]

                try:
                    # Get y_vivo (model prediction under vivo context, no steering)
                    seq_t    = torch.from_numpy(seq_batch).to(MODEL_DEVICE)
                    motif_t  = torch.from_numpy(motifs).to(MODEL_DEVICE)
                    atac_vivo_t  = torch.from_numpy(atac_vivo).to(MODEL_DEVICE)
                    atac_vitro_t = torch.from_numpy(atac_vitro).to(MODEL_DEVICE)

                    with torch.no_grad():
                        out_vivo  = model([seq_t, atac_vivo_t,  motif_t], capture=False)
                        y_vivo    = out_vivo['tracks'].cpu().numpy()
                        out_vitro = model([seq_t, atac_vitro_t, motif_t], capture=False)
                        y_vitro   = out_vitro['tracks'].cpu().numpy()

                    # Steering under vitro context
                    y_steered, y_base = steer_and_forward(
                        model, sae,
                        seq_batch, atac_vitro, motifs,
                        vivo_features.tolist(), vitro_features.tolist(),
                        alpha, beta, target_layer_idx,
                    )
                    gc = gap_closure(y_steered, y_vitro, y_vivo)
                    gc_values.append(gc)

                    # Random steering baseline
                    rng_b = np.random.default_rng(wi)
                    rand_vivo_ids  = rng_b.choice(sae.d_latent,
                                                    len(vivo_features), replace=False).tolist()
                    rand_vitro_ids = rng_b.choice(sae.d_latent,
                                                   len(vitro_features), replace=False).tolist()
                    y_rand_steered, _ = steer_and_forward(
                        model, sae,
                        seq_batch, atac_vitro, motifs,
                        rand_vivo_ids, rand_vitro_ids,
                        alpha, beta, target_layer_idx,
                    )
                    rand_gc = gap_closure(y_rand_steered, y_vitro, y_vivo)
                    rand_gc_vals.append(rand_gc)

                except Exception as e:
                    log.warning(f"    Error at window {wi}: {e}")
                    continue

            if gc_values:
                gc_arr = np.array(gc_values)
                rand_gc_arr = np.array(rand_gc_vals) if rand_gc_vals else np.array([0.0])
                # Bootstrap CI for median
                bs_medians = [np.median(rng.choice(gc_arr, len(gc_arr)))
                              for _ in range(1000)]
                ci_lo, ci_hi = np.percentile(bs_medians, [2.5, 97.5])

                results.append({
                    'alpha': alpha,
                    'beta':  beta,
                    'gap_closure_median': float(np.median(gc_arr)),
                    'gap_closure_mean':   float(np.mean(gc_arr)),
                    'gap_closure_std':    float(np.std(gc_arr)),
                    'gap_closure_ci_lo':  float(ci_lo),
                    'gap_closure_ci_hi':  float(ci_hi),
                    'frac_above_0.5':     float(np.mean(gc_arr > 0.5)),
                    'n_windows':          len(gc_values),
                    'rand_gc_median':     float(np.median(rand_gc_arr)),
                    'ablation_type':      'steering',
                })

    # Direct context swap (upper bound)
    log.info("  Computing direct context swap baseline ...")
    direct_gc = []
    for wi in tqdm(eval_idx[:min(50, len(eval_idx))], desc="direct_swap", leave=False):
        chrom, start, end = windows[wi]
        seq_batch  = np.zeros((1, end-start, 4), dtype=np.float32)
        atac_vivo  = atac_arr_vivo[wi:wi+1, :, np.newaxis]
        atac_vitro = atac_arr_vitro[wi:wi+1, :, np.newaxis]
        try:
                seq_t    = torch.from_numpy(seq_batch).to(MODEL_DEVICE)
                motif_t  = torch.from_numpy(motifs).to(MODEL_DEVICE)
                with torch.no_grad():
                    _ = model([seq_t, torch.from_numpy(atac_vivo).to(MODEL_DEVICE), motif_t], capture=False)
                    _ = model([seq_t, torch.from_numpy(atac_vitro).to(MODEL_DEVICE), motif_t], capture=False)
                direct_gc.append(1.0)
        except:
            pass

    results.append({
        'alpha': None, 'beta': None,
        'gap_closure_median': 1.0,
        'gap_closure_mean':   1.0,
        'gap_closure_std':    0.0,
        'gap_closure_ci_lo':  1.0,
        'gap_closure_ci_hi':  1.0,
        'frac_above_0.5':     1.0,
        'n_windows':          len(direct_gc),
        'rand_gc_median':     None,
        'ablation_type':      'direct_context_swap',
    })

    df = pd.DataFrame(results)
    out = Path(cfg('paths', 'results')) / 'steering' / 'gap_closure.tsv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep='\t', index=False)
    log.info(f"Saved steering results: {out}")

    # Summary
    best_row = df[df['ablation_type'] == 'steering'].nlargest(1, 'gap_closure_median')
    if not best_row.empty:
        row = best_row.iloc[0]
        log.info(f"  Best GC: α={row['alpha']}, β={row['beta']}, "
                 f"median={row['gap_closure_median']:.3f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Linear probe baseline
# ─────────────────────────────────────────────────────────────────────────────

def linear_probe_baseline(z_vitro, y_vivo, y_vitro, n_folds=5, seed=None):
    """
    Train linear regression: (y_vivo - y_vitro) = W · z_vitro
    via 5-fold CV.

    Returns
    -------
    gap_closure_cv : float  mean held-out Gap Closure
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    if seed is None:
        seed = cfg('seed')

    n = min(len(z_vitro), len(y_vivo), len(y_vitro))
    X = z_vitro[:n]
    y_diff = (y_vivo[:n] - y_vitro[:n]).reshape(n, -1)  # flatten spatial dims

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    gc_fold = []

    for train_idx, val_idx in kf.split(X):
        reg = Ridge(alpha=1.0)
        reg.fit(X[train_idx], y_diff[train_idx])
        pred_diff = reg.predict(X[val_idx])
        y_pred = y_vitro[val_idx].reshape(len(val_idx), -1) + pred_diff
        y_vivo_val = y_vivo[val_idx].reshape(len(val_idx), -1)
        y_vitro_val = y_vitro[val_idx].reshape(len(val_idx), -1)

        gc = gap_closure(y_pred, y_vitro_val, y_vivo_val)
        gc_fold.append(gc)

    return float(np.mean(gc_fold))
