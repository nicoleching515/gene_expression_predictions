"""
Ablation experiments: targeted vs. random vs. top-activation ablation.

Phase 6 of the pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, cfg, sae_path, activation_path, load_activations, LAYER_NAMES
from sae import BatchTopKSAE, DEVICE
from model_torch import get_model, DEVICE as MODEL_DEVICE

log = get_logger("ablation")


# ─────────────────────────────────────────────────────────────────────────────
# Core ablation functions
# ─────────────────────────────────────────────────────────────────────────────

def get_vivo_enriched_features(feature_df, top_k_total=100):
    """Return indices of vivo-enriched features sorted by CDS."""
    vivo_df = feature_df[feature_df['category'] == 'vivo_enriched'].copy()
    vivo_df = vivo_df.sort_values('cds_avg', ascending=False)
    return vivo_df['feature_id'].values[:top_k_total]


def get_vitro_enriched_features(feature_df, top_k_total=100):
    """Return indices of vitro-enriched features sorted by |CDS|."""
    vitro_df = feature_df[feature_df['category'] == 'vitro_enriched'].copy()
    vitro_df = vitro_df.sort_values('cds_avg', ascending=True)
    return vitro_df['feature_id'].values[:top_k_total]


def ablate_and_forward(
    model,
    sae,
    seq_batch,      # (B, L, 4) numpy
    atac_batch,     # (B, L, 1) numpy
    motif_batch,    # (B, 693)  numpy
    feature_ids,    # list/array of feature indices to zero out
    target_layer_idx,
):
    """
    Run one ablation experiment:
    1. Full forward pass → capture hidden state at target layer + ŷ_full
    2. Encode hidden state through SAE → z
    3. Zero out feature_ids in z → z'
    4. Decode z' → modified hidden state
    5. Run forward_from_layer → ŷ_ablated
    6. Return (ŷ_full, ŷ_ablated, z)

    Returns
    -------
    y_full    : ndarray (B, n_bins, 170)
    y_ablated : ndarray (B, n_bins, 170)
    z         : ndarray (B, d_latent)  SAE latents before ablation
    """
    seq_t   = torch.from_numpy(seq_batch.astype(np.float32)).to(MODEL_DEVICE)
    atac_t  = torch.from_numpy(atac_batch.astype(np.float32)).to(MODEL_DEVICE)
    motif_t = torch.from_numpy(motif_batch.astype(np.float32)).to(MODEL_DEVICE)
    inputs  = [seq_t, atac_t, motif_t]

    with torch.no_grad():
        out_full = model(inputs, capture=True)
        y_full = out_full['tracks'].cpu().numpy()

        # Get hidden state at target layer (B, S, 1024)
        layer_name_map = {v: k for k, v in cfg('model', 'hook_layers').items()}
        layer_name = layer_name_map.get(target_layer_idx)
        if layer_name is None or layer_name not in model.activation_cache:
            raise ValueError(f"Layer {target_layer_idx} not captured. "
                             f"Available: {list(model.activation_cache.keys())}")

        hidden_seq = model.activation_cache[layer_name]  # (B, S, 1024) PyTorch tensor
        B, S, D = hidden_seq.shape
        hidden_pooled = hidden_seq.mean(dim=1)  # (B, D)

        # Encode through SAE
        sae.eval()
        x_sae = hidden_pooled.to(DEVICE)
        z = sae.encode(x_sae)          # (B, d_latent)
        z_ablated = z.clone()
        z_ablated[:, feature_ids] = 0.0
        x_recon = sae.decode(z_ablated)  # (B, D)

        # Add difference back to sequence positions
        delta = (x_recon.to(MODEL_DEVICE) - hidden_pooled)  # (B, D)
        hidden_modified = hidden_seq + delta.unsqueeze(1)    # (B, S, D)

        out_ablated = model.forward_from_layer(hidden_modified, target_layer_idx)
        y_ablated = out_ablated['tracks'].cpu().numpy()

    z_np = z.cpu().numpy()
    return y_full, y_ablated, z_np


# ─────────────────────────────────────────────────────────────────────────────
# Dose-response sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_dose_response(
    model,
    sae,
    feature_df,
    windows,
    atac_arr_vitro,
    atac_arr_vivo,
    target_layer_name='mid',
    k_sweep=None,
    n_eval_windows=None,
    random_seeds=None,
):
    """
    Run ablation dose-response sweep for k in k_sweep.

    For each k, computes:
      - targeted_delta: mean |Δŷ| for ablating top-k vivo-enriched features
      - random_delta:   mean |Δŷ| for ablating k random features (5 seeds)
      - topact_delta:   mean |Δŷ| for ablating top-k highest-activation features

    Returns
    -------
    results : DataFrame with columns [k, ablation_type, delta_mean, delta_std, seed]
    """
    if k_sweep is None:
        k_sweep = cfg('ablation', 'k_sweep')
    if n_eval_windows is None:
        n_eval_windows = min(cfg('ablation', 'n_eval_genes'), len(windows))
    if random_seeds is None:
        random_seeds = list(range(cfg('ablation', 'random_ablation_seeds')))

    target_layer_idx = cfg('model', 'hook_layers')[target_layer_name]
    vivo_features    = get_vivo_enriched_features(feature_df)

    # Sample eval windows
    rng = np.random.default_rng(cfg('seed'))
    eval_idx = rng.choice(len(windows), n_eval_windows, replace=False)

    # Precompute SAE latents for eval windows under vivo context
    log.info(f"Precomputing SAE latents for {n_eval_windows} eval windows ...")
    motifs = np.zeros((1, cfg('model', 'num_motifs')), dtype=np.float32)

    results = []

    for k in k_sweep:
        log.info(f"  k={k} ...")

        # Sample eval windows in mini-batches
        targeted_deltas = []
        random_deltas_by_seed = {s: [] for s in random_seeds}
        topact_deltas  = []

        for wi in tqdm(eval_idx, desc=f"k={k}", leave=False):
            chrom, start, end = windows[wi]
            # Build batch of size 1
            seq_batch   = np.zeros((1, end-start, 4), dtype=np.float32)
            atac_vivo   = atac_arr_vivo[wi:wi+1, :, np.newaxis]    # (1, L, 1)
            atac_vitro  = atac_arr_vitro[wi:wi+1, :, np.newaxis]

            try:
                # Targeted ablation (vivo context, ablate vivo-enriched features)
                feat_ids = vivo_features[:k].tolist()
                y_full, y_abl, z = ablate_and_forward(
                    model, sae, seq_batch, atac_vivo, motifs, feat_ids, target_layer_idx
                )
                targeted_deltas.append(float(np.mean(np.abs(y_abl - y_full))))

                # Random ablation (k random features, matched magnitude)
                for seed in random_seeds:
                    rng_s = np.random.default_rng(seed + wi)
                    rand_ids = rng_s.choice(sae.d_latent, k, replace=False).tolist()
                    y_full_r, y_abl_r, z_r = ablate_and_forward(
                        model, sae, seq_batch, atac_vivo, motifs, rand_ids, target_layer_idx
                    )
                    random_deltas_by_seed[seed].append(
                        float(np.mean(np.abs(y_abl_r - y_full_r)))
                    )

                # Top-activation ablation (k most-active features regardless of CDS)
                z_mean = np.mean(z, axis=0)
                topact_ids = np.argsort(z_mean)[-k:].tolist()
                y_full_t, y_abl_t, _ = ablate_and_forward(
                    model, sae, seq_batch, atac_vivo, motifs, topact_ids, target_layer_idx
                )
                topact_deltas.append(float(np.mean(np.abs(y_abl_t - y_full_t))))

            except Exception as e:
                log.warning(f"    Error at window {wi}: {e}")
                continue

        # Aggregate
        if targeted_deltas:
            results.append({
                'k': k,
                'ablation_type': 'targeted',
                'delta_mean': np.mean(targeted_deltas),
                'delta_std':  np.std(targeted_deltas),
                'seed': -1,
            })
        if topact_deltas:
            results.append({
                'k': k,
                'ablation_type': 'top_activation',
                'delta_mean': np.mean(topact_deltas),
                'delta_std':  np.std(topact_deltas),
                'seed': -1,
            })
        for seed in random_seeds:
            deltas = random_deltas_by_seed[seed]
            if deltas:
                results.append({
                    'k': k,
                    'ablation_type': 'random',
                    'delta_mean': np.mean(deltas),
                    'delta_std':  np.std(deltas),
                    'seed': seed,
                })

    df = pd.DataFrame(results)
    out = Path(cfg('paths', 'results')) / 'ablation' / 'effects.tsv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep='\t', index=False)
    log.info(f"Saved ablation results: {out}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_ablation_test(effects_df, k=25):
    """
    Paired Wilcoxon signed-rank test: targeted vs. random.

    Builds paired observations across all k values × random seeds
    (n = len(k_sweep) × n_seeds), testing whether targeted Δŷ
    consistently exceeds random Δŷ. This gives sufficient n for the test
    when effects.tsv only has aggregate delta_mean per (k, type, seed).

    Returns (stat, pval, cohens_d).
    """
    from scipy import stats

    k_vals = sorted(effects_df['k'].unique())
    targeted_vals, random_vals = [], []

    for kv in k_vals:
        t = effects_df[(effects_df['k'] == kv) &
                       (effects_df['ablation_type'] == 'targeted')]['delta_mean'].values
        r = effects_df[(effects_df['k'] == kv) &
                       (effects_df['ablation_type'] == 'random')]['delta_mean'].values
        if len(t) == 0 or len(r) == 0:
            continue
        t_mean = t.mean()
        for rv in r:
            targeted_vals.append(t_mean)
            random_vals.append(rv)

    targeted_arr = np.array(targeted_vals)
    random_arr   = np.array(random_vals)

    if len(targeted_arr) < 2:
        log.warning("  Not enough paired observations for Wilcoxon")
        return None, None, None

    diffs = targeted_arr - random_arr
    cohens_d = diffs.mean() / max(diffs.std(ddof=1), 1e-8)

    stat, pval = stats.wilcoxon(targeted_arr, random_arr, alternative='greater')

    log.info(f"  Wilcoxon (all k × seeds, n={len(diffs)}): "
             f"targeted_mean={targeted_arr.mean():.4f}, "
             f"random_mean={random_arr.mean():.4f}, "
             f"p={pval:.2e}, d={cohens_d:.3f}")
    return stat, pval, cohens_d
