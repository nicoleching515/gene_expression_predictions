"""
Contrastive analysis: Context Divergence Score (CDS) and feature classification.

Phase 5: Encode all activations through SAE_pooled → compute CDS per feature per pair
→ classify features → export top-50 to Bio team.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (get_logger, cfg, load_activations, activation_path,
                   sae_path, LAYER_NAMES)
from sae import BatchTopKSAE, DEVICE

import torch

log = get_logger("analysis")


# ─────────────────────────────────────────────────────────────────────────────
# SAE encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_activations_through_sae(acts_np, sae, batch_size=4096):
    """
    Encode numpy activation array through SAE.

    Parameters
    ----------
    acts_np : ndarray (n_windows, d_input)
    sae     : BatchTopKSAE

    Returns
    -------
    z : ndarray (n_windows, d_latent)
    """
    sae.eval()
    sae = sae.to(DEVICE)
    n = len(acts_np)
    d_latent = sae.d_latent
    z_out = np.zeros((n, d_latent), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.from_numpy(acts_np[start:end]).float().to(DEVICE)
        with torch.no_grad():
            z = sae.encode(x)
        z_out[start:end] = z.cpu().numpy()

    return z_out


def build_z_tensor(layer_name, regime='pooled'):
    """
    Build Z tensor by encoding all condition activations through SAE_pooled.

    Returns
    -------
    z_dict : {pair: {'vitro': ndarray(n, d_lat), 'vivo': ndarray(n, d_lat)}}
    sae    : loaded SAE
    """
    sae_file = sae_path(layer_name, regime)
    if not os.path.isfile(sae_file):
        raise FileNotFoundError(f"SAE not found: {sae_file}. Run train_sae.py first.")

    sae = BatchTopKSAE.load(sae_file)
    log.info(f"Loaded SAE: {sae_file}")

    pairs_cfg = cfg('pairs')
    z_dict = {}

    for pair_name, pair_conds in pairs_cfg.items():
        z_dict[pair_name] = {}
        for side, cond in pair_conds.items():
            path = activation_path(pair_name, cond, layer_name)
            if not os.path.isfile(path):
                log.warning(f"  Missing activations: {path}")
                z_dict[pair_name][side] = None
                continue
            acts = load_activations(path).numpy()
            log.info(f"  Encoding {pair_name}/{cond}/{layer_name}: {acts.shape}")
            z = encode_activations_through_sae(acts, sae)
            z_dict[pair_name][side] = z

    return z_dict, sae


# ─────────────────────────────────────────────────────────────────────────────
# Context Divergence Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_cds(z_dict):
    """
    Compute Context Divergence Score for all features.

    CDS(f, p) = mean_w [ z_vivo(f, w) - z_vitro(f, w) ]

    Returns
    -------
    cds : ndarray (n_pairs, d_latent)  CDS per feature per pair
    cds_avg : ndarray (d_latent,)      mean CDS across pairs
    pair_names : list of pair names in order
    n_features : int
    """
    pair_names = list(z_dict.keys())
    # Get d_latent from first available tensor
    d_latent = None
    for pn in pair_names:
        for side in ['vitro', 'vivo']:
            if z_dict[pn].get(side) is not None:
                d_latent = z_dict[pn][side].shape[1]
                break
        if d_latent is not None:
            break

    cds_matrix = np.zeros((len(pair_names), d_latent), dtype=np.float32)

    for pi, pair_name in enumerate(pair_names):
        z_vivo  = z_dict[pair_name].get('vivo')
        z_vitro = z_dict[pair_name].get('vitro')
        if z_vivo is None or z_vitro is None:
            log.warning(f"  {pair_name}: missing data, CDS set to 0")
            continue
        # Ensure same n_windows (take minimum)
        n = min(len(z_vivo), len(z_vitro))
        cds_matrix[pi] = np.mean(z_vivo[:n] - z_vitro[:n], axis=0)
        log.info(f"  {pair_name}: CDS range [{cds_matrix[pi].min():.4f}, "
                 f"{cds_matrix[pi].max():.4f}]")

    cds_avg = cds_matrix.mean(axis=0)
    return cds_matrix, cds_avg, pair_names


def permutation_test_cds(z_dict, n_permutations=None, seed=None):
    """
    Permutation test: shuffle vitro/vivo labels → null CDS distribution.

    Vectorized via batched GPU matmul: for each pair, diff = z_vivo - z_vitro,
    then perm_cds_batch = (signs @ diff) / n where signs ∈ {-1, +1}.
    This turns 10K serial numpy loops into a handful of BLAS calls.

    Returns
    -------
    null_cds_abs : ndarray (n_permutations, d_latent)
    pvals        : ndarray (d_latent,) Bonferroni-corrected
    """
    if n_permutations is None:
        n_permutations = cfg('analysis', 'n_permutations')
    if seed is None:
        seed = cfg('seed')

    pair_names = list(z_dict.keys())

    # Real CDS
    cds_matrix, cds_avg, _ = compute_cds(z_dict)
    real_cds_abs = np.abs(cds_avg)
    d_latent = cds_avg.shape[0]
    log.info(f"Running {n_permutations:,} permutations on {d_latent} features (vectorized) ...")

    # Pre-compute diffs on GPU: diff[p] = z_vivo - z_vitro, shape (n_p, d_latent)
    diffs_gpu = []
    for pn in pair_names:
        z_vivo  = z_dict[pn].get('vivo')
        z_vitro = z_dict[pn].get('vitro')
        if z_vivo is None or z_vitro is None:
            continue
        n = min(len(z_vivo), len(z_vitro))
        diff = torch.from_numpy(z_vivo[:n] - z_vitro[:n]).float().to(DEVICE)
        diffs_gpu.append(diff)

    n_valid_pairs = len(diffs_gpu)
    if n_valid_pairs == 0:
        return np.zeros((n_permutations, d_latent), dtype=np.float32), np.ones(d_latent)

    torch.manual_seed(seed)
    null_cds_abs = np.zeros((n_permutations, d_latent), dtype=np.float32)

    # Batch across permutations so each matmul is (batch, n) @ (n, d_latent)
    batch_size = 500
    with torch.no_grad():
        for b_start in tqdm(range(0, n_permutations, batch_size),
                            desc="Permutation batches", leave=False):
            bs = min(batch_size, n_permutations - b_start)
            perm_cds = torch.zeros(bs, d_latent, device=DEVICE)
            for diff in diffs_gpu:
                n_w = diff.shape[0]
                # signs ∈ {-1, +1}, shape (bs, n_w)
                signs = (torch.randint(0, 2, (bs, n_w), device=DEVICE).float() * 2 - 1)
                perm_cds += (signs @ diff) / n_w
            perm_cds /= n_valid_pairs
            null_cds_abs[b_start:b_start + bs] = perm_cds.abs().cpu().numpy()

    real_t = torch.from_numpy(real_cds_abs).float()
    null_t  = torch.from_numpy(null_cds_abs).float()
    pvals   = (null_t >= real_t.unsqueeze(0)).float().mean(dim=0).numpy()
    pvals_bonf = np.minimum(pvals * d_latent, 1.0)

    frac_significant = np.mean(pvals_bonf < 0.05)
    log.info(f"  {frac_significant:.2%} of features significant (Bonferroni α=0.05)")

    return null_cds_abs, pvals_bonf


# ─────────────────────────────────────────────────────────────────────────────
# Feature classification
# ─────────────────────────────────────────────────────────────────────────────

def compute_jaccard_top_windows(z_vivo, z_vitro, top_n=1000):
    """
    Compute Jaccard overlap of top-1000 activating windows between conditions.
    Per feature.

    Returns
    -------
    jaccard : ndarray (d_latent,)
    """
    n_features = z_vivo.shape[1]
    jaccard = np.zeros(n_features, dtype=np.float32)

    for f in range(n_features):
        n = min(len(z_vivo), len(z_vitro), top_n * 5)  # consider more for robustness
        top_vivo  = set(np.argsort(z_vivo[:n, f])[-top_n:])
        top_vitro = set(np.argsort(z_vitro[:n, f])[-top_n:])
        inter = len(top_vivo & top_vitro)
        union = len(top_vivo | top_vitro)
        jaccard[f] = inter / union if union > 0 else 0.0

    return jaccard


def classify_features(z_dict, cds_matrix, cds_avg):
    """
    Classify SAE features into:
      - shared
      - vivo_enriched
      - vitro_enriched
      - context_switched
      - other

    Returns
    -------
    df : DataFrame with all features and their metrics
    """
    ana_cfg = cfg('analysis')
    shared_pct   = ana_cfg['cds_shared_pct']
    vivo_pct     = ana_cfg['cds_vivo_pct']
    vitro_pct    = ana_cfg['cds_vitro_pct']
    jaccard_thr  = ana_cfg['jaccard_switch_threshold']

    d_latent = cds_avg.shape[0]
    pair_names = list(z_dict.keys())
    n_pairs = len(pair_names)

    log.info(f"Classifying {d_latent} features ...")

    # Mean activation per condition (pooled across pairs)
    mean_vitro_all = []
    mean_vivo_all  = []
    jaccard_all    = []

    for pn in pair_names:
        z_vivo  = z_dict[pn].get('vivo')
        z_vitro = z_dict[pn].get('vitro')
        if z_vivo is None or z_vitro is None:
            continue
        n = min(len(z_vivo), len(z_vitro))
        mean_vitro_all.append(np.mean(z_vitro[:n], axis=0))
        mean_vivo_all.append(np.mean(z_vivo[:n],  axis=0))

        jac = compute_jaccard_top_windows(z_vivo[:n], z_vitro[:n])
        jaccard_all.append(jac)

    if not mean_vitro_all:
        log.error("No valid pair data for classification!")
        return pd.DataFrame()

    mean_vitro = np.mean(mean_vitro_all, axis=0)
    mean_vivo  = np.mean(mean_vivo_all,  axis=0)
    jaccard    = np.mean(jaccard_all,    axis=0)

    # Percentile thresholds on cds_avg
    cds_abs = np.abs(cds_avg)
    pct_shared = np.percentile(cds_abs, shared_pct)
    pct_vivo   = np.percentile(cds_avg, vivo_pct)
    pct_vitro  = np.percentile(cds_avg, vitro_pct)
    med_act    = np.percentile((mean_vitro + mean_vivo) / 2, 50)

    # Sign consistency: does CDS go the same direction in all valid pairs?
    sign_consistent = np.zeros(d_latent, dtype=bool)
    cds_signs = np.sign(cds_matrix)  # (n_pairs, d_latent)
    sign_consistent = (np.abs(cds_signs.sum(axis=0)) == n_pairs)

    # Classification
    categories = np.full(d_latent, 'other', dtype=object)

    # Shared: low CDS AND mean activation high in BOTH conditions
    mask_shared = (
        (cds_abs < pct_shared) &
        (mean_vitro > med_act) &
        (mean_vivo  > med_act)
    )
    categories[mask_shared] = 'shared'

    # vivo-enriched: high positive CDS AND mean_vivo > mean_vitro
    mask_vivo = (cds_avg > pct_vivo) & (mean_vivo > mean_vitro)
    categories[mask_vivo] = 'vivo_enriched'

    # vitro-enriched: very negative CDS AND mean_vitro > mean_vivo
    mask_vitro = (cds_avg < pct_vitro) & (mean_vitro > mean_vivo)
    categories[mask_vitro] = 'vitro_enriched'

    # Context-switched: moderate CDS but low Jaccard
    mask_switch = (
        (categories == 'other') &
        (~mask_shared) &
        (jaccard < jaccard_thr)
    )
    categories[mask_switch] = 'context_switched'

    # Build per-pair CDS columns
    data = {
        'feature_id': np.arange(d_latent),
        'cds_avg':    cds_avg,
        'mean_vitro': mean_vitro,
        'mean_vivo':  mean_vivo,
        'jaccard':    jaccard,
        'category':   categories,
        'sign_consistent': sign_consistent,
    }
    for pi, pn in enumerate(pair_names):
        data[f'cds_{pn}'] = cds_matrix[pi]

    df = pd.DataFrame(data)

    # Stats
    cat_counts = df['category'].value_counts()
    log.info(f"  Category counts: {dict(cat_counts)}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Export top features for Bio team
# ─────────────────────────────────────────────────────────────────────────────

def export_top_features(layer_name, z_dict, feature_df, windows):
    """
    Export top-50 highest-|CDS| features per layer for Bio team.
    """
    ana_cfg = cfg('analysis')
    top_n   = ana_cfg['top_n_features']
    top_win = ana_cfg['top_windows_n']

    # Top features by |CDS|
    top_idx = feature_df.nlargest(top_n, 'cds_avg', keep='all')['feature_id'].values
    # Also include most negative CDS
    bot_idx = feature_df.nsmallest(top_n // 2, 'cds_avg', keep='all')['feature_id'].values
    top_idx = np.unique(np.concatenate([top_idx, bot_idx]))[:top_n]

    out_base = Path(cfg('paths', 'outputs')) / 'top_features' / f'layer_{layer_name}'
    out_base.mkdir(parents=True, exist_ok=True)

    pair_names = list(z_dict.keys())

    for fid in top_idx:
        feat_dir = out_base / f'feature_{fid}'
        feat_dir.mkdir(exist_ok=True)

        # Activations file
        act_rows = []
        for pn in pair_names:
            for side in ['vitro', 'vivo']:
                z = z_dict[pn].get(side)
                if z is None:
                    continue
                vals = z[:, fid]
                act_rows.append(pd.DataFrame({
                    'pair': pn,
                    'condition': side,
                    'window_idx': np.arange(len(vals)),
                    'activation': vals,
                }))
        if act_rows:
            pd.concat(act_rows).to_csv(feat_dir / 'activations.tsv', sep='\t', index=False)

        # Top windows BED files per condition pair
        for pn in pair_names:
            for side in ['vitro', 'vivo']:
                z = z_dict[pn].get(side)
                if z is None:
                    continue
                top_win_idx = np.argsort(z[:, fid])[-top_win:][::-1]
                bed_rows = []
                for wi in top_win_idx:
                    if wi < len(windows):
                        chrom, start, end = windows[wi]
                        bed_rows.append(f"{chrom}\t{start}\t{end}\t{z[wi, fid]:.4f}")
                bed_path = feat_dir / f'top_windows_{side}_{pn}.bed'
                with open(bed_path, 'w') as f:
                    f.write('\n'.join(bed_rows) + '\n')

    log.info(f"Exported {len(top_idx)} top features to {out_base}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(layer_name, windows, regime='pooled'):
    """
    Full analysis pipeline for one layer.
    Returns (feature_df, cds_matrix, cds_avg, null_cds_abs, pvals)
    """
    results_dir = Path(cfg('paths', 'results')) / 'cds'
    results_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = results_dir / f'layer_{layer_name}_features.tsv'

    log.info(f"\n{'='*60}")
    log.info(f"Analysis: layer={layer_name}, regime={regime}")
    log.info(f"{'='*60}")

    # Build Z tensor
    z_dict, sae = build_z_tensor(layer_name, regime)

    # CDS
    cds_matrix, cds_avg, pair_names = compute_cds(z_dict)

    # Classification
    feature_df = classify_features(z_dict, cds_matrix, cds_avg)

    # Save feature table
    feature_df.to_csv(out_tsv, sep='\t', index=False)
    log.info(f"Saved feature table: {out_tsv} ({len(feature_df)} features)")

    # Permutation test
    log.info("Running permutation test ...")
    null_cds_abs, pvals = permutation_test_cds(z_dict)

    # Save null distribution
    np.save(results_dir / f'layer_{layer_name}_null_cds.npy', null_cds_abs)
    np.save(results_dir / f'layer_{layer_name}_pvals.npy', pvals)

    frac_sig = np.mean(pvals < 0.05)
    log.info(f"  Significant features (Bonferroni): {frac_sig:.2%}")

    # Export top features for Bio team
    export_top_features(layer_name, z_dict, feature_df, windows)

    return feature_df, cds_matrix, cds_avg, null_cds_abs, pvals, z_dict


if __name__ == '__main__':
    import argparse
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data import load_windows

    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', choices=['early', 'mid', 'late'], default='mid')
    parser.add_argument('--regime', default='pooled')
    args = parser.parse_args()

    windows = load_windows()
    run_analysis(args.layer, windows, args.regime)
