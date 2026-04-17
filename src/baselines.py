"""
Baseline analyses and statistical tests.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, cfg, LAYER_NAMES

log = get_logger("baselines")


def run_all_stats(effects_df=None, gc_df=None, feature_dfs=None, pvals_dict=None):
    """
    Run all statistical tests and collect into stats.tsv.

    stats_dict keys:
      1. perm_test_frac_sig   fraction of features with sig CDS
      2. wilcoxon_targeted_vs_random   p-value, Cohen's d
      3. bootstrap_gc_ci       95% CI for median Gap Closure
      4. hypergeom_cross_layer cross-layer feature overlap
      5. BH-FDR where applicable
    """
    from statsmodels.stats.multitest import multipletests

    results = []

    # 1. Permutation test fraction
    if pvals_dict is not None:
        for layer, pvals in pvals_dict.items():
            frac_sig = float(np.mean(pvals < 0.05))
            results.append({
                'test': 'permutation_bonferroni',
                'layer': layer,
                'metric': 'frac_significant',
                'value': frac_sig,
                'pval': None,
            })
            log.info(f"  Permutation [{layer}]: {frac_sig:.2%} significant")

    # 2. Wilcoxon: targeted vs. random ablation
    if effects_df is not None and not effects_df.empty:
        from ablation import wilcoxon_ablation_test
        for k in [25]:
            stat, pval, cohens_d = wilcoxon_ablation_test(effects_df, k=k)
            if pval is not None:
                results.append({
                    'test': 'wilcoxon_targeted_vs_random',
                    'layer': 'mid',
                    'metric': f'k{k}_pval',
                    'value': cohens_d,
                    'pval': pval,
                })

    # 3. Bootstrap CI for median Gap Closure
    if gc_df is not None and not gc_df.empty:
        steering_rows = gc_df[gc_df['ablation_type'] == 'steering']
        if not steering_rows.empty:
            # Best (α=2.0, β=0.25) row
            best = steering_rows[(steering_rows['alpha'] == 2.0) &
                                  (steering_rows['beta'] == 0.25)]
            if not best.empty:
                row = best.iloc[0]
                results.append({
                    'test': 'bootstrap_gc_ci',
                    'layer': 'mid',
                    'metric': 'median_gap_closure',
                    'value': row['gap_closure_median'],
                    'pval': None,
                })
                results.append({
                    'test': 'bootstrap_gc_ci',
                    'layer': 'mid',
                    'metric': 'ci_lo',
                    'value': row['gap_closure_ci_lo'],
                    'pval': None,
                })
                results.append({
                    'test': 'bootstrap_gc_ci',
                    'layer': 'mid',
                    'metric': 'ci_hi',
                    'value': row['gap_closure_ci_hi'],
                    'pval': None,
                })

    # 4. Hypergeometric test for cross-layer overlap
    if feature_dfs is not None and len(feature_dfs) >= 2:
        layer_names_list = list(feature_dfs.keys())
        for i in range(len(layer_names_list)):
            for j in range(i+1, len(layer_names_list)):
                la, lb = layer_names_list[i], layer_names_list[j]
                df_a = feature_dfs[la]
                df_b = feature_dfs[lb]

                if df_a.empty or df_b.empty:
                    continue

                top_a = set(df_a.nlargest(50, 'cds_avg')['feature_id'].values)
                top_b = set(df_b.nlargest(50, 'cds_avg')['feature_id'].values)

                n_total = max(df_a['feature_id'].max(), df_b['feature_id'].max()) + 1
                n_success = len(top_a & top_b)
                pval_hyp = stats.hypergeom.sf(
                    n_success - 1,
                    n_total,    # population
                    50,         # K: top features in A
                    50,         # n: top features in B
                )
                jaccard = n_success / len(top_a | top_b) if (top_a | top_b) else 0

                results.append({
                    'test': 'hypergeometric_cross_layer',
                    'layer': f'{la}_vs_{lb}',
                    'metric': 'overlap_jaccard',
                    'value': jaccard,
                    'pval': pval_hyp,
                })
                log.info(f"  Cross-layer [{la}|{lb}]: Jaccard={jaccard:.3f}, p={pval_hyp:.4f}")

    # 5. BH-FDR for tests with p-values
    pval_rows = [r for r in results if r['pval'] is not None]
    if len(pval_rows) >= 2:
        pvals_arr = [r['pval'] for r in pval_rows]
        reject, pvals_adj, _, _ = multipletests(pvals_arr, method='fdr_bh')
        for r, p_adj in zip(pval_rows, pvals_adj):
            r['pval_bh'] = p_adj
    for r in results:
        if 'pval_bh' not in r:
            r['pval_bh'] = None

    df = pd.DataFrame(results)
    out = Path(cfg('paths', 'results')) / 'stats.tsv'
    df.to_csv(out, sep='\t', index=False)
    log.info(f"Saved stats: {out}")
    return df
