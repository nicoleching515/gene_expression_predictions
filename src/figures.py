"""
Figure generation for ICML paper.

Reads cached results — no new GPU runs.
All figures saved at 300 DPI in PDF + PNG.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, cfg, LAYER_NAMES

log = get_logger("figures")

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

FIGURES_DIR = None
DPI = None
FORMATS = None


def setup_figures():
    global FIGURES_DIR, DPI, FORMATS
    FIGURES_DIR = Path(cfg('paths', 'results')) / 'figures'
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DPI = cfg('figures', 'dpi')
    FORMATS = cfg('figures', 'formats')
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")


def save_fig(fig, name):
    for fmt in FORMATS:
        path = FIGURES_DIR / f'{name}.{fmt}'
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        log.info(f"  Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: CDS distribution
# ─────────────────────────────────────────────────────────────────────────────

def figure2_cds(feature_dfs, null_cds_dict):
    """
    Fig 2: CDS distribution histogram + null overlay + stacked bar of categories.

    Parameters
    ----------
    feature_dfs : {layer_name: DataFrame}
    null_cds_dict: {layer_name: ndarray (n_perm, d_latent)}
    """
    setup_figures()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    layer_names = LAYER_NAMES

    categories  = ['shared', 'vivo_enriched', 'vitro_enriched', 'context_switched', 'other']
    cat_colors  = {'shared': '#4878CF', 'vivo_enriched': '#D65F5F',
                   'vitro_enriched': '#67BF5C', 'context_switched': '#B47CC7', 'other': '#AAB7B8'}

    for col, layer in enumerate(layer_names):
        if layer not in feature_dfs:
            continue
        df = feature_dfs[layer]

        # Top row: CDS histogram + null
        ax = axes[0, col]
        cds = df['cds_avg'].values
        ax.hist(cds, bins=60, density=True, alpha=0.7, color='steelblue', label='Real CDS')

        if layer in null_cds_dict and null_cds_dict[layer] is not None:
            null_flat = null_cds_dict[layer].flatten()
            ax.hist(null_flat, bins=60, density=True, alpha=0.4,
                    color='gray', label='Permuted null')

        ax.set_xlabel('CDS (mean vivo − vitro activation)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'Layer: {layer}', fontsize=10)
        ax.legend(fontsize=8)

        # Vertical lines for category thresholds
        for pct, color, label in [
            (cfg('analysis', 'cds_vivo_pct'),  'red',   '90th pct'),
            (cfg('analysis', 'cds_vitro_pct'), 'green', '10th pct'),
        ]:
            thresh = np.percentile(np.abs(cds), pct)
            ax.axvline(thresh,  color=color, ls='--', lw=1, alpha=0.7)
            ax.axvline(-thresh, color=color, ls='--', lw=1, alpha=0.7)

        # Bottom row: stacked bar of feature categories
        ax2 = axes[1, col]
        cat_counts = df['category'].value_counts()
        total = len(df)
        fractions = [cat_counts.get(c, 0) / total for c in categories]
        bottom = 0
        for cat, frac in zip(categories, fractions):
            ax2.bar(0, frac, bottom=bottom, color=cat_colors[cat], label=cat, width=0.5)
            if frac > 0.03:
                ax2.text(0, bottom + frac / 2, f'{frac:.1%}',
                         ha='center', va='center', fontsize=8, color='white', weight='bold')
            bottom += frac

        ax2.set_ylim(0, 1)
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_ylabel('Fraction of features', fontsize=9)
        ax2.set_xticks([])
        if col == 2:
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    fig.suptitle('Context Divergence Score (CDS) Distribution', fontsize=12, weight='bold')
    plt.tight_layout()
    save_fig(fig, 'fig2_cds_distribution')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Ablation
# ─────────────────────────────────────────────────────────────────────────────

def figure4_ablation(effects_df):
    """
    Fig 4: Ablation bar chart at k=25 + dose-response line plot.
    """
    setup_figures()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: bar chart at k=25
    k25 = effects_df[effects_df['k'] == 25]
    abl_types = ['targeted', 'top_activation', 'random']
    colors = {'targeted': '#D65F5F', 'top_activation': '#B47CC7', 'random': '#AAB7B8'}

    means = []
    stds  = []
    labels = []
    for at in abl_types:
        sub = k25[k25['ablation_type'] == at]
        if sub.empty:
            means.append(0); stds.append(0)
        else:
            means.append(sub['delta_mean'].mean())
            stds.append(sub['delta_std'].mean() if len(sub) > 1 else sub['delta_mean'].std())
        labels.append(at.replace('_', ' ').title())

    x = np.arange(len(abl_types))
    bars = ax1.bar(x, means, yerr=stds, capsize=4,
                   color=[colors[at] for at in abl_types], alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('Mean |Δŷ| (prediction change)', fontsize=9)
    ax1.set_title('Ablation at k=25 features', fontsize=10)

    # Right: dose-response line plot
    k_vals = sorted(effects_df['k'].unique())
    for at, color in colors.items():
        sub = effects_df[effects_df['ablation_type'] == at]
        if sub.empty:
            continue
        means_k = []
        stds_k  = []
        for k in k_vals:
            sk = sub[sub['k'] == k]
            if sk.empty:
                means_k.append(np.nan); stds_k.append(0)
            else:
                means_k.append(sk['delta_mean'].mean())
                stds_k.append(sk['delta_std'].mean())

        means_k = np.array(means_k)
        stds_k  = np.array(stds_k)
        ax2.plot(k_vals, means_k, marker='o', label=at.replace('_',' ').title(),
                 color=color, linewidth=2)
        ax2.fill_between(k_vals, means_k - stds_k, means_k + stds_k,
                         alpha=0.2, color=color)

    ax2.set_xlabel('k (features ablated)', fontsize=9)
    ax2.set_ylabel('Mean |Δŷ|', fontsize=9)
    ax2.set_title('Dose-response: ablation strength', fontsize=10)
    ax2.legend(fontsize=8)

    fig.suptitle('Ablation Experiments', fontsize=12, weight='bold')
    plt.tight_layout()
    save_fig(fig, 'fig4_ablation')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Steering Gap Closure
# ─────────────────────────────────────────────────────────────────────────────

def figure5_steering(gc_df):
    """
    Fig 5: Gap Closure scatter + box plots with baselines.
    """
    setup_figures()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: heatmap of (α, β) sweep
    steering = gc_df[gc_df['ablation_type'] == 'steering'].copy()
    if not steering.empty:
        pivot = steering.pivot(index='alpha', columns='beta', values='gap_closure_median')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1,
                    vmin=0, vmax=1, linewidths=0.5)
        ax1.set_title('Gap Closure by (α, β)', fontsize=10)
        ax1.set_xlabel('β (vitro suppression)', fontsize=9)
        ax1.set_ylabel('α (vivo amplification)', fontsize=9)

    # Right: bar chart comparing baselines
    baseline_data = {}
    if not steering.empty:
        row = steering.nlargest(1, 'gap_closure_median').iloc[0]
        baseline_data['SAE Steering\n(best α,β)'] = row['gap_closure_median']

    # Fixed (α=2, β=0.25) row
    fixed = steering[(steering['alpha'] == 2.0) & (steering['beta'] == 0.25)]
    if not fixed.empty:
        baseline_data['SAE Steering\n(α=2, β=0.25)'] = fixed.iloc[0]['gap_closure_median']

    # Random steering
    if not steering.empty:
        rand_gc = steering['rand_gc_median'].mean()
        baseline_data['Random Steering'] = rand_gc

    # Direct context swap
    direct = gc_df[gc_df['ablation_type'] == 'direct_context_swap']
    if not direct.empty:
        baseline_data['Direct\nContext Swap'] = direct.iloc[0]['gap_closure_median']

    if baseline_data:
        x = np.arange(len(baseline_data))
        colors = ['#D65F5F', '#F47F72', '#AAB7B8', '#4878CF']
        bars = ax2.bar(x, list(baseline_data.values()),
                       color=colors[:len(baseline_data)], alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(list(baseline_data.keys()), fontsize=8)
        ax2.set_ylabel('Median Gap Closure', fontsize=9)
        ax2.set_ylim(0, 1.1)
        ax2.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5, label='Perfect')
        ax2.set_title('Gap Closure vs. Baselines', fontsize=10)
        ax2.legend(fontsize=8)

    fig.suptitle('Context Steering: Gap Closure', fontsize=12, weight='bold')
    plt.tight_layout()
    save_fig(fig, 'fig5_steering')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Cross-layer validation (Jaccard of top-50 features)
# ─────────────────────────────────────────────────────────────────────────────

def figure7_cross_layer(feature_dfs):
    """
    Fig 7: Jaccard overlap of top-50 divergent features across layers.
    """
    setup_figures()
    fig, ax = plt.subplots(figsize=(7, 6))

    n_layers = len(LAYER_NAMES)
    jaccard_matrix = np.eye(n_layers)

    for i, la in enumerate(LAYER_NAMES):
        for j, lb in enumerate(LAYER_NAMES):
            if i == j or la not in feature_dfs or lb not in feature_dfs:
                continue
            df_a = feature_dfs[la]
            df_b = feature_dfs[lb]
            if df_a.empty or df_b.empty:
                continue
            top_a = set(df_a.nlargest(50, 'cds_avg')['feature_id'].values)
            top_b = set(df_b.nlargest(50, 'cds_avg')['feature_id'].values)
            inter = len(top_a & top_b)
            union = len(top_a | top_b)
            jaccard_matrix[i, j] = inter / union if union > 0 else 0.0

    sns.heatmap(
        jaccard_matrix,
        annot=True, fmt='.3f', cmap='Blues',
        xticklabels=LAYER_NAMES, yticklabels=LAYER_NAMES,
        ax=ax, vmin=0, vmax=1, linewidths=0.5,
        annot_kws={'size': 12},
    )
    ax.set_title('Cross-layer Jaccard Overlap of Top-50 Divergent Features', fontsize=11)
    ax.set_xlabel('Layer', fontsize=9)
    ax.set_ylabel('Layer', fontsize=9)

    plt.tight_layout()
    save_fig(fig, 'fig7_cross_layer')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Generate all available figures
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures():
    """
    Load cached results and generate all available figures.
    Figures requiring Bio team output are marked [BIO-PENDING].
    """
    setup_figures()
    results_dir = Path(cfg('paths', 'results'))

    # Load feature tables
    feature_dfs = {}
    null_cds_dict = {}
    for layer in LAYER_NAMES:
        feat_path = results_dir / 'cds' / f'layer_{layer}_features.tsv'
        null_path = results_dir / 'cds' / f'layer_{layer}_null_cds.npy'
        if feat_path.exists():
            feature_dfs[layer] = pd.read_csv(feat_path, sep='\t')
            log.info(f"Loaded feature table: {feat_path}")
        if null_path.exists():
            null_cds_dict[layer] = np.load(null_path)

    # Load ablation results
    abl_path = results_dir / 'ablation' / 'effects.tsv'
    effects_df = pd.read_csv(abl_path, sep='\t') if abl_path.exists() else pd.DataFrame()

    # Load steering results
    gc_path = results_dir / 'steering' / 'gap_closure.tsv'
    gc_df = pd.read_csv(gc_path, sep='\t') if gc_path.exists() else pd.DataFrame()

    generated = []

    if feature_dfs:
        log.info("Generating Fig 2 (CDS distribution) ...")
        figure2_cds(feature_dfs, null_cds_dict)
        generated.append('fig2')

    if not effects_df.empty:
        log.info("Generating Fig 4 (ablation) ...")
        figure4_ablation(effects_df)
        generated.append('fig4')

    if not gc_df.empty:
        log.info("Generating Fig 5 (steering) ...")
        figure5_steering(gc_df)
        generated.append('fig5')

    if len(feature_dfs) >= 2:
        log.info("Generating Fig 7 (cross-layer) ...")
        figure7_cross_layer(feature_dfs)
        generated.append('fig7')

    # Bio-pending placeholders
    for fig_name, title in [('fig3_feature_annotation', 'Feature Annotation Heatmap'),
                              ('fig6_case_studies', 'Case Studies')]:
        _write_bio_pending(fig_name, title)
        generated.append(fig_name)

    log.info(f"Generated figures: {generated}")
    return generated


def _write_bio_pending(name, title):
    """Write a placeholder figure for bio-team-dependent figures."""
    setup_figures()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, f'[BIO-PENDING]\n{title}\n\nAwating Bio team annotation data',
            ha='center', va='center', fontsize=14, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', lw=2))
    ax.set_axis_off()
    fig.suptitle(title, fontsize=12)
    save_fig(fig, name)


if __name__ == '__main__':
    generate_all_figures()
