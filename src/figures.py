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

        # Vertical lines for category thresholds (raw CDS, not |CDS|)
        vivo_pct  = cfg('analysis', 'cds_vivo_pct')
        vitro_pct = cfg('analysis', 'cds_vitro_pct')
        vivo_thresh  = np.percentile(cds, vivo_pct)
        vitro_thresh = np.percentile(cds, vitro_pct)
        ax.axvline(vivo_thresh,  color='red',   ls='--', lw=1, alpha=0.7, label=f'vivo {vivo_pct}th pct')
        ax.axvline(vitro_thresh, color='green', ls='--', lw=1, alpha=0.7, label=f'vitro {vitro_pct}th pct')

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
    # Clip error bars so lower bound never goes below zero (|Δŷ| is non-negative)
    err_lo = [min(s, m) for m, s in zip(means, stds)]
    bars = ax1.bar(x, means, yerr=[err_lo, stds], capsize=4,
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
        vmax_data = pivot.values[~np.isnan(pivot.values)].max() if pivot.size > 0 else 0.2
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1,
                    vmin=0, vmax=max(vmax_data, 0.01), linewidths=0.5)
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
        all_vals = list(baseline_data.values())
        y_min = min(0, min(all_vals) - 0.05)
        ax2.set_ylim(y_min, 1.1)
        ax2.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5, label='Perfect')
        ax2.set_title('Gap Closure vs. Baselines', fontsize=10)
        ax2.legend(fontsize=8)

    fig.suptitle('Context Steering: Gap Closure', fontsize=12, weight='bold')
    plt.tight_layout()
    save_fig(fig, 'fig5_steering')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Feature annotation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def figure3_feature_annotation(feature_dfs, top_n=20):
    """
    Fig 3: CDS heatmap of top divergent features per layer.

    Each subplot shows the top `top_n` features (by |Cds_avg|) for one layer,
    with per-context-pair CDS values as columns and features as rows.
    Features are sorted by cds_avg (vivo-enriched at top, vitro-enriched at
    bottom) and annotated with their category.
    """
    setup_figures()
    pair_cols = ['cds_blood', 'cds_liver', 'cds_lymph']
    pair_labels = ['Blood', 'Liver', 'Lymph']
    cat_colors = {
        'vivo_enriched':    '#D65F5F',
        'vitro_enriched':   '#4878CF',
        'context_switched': '#B47CC7',
        'shared':           '#67BF5C',
        'other':            '#AAB7B8',
    }

    n_layers = len(LAYER_NAMES)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 8))
    if n_layers == 1:
        axes = [axes]

    for ax, layer in zip(axes, LAYER_NAMES):
        if layer not in feature_dfs:
            ax.set_visible(False)
            continue
        df = feature_dfs[layer]

        # Top features by |CDS avg|; sort descending so vivo-enriched is on top
        top_df = df.reindex(df['cds_avg'].abs().nlargest(top_n).index)
        top_df = top_df.sort_values('cds_avg', ascending=False)

        # Only keep pair columns that actually exist
        present_cols = [c for c in pair_cols if c in top_df.columns]
        present_labels = [pair_labels[pair_cols.index(c)] for c in present_cols]

        mat = top_df[present_cols].values.astype(float)
        abs_max = np.abs(mat).max() if mat.size else 1.0

        im = ax.imshow(mat, aspect='auto', cmap='RdBu_r',
                       vmin=-abs_max, vmax=abs_max)

        ax.set_xticks(np.arange(len(present_labels)))
        ax.set_xticklabels(present_labels, fontsize=9)
        ax.set_yticks(np.arange(len(top_df)))

        # Row labels: feature id + category badge
        y_labels = []
        for _, row in top_df.iterrows():
            y_labels.append(f"F{int(row['feature_id'])}")
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.set_title(f'Layer: {layer}', fontsize=10)

        # Annotate cells with CDS value
        for ri in range(mat.shape[0]):
            for ci in range(mat.shape[1]):
                val = mat[ri, ci]
                color = 'white' if abs(val) > abs_max * 0.5 else 'black'
                ax.text(ci, ri, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)

        # Right-side category color strip
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(np.arange(len(top_df)))
        ax2.set_yticklabels(
            [row['category'] for _, row in top_df.iterrows()],
            fontsize=6,
        )
        for ticklabel, (_, row) in zip(ax2.get_yticklabels(), top_df.iterrows()):
            ticklabel.set_color(cat_colors.get(row['category'], '#AAB7B8'))

        plt.colorbar(im, ax=ax, shrink=0.6, pad=0.18,
                     label='CDS (vivo − vitro)')

    fig.suptitle(
        'Feature Annotation Heatmap: Context Divergence Score by Condition Pair',
        fontsize=12, weight='bold',
    )
    plt.tight_layout()
    save_fig(fig, 'fig3_feature_annotation')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Case studies
# ─────────────────────────────────────────────────────────────────────────────

def figure6_case_studies(feature_dfs):
    """
    Fig 6: Activation profiles for 3 illustrative case-study features.

    Selects:
      1. Top vivo-enriched feature from the late layer
      2. Top vitro-enriched feature from the late layer
      3. Top vivo-enriched feature from the mid layer (cross-layer contrast)

    For each feature shows:
      Left  — mean activation (vitro vs vivo) per context pair
      Right — CDS per context pair
    """
    setup_figures()
    pair_cols   = ['cds_blood', 'cds_liver', 'cds_lymph']
    pair_labels = ['Blood', 'Liver', 'Lymph']
    act_pairs = {
        'cds_blood': ('mean_vitro', 'mean_vivo', 'Blood'),
        'cds_liver': ('mean_vitro', 'mean_vivo', 'Liver'),
        'cds_lymph': ('mean_vitro', 'mean_vivo', 'Lymph'),
    }

    def pick_feature(layer, category, largest=True):
        df = feature_dfs.get(layer)
        if df is None:
            return None, None
        sub = df[df['category'] == category]
        if sub.empty:
            return None, None
        col = 'cds_avg'
        row = sub.nlargest(1, col).iloc[0] if largest else sub.nsmallest(1, col).iloc[0]
        return layer, row

    candidates = [
        pick_feature('late', 'vivo_enriched',  largest=True),
        pick_feature('late', 'vitro_enriched', largest=False),
        pick_feature('mid',  'vivo_enriched',  largest=True),
    ]
    candidates = [(l, r) for l, r in candidates if l is not None]

    if not candidates:
        _write_bio_pending('fig6_case_studies', 'Case Studies')
        return None

    n_cases = len(candidates)
    fig, axes = plt.subplots(n_cases, 2, figsize=(11, 4 * n_cases))
    if n_cases == 1:
        axes = [axes]

    colors_vitro = '#4878CF'
    colors_vivo  = '#D65F5F'

    for row_i, (layer, feat_row) in enumerate(candidates):
        fid      = int(feat_row['feature_id'])
        category = feat_row['category']
        cds_avg  = feat_row['cds_avg']

        present_pairs = [c for c in pair_cols if c in feat_row.index]
        present_labels = [pair_labels[pair_cols.index(c)] for c in present_pairs]
        cds_vals = [feat_row[c] for c in present_pairs]

        # Left panel: mean activation vitro vs vivo (pooled across pairs shown as groups)
        ax_act = axes[row_i][0]
        x = np.arange(len(present_labels))
        w = 0.35
        ax_act.bar(x - w / 2, [feat_row['mean_vitro']] * len(present_labels),
                   w, color=colors_vitro, alpha=0.8, label='In vitro')
        ax_act.bar(x + w / 2, [feat_row['mean_vivo']] * len(present_labels),
                   w, color=colors_vivo,  alpha=0.8, label='In vivo')
        ax_act.set_xticks(x)
        ax_act.set_xticklabels(present_labels, fontsize=9)
        ax_act.set_ylabel('Mean SAE activation', fontsize=9)
        ax_act.set_title(
            f'Feature {fid} ({layer}, {category})\nMean activation by context',
            fontsize=9,
        )
        ax_act.legend(fontsize=8)

        # Right panel: CDS per context pair
        ax_cds = axes[row_i][1]
        bar_colors = [colors_vivo if v > 0 else colors_vitro for v in cds_vals]
        ax_cds.bar(x, cds_vals, color=bar_colors, alpha=0.8)
        ax_cds.axhline(0, color='black', lw=0.8)
        ax_cds.axhline(cds_avg, color='gray', lw=1, ls='--',
                       label=f'Mean CDS = {cds_avg:.2f}')
        ax_cds.set_xticks(x)
        ax_cds.set_xticklabels(present_labels, fontsize=9)
        ax_cds.set_ylabel('CDS (vivo − vitro)', fontsize=9)
        ax_cds.set_title(
            f'Feature {fid} ({layer}, {category})\nCDS per context pair',
            fontsize=9,
        )
        ax_cds.legend(fontsize=8)

    fig.suptitle('Case Studies: Top Context-Divergent SAE Features',
                 fontsize=12, weight='bold')
    plt.tight_layout()
    save_fig(fig, 'fig6_case_studies')
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

    if feature_dfs:
        log.info("Generating Fig 3 (feature annotation heatmap) ...")
        figure3_feature_annotation(feature_dfs)
        generated.append('fig3')

        log.info("Generating Fig 6 (case studies) ...")
        figure6_case_studies(feature_dfs)
        generated.append('fig6')

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
