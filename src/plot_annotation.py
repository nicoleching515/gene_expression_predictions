"""
Generate annotation figures (Fig 3 heatmap + Fig 6 case studies)
from HOMER motif, ChromHMM, and GO enrichment results.
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')

CHROMHMM_STATE_NAMES = {
    '1': 'TssA', '2': 'TssAFlnk', '3': 'TxFlnk', '4': 'Tx',
    '5': 'TxWk', '6': 'EnhG', '7': 'Enh', '8': 'ZNF/Rpts',
    '9': 'Het', '10': 'TssBiv', '11': 'BivFlnk', '12': 'EnhBiv',
    '13': 'ReprPC', '14': 'ReprPCWk', '15': 'Quies',
}

# Functional groups for ChromHMM states
ACTIVE_STATES   = {'TssA', 'TssAFlnk', 'TxFlnk', 'Tx', 'TxWk', 'EnhG', 'Enh'}
REPRESSED_STATES = {'Het', 'TssBiv', 'BivFlnk', 'EnhBiv', 'ReprPC', 'ReprPCWk', 'Quies'}


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_homer_top_motifs(homer_dir: Path, layer, side, pair, n=5):
    """Return top-n known motifs by p-value for a given combination."""
    result_file = homer_dir / f'{layer}_{side}_{pair}' / 'knownResults.txt'
    if not result_file.exists():
        return []
    try:
        df = pd.read_csv(result_file, sep='\t')
        # Column names vary; find p-value column
        pval_col = next((c for c in df.columns if 'p-value' in c.lower() or 'pvalue' in c.lower()), None)
        name_col = df.columns[0]
        if pval_col is None:
            return []
        df = df.sort_values(pval_col).head(n)
        return list(df[name_col].str.split('(').str[0].str.strip())
    except Exception:
        return []


def load_chromhmm_fractions(chromhmm_dir: Path, layer, side, pair):
    """Return dict state_name -> fraction for a given combination."""
    tsv = chromhmm_dir / f'{layer}_{side}_{pair}_states.tsv'
    if not tsv.exists():
        return {}
    try:
        df = pd.read_csv(tsv, sep='\t')
        df['state_name'] = df['state'].astype(str).map(CHROMHMM_STATE_NAMES).fillna(df['state'].astype(str))
        return dict(zip(df['state_name'], df['fraction']))
    except Exception:
        return {}


def load_go_top_terms(go_dir: Path, layer, side, pair, n=5):
    """Return top-n GO:BP terms by adjusted p-value."""
    tsv = go_dir / f'{layer}_{side}_{pair}_go.tsv'
    if not tsv.exists():
        return []
    try:
        df = pd.read_csv(tsv, sep='\t')
        if df.empty:
            return []
        pval_col = next((c for c in df.columns if 'adj' in c.lower() or 'fdr' in c.lower()), df.columns[-1])
        name_col = next((c for c in df.columns if 'name' in c.lower() or 'term' in c.lower()), df.columns[0])
        df = df.sort_values(pval_col).head(n)
        terms = []
        for t in df[name_col]:
            t = str(t)
            if len(t) > 35:
                t = t[:33] + '…'
            terms.append(t)
        return terms
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: ChromHMM annotation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def figure3_annotation_heatmap(annotation_dir: Path, figures_dir: Path, layers, pairs=None):
    """
    Fig 3 annotation heatmap: fraction of top-feature windows overlapping each
    ChromHMM state, broken down by layer × context pair × vitro/vivo.
    """
    if pairs is None:
        pairs = ['blood', 'liver', 'lymph']

    chromhmm_dir = annotation_dir / 'chromhmm'
    homer_dir    = annotation_dir / 'homer'

    active_states = ['TssA', 'TssAFlnk', 'Enh', 'EnhG', 'Tx', 'TxWk', 'TxFlnk']
    repressed     = ['ReprPC', 'ReprPCWk', 'Het', 'Quies']
    show_states   = active_states + repressed

    # Build matrix: rows = (layer, side, pair), cols = states
    row_labels = []
    mat_rows   = []
    for layer in layers:
        for pair in pairs:
            for side in ['vivo', 'vitro']:
                fracs = load_chromhmm_fractions(chromhmm_dir, layer, side, pair)
                row = [fracs.get(s, 0.0) for s in show_states]
                mat_rows.append(row)
                row_labels.append(f'{layer}/{pair}/{side}')

    if not mat_rows or all(sum(r) == 0 for r in mat_rows):
        _write_placeholder(figures_dir, 'fig3_annotation_heatmap',
                           'ChromHMM Annotation\n(no data yet — run ChromHMM step first)')
        return

    mat = np.array(mat_rows)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(row_labels) * 0.35 + 2)),
                             gridspec_kw={'width_ratios': [3, 1]})

    # Left: ChromHMM state fractions
    ax = axes[0]
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=min(mat.max(), 1.0))
    ax.set_xticks(range(len(show_states)))
    ax.set_xticklabels(show_states, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title('ChromHMM State Enrichment\n(fraction of top-feature windows)', fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.6, label='Fraction of windows')

    # Right: active vs repressed summary bars
    ax2 = axes[1]
    active_idx   = [show_states.index(s) for s in active_states if s in show_states]
    repressed_idx = [show_states.index(s) for s in repressed  if s in show_states]
    active_frac   = mat[:, active_idx].sum(axis=1)
    repressed_frac = mat[:, repressed_idx].sum(axis=1)

    y = np.arange(len(row_labels))
    ax2.barh(y, active_frac,   color='#D65F5F', alpha=0.8, label='Active chromatin')
    ax2.barh(y, -repressed_frac, color='#4878CF', alpha=0.8, label='Repressed chromatin')
    ax2.axvline(0, color='black', lw=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(row_labels, fontsize=7)
    ax2.set_xlabel('Fraction', fontsize=8)
    ax2.set_title('Active vs Repressed', fontsize=10)
    ax2.legend(fontsize=7, loc='lower right')

    fig.suptitle('Feature Annotation: ChromHMM Chromatin State Enrichment',
                 fontsize=12, weight='bold')
    plt.tight_layout()
    _save(fig, figures_dir, 'fig3_annotation_heatmap')
    print(f"  Saved fig3_annotation_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Case studies — motif + ChromHMM for top features
# ─────────────────────────────────────────────────────────────────────────────

def figure6_case_studies(annotation_dir: Path, figures_dir: Path, layers, pairs=None):
    """
    Fig 6 case studies: for each layer, show the top HOMER motifs (vivo vs vitro)
    and ChromHMM state breakdown side-by-side for the most divergent context pair.
    """
    if pairs is None:
        pairs = ['blood', 'liver', 'lymph']

    homer_dir    = annotation_dir / 'homer'
    chromhmm_dir = annotation_dir / 'chromhmm'
    go_dir       = annotation_dir / 'go'

    # One row per layer; 3 columns: motifs (vivo), motifs (vitro), ChromHMM bars
    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 3, figsize=(16, 4.5 * n_layers))
    if n_layers == 1:
        axes = axes[None, :]

    # Pair with most ChromHMM signal per layer (use lymph as default — highest CDS)
    priority_pairs = ['lymph', 'liver', 'blood']

    for ri, layer in enumerate(layers):
        # Pick the pair with most chromhmm data
        chosen_pair = next(
            (p for p in priority_pairs
             if (chromhmm_dir / f'{layer}_vivo_{p}_states.tsv').exists()),
            pairs[0]
        )

        # ── Motifs: vivo ──────────────────────────────────────────────────────
        ax_mv = axes[ri][0]
        vivo_motifs = load_homer_top_motifs(homer_dir, layer, 'vivo', chosen_pair, n=8)
        if vivo_motifs:
            y = range(len(vivo_motifs))
            ax_mv.barh(y, [1.0] * len(vivo_motifs), color='#D65F5F', alpha=0.7)
            ax_mv.set_yticks(y)
            ax_mv.set_yticklabels(vivo_motifs, fontsize=8)
            ax_mv.set_xlim(0, 1.5)
            ax_mv.set_xticks([])
        else:
            ax_mv.text(0.5, 0.5, 'No HOMER results\n(run Step 2 first)',
                       ha='center', va='center', fontsize=9, transform=ax_mv.transAxes,
                       color='gray')
        ax_mv.set_title(f'{layer} / {chosen_pair} — vivo top motifs', fontsize=9)
        ax_mv.invert_yaxis()

        # ── Motifs: vitro ─────────────────────────────────────────────────────
        ax_mc = axes[ri][1]
        vitro_motifs = load_homer_top_motifs(homer_dir, layer, 'vitro', chosen_pair, n=8)
        if vitro_motifs:
            y = range(len(vitro_motifs))
            ax_mc.barh(y, [1.0] * len(vitro_motifs), color='#4878CF', alpha=0.7)
            ax_mc.set_yticks(y)
            ax_mc.set_yticklabels(vitro_motifs, fontsize=8)
            ax_mc.set_xlim(0, 1.5)
            ax_mc.set_xticks([])
        else:
            ax_mc.text(0.5, 0.5, 'No HOMER results\n(run Step 2 first)',
                       ha='center', va='center', fontsize=9, transform=ax_mc.transAxes,
                       color='gray')
        ax_mc.set_title(f'{layer} / {chosen_pair} — vitro top motifs', fontsize=9)
        ax_mc.invert_yaxis()

        # ── ChromHMM: vivo vs vitro stacked bars ──────────────────────────────
        ax_ch = axes[ri][2]
        show_states = ['TssA', 'TssAFlnk', 'Enh', 'EnhG', 'Tx', 'TxWk',
                       'ReprPC', 'ReprPCWk', 'Het', 'Quies']
        state_colors = {
            'TssA': '#E41A1C', 'TssAFlnk': '#FF7F00', 'Enh': '#FFFF33',
            'EnhG': '#F781BF', 'Tx': '#4DAF4A', 'TxWk': '#A6D96A',
            'ReprPC': '#377EB8', 'ReprPCWk': '#6BAED6',
            'Het': '#984EA3', 'Quies': '#CCCCCC', 'TxFlnk': '#B2DF8A',
        }
        fracs_vivo  = load_chromhmm_fractions(chromhmm_dir, layer, 'vivo',  chosen_pair)
        fracs_vitro = load_chromhmm_fractions(chromhmm_dir, layer, 'vitro', chosen_pair)

        if fracs_vivo or fracs_vitro:
            x       = np.arange(2)
            labels  = ['vivo', 'vitro']
            bottoms = [0.0, 0.0]
            for state in show_states:
                vals = [fracs_vivo.get(state, 0), fracs_vitro.get(state, 0)]
                ax_ch.bar(x, vals, bottom=bottoms,
                          color=state_colors.get(state, '#AAAAAA'),
                          label=state, alpha=0.85)
                bottoms = [bottoms[i] + vals[i] for i in range(2)]
            ax_ch.set_xticks(x)
            ax_ch.set_xticklabels(labels, fontsize=9)
            ax_ch.set_ylabel('Fraction of windows', fontsize=8)
            ax_ch.legend(fontsize=6, loc='upper right', ncol=2)
        else:
            ax_ch.text(0.5, 0.5, 'No ChromHMM data\n(run Step 3 first)',
                       ha='center', va='center', fontsize=9, transform=ax_ch.transAxes,
                       color='gray')
        ax_ch.set_title(f'{layer} / {chosen_pair} — ChromHMM states', fontsize=9)

    fig.suptitle('Case Studies: Motif Enrichment & Chromatin State Annotation\nof Top Context-Divergent SAE Features',
                 fontsize=11, weight='bold')
    plt.tight_layout()
    _save(fig, figures_dir, 'fig6_case_studies')
    print(f"  Saved fig6_case_studies")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, out_dir: Path, name: str, dpi=300):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ('pdf', 'png'):
        fig.savefig(out_dir / f'{name}.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _write_placeholder(figures_dir: Path, name: str, msg: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=13,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', lw=2))
    ax.set_axis_off()
    _save(fig, figures_dir, name)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--annotation_dir', default='/workspace/outputs/annotation')
    p.add_argument('--figures_dir',    default='/workspace/results/figures')
    p.add_argument('--layers',         default='early mid late')
    p.add_argument('--n_top',          type=int, default=50)
    args = p.parse_args()

    annotation_dir = Path(args.annotation_dir)
    figures_dir    = Path(args.figures_dir)
    layers         = args.layers.split()

    print(f"Generating annotation figures → {figures_dir}")
    figure3_annotation_heatmap(annotation_dir, figures_dir, layers)
    figure6_case_studies(annotation_dir, figures_dir, layers)
    print("Done.")


if __name__ == '__main__':
    main()
