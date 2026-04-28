#!/usr/bin/env python3
"""
src/plot_homer_go_figures.py
============================
Generates two publication-quality figures from HOMER and rGREAT GO:BP outputs:

  fig8_homer_motifs.pdf  --  Heatmap of -log10(p-value) for the top-N motifs
                             across all 18 conditions (3 layers x 3 pairs x 2 contexts).
                             Separate panels for vivo-enriched and vitro-enriched
                             conditions for clean comparison.

  fig9_go_enrichment.pdf --  Dot-plot of top GO:BP terms per layer x pair x side.
                             Dot size = fold enrichment, colour = -log10(FDR q-value).
                             Faceted as 3 x 3 grid (layers x pairs).

Usage:
    python3 src/plot_homer_go_figures.py \
        --annotation_dir outputs/annotation \
        --figures_dir    results/figures \
        --layers         "early mid late" \
        --n_top_motifs   10 \
        --n_top_go       10

Expected directory layout (produced by run_annotation.sh v2):
    outputs/annotation/
        homer/
            early_vitro_blood/knownResults.txt
            early_vivo_blood/knownResults.txt
            ...  (18 directories total)
        go/
            early_vitro_blood_go.tsv
            early_vivo_blood_go.tsv
            ...  (18 files total)
"""

import argparse
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=UserWarning)

# ── Colour palette ────────────────────────────────────────────────────────────
VITRO_COL = "#E04B4B"
VIVO_COL  = "#3A7DC9"
LAYERS    = ["early", "mid", "late"]
PAIRS     = ["blood", "liver", "lymph"]
SIDES     = ["vitro", "vivo"]
LAYER_LABELS = {"early": "Early (L/4)", "mid": "Mid (L/2)", "late": "Late (3L/4)"}
PAIR_LABELS  = {
    "blood": "Blood\n(K562/HSC)",
    "liver": "Liver\n(HepG2/Liver)",
    "lymph": "Lymph\n(GM12878/NaiveB)",
}


# =============================================================================
# Parsers
# =============================================================================

def parse_homer_known(homer_dir: str, n: int = 10) -> pd.DataFrame:
    """
    Parse HOMER knownResults.txt into a DataFrame with columns:
        motif_name, log_pvalue (negative, so larger = more significant)
    Returns empty DataFrame if file not found.
    """
    path = Path(homer_dir) / "knownResults.txt"
    if not path.exists():
        return pd.DataFrame(columns=["motif_name", "neg_log_p"])
    try:
        df = pd.read_csv(path, sep="\t", comment="#")
        # Normalise column names (HOMER uses varying capitalisations)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        # Find the p-value column
        pcol = next((c for c in df.columns if "p-value" in c or "pvalue" in c
                     or c == "p_value"), None)
        if pcol is None:
            return pd.DataFrame(columns=["motif_name", "neg_log_p"])
        namecol = df.columns[0]
        df = df[[namecol, pcol]].copy()
        df.columns = ["motif_name", "pval"]
        df["pval"] = pd.to_numeric(df["pval"], errors="coerce")
        df = df.dropna().sort_values("pval").head(n)
        # Clean motif name: keep only the TF name before the first "/"
        df["motif_name"] = df["motif_name"].str.split("(").str[0].str.strip()
        df["motif_name"] = df["motif_name"].str.split("/").str[0].str.strip()
        # Guard against log(0)
        df["pval"] = df["pval"].clip(lower=1e-300)
        df["neg_log_p"] = -np.log10(df["pval"])
        return df[["motif_name", "neg_log_p"]].reset_index(drop=True)
    except Exception as exc:
        print(f"  WARNING: could not parse {path}: {exc}")
        return pd.DataFrame(columns=["motif_name", "neg_log_p"])


def parse_go_tsv(tsv_path: str, n: int = 10) -> pd.DataFrame:
    """
    Parse rGREAT GO:BP TSV into a DataFrame with columns:
        description, fold_enrichment, neg_log_q
    Returns empty DataFrame if file not found.
    """
    path = Path(tsv_path)
    if not path.exists():
        return pd.DataFrame(columns=["description", "fold_enrichment", "neg_log_q"])
    try:
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        # Required columns
        if "description" not in df.columns:
            return pd.DataFrame(columns=["description", "fold_enrichment", "neg_log_q"])
        # Find fold enrichment column
        fcol = next((c for c in df.columns
                     if "fold" in c or "enrichment" in c), None)
        # Find adjusted p-value column
        qcol = next((c for c in df.columns
                     if "adjust" in c or "p_adjust" in c or "fdr" in c
                     or "q_value" in c), None)
        if fcol is None or qcol is None:
            return pd.DataFrame(columns=["description", "fold_enrichment", "neg_log_q"])
        df = df[["description", fcol, qcol]].copy()
        df.columns = ["description", "fold_enrichment", "q"]
        df["fold_enrichment"] = pd.to_numeric(df["fold_enrichment"], errors="coerce")
        df["q"] = pd.to_numeric(df["q"], errors="coerce")
        df = df.dropna().sort_values("q").head(n)
        df["q"] = df["q"].clip(lower=1e-300)
        df["neg_log_q"] = -np.log10(df["q"])
        # Truncate long GO term names
        df["description"] = df["description"].str[:45]
        return df[["description", "fold_enrichment", "neg_log_q"]].reset_index(drop=True)
    except Exception as exc:
        print(f"  WARNING: could not parse {path}: {exc}")
        return pd.DataFrame(columns=["description", "fold_enrichment", "neg_log_q"])


# =============================================================================
# Fig 8 — HOMER motif enrichment heatmap
# =============================================================================

def make_fig8_homer(annotation_dir: str, figures_dir: str,
                    layers: list, n_top: int = 10) -> None:
    """
    Two-panel heatmap:
      Left  panel: vivo conditions (3 layers x 3 pairs = 9 rows)
      Right panel: vitro conditions (3 layers x 3 pairs = 9 rows)
    Columns = top motifs (union of top-N across all conditions).
    Colour = -log10(p-value); grey = not tested / not significant.
    """
    homer_base = Path(annotation_dir) / "homer"

    # ── Collect all data ─────────────────────────────────────────────────────
    all_data = {}   # key: (layer, side, pair) -> {motif_name: neg_log_p}
    for layer in layers:
        for side in SIDES:
            for pair in PAIRS:
                tag   = f"{layer}_{side}_{pair}"
                hdir  = homer_base / tag
                df    = parse_homer_known(str(hdir), n=n_top)
                all_data[(layer, side, pair)] = dict(
                    zip(df["motif_name"], df["neg_log_p"])
                ) if not df.empty else {}

    # ── Build union motif list (top-N most frequently appearing) ─────────────
    from collections import Counter
    motif_counts: Counter = Counter()
    for d in all_data.values():
        motif_counts.update(d.keys())
    if not motif_counts:
        print("  WARNING: No HOMER results found. Saving empty placeholder for fig8.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No HOMER motif results found.\n"
                "Check outputs/annotation/homer/ for knownResults.txt files.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11,
                color="grey")
        ax.axis("off")
        fig.suptitle("Fig 8 — HOMER Motif Enrichment (no data)", fontsize=12)
        for ext in ("pdf", "png"):
            fig.savefig(Path(figures_dir) / f"fig8_homer_motifs.{ext}",
                        bbox_inches="tight", dpi=300)
        plt.close(fig)
        return

    top_motifs = [m for m, _ in motif_counts.most_common(n_top)]

    # Row labels: "Early / Blood" etc.
    row_labels = []
    row_keys   = []
    for side in SIDES:
        for layer in layers:
            for pair in PAIRS:
                row_labels.append(f"{LAYER_LABELS[layer]} | {pair.capitalize()} ({side})")
                row_keys.append((layer, side, pair))

    # Build matrix
    mat = np.full((len(row_keys), len(top_motifs)), np.nan)
    for ri, key in enumerate(row_keys):
        for ci, motif in enumerate(top_motifs):
            if motif in all_data[key]:
                mat[ri, ci] = all_data[key][motif]

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig_h = max(6, len(row_keys) * 0.45)
    fig_w = max(10, len(top_motifs) * 0.85)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 10.0
    cmap = plt.cm.YlOrRd
    cmap.set_bad(color="#E8E8E8")   # grey for NaN
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                   interpolation="nearest")

    # Annotate cells with rounded values where significant (> -log10(0.05) ≈ 1.3)
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            if not np.isnan(mat[ri, ci]) and mat[ri, ci] > 1.3:
                txt_col = "white" if mat[ri, ci] > vmax * 0.65 else "black"
                ax.text(ci, ri, f"{mat[ri, ci]:.1f}",
                        ha="center", va="center", fontsize=6.5,
                        color=txt_col, fontweight="bold")

    # Dividing line between vivo and vitro blocks
    n_vivo = len(layers) * len(PAIRS)
    ax.axhline(n_vivo - 0.5, color="white", lw=2.5)

    ax.set_xticks(range(len(top_motifs)))
    ax.set_xticklabels(top_motifs, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels(row_labels, fontsize=7.5)

    # Block labels on right
    mid_vivo  = n_vivo / 2 - 0.5
    mid_vitro = n_vivo + len(layers) * len(PAIRS) / 2 - 0.5
    for y, label, col in [(mid_vivo, "In vivo\n(tissue)", VIVO_COL),
                           (mid_vitro, "In vitro\n(cell line)", VITRO_COL)]:
        ax.annotate(label, xy=(1.01, 1 - (y + 0.5) / len(row_keys)),
                    xycoords="axes fraction", fontsize=9, color=col,
                    fontweight="bold", ha="left", va="center")

    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.12)
    cb.set_label("$-\\log_{10}(p\\text{-value})$", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    ax.set_title("HOMER Known Motif Enrichment — Top Context-Divergent SAE Features\n"
                 "(top-50 CDS features per layer per condition; 200 bp windows, hg38)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Transcription factor motif", fontsize=10)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(Path(figures_dir) / f"fig8_homer_motifs.{ext}",
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  fig8_homer_motifs saved.")


# =============================================================================
# Fig 9 — GO:BP enrichment dot-plot
# =============================================================================

def make_fig9_go(annotation_dir: str, figures_dir: str,
                 layers: list, n_top: int = 10) -> None:
    """
    3-column x 2-row facet grid (pairs x sides).
    Each facet: dot-plot of top GO:BP terms (y) vs. -log10(FDR q) (x),
    dot size = fold enrichment, colour = layer depth.
    Rows stacked per layer within each facet.
    """
    go_dir = Path(annotation_dir) / "go"

    # Colour per layer
    layer_colors = {
        "early": "#8ECFC9",
        "mid":   "#FFBE7A",
        "late":  "#FA7F6F",
    }

    any_data = False

    fig, axes = plt.subplots(
        len(SIDES), len(PAIRS),
        figsize=(16, 10),
        constrained_layout=True,
    )
    fig.suptitle(
        "GO:BP Enrichment — Top Context-Divergent SAE Features\n"
        "(rGREAT; top-50 CDS features per layer per condition; FDR < 0.05)",
        fontsize=12, fontweight="bold",
    )

    for ri, side in enumerate(SIDES):
        for ci, pair in enumerate(PAIRS):
            ax = axes[ri][ci]

            all_terms_rows = []
            for layer in layers:
                tag     = f"{layer}_{side}_{pair}"
                tsv_path = go_dir / f"{tag}_go.tsv"
                df      = parse_go_tsv(str(tsv_path), n=n_top)
                if df.empty:
                    continue
                df["layer"] = layer
                all_terms_rows.append(df)

            if not all_terms_rows:
                ax.text(0.5, 0.5, "No GO data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="grey")
                ax.set_title(f"{PAIR_LABELS[pair].replace(chr(10),' ')} / {side}",
                             fontsize=9, fontweight="bold",
                             color=VIVO_COL if side == "vivo" else VITRO_COL)
                ax.axis("off")
                continue

            any_data = True
            combined = pd.concat(all_terms_rows, ignore_index=True)
            # Union of unique GO terms across layers, ordered by best q
            term_order = (combined.groupby("description")["neg_log_q"]
                          .max().sort_values(ascending=True).index.tolist())

            y_ticks = []
            y_pos   = 0
            layer_gap = 0.6  # vertical separation between layer groups

            for layer in layers:
                sub = combined[combined["layer"] == layer].copy()
                sub = sub.set_index("description").reindex(term_order).dropna()
                col = layer_colors[layer]

                for _, row in sub.iterrows():
                    y_ticks.append((y_pos, row.name))
                    size = max(20, min(300, row["fold_enrichment"] * 40))
                    ax.scatter(row["neg_log_q"], y_pos, s=size, color=col,
                               edgecolors="white", linewidths=0.5,
                               alpha=0.85, zorder=3)
                    y_pos += 1
                y_pos += layer_gap

            if y_ticks:
                ys, labs = zip(*y_ticks)
                ax.set_yticks(list(ys))
                ax.set_yticklabels(list(labs), fontsize=7)
            ax.axvline(-np.log10(0.05), color="grey", lw=1, ls="--",
                       alpha=0.6, zorder=1)
            ax.set_xlabel("$-\\log_{10}$(FDR $q$)", fontsize=8)
            ax.grid(axis="x", alpha=0.25, linestyle=":")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            title_col = VIVO_COL if side == "vivo" else VITRO_COL
            ax.set_title(
                f"{PAIR_LABELS[pair].replace(chr(10),' ')} | {'In vivo' if side=='vivo' else 'In vitro'}",
                fontsize=9, fontweight="bold", color=title_col,
            )

    # Shared legend for layers and dot size
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=layer_colors[l],
               markersize=9, label=LAYER_LABELS[l])
        for l in layers
    ]
    size_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#999999",
               markersize=np.sqrt(fe * 40) * 0.6,
               label=f"Fold enrichment = {fe}")
        for fe in [2, 5, 10]
    ]
    fig.legend(handles=legend_handles + size_handles,
               loc="lower center", ncol=6, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.04), frameon=True, edgecolor="#CCCCCC")

    if not any_data:
        print("  WARNING: No GO results found. Saving empty placeholder for fig9.")
        for ax_row in axes:
            for ax in ax_row:
                ax.text(0.5, 0.5, "No GO:BP results found.\n"
                        "Check outputs/annotation/go/ for *_go.tsv files.",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=10, color="grey")
                ax.axis("off")

    for ext in ("pdf", "png"):
        fig.savefig(Path(figures_dir) / f"fig9_go_enrichment.{ext}",
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  fig9_go_enrichment saved.")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate HOMER motif and GO:BP enrichment figures."
    )
    parser.add_argument("--annotation_dir", required=True,
                        help="Path to outputs/annotation/")
    parser.add_argument("--figures_dir",    required=True,
                        help="Path to results/figures/")
    parser.add_argument("--layers",         default="early mid late",
                        help="Space-separated list of layer names")
    parser.add_argument("--n_top_motifs",   type=int, default=10,
                        help="Top N motifs to show per condition")
    parser.add_argument("--n_top_go",       type=int, default=10,
                        help="Top N GO terms to show per condition")
    args = parser.parse_args()

    layers = args.layers.split()
    os.makedirs(args.figures_dir, exist_ok=True)

    print("Generating Fig 8: HOMER motif enrichment heatmap...")
    make_fig8_homer(args.annotation_dir, args.figures_dir,
                    layers, n_top=args.n_top_motifs)

    print("Generating Fig 9: GO:BP enrichment dot-plot...")
    make_fig9_go(args.annotation_dir, args.figures_dir,
                 layers, n_top=args.n_top_go)

    print(f"\nDone. Figures saved to: {args.figures_dir}")


if __name__ == "__main__":
    main()
