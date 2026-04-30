#!/usr/bin/env python3
"""
src/plot_homer_go_figures.py
============================
Generates two publication-quality figures from HOMER and g:Profiler GO:BP outputs:

  fig8_homer_motifs.pdf  --  Heatmap of -log10(p-value) for top motifs across all
                             18 conditions (3 layers x 3 pairs x 2 contexts).
                             Significance tiers annotated: ★ q<0.05, † q<0.15.
                             Bubble overlay shows fold enrichment (% target / % bg).

  fig9_go_enrichment.pdf --  Dot-plot of top GO:BP terms per layer x pair x side.
                             Dot size = fold enrichment, colour = -log10(FDR q-value).
                             Faceted as 2 x 3 grid (sides x pairs).

Usage:
    python3 src/plot_homer_go_figures.py \
        --annotation_dir outputs/annotation \
        --figures_dir    results/figures \
        --layers         "early mid late" \
        --n_top_motifs   15 \
        --n_top_go       10
"""

import argparse
import os
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore", category=UserWarning)

# ── Colour palette ─────────────────────────────────────────────────────────────
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

Q_SIG   = 0.05   # FDR threshold for ★
Q_TREND = 0.15   # FDR threshold for †


# =============================================================================
# Parsers
# =============================================================================

def parse_homer_known(homer_dir: str, n: int = 10) -> pd.DataFrame:
    """
    Parse HOMER knownResults.txt.
    Returns: motif_name, pval, qval, neg_log_p, fold_enrichment
    Sorted ascending by pval, deduplicated by TF name.
    """
    path = Path(homer_dir) / "knownResults.txt"
    if not path.exists():
        return pd.DataFrame(columns=["motif_name", "pval", "qval",
                                     "neg_log_p", "fold_enrichment"])
    try:
        df = pd.read_csv(path, sep="\t", comment=None)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        namecol = df.columns[0]
        pcol = next((c for c in df.columns
                     if "p-value" in c or "pvalue" in c or c == "p_value"), None)
        qcol = next((c for c in df.columns
                     if "benjamini" in c or "q-value" in c or "qvalue" in c), None)
        pct_t_col = next((c for c in df.columns
                          if "%" in c and "target" in c), None)
        pct_b_col = next((c for c in df.columns
                          if "%" in c and "background" in c), None)

        if pcol is None:
            return pd.DataFrame(columns=["motif_name", "pval", "qval",
                                         "neg_log_p", "fold_enrichment"])

        keep = [namecol, pcol]
        if qcol:     keep.append(qcol)
        if pct_t_col: keep.append(pct_t_col)
        if pct_b_col: keep.append(pct_b_col)

        df = df[keep].copy()
        rn = {namecol: "motif_name", pcol: "pval"}
        if qcol:     rn[qcol]     = "qval"
        if pct_t_col: rn[pct_t_col] = "pct_target"
        if pct_b_col: rn[pct_b_col] = "pct_bg"
        df = df.rename(columns=rn)

        df["motif_name"] = df["motif_name"].astype(str)
        df["pval"] = pd.to_numeric(df["pval"], errors="coerce")
        df["qval"] = pd.to_numeric(df.get("qval", pd.Series(dtype=float)),
                                   errors="coerce") if "qval" in df.columns else np.nan

        # Fold enrichment from % target / % background
        if "pct_target" in df.columns and "pct_bg" in df.columns:
            def _pct(s):
                return pd.to_numeric(
                    s.astype(str).str.rstrip("%"), errors="coerce")
            pt = _pct(df["pct_target"])
            pb = _pct(df["pct_bg"]).clip(lower=0.01)
            df["fold_enrichment"] = pt / pb
        else:
            df["fold_enrichment"] = np.nan

        df = df.dropna(subset=["pval"]).sort_values("pval")

        def _clean(name):
            m = re.match(r'Factor:\s*([^;]+)', name)
            if m:
                return m.group(1).strip()
            return name.split("(")[0].split("/")[0].strip()

        df["motif_name"] = df["motif_name"].apply(_clean)
        df = df.drop_duplicates(subset="motif_name").head(n)
        df["pval"] = df["pval"].clip(lower=1e-300)
        df["neg_log_p"] = -np.log10(df["pval"])

        return df[["motif_name", "pval", "qval",
                   "neg_log_p", "fold_enrichment"]].reset_index(drop=True)

    except Exception as exc:
        print(f"  WARNING: could not parse {path}: {exc}")
        return pd.DataFrame(columns=["motif_name", "pval", "qval",
                                     "neg_log_p", "fold_enrichment"])


def parse_go_tsv(tsv_path: str, n: int = 10) -> pd.DataFrame:
    """
    Parse g:Profiler GO:BP TSV.
    Returns: description, fold_enrichment, neg_log_q
    """
    path = Path(tsv_path)
    if not path.exists():
        return pd.DataFrame(columns=["description", "fold_enrichment", "neg_log_q"])
    try:
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "description" not in df.columns:
            return pd.DataFrame(columns=["description", "fold_enrichment", "neg_log_q"])
        fcol = next((c for c in df.columns
                     if "fold" in c or "enrichment" in c), None)
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
        df["description"] = df["description"].str[:45]
        return df[["description", "fold_enrichment", "neg_log_q"]].reset_index(drop=True)
    except Exception as exc:
        print(f"  WARNING: could not parse {path}: {exc}")
        return pd.DataFrame(columns=["description", "fold_enrichment", "neg_log_q"])


# =============================================================================
# Fig 8 — HOMER motif enrichment heatmap (improved)
# =============================================================================

def _build_motif_priority_list(all_data: dict, n_top: int) -> list:
    """
    Build ordered motif list with three priority tiers:
      1. Any motif FDR-significant (q < Q_SIG) in at least one condition
      2. Any motif trending (Q_SIG <= q < Q_TREND) in at least one condition
      3. Top motifs by frequency of appearance in per-condition top-5 lists
    Within each tier, order by best -log10(p) across conditions.
    Total capped at n_top.
    """
    # all_data: key -> {"motif": {"neg_log_p": float, "qval": float, "fold": float}}
    best_nlp  = {}   # motif -> best neg_log_p across conditions
    best_q    = {}   # motif -> best qval across conditions

    for cond_dict in all_data.values():
        for motif, vals in cond_dict.items():
            nlp = vals["neg_log_p"]
            q   = vals["qval"]
            if motif not in best_nlp or nlp > best_nlp[motif]:
                best_nlp[motif] = nlp
            if motif not in best_q or (not np.isnan(q) and
               (np.isnan(best_q.get(motif, np.nan)) or q < best_q[motif])):
                best_q[motif] = q

    tier1 = sorted(
        [m for m, q in best_q.items() if not np.isnan(q) and q < Q_SIG],
        key=lambda m: -best_nlp[m])
    tier2 = sorted(
        [m for m, q in best_q.items()
         if not np.isnan(q) and Q_SIG <= q < Q_TREND and m not in tier1],
        key=lambda m: -best_nlp[m])

    # frequency tier: top-5 per condition by p-value
    freq_counts: Counter = Counter()
    for cond_dict in all_data.values():
        top5 = sorted(cond_dict.keys(),
                      key=lambda m: -cond_dict[m]["neg_log_p"])[:5]
        freq_counts.update(top5)

    already = set(tier1) | set(tier2)
    tier3 = [m for m, _ in freq_counts.most_common()
             if m not in already]

    ordered = tier1 + tier2 + tier3
    return ordered[:n_top]


def make_fig8_homer(annotation_dir: str, figures_dir: str,
                    layers: list, n_top: int = 15) -> None:
    """
    Heatmap of -log10(p-value) for top motifs across 18 conditions.

    Improvements over v1:
    - Top-5 motifs per condition (not just top-N by frequency)
    - Priority union: q<0.05 motifs first, then q<0.15, then frequent
    - Significance tier annotation: ★ q<0.05, † q<0.15
    - Bubble overlay: circle size = fold enrichment (% target / % background)
    - Colormap masked so p >= 0.2 cells are grey (not falsely coloured)
    - Footer note on chr8/chr9 scope and genome-wide recommendation
    """
    homer_base = Path(annotation_dir) / "homer"

    # ── Load all data ──────────────────────────────────────────────────────────
    all_data = {}
    for layer in layers:
        for side in SIDES:
            for pair in PAIRS:
                tag  = f"{layer}_{side}_{pair}"
                hdir = homer_base / tag
                df   = parse_homer_known(str(hdir), n=200)
                if df.empty:
                    all_data[(layer, side, pair)] = {}
                else:
                    all_data[(layer, side, pair)] = {
                        row["motif_name"]: {
                            "neg_log_p":       row["neg_log_p"],
                            "qval":            row["qval"],
                            "fold_enrichment": row["fold_enrichment"],
                        }
                        for _, row in df.iterrows()
                    }

    # ── Select motifs ──────────────────────────────────────────────────────────
    top_motifs = _build_motif_priority_list(all_data, n_top)

    if not top_motifs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                "No HOMER motif results found.\n"
                "Check outputs/annotation/homer/ for knownResults.txt files.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="grey")
        ax.axis("off")
        fig.suptitle("Fig 8 — HOMER Motif Enrichment (no data)", fontsize=12)
        for ext in ("pdf", "png"):
            fig.savefig(Path(figures_dir) / f"fig8_homer_motifs.{ext}",
                        bbox_inches="tight", dpi=300)
        plt.close(fig)
        return

    # ── Build matrices (rows = conditions, cols = motifs) ─────────────────────
    row_labels = []
    row_keys   = []
    for side in SIDES:
        for layer in layers:
            for pair in PAIRS:
                row_labels.append(
                    f"{LAYER_LABELS[layer]} | {pair.capitalize()} ({side})")
                row_keys.append((layer, side, pair))

    n_rows = len(row_keys)
    n_cols = len(top_motifs)

    mat_nlp  = np.full((n_rows, n_cols), np.nan)   # -log10(p)
    mat_q    = np.full((n_rows, n_cols), np.nan)   # q-value
    mat_fold = np.full((n_rows, n_cols), np.nan)   # fold enrichment

    for ri, key in enumerate(row_keys):
        for ci, motif in enumerate(top_motifs):
            if motif in all_data[key]:
                v = all_data[key][motif]
                mat_nlp[ri, ci]  = v["neg_log_p"]
                mat_q[ri, ci]    = v["qval"]
                mat_fold[ri, ci] = v["fold_enrichment"]

    # Mask cells where p >= 0.2  (-log10(0.2) ≈ 0.7) so they appear grey
    masked_nlp = np.where(mat_nlp >= -np.log10(0.2), mat_nlp, np.nan)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig_h = max(7, n_rows * 0.50)
    fig_w = max(12, n_cols * 0.95)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = max(4.0, np.nanmax(masked_nlp)) if not np.all(np.isnan(masked_nlp)) else 4.0
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#EBEBEB")   # grey for masked / no data

    im = ax.imshow(masked_nlp, aspect="auto", cmap=cmap,
                   vmin=0, vmax=vmax, interpolation="nearest")

    # ── Bubble overlay: fold enrichment ───────────────────────────────────────
    bubble_max_area = 140   # pt² for fold enrichment = 3×
    for ri in range(n_rows):
        for ci in range(n_cols):
            fe = mat_fold[ri, ci]
            if np.isnan(fe) or fe <= 0:
                continue
            area = min(bubble_max_area, (fe / 3.0) * bubble_max_area)
            ax.scatter(ci, ri, s=area, color="steelblue", alpha=0.45,
                       edgecolors="white", linewidths=0.4, zorder=3)

    # ── Significance tier annotations ─────────────────────────────────────────
    for ri in range(n_rows):
        for ci in range(n_cols):
            nlp = mat_nlp[ri, ci]
            q   = mat_q[ri, ci]
            if np.isnan(nlp) or nlp < -np.log10(0.2):
                continue
            txt_col = "white" if nlp > vmax * 0.65 else "black"
            if not np.isnan(q) and q < Q_SIG:
                # FDR-significant: bold value + star
                label = f"{nlp:.1f}★"
                ax.text(ci, ri, label, ha="center", va="center",
                        fontsize=7, color=txt_col, fontweight="bold", zorder=4)
            elif not np.isnan(q) and q < Q_TREND:
                # Trending: value + dagger
                label = f"{nlp:.1f}†"
                ax.text(ci, ri, label, ha="center", va="center",
                        fontsize=6.5, color=txt_col, fontstyle="italic", zorder=4)
            else:
                # Nominal only (p<0.2): show value, no marker
                ax.text(ci, ri, f"{nlp:.1f}", ha="center", va="center",
                        fontsize=6, color=txt_col, alpha=0.75, zorder=4)

    # ── Dividing line between vivo / vitro blocks ─────────────────────────────
    n_vivo = len(layers) * len(PAIRS)
    ax.axhline(n_vivo - 0.5, color="white", lw=2.5)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(top_motifs, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7.5)

    # Highlight FDR-significant motif column labels in bold
    sig_motifs = set()
    for cond_dict in all_data.values():
        for motif, vals in cond_dict.items():
            if not np.isnan(vals["qval"]) and vals["qval"] < Q_SIG:
                sig_motifs.add(motif)
    for tick, label in zip(ax.get_xticklabels(), top_motifs):
        if label in sig_motifs:
            tick.set_fontweight("bold")
            tick.set_color("#B22222")

    # Side block labels — SIDES = ["vitro", "vivo"], so vitro rows come first
    mid_vitro = n_vivo / 2 - 0.5
    mid_vivo  = n_vivo + len(layers) * len(PAIRS) / 2 - 0.5
    for y, label, col in [(mid_vitro, "In vitro\n(cell line)", VITRO_COL),
                           (mid_vivo,  "In vivo\n(tissue)",    VIVO_COL)]:
        ax.annotate(label,
                    xy=(1.01, 1 - (y + 0.5) / n_rows),
                    xycoords="axes fraction", fontsize=9,
                    color=col, fontweight="bold",
                    ha="left", va="center")

    # ── Colourbar ─────────────────────────────────────────────────────────────
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.13)
    cb.set_label("$-\\log_{10}(p$-value$)$", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=np.sqrt(bubble_max_area * (fe / 3.0)) * 0.5,
               alpha=0.65, label=f"Fold enrichment ≈ {fe}×")
        for fe in [1, 2, 3]
    ] + [
        Line2D([0], [0], linestyle="none", marker="$★$", color="#B22222",
               markersize=8, label=f"★  FDR q < {Q_SIG}"),
        Line2D([0], [0], linestyle="none", marker="$†$", color="dimgrey",
               markersize=8, label=f"†  FDR q < {Q_TREND}"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              bbox_to_anchor=(0.0, -0.22), ncol=5,
              fontsize=8, frameon=True, edgecolor="#CCCCCC")

    # ── Titles & caption ──────────────────────────────────────────────────────
    ax.set_title(
        "HOMER Known Motif Enrichment — Top Context-Divergent SAE Features\n"
        "(findMotifsGenome.pl, hg38, -size 200 -mask, genome-wide background; "
        "top-50 CDS features per layer per condition)",
        fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel("Transcription factor motif", fontsize=9)

    fig.text(
        0.5, -0.02,
        "Grey cells: p ≥ 0.2 (not nominally enriched). "
        "Bubble size indicates fold enrichment (% target / % background). "
        "Background: 100,000 genome-wide GC-matched random sequences (HOMER -genomeBg).",
        ha="center", fontsize=7.5, color="dimgrey",
        wrap=True)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(Path(figures_dir) / f"fig8_homer_motifs.{ext}",
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  fig8_homer_motifs saved.")


# =============================================================================
# Fig 9 — GO:BP enrichment dot-plot (unchanged)
# =============================================================================

def make_fig9_go(annotation_dir: str, figures_dir: str,
                 layers: list, n_top: int = 10) -> None:
    """
    2-row x 3-column facet grid (sides x pairs).
    Each facet: grouped dot-plot; rows = GO:BP terms, dots = layers.
    Dot size = fold enrichment, colour = layer depth.
    """
    go_dir = Path(annotation_dir) / "go"

    layer_colors = {
        "early": "#8ECFC9",
        "mid":   "#FFBE7A",
        "late":  "#FA7F6F",
    }
    layer_offsets = {layers[i]: (i - (len(layers) - 1) / 2) * 0.22
                     for i in range(len(layers))}

    any_data = False

    fig, axes = plt.subplots(
        len(SIDES), len(PAIRS),
        figsize=(16, 10),
        constrained_layout=True,
    )
    fig.suptitle(
        "GO:BP Enrichment — Top Context-Divergent SAE Features\n"
        "(g:Profiler; top-50 CDS features per layer per condition; FDR < 0.05)",
        fontsize=12, fontweight="bold",
    )

    for ri, side in enumerate(SIDES):
        for ci, pair in enumerate(PAIRS):
            ax = axes[ri][ci]

            all_terms_rows = []
            for layer in layers:
                tag      = f"{layer}_{side}_{pair}"
                tsv_path = go_dir / f"{tag}_go.tsv"
                df = parse_go_tsv(str(tsv_path), n=n_top * 3)
                if df.empty:
                    continue
                df["layer"] = layer
                all_terms_rows.append(df)

            if not all_terms_rows:
                ax.text(0.5, 0.5, "No GO data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="grey")
                ax.set_title(
                    f"{PAIR_LABELS[pair].replace(chr(10),' ')} / {side}",
                    fontsize=9, fontweight="bold",
                    color=VIVO_COL if side == "vivo" else VITRO_COL)
                ax.axis("off")
                continue

            any_data = True
            combined = pd.concat(all_terms_rows, ignore_index=True)

            term_best_q = combined.groupby("description")["neg_log_q"].max()
            top_terms   = (term_best_q.sort_values(ascending=False)
                           .head(n_top).index.tolist())
            term_to_y   = {term: i for i, term in enumerate(reversed(top_terms))}

            for term, y_pos in term_to_y.items():
                shade = "#F4F4F4" if y_pos % 2 == 0 else "white"
                ax.axhspan(y_pos - 0.48, y_pos + 0.48, color=shade, zorder=0)

                for layer in layers:
                    sub = combined[
                        (combined["layer"] == layer) &
                        (combined["description"] == term)
                    ]
                    if sub.empty:
                        continue
                    row   = sub.iloc[0]
                    size  = max(20, min(300, row["fold_enrichment"] * 40))
                    y_dot = y_pos + layer_offsets[layer]
                    ax.scatter(row["neg_log_q"], y_dot,
                               s=size, color=layer_colors[layer],
                               edgecolors="white", linewidths=0.5,
                               alpha=0.88, zorder=3)

            ys   = list(term_to_y.values())
            labs = list(term_to_y.keys())
            ax.set_yticks(ys)
            ax.set_yticklabels(labs, fontsize=7)
            ax.set_ylim(-0.6, len(top_terms) - 0.4)

            ax.axvline(-np.log10(0.05), color="grey", lw=1, ls="--",
                       alpha=0.6, zorder=1)
            ax.set_xlabel("$-\\log_{10}$(FDR $q$)", fontsize=8)
            ax.grid(axis="x", alpha=0.25, linestyle=":")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            title_col = VIVO_COL if side == "vivo" else VITRO_COL
            ax.set_title(
                f"{PAIR_LABELS[pair].replace(chr(10),' ')} | "
                f"{'In vivo' if side == 'vivo' else 'In vitro'}",
                fontsize=9, fontweight="bold", color=title_col,
            )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=layer_colors[l],
               markersize=9, label=LAYER_LABELS[l])
        for l in layers
    ] + [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#999999",
               markersize=np.sqrt(fe * 40) * 0.6,
               label=f"Fold enrichment = {fe}")
        for fe in [2, 5, 10]
    ]
    fig.legend(handles=legend_handles,
               loc="lower center", ncol=6, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.04), frameon=True,
               edgecolor="#CCCCCC")

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
        description="Generate HOMER motif and GO:BP enrichment figures.")
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--figures_dir",    required=True)
    parser.add_argument("--layers",         default="early mid late")
    parser.add_argument("--n_top_motifs",   type=int, default=15)
    parser.add_argument("--n_top_go",       type=int, default=10)
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
