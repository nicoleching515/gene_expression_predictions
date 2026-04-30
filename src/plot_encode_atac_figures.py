#!/usr/bin/env python3
"""
src/plot_encode_atac_figures.py
================================
Publication-quality HOMER motif enrichment figures from genome-wide ENCODE
ATAC-seq condition-specific peaks.

Generates:
  fig11_encode_atac_homer.pdf/png — heatmap of -log10(p) for top motifs across
      all 6 cell-type conditions (3 pairs × 2 sides), with fold-enrichment
      bubbles and FDR annotation.

Conditions:
  blood  vitro : K562-specific peaks (K562 vs HSC)
  blood  vivo  : HSC-specific peaks  (HSC vs K562, mm10→hg38 liftover)
  liver  vitro : HepG2-specific peaks
  liver  vivo  : Liver-specific peaks
  lymph  vitro : GM12878-specific peaks
  lymph  vivo  : NaiveB-specific peaks (2 donors merged)

Usage:
    python3 src/plot_encode_atac_figures.py \
        --homer_dir  outputs/annotation/homer_encode/homer \
        --figures_dir results/figures
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── Condition metadata ─────────────────────────────────────────────────────────
CONDITIONS = [
    # (homer_tag,                  pair,    side,   label)
    ("K562_specific_blood",   "blood",  "vitro", "K562 (erythroid, vs HSC)"),
    ("HSC_specific_blood",    "blood",  "vivo",  "HSC (stem cell, vs K562)"),
    ("HepG2_vitro_liver",     "liver",  "vitro", "HepG2 (hepatoma, vs Liver)"),
    ("Liver_vivo_liver",      "liver",  "vivo",  "Liver tissue (vs HepG2)"),
    ("GM12878_vitro_lymph",   "lymph",  "vitro", "GM12878 (EBV-B, vs NaiveB)"),
    ("NaiveB_vivo_lymph",     "lymph",  "vivo",  "Naive B cell (vs GM12878)"),
]

PAIR_COLORS  = {"blood": "#E05C5C", "liver": "#5C9BE0", "lymph": "#5CC47A"}
VITRO_COL    = "#E04B4B"
VIVO_COL     = "#3A7DC9"
Q_SIG        = 0.05


def parse_homer(path: Path, n: int = 300) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip() for c in df.columns]

        name_col = df.columns[0]
        p_col    = next((c for c in df.columns if "p-value" in c.lower()), None)
        q_col    = next((c for c in df.columns if "benjamini" in c.lower()), None)
        pt_col   = next((c for c in df.columns if "% of target" in c.lower()), None)
        pb_col   = next((c for c in df.columns if "% of background" in c.lower()), None)

        if p_col is None:
            return pd.DataFrame()

        df = df[[c for c in [name_col, p_col, q_col, pt_col, pb_col] if c]].copy()
        df.columns = ["motif", "pval", "qval", "pct_t", "pct_b"][:len(df.columns)]

        def clean(s):
            m = re.match(r'Factor:\s*([^;]+)', str(s))
            if m: return m.group(1).strip()
            return str(s).split("(")[0].split("/")[0].strip()

        df["motif"] = df["motif"].apply(clean)
        df = df.drop_duplicates("motif")

        df["pval"] = pd.to_numeric(df["pval"], errors="coerce").clip(lower=1e-1500)
        df["qval"] = pd.to_numeric(df.get("qval", pd.Series(dtype=float)),
                                    errors="coerce") if "qval" in df.columns else np.nan
        df["neg_log_p"] = -np.log10(df["pval"])

        if "pct_t" in df.columns and "pct_b" in df.columns:
            pt = pd.to_numeric(df["pct_t"].astype(str).str.rstrip("%"), errors="coerce")
            pb = pd.to_numeric(df["pct_b"].astype(str).str.rstrip("%"), errors="coerce").clip(lower=0.01)
            df["fold"] = pt / pb
        else:
            df["fold"] = np.nan

        return df[["motif","pval","qval","neg_log_p","fold"]].head(n).reset_index(drop=True)
    except Exception as e:
        print(f"  WARNING: {path}: {e}")
        return pd.DataFrame()


def select_top_motifs(all_data: dict, n: int = 20) -> list:
    """
    Pick top N motifs by:
      1. Motifs FDR-significant (q < 0.05) in any condition — ordered by best -log10(p)
      2. Top motifs by -log10(p) per condition, union of top-5 per condition
    Cap at n total.
    """
    best_nlp = {}
    best_q   = {}
    for cond_dict in all_data.values():
        for motif, v in cond_dict.items():
            if v["neg_log_p"] > best_nlp.get(motif, 0):
                best_nlp[motif] = v["neg_log_p"]
            q = v.get("qval", np.nan)
            if not np.isnan(q) and (motif not in best_q or q < best_q[motif]):
                best_q[motif] = q

    sig = sorted([m for m, q in best_q.items() if q < Q_SIG],
                 key=lambda m: -best_nlp[m])
    per_cond_top5 = set()
    for cd in all_data.values():
        for m in sorted(cd, key=lambda x: -cd[x]["neg_log_p"])[:5]:
            per_cond_top5.add(m)

    ordered = sig[:]
    for m in sorted(per_cond_top5 - set(sig), key=lambda m: -best_nlp.get(m, 0)):
        ordered.append(m)
        if len(ordered) >= n:
            break

    # fill remaining with globally highest -log10(p)
    if len(ordered) < n:
        remaining = sorted(set(best_nlp) - set(ordered), key=lambda m: -best_nlp[m])
        ordered.extend(remaining[:n - len(ordered)])

    return ordered[:n]


def make_fig11(homer_dir: Path, figures_dir: Path, n_top: int = 20) -> None:
    # ── Load data ──────────────────────────────────────────────────────────────
    all_data = {}
    for tag, pair, side, label in CONDITIONS:
        path = homer_dir / tag / "knownResults.txt"
        df = parse_homer(path, n=300)
        if df.empty:
            all_data[tag] = {}
        else:
            all_data[tag] = {
                row["motif"]: {
                    "neg_log_p": row["neg_log_p"],
                    "qval":      row["qval"] if "qval" in df.columns else np.nan,
                    "fold":      row["fold"],
                }
                for _, row in df.iterrows()
            }

    top_motifs = select_top_motifs(all_data, n=n_top)
    if not top_motifs:
        print("  No motif data found.")
        return

    n_rows = len(CONDITIONS)
    n_cols = len(top_motifs)

    mat_nlp  = np.full((n_rows, n_cols), np.nan)
    mat_q    = np.full((n_rows, n_cols), np.nan)
    mat_fold = np.full((n_rows, n_cols), np.nan)

    for ri, (tag, *_) in enumerate(CONDITIONS):
        for ci, motif in enumerate(top_motifs):
            if motif in all_data[tag]:
                v = all_data[tag][motif]
                mat_nlp[ri, ci]  = v["neg_log_p"]
                mat_q[ri, ci]    = v["qval"]
                mat_fold[ri, ci] = v["fold"]

    # Cap display at 300 so colour scale isn't dominated by extreme values
    vmax = min(300, float(np.nanmax(mat_nlp))) if not np.all(np.isnan(mat_nlp)) else 100
    # Only colour cells with nominal significance (p < 0.05, i.e. -log10p > 1.3)
    masked_nlp = np.where(mat_nlp > 1.3, mat_nlp, np.nan)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig_w = max(14, n_cols * 0.85)
    fig_h = max(5,  n_rows * 0.70)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("#EBEBEB")
    im = ax.imshow(masked_nlp, aspect="auto", cmap=cmap,
                   vmin=0, vmax=vmax, interpolation="nearest")

    # ── Bubble overlay: fold enrichment ───────────────────────────────────────
    bubble_ref = 120   # area for fold = 3×
    for ri in range(n_rows):
        for ci in range(n_cols):
            fe = mat_fold[ri, ci]
            if np.isnan(fe) or fe <= 0 or np.isnan(mat_nlp[ri, ci]) or mat_nlp[ri, ci] <= 1.3:
                continue
            area = min(bubble_ref * 2, (fe / 3.0) * bubble_ref)
            ax.scatter(ci, ri, s=area, color="steelblue", alpha=0.40,
                       edgecolors="white", linewidths=0.4, zorder=3)

    # ── Cell annotations ──────────────────────────────────────────────────────
    for ri in range(n_rows):
        for ci in range(n_cols):
            nlp = mat_nlp[ri, ci]
            q   = mat_q[ri, ci]
            if np.isnan(nlp) or nlp <= 1.3:
                continue
            txt_col = "white" if nlp > vmax * 0.65 else "black"
            disp = f"{min(nlp, 999):.0f}"
            if not np.isnan(q) and q < Q_SIG:
                ax.text(ci, ri, f"{disp}★", ha="center", va="center",
                        fontsize=6.5, color=txt_col, fontweight="bold", zorder=4)
            else:
                ax.text(ci, ri, disp, ha="center", va="center",
                        fontsize=6, color=txt_col, alpha=0.8, zorder=4)

    # ── Divider lines between pairs ───────────────────────────────────────────
    ax.axhline(1.5, color="white", lw=2.0)   # blood / liver
    ax.axhline(3.5, color="white", lw=2.0)   # liver / lymph

    # ── Pair bracket labels on right ─────────────────────────────────────────
    for mid_y, pair, label_txt in [(0.5, "blood", "Blood"),
                                    (2.5, "liver", "Liver"),
                                    (4.5, "lymph", "Lymph")]:
        ax.annotate(label_txt,
                    xy=(1.015, 1 - (mid_y + 0.5) / n_rows),
                    xycoords="axes fraction",
                    fontsize=9, color=PAIR_COLORS[pair],
                    fontweight="bold", ha="left", va="center")

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(top_motifs, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    row_labels = [label for _, _, _, label in CONDITIONS]
    ax.set_yticklabels(row_labels, fontsize=8)

    # Colour y-tick labels by vitro/vivo
    for tick, (_, _, side, _) in zip(ax.get_yticklabels(), CONDITIONS):
        tick.set_color(VITRO_COL if side == "vitro" else VIVO_COL)

    # ── Colourbar ─────────────────────────────────────────────────────────────
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.12)
    cb.set_label("$-\\log_{10}(p$-value$)$  [capped at 300]", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elems = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="steelblue",
               markersize=np.sqrt(bubble_ref*(fe/3.0))*0.55,
               alpha=0.65, label=f"Fold enrichment ≈ {fe}×")
        for fe in [1, 2, 3]
    ] + [
        Line2D([0],[0], linestyle="none", marker="$★$", color="#B22222",
               markersize=8, label=f"★  FDR q < {Q_SIG}"),
        Line2D([0],[0], color=VITRO_COL, lw=2, label="In vitro (cell line)"),
        Line2D([0],[0], color=VIVO_COL,  lw=2, label="In vivo (primary)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left",
              bbox_to_anchor=(0.0, -0.22), ncol=6,
              fontsize=7.5, frameon=True, edgecolor="#CCCCCC")

    # ── Title & caption ───────────────────────────────────────────────────────
    ax.set_title(
        "Genome-Wide HOMER Motif Enrichment — ENCODE ATAC-seq Condition-Specific Peaks\n"
        "(findMotifsGenome.pl, hg38, -size 200 -mask -genomeBg; "
        "IDR peaks, bedtools subtract for condition specificity)",
        fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel("Transcription factor motif", fontsize=9)

    fig.text(
        0.5, -0.02,
        "Grey: p ≥ 0.05. Bubble = fold enrichment (% target / % background). "
        "Blood-vivo HSC peaks: mouse ENCSR366VBB mm10→hg38 liftover (46% converted). "
        "★ FDR q < 0.05 (Benjamini–Hochberg).",
        ha="center", fontsize=7.5, color="dimgrey")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = figures_dir / f"fig11_encode_atac_homer.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"  Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--homer_dir",   required=True)
    parser.add_argument("--figures_dir", required=True)
    parser.add_argument("--n_top",       type=int, default=20)
    args = parser.parse_args()

    homer_dir   = Path(args.homer_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(exist_ok=True)

    print("Generating Fig 11: ENCODE ATAC genome-wide HOMER heatmap...")
    make_fig11(homer_dir, figures_dir, n_top=args.n_top)
    print("Done.")


if __name__ == "__main__":
    main()
