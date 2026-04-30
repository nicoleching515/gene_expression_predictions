#!/usr/bin/env python3
"""
src/motif_meta_analysis.py
==========================
Three-part validation of HOMER motif enrichment in SAE features:

  1. Fisher meta-analysis — combines p-values across conditions grouped by
     cell-type pair and context (vivo/vitro), boosting power beyond any single
     condition.

  2. Null comparison — for each motif uses the hypergeometric distribution
     to estimate expected fold enrichment if target peaks were drawn at random
     from the background. Observed fold enrichment is plotted against the null
     CI to show which motifs are genuinely above chance.

  3. Cell-type specificity — for motifs significant in at least one cell-type
     group, tests whether enrichment is restricted to that cell type vs shared
     across all three (blood / liver / lymph).

Outputs:
  outputs/annotation/meta_analysis.tsv     — per-motif Fisher p-values
  results/figures/fig10_meta_analysis.pdf/png

Usage:
    python3 src/motif_meta_analysis.py \
        --annotation_dir outputs/annotation \
        --figures_dir    results/figures
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

LAYERS = ["early", "mid", "late"]
PAIRS  = ["blood", "liver", "lymph"]
SIDES  = ["vitro", "vivo"]

PAIR_COLORS = {"blood": "#E05C5C", "liver": "#5C9BE0", "lymph": "#5CC47A"}
SIDE_MARKERS = {"vitro": "o", "vivo": "s"}

Q_SIG   = 0.05
Q_TREND = 0.15

CELL_LABELS = {
    "blood": "Blood (K562/HSC)",
    "liver": "Liver (HepG2/Liver)",
    "lymph": "Lymph (GM12878/NaiveB)",
}


# =============================================================================
# Data loading
# =============================================================================

def _clean_name(name: str) -> str:
    m = re.match(r'Factor:\s*([^;]+)', name)
    if m:
        return m.group(1).strip()
    return name.split("(")[0].split("/")[0].strip()


def load_homer(homer_dir: Path, tag: str, n: int = 600) -> pd.DataFrame:
    """
    Load HOMER knownResults.txt for one condition.
    Returns DataFrame with columns:
      motif, pval, qval, n_target, n_bg, k_target, pct_target, k_bg, pct_bg,
      fold_enrichment, condition, pair, side, layer
    """
    path = homer_dir / tag / "knownResults.txt"
    if not path.exists():
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path, sep="\t", comment=None)
        raw.columns = [c.strip() for c in raw.columns]

        # Parse N_target and N_bg from column headers like
        # "# of Target Sequences with Motif(of 317)"
        header = list(raw.columns)
        n_target = n_bg = None
        for col in header:
            m = re.search(r'Target.*of (\d+)', col)
            if m:
                n_target = int(m.group(1))
            m = re.search(r'Background.*of (\d+)', col)
            if m:
                n_bg = int(m.group(1))

        # Normalise column names for lookup
        col_map = {}
        for col in header:
            lc = col.lower().replace(" ", "_")
            col_map[lc] = col
            col_map[col] = col  # identity

        def _find(keywords):
            for kw in keywords:
                for key, orig in col_map.items():
                    if kw in key:
                        return orig
            return None

        name_col    = header[0]
        pval_col    = _find(["p-value", "p_value"])
        qval_col    = _find(["benjamini", "q-value"])
        k_tgt_col   = _find(["#_of_target", "# of target"])
        pct_tgt_col = _find(["%_of_target", "% of target"])
        k_bg_col    = _find(["#_of_background", "# of background"])
        pct_bg_col  = _find(["%_of_background", "% of background"])

        df = raw[[c for c in [name_col, pval_col, qval_col,
                               k_tgt_col, pct_tgt_col, k_bg_col, pct_bg_col]
                  if c is not None]].copy()
        df.columns = ["motif", "pval", "qval",
                      "k_target", "pct_target", "k_bg", "pct_bg"
                      ][:len(df.columns)]
        for col in ["pval", "qval", "k_target", "k_bg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["pct_target", "pct_bg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.rstrip("%"), errors="coerce")

        df["motif"] = df["motif"].astype(str).apply(_clean_name)
        df = df.drop_duplicates(subset="motif").head(n)
        df["pval"] = df["pval"].clip(lower=1e-300)
        df["n_target"] = n_target
        df["n_bg"]     = n_bg

        # Fold enrichment
        if "pct_target" in df.columns and "pct_bg" in df.columns:
            df["fold_enrichment"] = df["pct_target"] / df["pct_bg"].clip(lower=0.01)
        else:
            df["fold_enrichment"] = np.nan

        parts = tag.split("_")   # layer_side_pair
        df["layer"]     = parts[0]
        df["side"]      = parts[1]
        df["pair"]      = parts[2]
        df["condition"] = tag

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"  WARNING: {tag}: {e}")
        return pd.DataFrame()


def load_all(homer_dir: Path) -> pd.DataFrame:
    frames = []
    for layer in LAYERS:
        for side in SIDES:
            for pair in PAIRS:
                tag = f"{layer}_{side}_{pair}"
                # n=600: load all HOMER motifs (typically 472) so every motif
                # gets a p-value in every condition for the Fisher test
                df  = load_homer(homer_dir, tag, n=600)
                if not df.empty:
                    frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =============================================================================
# 1. Fisher meta-analysis
# =============================================================================

def fisher_combined(pvals):
    """Fisher's combined probability test."""
    pvals = np.asarray(pvals, dtype=float)
    pvals = pvals[~np.isnan(pvals)]
    pvals = np.clip(pvals, 1e-300, 1.0)
    if len(pvals) == 0:
        return np.nan
    stat = -2.0 * np.sum(np.log(pvals))
    return stats.chi2.sf(stat, df=2 * len(pvals))


def run_meta_analysis(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Fisher combined p-values per motif grouped by:
      - pair (blood / liver / lymph) across all 6 conditions
      - side (vitro / vivo) across all 9 conditions
      - all 18 conditions combined
    Returns one row per motif.
    """
    records = []
    motifs = all_df["motif"].unique()
    for motif in motifs:
        sub = all_df[all_df["motif"] == motif]
        row = {"motif": motif}

        # Per cell-type pair
        for pair in PAIRS:
            ps = sub[sub["pair"] == pair]["pval"].values
            row[f"fisher_p_{pair}"] = fisher_combined(ps)

        # Per side (context)
        for side in SIDES:
            ps = sub[sub["side"] == side]["pval"].values
            row[f"fisher_p_{side}"] = fisher_combined(ps)

        # Combined across all 18
        row["fisher_p_all"] = fisher_combined(sub["pval"].values)

        # Best single-condition stats
        best_idx = sub["pval"].idxmin()
        row["best_pval"]          = sub.loc[best_idx, "pval"]
        row["best_qval"]          = sub.loc[best_idx, "qval"] if "qval" in sub.columns else np.nan
        row["best_condition"]     = sub.loc[best_idx, "condition"]
        row["mean_fold"]          = sub["fold_enrichment"].mean()
        row["max_fold"]           = sub["fold_enrichment"].max()
        row["n_conditions_tested"] = len(sub)

        records.append(row)

    result = pd.DataFrame(records)

    # Best per-pair Fisher p (the cell-type-specific signal)
    result["fisher_p_best_pair"] = result[
        [f"fisher_p_{p}" for p in PAIRS]].min(axis=1)
    result["fisher_best_pair"] = result[
        [f"fisher_p_{p}" for p in PAIRS]].idxmin(axis=1).str.replace(
        "fisher_p_", "")

    # BH FDR correction on best-pair Fisher p
    def _bh(pvals):
        n = len(pvals)
        order = np.argsort(pvals)
        bh = pvals[order] * n / (np.arange(n) + 1)
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        out = np.empty(n)
        out[order] = np.minimum(bh, 1.0)
        return out

    valid = result["fisher_p_best_pair"].notna()
    fdr = np.full(len(result), np.nan)
    pv  = result.loc[valid, "fisher_p_best_pair"].values
    if len(pv) > 0:
        fdr[valid] = _bh(pv)
    result["fisher_q_best_pair"] = fdr

    return result.sort_values("fisher_p_best_pair")


# =============================================================================
# 2. Null comparison (hypergeometric)
# =============================================================================

def null_fold_enrichment_ci(n_target: float, n_bg: float,
                              k_bg: float, alpha: float = 0.05):
    """
    Under the null (target peaks drawn at random from background):
      E[k_target] = n_target * (k_bg / n_bg)
      Fold enrichment = 1.0 by construction.
    Returns (fold_null_mean=1.0, fold_null_lo, fold_null_hi) using
    the Poisson approximation for the CI.
    """
    if n_bg <= 0 or k_bg <= 0:
        return 1.0, np.nan, np.nan
    expected = n_target * k_bg / n_bg
    if expected <= 0:
        return 1.0, np.nan, np.nan
    # Poisson CI on count → fold enrichment CI
    lo_count = stats.poisson.ppf(alpha / 2, expected)
    hi_count = stats.poisson.ppf(1 - alpha / 2, expected)
    base_rate = k_bg / n_bg
    lo_fold = lo_count / (n_target * base_rate) if base_rate > 0 else np.nan
    hi_fold = hi_count / (n_target * base_rate) if base_rate > 0 else np.nan
    return 1.0, lo_fold, hi_fold


def compute_null_comparison(all_df: pd.DataFrame,
                             focus_motifs: list) -> pd.DataFrame:
    """
    For each (motif, condition) in focus_motifs, compute:
      observed fold enrichment, null mean, null 95% CI.
    """
    records = []
    for motif in focus_motifs:
        sub = all_df[all_df["motif"] == motif]
        for _, row in sub.iterrows():
            if pd.isna(row.get("k_bg")) or pd.isna(row.get("n_target")):
                continue
            null_mean, null_lo, null_hi = null_fold_enrichment_ci(
                row["n_target"], row["n_bg"], row["k_bg"])
            records.append({
                "motif":           motif,
                "condition":       row["condition"],
                "pair":            row["pair"],
                "side":            row["side"],
                "layer":           row["layer"],
                "fold_observed":   row["fold_enrichment"],
                "null_mean":       null_mean,
                "null_lo":         null_lo,
                "null_hi":         null_hi,
                "pval":            row["pval"],
                "qval":            row.get("qval", np.nan),
            })
    return pd.DataFrame(records)


# =============================================================================
# 3. Cell-type specificity
# =============================================================================

def specificity_score(meta_df: pd.DataFrame, motif: str) -> dict:
    """
    For one motif: return Fisher p-values for all three cell types +
    a specificity index: -log10(p_best) / sum(-log10(p_all)).
    Higher = more specific to one cell type.
    """
    row  = meta_df[meta_df["motif"] == motif]
    if row.empty:
        return {}
    row = row.iloc[0]
    ps   = {pair: row.get(f"fisher_p_{pair}", np.nan) for pair in PAIRS}
    nlps = {pair: -np.log10(max(p, 1e-300)) if not np.isnan(p) else 0.0
            for pair, p in ps.items()}
    total = sum(nlps.values())
    best  = max(nlps.values()) if total > 0 else 0.0
    spec  = best / total if total > 0 else 0.0
    return {**nlps, "specificity": spec,
            "best_pair": max(nlps, key=nlps.get) if total > 0 else None}


# =============================================================================
# Figure generation
# =============================================================================

def make_fig10(meta_df: pd.DataFrame, all_df: pd.DataFrame,
               figures_dir: Path) -> None:
    """
    Three-panel figure:
      Panel A: Fisher meta-analysis — top 20 motifs by combined p-value,
               coloured by which cell type drives the signal
      Panel B: Null comparison — observed vs null fold enrichment for the
               top-5 motifs in each cell type
      Panel C: Cell-type specificity heatmap — -log10(Fisher p) per motif
               per cell type for the top meta-significant motifs
    """
    # ── Select motifs to display ──────────────────────────────────────────────
    top_all  = meta_df.dropna(subset=["fisher_p_all"]).head(20)

    # Motifs for null comparison: significant or trending in any single cond.
    focus_motifs = list(
        all_df[all_df["pval"] < 0.05]["motif"].value_counts().head(15).index)
    # Always include the three FDR-significant ones
    sig_motifs = list(all_df[all_df.get("qval", pd.Series(dtype=float)).lt(Q_SIG)
                              ]["motif"].unique()) if "qval" in all_df.columns else []
    for m in sig_motifs:
        if m not in focus_motifs:
            focus_motifs.insert(0, m)
    focus_motifs = focus_motifs[:15]

    null_df = compute_null_comparison(all_df, focus_motifs)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38,
                            height_ratios=[1, 1.1])
    ax_a = fig.add_subplot(gs[0, 0])   # Fisher meta-analysis bar chart
    ax_b = fig.add_subplot(gs[0, 1])   # Null comparison strip-plot
    ax_c = fig.add_subplot(gs[1, :])   # Specificity heatmap (full width)

    # ── Panel A: Fisher per-cell-type p-values ────────────────────────────────
    top_all = meta_df.dropna(subset=["fisher_p_best_pair"]).head(20)
    if not top_all.empty:
        top_a = top_all.copy()
        top_a["neg_log_p"] = -np.log10(
            top_a["fisher_p_best_pair"].clip(1e-30))
        colors = [PAIR_COLORS[r["fisher_best_pair"]]
                  for _, r in top_a.iterrows()]

        ax_a.barh(range(len(top_a)), top_a["neg_log_p"].values,
                  color=colors, alpha=0.82, edgecolor="white", height=0.7)

        # Significance thresholds
        bonf_line = -np.log10(0.05 / len(meta_df))
        ax_a.axvline(bonf_line, color="grey", lw=1.2, ls="--", alpha=0.7)
        ax_a.axvline(-np.log10(0.05), color="#AAAAAA", lw=0.8, ls=":", alpha=0.6)

        ax_a.set_yticks(range(len(top_a)))
        ax_a.set_yticklabels(top_a["motif"].values, fontsize=8)
        ax_a.invert_yaxis()
        ax_a.set_xlabel("$-\\log_{10}$(Fisher $p$, best cell type)", fontsize=9)
        ax_a.set_title(
            "A   Fisher Meta-Analysis\n"
            "(6 conditions per cell type; colour = driving cell type)",
            fontsize=9, fontweight="bold", loc="left")
        ax_a.spines["top"].set_visible(False)
        ax_a.spines["right"].set_visible(False)

        legend_handles = [mpatches.Patch(color=PAIR_COLORS[p],
                          label=CELL_LABELS[p]) for p in PAIRS]
        ax_a.legend(handles=legend_handles, fontsize=7.5,
                    loc="lower right", frameon=True, edgecolor="#CCCCCC")
        ax_a.text(bonf_line + 0.05, -0.8, "Bonferroni",
                  fontsize=6.5, color="grey", va="center")

    # ── Panel B: Null comparison strip-plot ───────────────────────────────────
    if not null_df.empty:
        # Average fold enrichment per (motif, pair)
        grp = null_df.groupby(["motif", "pair"]).agg(
            fold_mean=("fold_observed", "mean"),
            null_hi=("null_hi", "mean"),
            null_lo=("null_lo", "mean"),
            best_pval=("pval", "min"),
        ).reset_index()

        # Order motifs by mean fold enrichment
        motif_order = (grp.groupby("motif")["fold_mean"]
                       .max().sort_values(ascending=False).index.tolist())
        motif_y = {m: i for i, m in enumerate(motif_order)}

        # Null CI band (grey, same for all motifs since it's ~1.0 ± small)
        for motif, yi in motif_y.items():
            sub = grp[grp["motif"] == motif]
            null_lo = sub["null_lo"].mean()
            null_hi = sub["null_hi"].mean()
            if not np.isnan(null_lo) and not np.isnan(null_hi):
                ax_b.fill_betweenx([yi - 0.35, yi + 0.35],
                                    null_lo, null_hi,
                                    color="#DDDDDD", alpha=0.6, zorder=1)

        # Null mean line
        ax_b.axvline(1.0, color="#AAAAAA", lw=1.5, ls="--", zorder=0)

        jitter = np.linspace(-0.25, 0.25, len(PAIRS))
        for pi, pair in enumerate(PAIRS):
            sub = grp[grp["pair"] == pair]
            ys  = [motif_y[m] + jitter[pi] for m in sub["motif"]]
            ax_b.scatter(sub["fold_mean"], ys,
                         color=PAIR_COLORS[pair], s=55,
                         edgecolors="white", linewidths=0.5,
                         alpha=0.85, zorder=3,
                         label=CELL_LABELS[pair])
            # Star for significant hits
            for _, row in sub[sub["best_pval"] < 0.05].iterrows():
                yi = motif_y[row["motif"]] + jitter[pi]
                ax_b.text(row["fold_mean"] + 0.04, yi, "★",
                          fontsize=7, color=PAIR_COLORS[pair],
                          va="center", fontweight="bold")

        ax_b.set_yticks(list(motif_y.values()))
        ax_b.set_yticklabels(list(motif_y.keys()), fontsize=8)
        ax_b.set_xlabel("Mean fold enrichment (% target / % background)", fontsize=9)
        ax_b.set_title("B   Null Comparison\n(grey band = 95% CI under random sampling)",
                       fontsize=9, fontweight="bold", loc="left")
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
        ax_b.legend(fontsize=7, loc="lower right", frameon=True,
                    edgecolor="#CCCCCC")

    # ── Panel C: Cell-type specificity heatmap ────────────────────────────────
    # Select motifs with best-pair Fisher p < 0.15
    spec_motifs = meta_df[
        meta_df["fisher_p_best_pair"].fillna(1.0) < 0.15
    ]["motif"].tolist()[:25]

    if spec_motifs:
        mat = np.zeros((len(spec_motifs), len(PAIRS)))
        for ri, motif in enumerate(spec_motifs):
            row = meta_df[meta_df["motif"] == motif]
            if row.empty:
                continue
            row = row.iloc[0]
            for ci, pair in enumerate(PAIRS):
                p = row.get(f"fisher_p_{pair}", np.nan)
                mat[ri, ci] = -np.log10(max(p, 1e-30)) if not np.isnan(p) else 0.0

        vmax = max(mat.max(), 2.0)
        im = ax_c.imshow(mat.T, aspect="auto", cmap="Blues",
                         vmin=0, vmax=vmax, interpolation="nearest")

        # Annotate each cell
        for ri in range(len(spec_motifs)):
            for ci, pair in enumerate(PAIRS):
                val = mat[ri, ci]
                if val > 0.5:
                    txt_col = "white" if val > vmax * 0.6 else "#333333"
                    p_raw = meta_df[meta_df["motif"] == spec_motifs[ri]].iloc[0].get(
                        f"fisher_p_{pair}", np.nan)
                    if not np.isnan(p_raw) and p_raw < Q_SIG:
                        label = f"{val:.1f}★"
                        ax_c.text(ri, ci, label, ha="center", va="center",
                                  fontsize=7, color=txt_col, fontweight="bold")
                    elif not np.isnan(p_raw) and p_raw < 0.1:
                        ax_c.text(ri, ci, f"{val:.1f}", ha="center", va="center",
                                  fontsize=6.5, color=txt_col, fontstyle="italic")

        ax_c.set_xticks(range(len(spec_motifs)))
        ax_c.set_xticklabels(spec_motifs, rotation=45, ha="right", fontsize=8)
        ax_c.set_yticks(range(len(PAIRS)))
        ax_c.set_yticklabels([CELL_LABELS[p] for p in PAIRS], fontsize=9)

        cb = fig.colorbar(im, ax=ax_c, fraction=0.015, pad=0.01)
        cb.set_label("$-\\log_{10}$(Fisher $p$) per cell type", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        # Highlight columns where one cell type clearly dominates
        for ri, motif in enumerate(spec_motifs):
            row = meta_df[meta_df["motif"] == motif]
            if row.empty:
                continue
            row = row.iloc[0]
            nlps = [-np.log10(max(row.get(f"fisher_p_{p}", 1.0), 1e-30))
                    for p in PAIRS]
            total = sum(nlps)
            if total > 0 and max(nlps) / total > 0.7:
                # dominant cell type — add outline
                best_ci = int(np.argmax(nlps))
                rect = plt.Rectangle(
                    (ri - 0.5, best_ci - 0.5), 1, 1,
                    fill=False, edgecolor=PAIR_COLORS[PAIRS[best_ci]],
                    lw=2.2, zorder=4)
                ax_c.add_patch(rect)

        ax_c.set_title(
            "C   Cell-Type Specificity\n"
            "($-\\log_{10}$ Fisher combined $p$ per cell type; "
            "coloured outline = signal dominated by one cell type)",
            fontsize=9, fontweight="bold", loc="left")

    # ── Overall title + notes ─────────────────────────────────────────────────
    fig.suptitle(
        "HOMER Motif Enrichment Validation — SAE Context-Divergent Features\n"
        "Meta-analysis, null comparison, and cell-type specificity",
        fontsize=12, fontweight="bold", y=0.98)

    fig.text(
        0.5, 0.005,
        "★ Fisher p < 0.05.  "
        "Null 95% CI derived from hypergeometric distribution "
        "(target peaks drawn at random from background windows).  "
        "Analysis restricted to chr8/chr9; genome-wide extension recommended.",
        ha="center", fontsize=7.5, color="dimgrey")

    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"fig10_meta_analysis.{ext}",
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  fig10_meta_analysis saved.")


# =============================================================================
# ENCODE positive control (download script)
# =============================================================================

ENCODE_EXPERIMENTS = {
    # cell_type: [(accession, TF/mark, description)]
    "K562": [
        ("ENCFF002CEL", "RUNX1",   "RUNX1 ChIP-seq K562"),
        ("ENCFF496HZP", "SPI1",    "SPI1/PU.1 ChIP-seq K562"),
        ("ENCFF828IEW", "H3K27ac", "H3K27ac K562"),
    ],
    "HepG2": [
        ("ENCFF613PYA", "FOXA1",   "FOXA1 ChIP-seq HepG2"),
        ("ENCFF114YUS", "HNF4A",   "HNF4A ChIP-seq HepG2"),
        ("ENCFF617QIP", "H3K27ac", "H3K27ac HepG2"),
    ],
    "GM12878": [
        ("ENCFF001VCU", "EBF1",    "EBF1 ChIP-seq GM12878"),
        ("ENCFF002DAL", "CTCF",    "CTCF ChIP-seq GM12878"),
        ("ENCFF796WRU", "H3K27ac", "H3K27ac GM12878"),
    ],
}


def write_encode_script(out_path: Path) -> None:
    """Write a shell script to download ENCODE peak files and run HOMER."""
    lines = [
        "#!/usr/bin/env bash",
        "# ENCODE positive-control HOMER analysis",
        "# Download narrowPeak files for K562/HepG2/GM12878, filter to chr8/chr9,",
        "# run HOMER with same settings, compare enrichment to SAE feature peaks.",
        "#",
        "# Requires: HOMER, bedtools, wget",
        "# Run from repo root: bash src/run_encode_comparison.sh",
        "",
        "set -euo pipefail",
        'REPO="$(cd "$(dirname "$0")/.." && pwd)"',
        'BG_BED="$REPO/data/windows.bed"',
        'GENOME="hg38"',
        'OUT="$REPO/outputs/annotation/encode_comparison"',
        "mkdir -p \"$OUT\"",
        "",
    ]
    for cell, experiments in ENCODE_EXPERIMENTS.items():
        lines.append(f"# ── {cell} ─────────────────────────────────────")
        for acc, tf, desc in experiments:
            tag = f"{cell}_{tf}"
            lines += [
                f"# {desc}",
                f'BED="$OUT/{tag}.bed"',
                f'if [[ ! -f "$BED" ]]; then',
                f'  echo "Downloading {acc} ({desc})..."',
                f'  wget -q "https://www.encodeproject.org/files/{acc}/@@download/{acc}.bed.gz" \\',
                f'       -O "$BED.gz"',
                f'  gunzip "$BED.gz"',
                f'fi',
                f'# Filter to chr8/chr9 to match SAE window universe',
                f'grep -E "^chr[89]\\b" "$BED" > "$OUT/{tag}_chr89.bed" || true',
                f'N=$(wc -l < "$OUT/{tag}_chr89.bed")',
                f'echo "  {tag}: $N peaks on chr8/chr9"',
                f'if [[ $N -gt 10 && ! -d "$OUT/{tag}_homer/homerResults" ]]; then',
                f'  mkdir -p "$OUT/{tag}_homer"',
                f'  findMotifsGenome.pl "$OUT/{tag}_chr89.bed" "$GENOME" \\',
                f'    "$OUT/{tag}_homer" -size 200 -mask \\',
                f'    -bg "$BG_BED" -p 8 2>"$OUT/{tag}_homer/homer.log"',
                f'  echo "  HOMER done: {tag}"',
                f'fi',
                "",
            ]
    lines += [
        "echo 'ENCODE comparison complete.'",
        "echo 'Results in: $OUT'",
        "echo 'Compare knownResults.txt p-values vs SAE feature peaks'",
        "echo 'to establish upper bound on expected motif enrichment.'",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    out_path.chmod(0o755)
    print(f"  ENCODE comparison script written to {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--figures_dir",    required=True)
    args = parser.parse_args()

    homer_dir   = Path(args.annotation_dir) / "homer"
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(exist_ok=True)
    out_tsv     = Path(args.annotation_dir) / "meta_analysis.tsv"

    print("Loading HOMER results...")
    all_df = load_all(homer_dir)
    if all_df.empty:
        print("ERROR: no HOMER data found.")
        return

    print(f"  {len(all_df)} motif×condition rows loaded")

    print("Running Fisher meta-analysis...")
    meta_df = run_meta_analysis(all_df)
    meta_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"  Meta-analysis saved to {out_tsv}")

    # Print top hits
    top = meta_df.dropna(subset=["fisher_p_best_pair"]).head(10)
    print("\nTop 10 motifs by per-cell-type Fisher p-value:")
    for _, row in top.iterrows():
        print(f"  {row['motif']:<25} best_pair_p={row['fisher_p_best_pair']:.2e} "
              f"({row['fisher_best_pair']})  "
              f"best_single={row['best_pval']:.2e} ({row['best_condition']})")

    print("\nGenerating Fig 10...")
    make_fig10(meta_df, all_df, figures_dir)

    print("\nWriting ENCODE download script...")
    encode_script = Path(args.annotation_dir).parent.parent / "src" / "run_encode_comparison.sh"
    write_encode_script(encode_script)

    print("\nDone.")


if __name__ == "__main__":
    main()
