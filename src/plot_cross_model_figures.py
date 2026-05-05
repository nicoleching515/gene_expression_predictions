#!/usr/bin/env python3
"""
src/plot_cross_model_figures.py  (v2 — corrected analyses)
===========================================================
Figures 12–16 for the cross-model SAE interpretability paper.

Fig 12  Row A: % Bonferroni-significant CDS features per model × layer × pair.
        Row B: max |CDS| (log y) — visible contrast when Bonferroni % is ~0.
Fig 13  Window-space Jaccard heatmap between model pairs (corrected)
Fig 14  CDS score distributions (violin) per model × layer
Fig 15  HOMER motif enrichment dot-plot for top-5 motifs per model/layer
Fig 16  Layer-depth signatures: directional asymmetry (vitro vs vivo enriched)
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

LAYERS = ["early", "mid", "late"]
PAIRS  = ["blood", "liver", "lymph"]
LAYER_LABELS = {"early": "Early\n(L/4)", "mid": "Mid\n(L/2)", "late": "Late\n(3L/4)"}
PAIR_LABELS  = {"blood": "Blood\n(K562/HSC)", "liver": "Liver\n(HepG2/Liver)",
                "lymph": "Lymph\n(GM12878/NaiveB)"}
MODEL_COLORS = {
    "epibert":                "#4E79A7",
    "enformer":               "#F28E2B",
    "hyenadna":               "#59A14F",
    "nucleotide_transformer": "#E15759",
}
MODEL_LABELS = {
    "epibert":                "EpiBERT",
    "enformer":               "Enformer",
    "hyenadna":               "HyenaDNA",
    "nucleotide_transformer": "NT",
}
MODEL_ORDER = ["epibert", "enformer", "hyenadna", "nucleotide_transformer"]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_summary(d: Path) -> pd.DataFrame:
    p = d / "cross_model_summary.tsv"
    return pd.read_csv(p, sep="\t") if p.exists() else pd.DataFrame()


def load_window_jaccard(d: Path) -> pd.DataFrame:
    p = d / "window_jaccard.tsv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, sep="\t")
    except Exception:
        return pd.DataFrame()


def load_cds_tables(d: Path) -> dict[tuple, pd.DataFrame]:
    out = {}
    for f in sorted(d.glob("*_cds.tsv")):
        stem = f.stem  # e.g. enformer_layer_mid_blood_cds
        parts = stem.split("_layer_")
        if len(parts) != 2:
            continue
        model = parts[0]
        rest  = parts[1].split("_")
        if len(rest) < 2:
            continue
        layer, pair = rest[0], rest[1]
        out[(model, layer, pair)] = pd.read_csv(f, sep="\t")
    return out


def load_motifs(d: Path) -> pd.DataFrame:
    p = d / "sae_motif_summary.tsv"
    return pd.read_csv(p, sep="\t") if p.exists() else pd.DataFrame()


# ── Fig 12: % significant features ────────────────────────────────────────────

def _fig12_panel_grouped_bars(
    ax,
    summary: pd.DataFrame,
    pair: str,
    models: list,
    value_col: str,
    *,
    ylabel: str,
    pct_formatter: bool,
    annotate_nonzero: bool,
    fmt_cell: str,
    offset_frac: float = 0.02,
    log_y: bool = False,
    pct_ymax_cap: float | None = None,
) -> float:
    """Grouped bars: models × layers for one cell-type pair.

    Returns the panel ymax used (before optional pct cap).
    """
    sub = summary[summary["pair"] == pair]
    n_m = len(models)
    w = 0.18
    x = np.arange(len(LAYERS))
    ymax = 0.0
    all_plot_heights: list[float] = []

    for mi, model in enumerate(models):
        ms = sub[sub["model"] == model]
        vals: list[float] = []
        for layer in LAYERS:
            row = ms[ms["layer"] == layer]
            if len(row):
                v = float(row[value_col].values[0])
            else:
                v = 0.0
            vals.append(v)
            ymax = max(ymax, v)

        if log_y:
            plot_vals = [max(v, 1e-12) for v in vals]
        else:
            plot_vals = vals

        all_plot_heights.extend(plot_vals)

        off = (mi - (n_m - 1) / 2) * w
        ax.bar(
            x + off,
            plot_vals,
            width=w,
            color=MODEL_COLORS.get(model, "#999"),
            label=MODEL_LABELS.get(model, model),
            edgecolor="white",
            linewidth=0.5,
        )
        if annotate_nonzero:
            for xi, v in zip(x + off, vals):
                if v > 0:
                    ax.text(
                        xi,
                        v + offset_frac * max(ymax, 1e-9),
                        fmt_cell.format(v),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=45,
                    )

    ax.set_xticks(x)
    ax.set_xticklabels([LAYER_LABELS[l] for l in LAYERS], fontsize=9)
    ax.set_title(PAIR_LABELS[pair], fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    if pct_formatter:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    if log_y:
        ax.set_yscale("log")
        lo = min(all_plot_heights)
        hi = max(all_plot_heights)
        ax.set_ylim(lo * 0.35, hi * 2.5)
    else:
        top = ymax * 1.15 if ymax > 0 else 1.0
        cap = pct_ymax_cap if pct_ymax_cap is not None else top
        ax.set_ylim(0, max(top, cap, 0.02 if pct_formatter else 1e-6))

    return ymax


def make_fig12(summary: pd.DataFrame, fdir: Path) -> None:
    if summary.empty:
        return _placeholder("fig12_cross_model_cds", fdir, "No summary data.")
    models = [m for m in MODEL_ORDER if m in summary["model"].unique()]
    n_m = len(models)
    fig, axes = plt.subplots(
        2,
        len(PAIRS),
        figsize=(5 * len(PAIRS), 8.5),
        constrained_layout=True,
    )
    if len(PAIRS) == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        "Context-Divergent SAE Features Across Cell-Type Pairs\n"
        "(vitro-only vs vivo-only ATAC windows)",
        fontsize=12,
        fontweight="bold",
    )

    pct_global_max = 0.0
    for pair in PAIRS:
        sub = summary[summary["pair"] == pair]
        for model in models:
            ms = sub[sub["model"] == model]
            for layer in LAYERS:
                row = ms[ms["layer"] == layer]
                if len(row):
                    pct_global_max = max(pct_global_max, float(row["pct_significant"].values[0]))

    pct_cap = max(pct_global_max * 1.2, 0.08) if pct_global_max > 0 else 0.08

    for ci, pair in enumerate(PAIRS):
        _fig12_panel_grouped_bars(
            axes[0][ci],
            summary,
            pair,
            models,
            "pct_significant",
            ylabel="% Bonferroni-significant features" if ci == 0 else "",
            pct_formatter=True,
            annotate_nonzero=True,
            fmt_cell="{:.2f}%",
            offset_frac=0.05,
            log_y=False,
            pct_ymax_cap=pct_cap,
        )
        axes[0][ci].text(
            0.02,
            0.98,
            "Bonferroni p < 0.05",
            transform=axes[0][ci].transAxes,
            fontsize=8,
            va="top",
            ha="left",
            color="#444",
        )

    fig.text(
        0.5,
        0.485,
        "Row B — max |CDS| (log scale); comparable effect magnitudes across architectures",
        ha="center",
        fontsize=9,
        style="italic",
        color="#333",
    )

    for ci, pair in enumerate(PAIRS):
        _fig12_panel_grouped_bars(
            axes[1][ci],
            summary,
            pair,
            models,
            "max_abs_cds",
            ylabel="Max |CDS|" if ci == 0 else "",
            pct_formatter=False,
            annotate_nonzero=False,
            fmt_cell="",
            log_y=True,
        )

    handles = [
        Line2D([0], [0], color=MODEL_COLORS.get(m, "#999"), linewidth=6,
               label=MODEL_LABELS.get(m, m))
        for m in models
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=n_m,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
        frameon=True,
        edgecolor="#CCC",
    )
    _save(fig, fdir / "fig12_cross_model_cds")


# ── Fig 13: Window-space Jaccard heatmap ─────────────────────────────────────

def make_fig13(jaccard: pd.DataFrame, fdir: Path) -> None:
    if jaccard.empty:
        return _placeholder("fig13_window_jaccard", fdir, "No window Jaccard data.")
    models_all = list({m for col in ("model_1","model_2") for m in jaccard[col].unique()})
    models = [m for m in MODEL_ORDER if m in models_all]
    n_m = len(models)
    has_pairs = [p for p in PAIRS if
                 not jaccard[(jaccard["pair"]==p)].empty]
    n_cols, n_rows = len(LAYERS), len(has_pairs)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4*n_cols, 3.5*n_rows), constrained_layout=True)
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [list(axes)]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    fig.suptitle("Window-Space Jaccard: Top-100 CDS-weighted Windows per Model\n"
                 "(corrected: overlap in genomic window space, not feature-index space)",
                 fontsize=11, fontweight="bold")
    for ri, pair in enumerate(has_pairs):
        for ci, layer in enumerate(LAYERS):
            ax = axes[ri][ci]
            mat = np.full((n_m, n_m), np.nan)
            sub = jaccard[(jaccard["layer"]==layer) & (jaccard["pair"]==pair)]
            for _, row in sub.iterrows():
                m1, m2 = row["model_1"], row["model_2"]
                if m1 in models and m2 in models:
                    i1, i2 = models.index(m1), models.index(m2)
                    mat[i1,i2] = mat[i2,i1] = row["jaccard"]
            np.fill_diagonal(mat, 1.0)
            cmap = plt.cm.YlOrRd.copy(); cmap.set_bad("#F0F0F0")
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=0.5,
                           aspect="auto", interpolation="nearest")
            for i in range(n_m):
                for j in range(n_m):
                    v = mat[i,j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=9, color="white" if v>0.3 else "black",
                                fontweight="bold" if v>0 and i!=j else "normal")
            sl = [MODEL_LABELS.get(m,m) for m in models]
            ax.set_xticks(range(n_m)); ax.set_yticks(range(n_m))
            ax.set_xticklabels(sl, rotation=30, ha="right", fontsize=8)
            ax.set_yticklabels(sl, fontsize=8)
            ax.set_title(f"{LAYER_LABELS[layer].replace(chr(10),' ')} | {pair}", fontsize=9)
            if ci == n_cols-1:
                plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03).ax.tick_params(labelsize=7)
    _save(fig, fdir/"fig13_window_jaccard")


# ── Fig 14: CDS distribution violins ─────────────────────────────────────────

def make_fig14(cds_tables: dict, fdir: Path) -> None:
    if not cds_tables:
        return _placeholder("fig14_cds_violin", fdir, "No CDS tables.")
    models_p = sorted({k[0] for k in cds_tables})
    models   = [m for m in MODEL_ORDER if m in models_p]
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(5*len(LAYERS), 5),
                             constrained_layout=True)
    fig.suptitle("|CDS| Distribution by Model and Layer\n"
                 "(|CDS| = |mean activation at K562-specific − HSC-specific windows|)",
                 fontsize=11, fontweight="bold")
    for li, layer in enumerate(LAYERS):
        ax = axes[li]
        pos, data, colors, labels = [], [], [], []
        for xi, model in enumerate(models, 1):
            vals = [cds_tables[(model,layer,p)]["cds"].abs().values
                    for p in PAIRS if (model,layer,p) in cds_tables]
            if not vals:
                continue
            data.append(np.concatenate(vals))
            pos.append(xi); colors.append(MODEL_COLORS.get(model,"#999"))
            labels.append(MODEL_LABELS.get(model,model))
        if not data:
            ax.set_visible(False); continue
        parts = ax.violinplot(data, positions=pos, showmedians=True, showextrema=False)
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col); pc.set_alpha(0.7); pc.set_edgecolor("white")
        parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.5)
        ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
        ax.set_title(LAYER_LABELS[layer], fontsize=10, fontweight="bold")
        ax.set_ylabel("|CDS|" if li==0 else "", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
    _save(fig, fdir/"fig14_cds_violin")


# ── Fig 15: HOMER motif enrichment dot-plot ───────────────────────────────────

def make_fig15(motifs: pd.DataFrame, fdir: Path) -> None:
    if motifs.empty:
        return _placeholder("fig15_homer_motifs", fdir, "No motif data.")
    blood = motifs[motifs["pair"] == "blood"].copy()
    if blood.empty:
        return _placeholder("fig15_homer_motifs", fdir, "No blood motif data.")

    blood["neg_logp"] = -blood["log_pval"]
    blood["pct_num"]  = pd.to_numeric(
        blood["pct_target"].astype(str).str.replace("%",""), errors="coerce")

    models  = [m for m in MODEL_ORDER if m in blood["model"].unique()]
    n_m     = len(models)
    fig, axes = plt.subplots(1, n_m, figsize=(3.5*n_m, 8), constrained_layout=True,
                             sharey=False)
    if n_m == 1:
        axes = [axes]
    fig.suptitle("Top HOMER Motifs in CDS-Significant Windows (blood pair)\n"
                 "Size = −log₁₀ p-value  |  Colour = % target sequences",
                 fontsize=11, fontweight="bold")

    for mi, model in enumerate(models):
        ax = axes[mi]
        sub = blood[blood["model"] == model].copy()
        if sub.empty:
            ax.set_visible(False); continue

        # One entry per motif per layer (take best p per motif×layer)
        sub = sub.sort_values("neg_logp", ascending=False)
        motif_order = sub["motif"].unique()[:12]   # top 12 unique motifs

        y_pos = {m: i for i, m in enumerate(reversed(motif_order))}
        layer_offsets = {"early": -0.2, "mid": 0.0, "late": 0.2}
        layer_shapes  = {"early": "^", "mid": "o", "late": "s"}

        for _, row in sub.iterrows():
            if row["motif"] not in y_pos:
                continue
            y = y_pos[row["motif"]] + layer_offsets.get(row["layer"], 0)
            sc = ax.scatter(row["neg_logp"], y,
                            s=max(20, row["neg_logp"]*15),
                            c=[row["pct_num"] if not pd.isna(row["pct_num"]) else 0],
                            cmap="YlOrRd", vmin=0, vmax=30,
                            marker=layer_shapes.get(row["layer"], "o"),
                            edgecolors="white", linewidths=0.4, zorder=3)

        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels(list(reversed(motif_order)), fontsize=8)
        ax.set_xlabel("−log p-value", fontsize=9)
        ax.set_title(MODEL_LABELS.get(model, model),
                     fontsize=10, fontweight="bold",
                     color=MODEL_COLORS.get(model, "#333"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linestyle=":")
        ax.axvline(x=5, color="#CCCCCC", linestyle="--", linewidth=0.8)

    # Legend for layers
    legend_els = [Line2D([0],[0], marker=layer_shapes[l], color="#555", linestyle="",
                         markersize=7, label=l.capitalize()) for l in LAYERS]
    fig.legend(handles=legend_els, title="Layer", loc="lower center",
               ncol=3, bbox_to_anchor=(0.5,-0.03), fontsize=8)
    _save(fig, fdir/"fig15_homer_motifs")


# ── Fig 16: Layer-depth directional asymmetry ─────────────────────────────────

def make_fig16(summary: pd.DataFrame, cds_tables: dict, fdir: Path) -> None:
    """
    For each model, plot the fraction of significant features that are
    vitro-enriched (CDS > 0) vs vivo-enriched (CDS < 0) across layers.
    Reveals architecture-specific representational biases.
    """
    models_p = {k[0] for k in cds_tables}
    models = [m for m in MODEL_ORDER if m in models_p]
    fig, axes = plt.subplots(1, len(models), figsize=(3.5*len(models), 5),
                             constrained_layout=True, sharey=True)
    if len(models) == 1:
        axes = [axes]
    fig.suptitle("Layer-Depth Directional Asymmetry of Significant CDS Features\n"
                 "(vitro-enriched = K562-specific; vivo-enriched = HSC-specific)",
                 fontsize=11, fontweight="bold")

    for mi, model in enumerate(models):
        ax = axes[mi]
        vitro_counts, vivo_counts = [], []
        for layer in LAYERS:
            n_pos = n_neg = 0
            for pair in PAIRS:
                key = (model, layer, pair)
                if key in cds_tables:
                    sig = cds_tables[key][cds_tables[key]["significant"]]
                    n_pos += (sig["cds"] > 0).sum()
                    n_neg += (sig["cds"] < 0).sum()
            vitro_counts.append(n_pos)
            vivo_counts.append(n_neg)

        x = np.arange(len(LAYERS))
        col = MODEL_COLORS.get(model, "#999")
        ax.bar(x - 0.18, vitro_counts, 0.35, label="Vitro-enriched (K562)",
               color=col, alpha=0.9, edgecolor="white")
        ax.bar(x + 0.18, vivo_counts, 0.35, label="Vivo-enriched (HSC)",
               color=col, alpha=0.4, edgecolor=col, linewidth=1)

        for xi, (v, u) in enumerate(zip(vitro_counts, vivo_counts)):
            total = v + u
            if total:
                ratio = v / total
                ax.text(xi, max(v,u)+0.5, f"{ratio:.0%}", ha="center",
                        va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([LAYER_LABELS[l].replace("\n"," ") for l in LAYERS],
                           fontsize=9)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=10, fontweight="bold",
                     color=col)
        ax.set_ylabel("# significant features" if mi==0 else "", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle=":")

        if mi == 0:
            ax.legend(fontsize=8, frameon=False)

    _save(fig, fdir/"fig16_layer_asymmetry")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _placeholder(name: str, fdir: Path, msg: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="grey")
    ax.axis("off")
    _save(fig, fdir/name)


def _save(fig, stem: Path) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {stem.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/cross_model")
    parser.add_argument("--figures_dir", default="results/figures")
    args = parser.parse_args()

    rdir = Path(args.results_dir)
    fdir = Path(args.figures_dir)
    fdir.mkdir(parents=True, exist_ok=True)

    summary    = load_summary(rdir)
    win_jac    = load_window_jaccard(rdir)
    cds_tables = load_cds_tables(rdir)
    motifs     = load_motifs(rdir)

    print("Generating Fig 12: % significant features …")
    make_fig12(summary, fdir)

    print("Generating Fig 13: Window-space Jaccard heatmap …")
    make_fig13(win_jac, fdir)

    print("Generating Fig 14: CDS violin distributions …")
    make_fig14(cds_tables, fdir)

    print("Generating Fig 15: HOMER motif dot-plot …")
    make_fig15(motifs, fdir)

    print("Generating Fig 16: Layer-depth directional asymmetry …")
    make_fig16(summary, cds_tables, fdir)

    print(f"\nAll figures → {fdir}/")


if __name__ == "__main__":
    main()
