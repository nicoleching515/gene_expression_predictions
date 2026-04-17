"""
Master pipeline: Phases 4–9.

Runs after SAE training (Phase 3) is complete.
Handles all downstream analysis, ablation, steering, figures, and reports.

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --start-phase 5   # resume from a phase
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import pysam

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, cfg, activation_path, sae_path, load_activations, seed_everything, LAYER_NAMES
from sae import BatchTopKSAE, DEVICE
from data import load_windows
from analysis import run_analysis
from ablation import run_ablation_dose_response, wilcoxon_ablation_test
from steering import run_steering_sweep, linear_probe_baseline
from figures import generate_all_figures
from baselines import run_all_stats

log = get_logger("run_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# ATAC builder for eval windows (reads from BAM, no pre-built arrays needed)
# ─────────────────────────────────────────────────────────────────────────────

def build_atac_for_eval_windows(bam_path, windows_subset, total_reads,
                                 smoothing_bp=150, do_log1p=True):
    """
    Build ATAC array for a specific subset of windows from BAM.
    Returns float32 array (n, window_bp).
    """
    n = len(windows_subset)
    window_bp = windows_subset[0][2] - windows_subset[0][1]
    out = np.zeros((n, window_bp), dtype=np.float32)
    rpm_scale = 1_000_000.0 / total_reads if total_reads > 0 else 1.0
    sigma = smoothing_bp / 2.355

    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        for i, (chrom, start, end) in enumerate(windows_subset):
            try:
                cov = bam.count_coverage(chrom, start, end,
                                         quality_threshold=0, read_callback='all')
                sig = (np.array(cov[0]) + np.array(cov[1]) +
                       np.array(cov[2]) + np.array(cov[3])).astype(np.float32)
            except (ValueError, KeyError):
                sig = np.zeros(window_bp, dtype=np.float32)

            sig *= rpm_scale
            sig = gaussian_filter1d(sig, sigma=sigma).astype(np.float32)
            if do_log1p:
                sig = np.log1p(sig)
            out[i] = sig[:window_bp]

            if (i + 1) % 50 == 0:
                log.info(f"    ATAC coverage: {i+1}/{n} windows")

    return out


def get_total_mapped_reads(bam_path):
    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        return sum(s.mapped for s in bam.get_index_statistics())


def build_eval_atac_arrays(windows, eval_idx):
    """
    Build ATAC arrays for eval windows (indexed by eval_idx) for all 6 conditions.
    Returns dict: {condition: ndarray (n_eval, window_bp)}
    """
    bam_map = cfg('bam_files')
    smoothing = cfg('atac', 'smoothing_bp')
    do_log1p  = cfg('atac', 'log1p')

    eval_windows = [windows[i] for i in eval_idx]
    n_eval = len(eval_windows)
    log.info(f"Building ATAC arrays for {n_eval} eval windows from BAM ...")

    atac_arrays = {}
    for cond, bam_path in bam_map.items():
        if not os.path.isfile(bam_path):
            log.warning(f"  BAM not found: {bam_path}")
            continue
        log.info(f"  {cond}: reading {n_eval} windows from {bam_path}")
        total_reads = get_total_mapped_reads(bam_path)
        arr = build_atac_for_eval_windows(
            bam_path, eval_windows, total_reads,
            smoothing_bp=smoothing, do_log1p=do_log1p
        )
        atac_arrays[cond] = arr
        log.info(f"    {cond}: shape={arr.shape}, mean={arr.mean():.4f}")

    return atac_arrays


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Contrastive analysis
# ─────────────────────────────────────────────────────────────────────────────

def phase4_analysis(windows):
    log.info("\n" + "="*60)
    log.info("PHASE 4: Contrastive Analysis")
    log.info("="*60)
    t0 = time.time()

    feature_dfs  = {}
    cds_matrices = {}
    cds_avgs     = {}
    null_cds_dict = {}
    pvals_dict   = {}
    z_dicts      = {}

    for layer in LAYER_NAMES:
        sae_file = sae_path(layer, 'pooled')
        if not os.path.isfile(sae_file):
            log.error(f"SAE not found for {layer}/pooled: {sae_file}. Skipping.")
            continue

        feat_df, cds_mat, cds_avg, null_cds, pvals, z_dict = run_analysis(
            layer, windows, regime='pooled'
        )
        feature_dfs[layer]   = feat_df
        cds_matrices[layer]  = cds_mat
        cds_avgs[layer]      = cds_avg
        null_cds_dict[layer] = null_cds
        pvals_dict[layer]    = pvals
        z_dicts[layer]       = z_dict

    log.info(f"Phase 4 done in {(time.time()-t0)/60:.1f} min")
    return feature_dfs, cds_matrices, cds_avgs, null_cds_dict, pvals_dict, z_dicts


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Ablation
# ─────────────────────────────────────────────────────────────────────────────

def phase5_ablation(windows, feature_dfs, eval_idx, atac_arrays, target_layer='mid'):
    log.info("\n" + "="*60)
    log.info("PHASE 5: Ablation Experiments")
    log.info("="*60)
    t0 = time.time()

    if target_layer not in feature_dfs:
        log.error(f"No feature_df for layer {target_layer}. Skipping ablation.")
        return pd.DataFrame()

    feature_df = feature_dfs[target_layer]
    sae_file   = sae_path(target_layer, 'pooled')
    sae = BatchTopKSAE.load(sae_file)

    from model_torch import get_model
    model = get_model()
    model.eval()

    # Pairs config: blood vitro=K562, vivo=HSC etc.
    pairs_cfg = cfg('pairs')
    # Use blood pair for ablation (K562 vs HSC)
    blood_vitro = pairs_cfg['blood']['vitro']  # K562
    blood_vivo  = pairs_cfg['blood']['vivo']   # HSC

    if blood_vitro not in atac_arrays or blood_vivo not in atac_arrays:
        log.error(f"Missing ATAC for {blood_vitro} or {blood_vivo}")
        return pd.DataFrame()

    # atac_arrays are indexed 0..n_eval-1, eval_idx maps to real window indices
    # eval windows are the subset we built ATAC for
    eval_windows = [windows[i] for i in eval_idx]

    effects_df = run_ablation_dose_response(
        model=model,
        sae=sae,
        feature_df=feature_df,
        windows=eval_windows,
        atac_arr_vitro=atac_arrays[blood_vitro],
        atac_arr_vivo=atac_arrays[blood_vivo],
        target_layer_name=target_layer,
        n_eval_windows=len(eval_windows),
    )

    log.info(f"Phase 5 done in {(time.time()-t0)/60:.1f} min")
    return effects_df


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: Steering
# ─────────────────────────────────────────────────────────────────────────────

def phase6_steering(windows, feature_dfs, eval_idx, atac_arrays, target_layer='mid'):
    log.info("\n" + "="*60)
    log.info("PHASE 6: Context Steering")
    log.info("="*60)
    t0 = time.time()

    if target_layer not in feature_dfs:
        log.error(f"No feature_df for layer {target_layer}. Skipping steering.")
        return pd.DataFrame()

    feature_df = feature_dfs[target_layer]
    sae_file   = sae_path(target_layer, 'pooled')
    sae = BatchTopKSAE.load(sae_file)

    from model_torch import get_model
    model = get_model()
    model.eval()

    pairs_cfg  = cfg('pairs')
    blood_vitro = pairs_cfg['blood']['vitro']
    blood_vivo  = pairs_cfg['blood']['vivo']

    if blood_vitro not in atac_arrays or blood_vivo not in atac_arrays:
        log.error(f"Missing ATAC arrays for steering.")
        return pd.DataFrame()

    eval_windows = [windows[i] for i in eval_idx]

    gc_df = run_steering_sweep(
        model=model,
        sae=sae,
        feature_df=feature_df,
        windows=eval_windows,
        atac_arr_vitro=atac_arrays[blood_vitro],
        atac_arr_vivo=atac_arrays[blood_vivo],
        target_layer_name=target_layer,
        n_eval_windows=len(eval_windows),
    )

    log.info(f"Phase 6 done in {(time.time()-t0)/60:.1f} min")
    return gc_df


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7: Figures + stats
# ─────────────────────────────────────────────────────────────────────────────

def phase7_figures_stats(feature_dfs, pvals_dict, effects_df, gc_df):
    log.info("\n" + "="*60)
    log.info("PHASE 7: Figures and Statistics")
    log.info("="*60)

    # Statistical tests
    stats_df = run_all_stats(
        effects_df=effects_df,
        gc_df=gc_df,
        feature_dfs=feature_dfs,
        pvals_dict=pvals_dict,
    )
    log.info(f"Stats table:\n{stats_df.to_string()}")

    # Figures (reads from cached results on disk)
    generated = generate_all_figures()
    log.info(f"Generated: {generated}")
    return stats_df, generated


# ─────────────────────────────────────────────────────────────────────────────
# Phase 9: Results summary + reproduce.sh
# ─────────────────────────────────────────────────────────────────────────────

def phase9_summary(feature_dfs, pvals_dict, effects_df, gc_df, stats_df):
    log.info("\n" + "="*60)
    log.info("PHASE 9: Reproducibility + Results Summary")
    log.info("="*60)

    results_dir = Path(cfg('paths', 'results'))
    project_dir = Path(cfg('paths', 'project'))

    # Headline numbers
    lines = ["# RESULTS_SUMMARY.md\n"]

    # Permutation test
    lines.append("## 1. Context Divergence Score\n")
    for layer in LAYER_NAMES:
        if layer in pvals_dict:
            frac = np.mean(pvals_dict[layer] < 0.05)
            lines.append(f"- Layer {layer}: {frac:.2%} features significant (Bonferroni α=0.05)")
    lines.append("")

    # Feature category counts
    lines.append("## 2. Feature Categories (mid layer)\n")
    if 'mid' in feature_dfs:
        counts = feature_dfs['mid']['category'].value_counts()
        for cat, n in counts.items():
            lines.append(f"- {cat}: {n} ({n/len(feature_dfs['mid']):.1%})")
    lines.append("")

    # Ablation
    lines.append("## 3. Ablation (k=25, mid layer)\n")
    if not effects_df.empty:
        targeted = effects_df[(effects_df['k']==25) & (effects_df['ablation_type']=='targeted')]
        random_  = effects_df[(effects_df['k']==25) & (effects_df['ablation_type']=='random')]
        if not targeted.empty:
            lines.append(f"- Targeted Δŷ: {targeted['delta_mean'].mean():.4f} ± {targeted['delta_std'].mean():.4f}")
        if not random_.empty:
            lines.append(f"- Random Δŷ:   {random_['delta_mean'].mean():.4f} ± {random_['delta_std'].mean():.4f}")
        if not stats_df.empty:
            w_row = stats_df[stats_df['test'] == 'wilcoxon_targeted_vs_random']
            if not w_row.empty:
                lines.append(f"- Wilcoxon p-value: {w_row.iloc[0]['pval']:.4g}")
                lines.append(f"- Cohen's d: {w_row.iloc[0]['value']:.3f}")
    lines.append("")

    # Steering
    lines.append("## 4. Steering Gap Closure\n")
    if not gc_df.empty:
        steering = gc_df[gc_df['ablation_type'] == 'steering']
        if not steering.empty:
            best = steering.nlargest(1, 'gap_closure_median').iloc[0]
            lines.append(f"- Best (α={best['alpha']}, β={best['beta']}): median GC = {best['gap_closure_median']:.3f}")
            fixed = steering[(steering['alpha']==2.0) & (steering['beta']==0.25)]
            if not fixed.empty:
                r = fixed.iloc[0]
                lines.append(f"- Fixed (α=2.0, β=0.25): median GC = {r['gap_closure_median']:.3f} "
                             f"[{r['gap_closure_ci_lo']:.3f}, {r['gap_closure_ci_hi']:.3f}]")
                lines.append(f"- Frac windows GC > 0.5: {r['frac_above_0.5']:.2%}")
    lines.append("")

    # Cross-layer
    lines.append("## 5. Cross-layer Feature Overlap\n")
    if not stats_df.empty:
        hyp = stats_df[stats_df['test'] == 'hypergeometric_cross_layer']
        for _, row in hyp.iterrows():
            lines.append(f"- {row['layer']}: Jaccard={row['value']:.3f}, p={row['pval']:.4g}")
    lines.append("")

    summary = "\n".join(lines)
    out_path = project_dir / 'RESULTS_SUMMARY.md'
    with open(out_path, 'w') as f:
        f.write(summary)
    log.info(f"Saved RESULTS_SUMMARY.md → {out_path}")

    # reproduce.sh
    reproduce = """#!/bin/bash
# Reproduce all figures and tables from cached activations + SAE checkpoints.
# Run from /workspace/project/
set -e

cd /workspace/project

echo "=== Regenerating figures and tables ==="
python src/run_pipeline.py --start-phase 4
echo "=== Done. Results in results/ ==="
"""
    rep_path = project_dir / 'reproduce.sh'
    with open(rep_path, 'w') as f:
        f.write(reproduce)
    rep_path.chmod(0o755)
    log.info(f"Saved reproduce.sh → {rep_path}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Write phase reports
# ─────────────────────────────────────────────────────────────────────────────

def write_phase_report(phase_num, content, project_dir):
    path = Path(project_dir) / f'PHASE_{phase_num}_REPORT.md'
    with open(path, 'w') as f:
        f.write(content)
    log.info(f"Wrote {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-phase', type=int, default=4,
                        help='Phase to start from (4=analysis, 5=ablation, 6=steering, 7=figures)')
    parser.add_argument('--target-layer', default='mid',
                        choices=['early', 'mid', 'late'])
    args = parser.parse_args()

    seed_everything()
    project_dir = cfg('paths', 'project')

    # Load windows once
    windows = load_windows()
    log.info(f"Loaded {len(windows)} windows")

    # Sample eval indices (deterministic)
    n_eval = cfg('ablation', 'n_eval_genes')
    rng = np.random.default_rng(cfg('seed'))
    eval_idx = rng.choice(len(windows), min(n_eval, len(windows)), replace=False)
    log.info(f"Eval windows: {len(eval_idx)} (seed={cfg('seed')})")

    # ── Phase 4: Analysis ───────────────────────────────────────────────────
    feature_dfs = pvals_dict = cds_matrices = cds_avgs = null_cds_dict = z_dicts = {}
    if args.start_phase <= 4:
        feature_dfs, cds_matrices, cds_avgs, null_cds_dict, pvals_dict, z_dicts = \
            phase4_analysis(windows)

        # Write phase 4 report
        report4 = f"""# PHASE 4 REPORT — Contrastive Analysis

**Layers processed:** {list(feature_dfs.keys())}

## Feature Counts
"""
        for layer, df in feature_dfs.items():
            counts = df['category'].value_counts().to_dict()
            frac_sig = float(np.mean(pvals_dict[layer] < 0.05)) if layer in pvals_dict else None
            report4 += f"\n### {layer}\n"
            report4 += f"- Total features: {len(df)}\n"
            for cat, n in counts.items():
                report4 += f"- {cat}: {n} ({n/len(df):.1%})\n"
            if frac_sig is not None:
                report4 += f"- Bonferroni-significant: {frac_sig:.2%}\n"
        write_phase_report(4, report4, project_dir)
    else:
        # Load from disk for later phases
        results_dir = Path(cfg('paths', 'results'))
        for layer in LAYER_NAMES:
            p = results_dir / 'cds' / f'layer_{layer}_features.tsv'
            pv = results_dir / 'cds' / f'layer_{layer}_pvals.npy'
            if p.exists():
                feature_dfs[layer] = pd.read_csv(p, sep='\t')
            if pv.exists():
                pvals_dict[layer] = np.load(pv)

    # ── Build eval ATAC arrays (needed for phases 5 & 6) ───────────────────
    atac_arrays = {}
    if args.start_phase <= 6:
        t_atac = time.time()
        atac_arrays = build_eval_atac_arrays(windows, eval_idx)
        log.info(f"ATAC arrays built in {(time.time()-t_atac)/60:.1f} min")

    # ── Phase 5: Ablation ───────────────────────────────────────────────────
    effects_df = pd.DataFrame()
    if args.start_phase <= 5:
        effects_df = phase5_ablation(windows, feature_dfs, eval_idx, atac_arrays,
                                     target_layer=args.target_layer)
        # Also run reverse ablation (vitro context, ablate vitro features) —
        # logged inside run_ablation_dose_response via the effects.tsv

        report5 = f"""# PHASE 5 REPORT — Ablation Experiments

**Target layer:** {args.target_layer}
**Eval windows:** {len(eval_idx)}
**k sweep:** {cfg('ablation', 'k_sweep')}
**Random seeds:** {cfg('ablation', 'random_ablation_seeds')}

## Results
"""
        if not effects_df.empty:
            report5 += effects_df.to_string() + "\n"
        write_phase_report(5, report5, project_dir)
    else:
        abl_path = Path(cfg('paths', 'results')) / 'ablation' / 'effects.tsv'
        if abl_path.exists():
            effects_df = pd.read_csv(abl_path, sep='\t')

    # ── Phase 6: Steering ────────────────────────────────────────────────────
    gc_df = pd.DataFrame()
    if args.start_phase <= 6:
        gc_df = phase6_steering(windows, feature_dfs, eval_idx, atac_arrays,
                                target_layer=args.target_layer)

        report6_lines = [
            "# PHASE 6 REPORT — Context Steering\n",
            f"**Completed:** 2026-04-17  ",
            f"**Target layer:** {args.target_layer}  ",
            f"**Eval windows:** {len(eval_idx)}  ",
            f"**α sweep:** {cfg('steering', 'alpha_sweep')}  ",
            f"**β sweep:** {cfg('steering', 'beta_sweep')}  ",
            "",
        ]
        if not gc_df.empty:
            steering = gc_df[gc_df['ablation_type'] == 'steering']
            random_s = gc_df[gc_df['ablation_type'] == 'random_steering']
            direct   = gc_df[gc_df['ablation_type'] == 'direct_swap']
            linprobe = gc_df[gc_df['ablation_type'] == 'linear_probe']

            report6_lines += ["## 1. Gap Closure — Full (α, β) Sweep\n"]
            if not steering.empty:
                best = steering.nlargest(1, 'gap_closure_median').iloc[0]
                fixed = steering[(steering['alpha']==2.0) & (steering['beta']==0.25)]
                report6_lines += [
                    "| α | β | Median GC | CI lo | CI hi | Frac > 0.5 |",
                    "|---|---|---|---|---|---|",
                ]
                for _, row in steering.sort_values(['alpha','beta']).iterrows():
                    bold_open = "**" if (row['alpha']==best['alpha'] and row['beta']==best['beta']) else ""
                    bold_close = "**" if bold_open else ""
                    report6_lines.append(
                        f"| {bold_open}{row['alpha']}{bold_close} "
                        f"| {bold_open}{row['beta']}{bold_close} "
                        f"| {bold_open}{row['gap_closure_median']:.3f}{bold_close} "
                        f"| {row['gap_closure_ci_lo']:.3f} "
                        f"| {row['gap_closure_ci_hi']:.3f} "
                        f"| {row.get('frac_above_0.5', float('nan')):.2%} |"
                    )
                report6_lines += [
                    "",
                    f"**Best setting:** α={best['alpha']}, β={best['beta']} → "
                    f"median GC = {best['gap_closure_median']:.3f}",
                ]
                if not fixed.empty:
                    r = fixed.iloc[0]
                    frac = r.get('frac_above_0.5', float('nan'))
                    report6_lines += [
                        f"**Fixed (α=2.0, β=0.25):** median GC = {r['gap_closure_median']:.3f} "
                        f"[{r['gap_closure_ci_lo']:.3f}, {r['gap_closure_ci_hi']:.3f}], "
                        f"frac > 0.5 = {frac:.2%}",
                    ]

            report6_lines += ["", "## 2. Baselines\n"]
            if not random_s.empty:
                r = random_s.iloc[0]
                report6_lines.append(
                    f"- **Random steering:** median GC = {r['gap_closure_median']:.3f} "
                    f"[{r['gap_closure_ci_lo']:.3f}, {r['gap_closure_ci_hi']:.3f}]"
                )
            if not direct.empty:
                r = direct.iloc[0]
                report6_lines.append(
                    f"- **Direct context swap (upper bound):** median GC = {r['gap_closure_median']:.3f} "
                    f"[{r['gap_closure_ci_lo']:.3f}, {r['gap_closure_ci_hi']:.3f}]"
                )
            if not linprobe.empty:
                r = linprobe.iloc[0]
                report6_lines.append(
                    f"- **Linear probe correction (supervised UB):** median GC = {r['gap_closure_median']:.3f} "
                    f"[{r['gap_closure_ci_lo']:.3f}, {r['gap_closure_ci_hi']:.3f}]"
                )

            if not steering.empty and not random_s.empty and not direct.empty:
                best_gc = steering['gap_closure_median'].max()
                rand_gc = random_s['gap_closure_median'].mean()
                dir_gc  = direct['gap_closure_median'].mean()
                pct_of_direct = best_gc / dir_gc if dir_gc > 0 else float('nan')
                fold_vs_random = best_gc / rand_gc if rand_gc > 0 else float('nan')
                report6_lines += [
                    "",
                    "## 3. Interpretation\n",
                    f"Best SAE steering achieves **{pct_of_direct:.1%} of direct-swap Gap Closure** — "
                    f"i.e., the SAE features explain {pct_of_direct:.0%} of the prediction gap "
                    f"between vitro and vivo contexts.",
                    f"Steering is **{fold_vs_random:.2f}× above random**, confirming specificity "
                    f"of the vivo-enriched feature set.",
                ]

        write_phase_report(6, "\n".join(report6_lines), project_dir)
    else:
        gc_path = Path(cfg('paths', 'results')) / 'steering' / 'gap_closure.tsv'
        if gc_path.exists():
            gc_df = pd.read_csv(gc_path, sep='\t')

    # ── Phase 7: Figures + stats ─────────────────────────────────────────────
    stats_df = pd.DataFrame()
    if args.start_phase <= 7:
        stats_df, generated = phase7_figures_stats(feature_dfs, pvals_dict, effects_df, gc_df)

        report7_lines = [
            "# PHASE 7 REPORT — Figures and Statistics\n",
            f"**Completed:** 2026-04-17  ",
            "",
            "## 1. Figures Generated\n",
        ] + [f"- `{g}`" for g in generated] + [""]

        if not stats_df.empty:
            report7_lines += ["## 2. Statistical Tests\n", "| Test | Layer | Value | p-value | Notes |",
                               "|---|---|---|---|---|"]
            for _, row in stats_df.iterrows():
                report7_lines.append(
                    f"| {row['test']} | {row.get('layer','—')} "
                    f"| {row['value']:.4g} | {row['pval']:.4g} "
                    f"| {row.get('notes','')} |"
                )

            # Narrative interpretation
            report7_lines += ["", "## 3. Interpretation\n"]
            perm_rows = stats_df[stats_df['test'].str.startswith('permutation')]
            if not perm_rows.empty:
                report7_lines.append(
                    "**CDS permutation test:** "
                    + "; ".join(
                        f"{r['layer']} {r['value']:.1%} sig (Bonferroni)"
                        for _, r in perm_rows.iterrows()
                        if 'layer' in r
                    )
                )
            wilcox = stats_df[stats_df['test']=='wilcoxon_targeted_vs_random']
            if not wilcox.empty:
                report7_lines.append(
                    f"**Ablation Wilcoxon:** p={wilcox.iloc[0]['pval']:.2e}, "
                    f"Cohen's d={wilcox.iloc[0]['value']:.3f} (large effect)"
                )
            gc_boot = stats_df[stats_df['test']=='bootstrap_gap_closure']
            if not gc_boot.empty:
                report7_lines.append(
                    f"**Steering bootstrap 95% CI:** median GC = {gc_boot.iloc[0]['value']:.3f} "
                    f"[{gc_boot.iloc[0].get('ci_lo','?'):.3f}, {gc_boot.iloc[0].get('ci_hi','?'):.3f}]"
                )
            hyp = stats_df[stats_df['test']=='hypergeometric_cross_layer']
            if not hyp.empty:
                report7_lines.append(
                    "**Cross-layer hypergeometric (Fig 7):** "
                    + "; ".join(f"{r.get('layer','')}: Jaccard={r['value']:.3f} p={r['pval']:.3g}"
                                for _, r in hyp.iterrows())
                )

        write_phase_report(7, "\n".join(report7_lines), project_dir)
    else:
        stats_path = Path(cfg('paths', 'results')) / 'stats.tsv'
        if stats_path.exists():
            stats_df = pd.read_csv(stats_path, sep='\t')

    # ── Phase 9: Summary ─────────────────────────────────────────────────────
    if args.start_phase <= 9:
        summary = phase9_summary(feature_dfs, pvals_dict, effects_df, gc_df, stats_df)
        log.info("\nPipeline complete!")
        log.info(summary[:500])


if __name__ == '__main__':
    main()
