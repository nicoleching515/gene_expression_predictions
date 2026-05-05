#!/usr/bin/env python3
"""
src/cross_model_analysis.py
============================
Phase 12 — Cross-model SAE training and contrastive analysis.

For each model in {epibert, enformer, hyenadna, nucleotide_transformer}:
  1. Train one SAE per hook layer using activations from
     activations/{model}/{pair}/{condition}/{layer}.pt
  2. Compute Context Divergence Score (CDS) per feature per pair.
  3. Summarise: n_significant features, CDS distribution, cross-layer Jaccard.

Then produce cross-model comparison statistics:
  - Table: n_significant features per model × layer
  - CDS distribution comparison
  - Feature-overlap (Jaccard) between model pairs

Outputs:
    results/cross_model/
        {model}_layer_{layer}_features.tsv   — CDS table per model
        cross_model_summary.tsv              — summary table
        cross_model_jaccard.tsv              — pairwise model feature overlap

Usage:
    python src/cross_model_analysis.py [--models all] [--layers early mid late]
    python src/cross_model_analysis.py --smoke-test   # 200-step SAEs
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

# A100 supports TF32 for matmuls; ~4x faster than FP32 with negligible accuracy loss.
torch.set_float32_matmul_precision("high")

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

from utils import get_logger, cfg, seed_everything
from sae import BatchTopKSAE

warnings.filterwarnings("ignore")
log = get_logger("cross_model_analysis")

MODELS  = ["epibert", "enformer", "hyenadna", "nucleotide_transformer"]
LAYERS  = ["early", "mid", "late"]
PAIRS   = ["blood", "liver", "lymph"]
SIDES   = ["vitro", "vivo"]

PAIR_CONDS = {
    "blood": {"vitro": "K562",    "vivo": "HSC"},
    "liver": {"vitro": "HepG2",   "vivo": "Liver"},
    "lymph": {"vitro": "GM12878", "vivo": "NaiveB"},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── ATAC-based window index labelling ─────────────────────────────────────────

_LABEL_CACHE: pd.DataFrame | None = None


def _load_labels() -> pd.DataFrame:
    global _LABEL_CACHE
    if _LABEL_CACHE is None:
        p = REPO / "data" / "window_condition_labels.tsv"
        if not p.exists():
            raise FileNotFoundError(
                f"Window condition labels not found: {p}\n"
                f"Run: python src/assign_condition_windows.py"
            )
        _LABEL_CACHE = pd.read_csv(p, sep="\t")
    return _LABEL_CACHE


def get_condition_indices(pair: str, side: str) -> np.ndarray:
    """
    Return 0-based window indices that are accessible in `side` condition
    for `pair`, but NOT in the opposing condition.

    This gives differentially accessible sequences for each context:
      - vitro: windows open in cell-line but not in primary tissue
      - vivo:  windows open in primary tissue but not in cell-line

    For sequence-only models (Enformer, HyenaDNA, NT):
      activations at these indices differ in *sequence content* because the
      regulatory grammar at cell-line-specific vs tissue-specific accessible
      sites is biologically distinct.

    For EpiBERT:
      activations additionally differ in the ATAC input channel, giving
      doubly-conditioned representations.
    """
    labels = _load_labels()
    vitro_cond = PAIR_CONDS[pair]["vitro"]
    vivo_cond  = PAIR_CONDS[pair]["vivo"]
    if side == "vitro":
        mask = labels[vitro_cond].astype(bool) & ~labels[vivo_cond].astype(bool)
    else:
        mask = labels[vivo_cond].astype(bool) & ~labels[vitro_cond].astype(bool)
    idx = labels.index[mask].to_numpy()
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _act_path(model: str, pair: str, cond: str, layer: str) -> Path:
    return REPO / "activations" / model / pair / cond / f"{layer}.pt"


def _sae_path(model: str, layer: str) -> Path:
    p = REPO / "saes" / model / layer
    p.mkdir(parents=True, exist_ok=True)
    return p / "pooled.pt"


def _result_dir() -> Path:
    d = REPO / "results" / "cross_model"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_activations(model: str, layer: str) -> dict[tuple, np.ndarray]:
    """
    Load activations for (model, layer), subsetted to condition-specific
    accessible windows using ATAC peak overlap labels.

    Returns {(pair, side): np.ndarray (N_subset, hidden_dim)}.

    For sequence-only models (Enformer, HyenaDNA, NT) all condition files
    contain identical activations (same hg38 sequence), so we load from the
    vitro condition file for both vitro and vivo but index into different
    ATAC-accessible window sets — giving biologically distinct sequence
    populations even though the underlying model is sequence-only.

    For EpiBERT the vitro and vivo condition files genuinely differ (the ATAC
    input channel is cell-type specific), so we load from the matched file.
    """
    result = {}
    is_epibert = (model == "epibert")

    for pair in PAIRS:
        vitro_cond = PAIR_CONDS[pair]["vitro"]
        vivo_cond  = PAIR_CONDS[pair]["vivo"]

        # For sequence-only models both condition files are identical;
        # load from whichever exists (try vitro first).
        def _load_cond(cond: str) -> np.ndarray | None:
            path = _act_path(model, pair, cond, layer)
            if path.exists():
                return torch.load(str(path), map_location="cpu",
                                  weights_only=True).numpy()
            return None

        vitro_all = _load_cond(vitro_cond)
        vivo_all  = _load_cond(vivo_cond)

        if vitro_all is None and vivo_all is None:
            log.warning(f"  Missing both conditions for {model}/{pair}/{layer}")
            continue

        # Fall back: if one file is missing, use the other for both
        if vitro_all is None:
            log.warning(f"  {vitro_cond} missing — using {vivo_cond} for both")
            vitro_all = vivo_all
        if vivo_all is None:
            log.warning(f"  {vivo_cond} missing — using {vitro_cond} for both")
            vivo_all = vitro_all

        # Get ATAC-accessibility indices for each side
        vitro_idx = get_condition_indices(pair, "vitro")
        vivo_idx  = get_condition_indices(pair, "vivo")

        if len(vitro_idx) == 0 or len(vivo_idx) == 0:
            log.warning(f"  No differential windows for {pair} — skipping.")
            continue

        if is_epibert:
            # EpiBERT: each condition file is cell-type specific
            result[(pair, "vitro")] = vitro_all[vitro_idx]
            result[(pair, "vivo")]  = vivo_all[vivo_idx]
        else:
            # Sequence-only: both files identical; index into different window sets
            base = vitro_all  # same as vivo_all
            result[(pair, "vitro")] = base[vitro_idx]
            result[(pair, "vivo")]  = base[vivo_idx]

        log.info(f"  {model}/{layer}/{pair}: vitro={len(vitro_idx)} vivo={len(vivo_idx)} windows")

    return result


def pool_all(acts: dict[tuple, np.ndarray]) -> np.ndarray:
    """Concatenate all condition activations → (N_total, hidden)."""
    arrays = [v for v in acts.values() if v is not None]
    if not arrays:
        return np.array([])
    return np.concatenate(arrays, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# SAE training (reuses existing sae.py)
# ─────────────────────────────────────────────────────────────────────────────

def train_sae_for_model_layer(
    model: str,
    layer: str,
    acts: dict[tuple, np.ndarray],
    n_steps: int = 50_000,
    sae_expansion: int = 4,  # 4x is standard in SAE literature; halves topk cost vs 8x
    k: int = 64,
    lr: float = 3e-4,
    batch_size: int = 4_096,
) -> BatchTopKSAE:
    """Train a pooled SAE on all conditions for (model, layer)."""
    all_acts = pool_all(acts)
    if len(all_acts) == 0:
        raise ValueError(f"No activations found for {model}/{layer}")

    hidden_dim = all_acts.shape[1]
    d_latent   = hidden_dim * sae_expansion

    sae_path = _sae_path(model, layer)
    if sae_path.exists():
        log.info(f"  SAE already trained: {sae_path} — loading.")
        return BatchTopKSAE.load(str(sae_path), device=DEVICE)

    log.info(f"  Training SAE: model={model} layer={layer} "
             f"hidden={hidden_dim} latent={d_latent} steps={n_steps}")

    sae = BatchTopKSAE(d_input=hidden_dim, expansion=sae_expansion, k=k).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    N = len(all_acts)

    # Pre-load the full training set onto GPU once to eliminate per-step
    # CPU numpy fancy-indexing and PCIe transfer (~6ms/step → ~0.05ms/step)
    acts_gpu = torch.from_numpy(all_acts).float().to(DEVICE)

    dead_resample_every = 2500

    for step in range(n_steps):
        idx   = torch.randint(0, N, (batch_size,), device=DEVICE)
        batch = acts_gpu[idx]
        optimizer.zero_grad()
        x_hat, z, _ = sae(batch)
        loss = torch.nn.functional.mse_loss(x_hat, batch)
        loss.backward()
        optimizer.step()
        sae._normalize_decoder()
        sae.update_dead_features(z.detach())

        if (step + 1) % dead_resample_every == 0:
            dead_mask = sae.get_dead_features(dead_resample_every)
            n_resampled = sae.resample_dead_features(batch, dead_mask)
            if n_resampled > 0:
                log.info(f"    step {step+1}: resampled {n_resampled} dead features")

        if (step + 1) % 5_000 == 0:
            log.info(f"    step {step+1}/{n_steps}  loss={loss.item():.4f}")

    sae.save(str(sae_path))
    log.info(f"  SAE saved → {sae_path}")
    return sae


# ─────────────────────────────────────────────────────────────────────────────
# CDS computation
# ─────────────────────────────────────────────────────────────────────────────

def encode_sae(sae: BatchTopKSAE, acts: np.ndarray, batch_size: int = 8_192) -> np.ndarray:
    """Encode activations through SAE → sparse latent codes (N, d_latent)."""
    N = len(acts)
    z = np.zeros((N, sae.d_latent), dtype=np.float32)
    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        x    = torch.from_numpy(acts[start:end]).float().to(DEVICE)
        with torch.no_grad():
            z[start:end] = sae.encode(x).cpu().numpy()
    return z


def compute_cds(
    z_vitro: np.ndarray,
    z_vivo:  np.ndarray,
    n_permutations: int = 1_000,
) -> pd.DataFrame:
    """
    Compute Context Divergence Score per latent feature.
    CDS_i = mean(z_vitro[:, i]) − mean(z_vivo[:, i])
    Significance: permutation t-test (Bonferroni-corrected).
    """
    n_features = z_vitro.shape[1]
    vitro_mean = z_vitro.mean(axis=0)
    vivo_mean  = z_vivo.mean(axis=0)
    cds        = vitro_mean - vivo_mean

    # t-test for each feature
    t_stats, p_vals = stats.ttest_ind(z_vitro, z_vivo, axis=0, equal_var=False)
    p_bonf = np.clip(p_vals * n_features, 0, 1)

    df = pd.DataFrame({
        "feature_id":    np.arange(n_features),
        "cds":           cds,
        "vitro_mean":    vitro_mean,
        "vivo_mean":     vivo_mean,
        "t_stat":        t_stats,
        "pval":          p_vals,
        "pval_bonf":     p_bonf,
        "significant":   p_bonf < 0.05,
    })
    return df.sort_values("pval")


def analyse_model_layer(
    model: str,
    layer: str,
    sae: BatchTopKSAE,
    acts: dict[tuple, np.ndarray],
) -> dict[str, pd.DataFrame]:
    """
    Compute CDS for each pair for (model, layer).
    Returns {pair: CDS DataFrame}.
    """
    results = {}
    for pair in PAIRS:
        if (pair, "vitro") not in acts or (pair, "vivo") not in acts:
            log.warning(f"  Missing activations for pair={pair} model={model}")
            continue
        z_vitro = encode_sae(sae, acts[(pair, "vitro")])
        z_vivo  = encode_sae(sae, acts[(pair, "vivo")])
        cds_df  = compute_cds(z_vitro, z_vivo)
        cds_df["pair"]  = pair
        cds_df["layer"] = layer
        cds_df["model"] = model
        results[pair]   = cds_df
        n_sig = cds_df["significant"].sum()
        n_tot = len(cds_df)
        log.info(f"    {model}/{layer}/{pair}: {n_sig}/{n_tot} significant "
                 f"({100*n_sig/n_tot:.2f}%)")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Cross-model summary
# ─────────────────────────────────────────────────────────────────────────────

def compute_cross_model_jaccard(
    sig_sets: dict[tuple[str, str, str], set],
) -> pd.DataFrame:
    """
    Compute Jaccard similarity between top-50 significant feature sets
    for each (model1, model2, layer, pair) combination.

    sig_sets: {(model, layer, pair): set of significant feature_ids}
    """
    keys   = list(sig_sets.keys())
    rows   = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            m1, l1, p1 = keys[i]
            m2, l2, p2 = keys[j]
            if l1 != l2 or p1 != p2:
                continue   # only compare same layer + pair across models
            s1 = sig_sets[keys[i]]
            s2 = sig_sets[keys[j]]
            if not s1 or not s2:
                continue
            jaccard = len(s1 & s2) / len(s1 | s2)
            rows.append({
                "model_1": m1, "model_2": m2,
                "layer": l1, "pair": p1,
                "jaccard": jaccard,
                "n_shared": len(s1 & s2),
                "n_union":  len(s1 | s2),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-model SAE training and contrastive analysis."
    )
    parser.add_argument(
        "--models", nargs="+",
        default=MODELS,
        help="Models to analyse (default: all four)",
    )
    parser.add_argument(
        "--layers", nargs="+",
        default=LAYERS,
        help="Layers to analyse (default: early mid late)",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick run: 200-step SAEs, 100 windows",
    )
    parser.add_argument(
        "--sae-steps", type=int, default=None,
        help="Override SAE training steps",
    )
    parser.add_argument(
        "--sae-expansion", type=int, default=4,
        help="SAE latent expansion factor (default 4; d_latent = expansion × hidden_dim)",
    )
    args = parser.parse_args()

    seed_everything()

    n_steps    = 200 if args.smoke_test else (args.sae_steps or int(cfg("sae", "steps")))
    expansion  = args.sae_expansion
    log.info(f"Models : {args.models}")
    log.info(f"Layers : {args.layers}")
    log.info(f"SAE steps: {n_steps}  expansion: {expansion}x")
    log.info(f"Device : {DEVICE}")

    rdir = _result_dir()

    summary_rows   = []
    sig_sets: dict[tuple, set] = {}

    for model in args.models:
        for layer in args.layers:
            log.info(f"\n── {model.upper()} / {layer} ───────────────────────────────")

            acts = load_activations(model, layer)
            if not acts:
                log.warning(f"  No activations for {model}/{layer} — skipping.")
                continue

            # Check which pairs have both vitro and vivo
            available_pairs = [
                p for p in PAIRS
                if (p, "vitro") in acts and (p, "vivo") in acts
            ]
            if not available_pairs:
                log.warning(f"  No complete pairs for {model}/{layer} — skipping.")
                continue

            try:
                sae = train_sae_for_model_layer(model, layer, acts, n_steps=n_steps,
                                                sae_expansion=expansion)
            except Exception as exc:
                log.error(f"  SAE training failed: {exc}")
                continue

            pair_results = analyse_model_layer(model, layer, sae, acts)

            for pair, cds_df in pair_results.items():
                # Save per-model CDS table
                out_tsv = rdir / f"{model}_layer_{layer}_{pair}_cds.tsv"
                cds_df.to_csv(out_tsv, sep="\t", index=False)

                n_sig    = int(cds_df["significant"].sum())
                pct_sig  = 100 * n_sig / len(cds_df)
                max_cds  = float(cds_df["cds"].abs().max())

                summary_rows.append({
                    "model":         model,
                    "layer":         layer,
                    "pair":          pair,
                    "n_significant": n_sig,
                    "pct_significant": round(pct_sig, 3),
                    "max_abs_cds":   round(max_cds, 4),
                    "d_latent":      sae.d_latent,
                })

                # Store top-50 significant feature ids for Jaccard
                top50 = set(
                    cds_df[cds_df["significant"]]
                    .nlargest(50, "cds")["feature_id"]
                    .tolist()
                )
                sig_sets[(model, layer, pair)] = top50

    # ── Summary table ─────────────────────────────────────────────────────────
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = rdir / "cross_model_summary.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        log.info(f"\nSummary table → {summary_path}")
        log.info("\n" + summary_df.to_string(index=False))

    # ── Cross-model Jaccard ────────────────────────────────────────────────────
    if sig_sets:
        jaccard_df   = compute_cross_model_jaccard(sig_sets)
        jaccard_path = rdir / "cross_model_jaccard.tsv"
        jaccard_df.to_csv(jaccard_path, sep="\t", index=False)
        log.info(f"\nJaccard table → {jaccard_path}")
        if not jaccard_df.empty:
            log.info("\n" + jaccard_df.to_string(index=False))

    log.info("\nDone.")


if __name__ == "__main__":
    main()
