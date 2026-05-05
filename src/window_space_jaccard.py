#!/usr/bin/env python3
"""
src/window_space_jaccard.py
============================
Corrected cross-model comparison in WINDOW SPACE.

For each (model, layer, pair), compute a CDS-weighted activation score
for every genomic window:

    score(w) = Σ_{f ∈ sig_features} |CDS_f| × z_{w,f}

where z_{w,f} is the SAE latent activation of feature f at window w.

Windows with high scores are those most responsible for the observed
context divergence.  We then compute Jaccard similarity between models
by comparing their top-K scoring windows — a well-defined comparison
because window indices are shared across all models.

Outputs
-------
results/cross_model/window_jaccard.tsv
    pairwise Jaccard of top-K windows per (model1, model2, layer, pair)

results/cross_model/top_windows_{model}_{layer}_{pair}.bed
    BED files of top-100 windows per model/layer/pair (for HOMER input)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

from utils import get_logger
from sae import BatchTopKSAE

torch.set_float32_matmul_precision("high")

log = get_logger("window_jaccard")

MODELS  = ["enformer", "hyenadna", "nucleotide_transformer"]
LAYERS  = ["early", "mid", "late"]
PAIRS   = ["blood", "liver", "lymph"]
PAIR_CONDS = {
    "blood": {"vitro": "K562",    "vivo": "HSC"},
    "liver": {"vitro": "HepG2",   "vivo": "Liver"},
    "lymph": {"vitro": "GM12878", "vivo": "NaiveB"},
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _act_path(model: str, pair: str, cond: str, layer: str) -> Path:
    return REPO / "activations" / model / pair / cond / f"{layer}.pt"


def _sae_path(model: str, layer: str) -> Path:
    return REPO / "saes" / model / layer / "pooled.pt"


def _cds_path(model: str, layer: str, pair: str) -> Path:
    return REPO / "results" / "cross_model" / f"{model}_layer_{layer}_{pair}_cds.tsv"


def load_condition_indices(pair: str) -> tuple[np.ndarray, np.ndarray]:
    labels = pd.read_csv(REPO / "data" / "window_condition_labels.tsv", sep="\t")
    vitro_cond = PAIR_CONDS[pair]["vitro"]
    vivo_cond  = PAIR_CONDS[pair]["vivo"]
    vitro_idx = labels.index[labels[vitro_cond].astype(bool) & ~labels[vivo_cond].astype(bool)].to_numpy()
    vivo_idx  = labels.index[~labels[vitro_cond].astype(bool) & labels[vivo_cond].astype(bool)].to_numpy()
    return vitro_idx, vivo_idx


def encode_all(sae: BatchTopKSAE, acts: np.ndarray, batch: int = 2048) -> np.ndarray:
    """Encode full activation matrix → latent codes (N, d_latent)."""
    N = len(acts)
    z = np.zeros((N, sae.d_latent), dtype=np.float32)
    for s in range(0, N, batch):
        x = torch.from_numpy(acts[s:s+batch]).float().to(DEVICE)
        with torch.no_grad():
            z[s:s+batch] = sae.encode(x).cpu().numpy()
    return z


def cds_window_scores(
    z_all: np.ndarray,
    sig_features: list[int],
    cds_weights: np.ndarray,
    window_indices: np.ndarray,
) -> np.ndarray:
    """
    For each window in window_indices, compute:
        score(w) = Σ_f |CDS_f| × z_{w,f}
    Returns scores aligned to window_indices.
    """
    if len(sig_features) == 0:
        return np.zeros(len(window_indices))
    z_sub  = z_all[np.ix_(window_indices, sig_features)]   # (N_sub, n_sig)
    scores = z_sub @ np.abs(cds_weights)                    # (N_sub,)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=100,
                        help="Top-K windows per model/layer/pair for Jaccard and BED output")
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    windows   = pd.read_csv(REPO / "data" / "windows.bed", sep="\t", header=None,
                            names=["chrom", "start", "end"])
    rdir      = REPO / "results" / "cross_model"
    rdir.mkdir(parents=True, exist_ok=True)
    bed_dir   = rdir / "top_windows_bed"
    bed_dir.mkdir(exist_ok=True)

    K = args.top_k

    # ── Compute window scores and save BED files ──────────────────────────────
    window_sets: dict[tuple, set[int]] = {}   # (model, layer, pair) → set of window indices

    for model in args.models:
        for layer in LAYERS:
            sae_p = _sae_path(model, layer)
            if not sae_p.exists():
                log.warning(f"No SAE for {model}/{layer} — skipping.")
                continue
            sae = BatchTopKSAE.load(str(sae_p), device=DEVICE)

            for pair in PAIRS:
                cds_p = _cds_path(model, layer, pair)
                if not cds_p.exists():
                    continue
                cds_df = pd.read_csv(cds_p, sep="\t")
                sig    = cds_df[cds_df["significant"]]
                if sig.empty:
                    log.info(f"  {model}/{layer}/{pair}: 0 significant — skipping.")
                    continue

                sig_feats   = sig["feature_id"].to_numpy().astype(int)
                cds_weights = sig["cds"].to_numpy().astype(np.float32)

                # Load activations (use vitro condition file; for seq-only models
                # it equals the vivo file, but indices differ)
                vitro_cond = PAIR_CONDS[pair]["vitro"]
                act_p = _act_path(model, pair, vitro_cond, layer)
                if not act_p.exists():
                    log.warning(f"  Missing activations: {act_p}")
                    continue

                acts = torch.load(str(act_p), map_location="cpu",
                                  weights_only=True).numpy()
                z_all = encode_all(sae, acts)

                # Compute CDS scores over ALL windows (not just the differential subset)
                vitro_idx, vivo_idx = load_condition_indices(pair)

                # Score vitro-specific windows
                scores_vitro = cds_window_scores(z_all, list(sig_feats),
                                                 cds_weights, vitro_idx)
                top_vitro    = vitro_idx[np.argsort(scores_vitro)[::-1][:K]]

                # Score vivo-specific windows (load vivo act file)
                vivo_cond = PAIR_CONDS[pair]["vivo"]
                act_vivo  = _act_path(model, pair, vivo_cond, layer)
                if act_vivo.exists():
                    acts_vivo = torch.load(str(act_vivo), map_location="cpu",
                                           weights_only=True).numpy()
                    z_vivo = encode_all(sae, acts_vivo)
                else:
                    z_vivo = z_all   # fallback for seq-only

                scores_vivo = cds_window_scores(z_vivo, list(sig_feats),
                                                -cds_weights,   # vivo-enriched = negative CDS
                                                vivo_idx)
                top_vivo = vivo_idx[np.argsort(scores_vivo)[::-1][:K]]

                combined  = np.union1d(top_vitro, top_vivo)
                window_sets[(model, layer, pair)] = set(combined.tolist())

                # Write BED file
                bed_path = bed_dir / f"top{K}_{model}_{layer}_{pair}.bed"
                windows.iloc[combined].to_csv(
                    bed_path, sep="\t", header=False, index=False)

                n_sig = len(sig_feats)
                log.info(f"  {model}/{layer}/{pair}: {n_sig} sig features → "
                         f"{len(combined)} top windows → {bed_path.name}")

    # ── Pairwise window-space Jaccard ─────────────────────────────────────────
    rows = []
    keys = list(window_sets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            m1, l1, p1 = keys[i]
            m2, l2, p2 = keys[j]
            if l1 != l2 or p1 != p2:
                continue
            s1 = window_sets[keys[i]]
            s2 = window_sets[keys[j]]
            if not s1 or not s2:
                continue
            inter   = len(s1 & s2)
            union   = len(s1 | s2)
            jaccard = inter / union if union else 0.0
            rows.append({
                "model_1": m1, "model_2": m2,
                "layer": l1, "pair": p1,
                "jaccard": round(jaccard, 4),
                "n_shared": inter,
                "n_union":  union,
                "n_m1": len(s1),
                "n_m2": len(s2),
            })
            log.info(f"  Jaccard {m1} vs {m2} / {l1}/{p1}: "
                     f"{jaccard:.3f}  ({inter}/{union})")

    if rows:
        jdf = pd.DataFrame(rows)
        out = rdir / "window_jaccard.tsv"
        jdf.to_csv(out, sep="\t", index=False)
        log.info(f"\nWindow-space Jaccard → {out}")
        log.info("\n" + jdf.to_string(index=False))
    else:
        log.warning("No overlapping (layer, pair) combos with significant features.")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
