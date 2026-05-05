#!/usr/bin/env python3
"""
src/homer_sae_enrichment.py
============================
Run HOMER findMotifsGenome.pl on the top-K windows per significant SAE feature,
comparing each model's context-divergent windows against a matched background.

For each (model, layer, pair) combination with ≥1 significant feature:
  1. Take the top-100 CDS-score-weighted windows (from window_space_jaccard.py BED files)
  2. Use the full set of differential windows as background
  3. Run HOMER findMotifsGenome.pl
  4. Parse knownResults.txt → extract top-10 enriched TF motifs
  5. Aggregate into a cross-model motif comparison table

Outputs
-------
results/homer_sae/
    {model}_{layer}_{pair}/      HOMER output directory per combo
results/cross_model/sae_motif_summary.tsv
    Top motifs per model/layer/pair with enrichment statistics
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))
from utils import get_logger

log = get_logger("homer_sae")

HOMER_BIN    = Path(os.environ.get("HOMER_BIN", "/opt/homer/bin"))
GENOME       = str(REPO / "data" / "genome" / "hg38.fa")
GENOME_LABEL = "hg38"

PAIRS  = ["blood", "liver", "lymph"]
LAYERS = ["early", "mid", "late"]

PAIR_CONDS = {
    "blood": {"vitro": "K562",    "vivo": "HSC"},
    "liver": {"vitro": "HepG2",   "vivo": "Liver"},
    "lymph": {"vitro": "GM12878", "vivo": "NaiveB"},
}


def run_homer(target_bed: Path, background_bed: Path, out_dir: Path,
              genome: str = GENOME_LABEL, size: int = 200,
              p: int = 4) -> bool:
    """Run findMotifsGenome.pl; return True on success."""
    out_dir.mkdir(parents=True, exist_ok=True)
    homer_exe = HOMER_BIN / "findMotifsGenome.pl"
    cmd = [
        str(homer_exe),
        str(target_bed),
        genome,
        str(out_dir),
        "-size", str(size),
        "-p", str(p),
        "-nomotif",          # skip de novo for speed; known motifs only
        "-bg", str(background_bed),
    ]
    log.info(f"  Running HOMER: {target_bed.name} vs background ({size}bp windows)")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        log.error(f"  HOMER failed:\n{result.stderr[-500:]}")
        return False
    return True


def parse_known_results(homer_dir: Path, top_n: int = 10) -> pd.DataFrame:
    """Parse knownResults.txt → DataFrame with top_n motifs."""
    p = homer_dir / "knownResults.txt"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, sep="\t", comment="#")
    df.columns = [c.strip() for c in df.columns]
    # Standard HOMER columns
    rename = {
        "Motif Name": "motif",
        "P-value": "pval",
        "Log P-value": "log_pval",
        "q-value (Benjamini)": "qval",
        "% of Target Sequences with Motif": "pct_target",
        "% of Background Sequences with Motif": "pct_bg",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    # Clean motif name: keep just the TF name
    if "motif" in df.columns:
        df["motif"] = df["motif"].astype(str)
        df["motif_short"] = df["motif"].str.split("(").str[0].str.strip()
    df["log_pval"] = pd.to_numeric(df.get("log_pval", pd.Series(dtype=float)),
                                   errors="coerce")
    df = df.sort_values("log_pval").head(top_n)
    return df


def make_background_bed(pair: str, out_path: Path) -> Path:
    """
    Background: all windows that are differential for this pair
    (vitro-only ∪ vivo-only).
    """
    labels  = pd.read_csv(REPO / "data" / "window_condition_labels.tsv", sep="\t")
    windows = pd.read_csv(REPO / "data" / "windows.bed", sep="\t", header=None,
                          names=["chrom", "start", "end"])
    vitro_c = PAIR_CONDS[pair]["vitro"]
    vivo_c  = PAIR_CONDS[pair]["vivo"]
    mask = (labels[vitro_c].astype(bool) & ~labels[vivo_c].astype(bool)) | \
           (~labels[vitro_c].astype(bool) & labels[vivo_c].astype(bool))
    idx = labels.index[mask].to_numpy()
    windows.iloc[idx].to_csv(out_path, sep="\t", header=False, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["enformer", "hyenadna", "nucleotide_transformer"])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--homer-bin", default=str(HOMER_BIN))
    parser.add_argument("--genome", default=GENOME_LABEL)
    args = parser.parse_args()

    homer_bin = Path(args.homer_bin)
    bed_dir   = REPO / "results" / "cross_model" / "top_windows_bed"
    homer_out = REPO / "results" / "homer_sae"
    homer_out.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    # Add HOMER to PATH
    os.environ["PATH"] = str(homer_bin) + ":" + os.environ.get("PATH", "")

    for model in args.models:
        for layer in LAYERS:
            for pair in PAIRS:
                target_bed = bed_dir / f"top{args.top_k}_{model}_{layer}_{pair}.bed"
                if not target_bed.exists():
                    log.info(f"  No BED for {model}/{layer}/{pair} — skipping.")
                    continue

                # Background: all differential windows for this pair
                bg_bed = homer_out / f"bg_{pair}.bed"
                if not bg_bed.exists():
                    make_background_bed(pair, bg_bed)

                out_dir = homer_out / f"{model}_{layer}_{pair}"
                success = run_homer(target_bed, bg_bed, out_dir,
                                    genome=args.genome)
                if not success:
                    continue

                motifs = parse_known_results(out_dir, top_n=5)
                for _, row in motifs.iterrows():
                    summary_rows.append({
                        "model":       model,
                        "layer":       layer,
                        "pair":        pair,
                        "motif":       row.get("motif_short", ""),
                        "log_pval":    row.get("log_pval", np.nan),
                        "qval":        row.get("qval", np.nan),
                        "pct_target":  row.get("pct_target", ""),
                        "pct_bg":      row.get("pct_bg", ""),
                    })

    if summary_rows:
        out_tsv = REPO / "results" / "cross_model" / "sae_motif_summary.tsv"
        df = pd.DataFrame(summary_rows)
        df.to_csv(out_tsv, sep="\t", index=False)
        log.info(f"\nMotif summary → {out_tsv}")
        log.info("\n" + df.to_string(index=False))
    else:
        log.warning("No HOMER results produced.")

    log.info("Done.")


if __name__ == "__main__":
    main()
