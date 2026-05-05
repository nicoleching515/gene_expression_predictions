#!/usr/bin/env python3
"""
src/assign_condition_windows.py
================================
For each of the 9,004 genome-wide windows, determine which ATAC conditions
the window overlaps (using peak BED files from ENCODE).

Output:
    data/window_condition_labels.tsv
        columns: idx | chrom | start | end | K562 | HepG2 | GM12878 | HSC | Liver | NaiveB
        (bool columns: True if window overlaps ≥1 peak in that condition)

Usage:
    python src/assign_condition_windows.py [--min-overlap 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent

PEAK_FILES = {
    "K562":    REPO / "outputs/annotation/homer_encode/peaks/K562.bed",
    "HepG2":   REPO / "outputs/annotation/homer_encode/peaks/HepG2.bed",
    "GM12878": REPO / "outputs/annotation/homer_encode/peaks/GM12878.bed",
    "HSC":     REPO / "outputs/annotation/homer_encode/peaks/HSC.bed",
    "Liver":   REPO / "outputs/annotation/homer_encode/peaks/Liver.bed",
    "NaiveB":  REPO / "outputs/annotation/homer_encode/peaks/NaiveB.bed",
}

WINDOWS_BED = REPO / "data/windows.bed"
OUT_PATH    = REPO / "data/window_condition_labels.tsv"


def load_bed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, comment="#",
                     usecols=[0, 1, 2], names=["chrom", "start", "end"])
    return df


def overlap_flag(windows: pd.DataFrame, peaks: pd.DataFrame) -> np.ndarray:
    """
    Return boolean array of length len(windows): True if window overlaps any peak.
    Pure-pandas interval join — no bedtools required.
    """
    flags = np.zeros(len(windows), dtype=bool)
    for chrom, peak_grp in peaks.groupby("chrom"):
        win_mask = windows["chrom"] == chrom
        if not win_mask.any():
            continue
        win_idx   = np.where(win_mask)[0]
        w_starts  = windows.loc[win_mask, "start"].values
        w_ends    = windows.loc[win_mask, "end"].values
        p_starts  = peak_grp["start"].values
        p_ends    = peak_grp["end"].values

        for i, (ws, we) in enumerate(zip(w_starts, w_ends)):
            if np.any((p_starts < we) & (p_ends > ws)):
                flags[win_idx[i]] = True
    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default=str(WINDOWS_BED))
    parser.add_argument("--out",     default=str(OUT_PATH))
    args = parser.parse_args()

    windows = load_bed(Path(args.windows))
    windows.insert(0, "idx", np.arange(len(windows)))
    print(f"Loaded {len(windows):,} windows")

    for cond, peak_path in PEAK_FILES.items():
        if not peak_path.exists():
            print(f"  WARNING: {peak_path} not found — marking all False")
            windows[cond] = False
            continue
        peaks = load_bed(peak_path)
        flags = overlap_flag(windows, peaks)
        windows[cond] = flags
        print(f"  {cond}: {flags.sum():,} / {len(flags):,} windows overlap peaks "
              f"({100*flags.mean():.1f}%)")

    windows.to_csv(args.out, sep="\t", index=False)
    print(f"\nSaved → {args.out}")

    # Print a summary of pairwise vitro/vivo differential windows
    PAIRS = [
        ("K562",    "HSC",    "blood"),
        ("HepG2",   "Liver",  "liver"),
        ("GM12878", "NaiveB", "lymph"),
    ]
    print("\nDifferential window counts (pair):")
    for vitro, vivo, pair in PAIRS:
        vitro_only = (windows[vitro] & ~windows[vivo]).sum()
        vivo_only  = (~windows[vitro] & windows[vivo]).sum()
        both       = (windows[vitro] &  windows[vivo]).sum()
        neither    = (~windows[vitro] & ~windows[vivo]).sum()
        print(f"  {pair:6s}  vitro-only={vitro_only:4d}  vivo-only={vivo_only:4d}  "
              f"shared={both:4d}  neither={neither:4d}")


if __name__ == "__main__":
    main()
