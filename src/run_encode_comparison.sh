#!/usr/bin/env bash
# ENCODE positive-control HOMER analysis
# Download narrowPeak files for K562/HepG2/GM12878, filter to chr8/chr9,
# run HOMER with same settings, compare enrichment to SAE feature peaks.
#
# Requires: HOMER, bedtools, wget
# Run from repo root: bash src/run_encode_comparison.sh

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BG_BED="$REPO/data/windows.bed"
GENOME="hg38"
OUT="$REPO/outputs/annotation/encode_comparison"
mkdir -p "$OUT"

# ── K562 ─────────────────────────────────────
# RUNX1 ChIP-seq K562
BED="$OUT/K562_RUNX1.bed"
if [[ ! -f "$BED" ]]; then
  echo "Downloading ENCFF002CEL (RUNX1 ChIP-seq K562)..."
  wget -q "https://www.encodeproject.org/files/ENCFF002CEL/@@download/ENCFF002CEL.bed.gz" \
       -O "$BED.gz"
  gunzip "$BED.gz"
fi
# Filter to chr8/chr9 to match SAE window universe
grep -E "^chr[89]\b" "$BED" > "$OUT/K562_RUNX1_chr89.bed" || true
N=$(wc -l < "$OUT/K562_RUNX1_chr89.bed")
echo "  K562_RUNX1: $N peaks on chr8/chr9"
if [[ $N -gt 10 && ! -d "$OUT/K562_RUNX1_homer/homerResults" ]]; then
  mkdir -p "$OUT/K562_RUNX1_homer"
  findMotifsGenome.pl "$OUT/K562_RUNX1_chr89.bed" "$GENOME" \
    "$OUT/K562_RUNX1_homer" -size 200 -mask \
    -bg "$BG_BED" -p 8 2>"$OUT/K562_RUNX1_homer/homer.log"
  echo "  HOMER done: K562_RUNX1"
fi

# SPI1/PU.1 ChIP-seq K562
BED="$OUT/K562_SPI1.bed"
if [[ ! -f "$BED" ]]; then
  echo "Downloading ENCFF496HZP (SPI1/PU.1 ChIP-seq K562)..."
  wget -q "https://www.encodeproject.org/files/ENCFF496HZP/@@download/ENCFF496HZP.bed.gz" \
       -O "$BED.gz"
  gunzip "$BED.gz"
fi
# Filter to chr8/chr9 to match SAE window universe
grep -E "^chr[89]\b" "$BED" > "$OUT/K562_SPI1_chr89.bed" || true
N=$(wc -l < "$OUT/K562_SPI1_chr89.bed")
echo "  K562_SPI1: $N peaks on chr8/chr9"
if [[ $N -gt 10 && ! -d "$OUT/K562_SPI1_homer/homerResults" ]]; then
  mkdir -p "$OUT/K562_SPI1_homer"
  findMotifsGenome.pl "$OUT/K562_SPI1_chr89.bed" "$GENOME" \
    "$OUT/K562_SPI1_homer" -size 200 -mask \
    -bg "$BG_BED" -p 8 2>"$OUT/K562_SPI1_homer/homer.log"
  echo "  HOMER done: K562_SPI1"
fi

# H3K27ac K562
BED="$OUT/K562_H3K27ac.bed"
if [[ ! -f "$BED" ]]; then
  echo "Downloading ENCFF828IEW (H3K27ac K562)..."
  wget -q "https://www.encodeproject.org/files/ENCFF828IEW/@@download/ENCFF828IEW.bed.gz" \
       -O "$BED.gz"
  gunzip "$BED.gz"
fi
# Filter to chr8/chr9 to match SAE window universe
grep -E "^chr[89]\b" "$BED" > "$OUT/K562_H3K27ac_chr89.bed" || true
N=$(wc -l < "$OUT/K562_H3K27ac_chr89.bed")
echo "  K562_H3K27ac: $N peaks on chr8/chr9"
if [[ $N -gt 10 && ! -d "$OUT/K562_H3K27ac_homer/homerResults" ]]; then
  mkdir -p "$OUT/K562_H3K27ac_homer"
  findMotifsGenome.pl "$OUT/K562_H3K27ac_chr89.bed" "$GENOME" \
    "$OUT/K562_H3K27ac_homer" -size 200 -mask \
    -bg "$BG_BED" -p 8 2>"$OUT/K562_H3K27ac_homer/homer.log"
  echo "  HOMER done: K562_H3K27ac"
fi

# ── HepG2 ─────────────────────────────────────
# FOXA1 ChIP-seq HepG2
BED="$OUT/HepG2_FOXA1.bed"
if [[ ! -f "$BED" ]]; then
  echo "Downloading ENCFF613PYA (FOXA1 ChIP-seq HepG2)..."
  wget -q "https://www.encodeproject.org/files/ENCFF613PYA/@@download/ENCFF613PYA.bed.gz" \
       -O "$BED.gz"
  gunzip "$BED.gz"
fi
# Filter to chr8/chr9 to match SAE window universe
grep -E "^chr[89]\b" "$BED" > "$OUT/HepG2_FOXA1_chr89.bed" || true
N=$(wc -l < "$OUT/HepG2_FOXA1_chr89.bed")
echo "  HepG2_FOXA1: $N peaks on chr8/chr9"
if [[ $N -gt 10 && ! -d "$OUT/HepG2_FOXA1_homer/homerResults" ]]; then
  mkdir -p "$OUT/HepG2_FOXA1_homer"
  findMotifsGenome.pl "$OUT/HepG2_FOXA1_chr89.bed" "$GENOME" \
    "$OUT/HepG2_FOXA1_homer" -size 200 -mask \
    -bg "$BG_BED" -p 8 2>"$OUT/HepG2_FOXA1_homer/homer.log"
  echo "  HOMER done: HepG2_FOXA1"
fi

# HNF4A ChIP-seq HepG2
BED="$OUT/HepG2_HNF4A.bed"
if [[ ! -f "$BED" ]]; then
  echo "Downloading ENCFF114YUS (HNF4A ChIP-seq HepG2)..."
  wget -q "https://www.encodeproject.org/files/ENCFF114YUS/@@download/ENCFF114YUS.bed.gz" \
       -O "$BED.gz"
  gunzip "$BED.gz"
fi
# Filter to chr8/chr9 to match SAE window universe
grep -E "^chr[89]\b" "$BED" > "$OUT/HepG2_HNF4A_chr89.bed" || true
N=$(wc -l < "$OUT/HepG2_HNF4A_chr89.bed")
echo "  HepG2_HNF4A: $N peaks on chr8/chr9"
if [[ $N -gt 10 && ! -d "$OUT/HepG2_HNF4A_homer/homerResults" ]]; then
  mkdir -p "$OUT/HepG2_HNF4A_homer"
  findMotifsGenome.pl "$OUT/HepG2_HNF4A_chr89.bed" "$GENOME" \
    "$OUT/HepG2_HNF4A_homer" -size 200 -mask \
    -bg "$BG_BED" -p 8 2>"$OUT/HepG2_HNF4A_homer/homer.log"
  echo "  HOMER done: HepG2_HNF4A"
fi

# H3K27ac HepG2
BED="$OUT/HepG2_H3K27ac.bed"
if [[ ! -f "$BED" ]]; then
  echo "Downloading ENCFF617QIP (H3K27ac HepG2)..."
  wget -q "https://www.encodeproject.org/files/ENCFF617QIP/@@download/ENCFF617QIP.bed.gz" \
       -O "$BED.gz"
  gunzip "$BED.gz"
fi
# Filter to chr8/chr9 to match SAE window universe
grep -E "^chr[89]\b" "$BED" > "$OUT/HepG2_H3K27ac_chr89.bed" || true
N=$(wc -l < "$OUT/HepG2_H3K27ac_chr89.bed")
echo "  HepG2_H3K27ac: $N peaks on chr8/chr9"
if [[ $N -gt 10 && ! -d "$OUT/HepG2_H3K27ac_homer/homerResults" ]]; then
  mkdir -p "$OUT/HepG2_H3K27ac_homer"
  findMotifsGenome.pl "$OUT/HepG2_H3K27ac_chr89.bed" "$GENOME" \
    "$OUT/HepG2_H3K27ac_homer" -size 200 -mask \
    -bg "$BG_BED" -p 8 2>"$OUT/HepG2_H3K27ac_homer/homer.log"
  echo "  HOMER done: HepG2_H3K27ac"
fi

# ── GM12878 ─────────────────────────────────────
# EBF1 ChIP-seq GM12878
BED="$OUT/GM12878_EBF1.bed"
if [[ ! -f "$BED" ]]; then
  echo "Downloading ENCFF001VCU (EBF1 ChIP-seq GM12878)..."
  wget -q "https://www.encodeproject.org/files/ENCFF001VCU/@@download/ENCFF001VCU.bed.gz" \
       -O "$BED.gz"
  gunzip "$BED.gz"
fi
# Filter to chr8/chr9 to match SAE window universe
grep -E "^chr[89]\b" "$BED" > "$OUT/GM12878_EBF1_chr89.bed" || true
N=$(wc -l < "$OUT/GM12878_EBF1_chr89.bed")
echo "  GM12878_EBF1: $N peaks on chr8/chr9"
if [[ $N -gt 10 && ! -d "$OUT/GM12878_EBF1_homer/homerResults" ]]; then
  mkdir -p "$OUT/GM12878_EBF1_homer"
  findMotifsGenome.pl "$OUT/GM12878_EBF1_chr89.bed" "$GENOME" \
    "$OUT/GM12878_EBF1_homer" -size 200 -mask \
    -bg "$BG_BED" -p 8 2>"$OUT/GM12878_EBF1_homer/homer.log"
  echo "  HOMER done: GM12878_EBF1"
fi

# CTCF ChIP-seq GM12878
BED="$OUT/GM12878_CTCF.bed"
if [[ ! -f "$BED" ]]; then
  echo "Downloading ENCFF002DAL (CTCF ChIP-seq GM12878)..."
  wget -q "https://www.encodeproject.org/files/ENCFF002DAL/@@download/ENCFF002DAL.bed.gz" \
       -O "$BED.gz"
  gunzip "$BED.gz"
fi
# Filter to chr8/chr9 to match SAE window universe
grep -E "^chr[89]\b" "$BED" > "$OUT/GM12878_CTCF_chr89.bed" || true
N=$(wc -l < "$OUT/GM12878_CTCF_chr89.bed")
echo "  GM12878_CTCF: $N peaks on chr8/chr9"
if [[ $N -gt 10 && ! -d "$OUT/GM12878_CTCF_homer/homerResults" ]]; then
  mkdir -p "$OUT/GM12878_CTCF_homer"
  findMotifsGenome.pl "$OUT/GM12878_CTCF_chr89.bed" "$GENOME" \
    "$OUT/GM12878_CTCF_homer" -size 200 -mask \
    -bg "$BG_BED" -p 8 2>"$OUT/GM12878_CTCF_homer/homer.log"
  echo "  HOMER done: GM12878_CTCF"
fi

# H3K27ac GM12878
BED="$OUT/GM12878_H3K27ac.bed"
if [[ ! -f "$BED" ]]; then
  echo "Downloading ENCFF796WRU (H3K27ac GM12878)..."
  wget -q "https://www.encodeproject.org/files/ENCFF796WRU/@@download/ENCFF796WRU.bed.gz" \
       -O "$BED.gz"
  gunzip "$BED.gz"
fi
# Filter to chr8/chr9 to match SAE window universe
grep -E "^chr[89]\b" "$BED" > "$OUT/GM12878_H3K27ac_chr89.bed" || true
N=$(wc -l < "$OUT/GM12878_H3K27ac_chr89.bed")
echo "  GM12878_H3K27ac: $N peaks on chr8/chr9"
if [[ $N -gt 10 && ! -d "$OUT/GM12878_H3K27ac_homer/homerResults" ]]; then
  mkdir -p "$OUT/GM12878_H3K27ac_homer"
  findMotifsGenome.pl "$OUT/GM12878_H3K27ac_chr89.bed" "$GENOME" \
    "$OUT/GM12878_H3K27ac_homer" -size 200 -mask \
    -bg "$BG_BED" -p 8 2>"$OUT/GM12878_H3K27ac_homer/homer.log"
  echo "  HOMER done: GM12878_H3K27ac"
fi

echo 'ENCODE comparison complete.'
echo 'Results in: $OUT'
echo 'Compare knownResults.txt p-values vs SAE feature peaks'
echo 'to establish upper bound on expected motif enrichment.'
