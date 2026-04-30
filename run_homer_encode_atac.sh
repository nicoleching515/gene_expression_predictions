#!/usr/bin/env bash
# =============================================================================
# run_homer_encode_atac.sh
#
# Option B genome-wide HOMER: uses publicly available ENCODE ATAC-seq IDR peak
# BED files (hg38) as target regions instead of SAE-derived chr8/chr9 windows.
#
# For each cell-type pair, computes condition-specific peaks (bedtools subtract)
# and runs HOMER findMotifsGenome.pl with -genomeBg.
#
# Cell types & ENCODE files:
#   K562    (blood vitro)  ENCFF738NOA  ENCSR956DNB
#   GM12878 (lymph vitro)  ENCFF917REN  ENCSR095QNB
#   HepG2   (liver vitro)  ENCFF791RKW  ENCSR042AWH
#   Liver   (liver vivo)   ENCFF654SUU  ENCSR124NNL
#   NaiveB  (lymph vivo)   ENCFF380TOL  ENCSR685OFR (female, 39y)
#                          ENCFF590QLY  ENCSR903WVU (male,   40y) -- merged
#   HSC     (blood vivo)   NOT available on ENCODE -- K562 run vs genome only
#
# Usage:
#   cd /workspace/gene_expression_predictions
#   bash run_homer_encode_atac.sh
# =============================================================================

set -euo pipefail
export PATH="/workspace/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$REPO_ROOT/outputs/annotation/homer_encode"
GENOME="hg38"

mkdir -p "$OUT_DIR/peaks" "$OUT_DIR/homer"

ENCODE_BASE="https://www.encodeproject.org/files"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Download ENCODE ATAC-seq IDR peak files
# ─────────────────────────────────────────────────────────────────────────────
echo ">>> STEP 1: Downloading ENCODE ATAC-seq IDR peak files..."

declare -A FILES=(
  ["K562"]="ENCFF738NOA"
  ["GM12878"]="ENCFF917REN"
  ["HepG2"]="ENCFF791RKW"
  ["Liver"]="ENCFF654SUU"
  ["NaiveB_F"]="ENCFF380TOL"
  ["NaiveB_M"]="ENCFF590QLY"
)

for cell in "${!FILES[@]}"; do
  acc="${FILES[$cell]}"
  out="$OUT_DIR/peaks/${cell}.narrowPeak.gz"
  if [[ ! -f "$out" ]]; then
    echo "  Downloading $cell ($acc)..."
    wget -q "$ENCODE_BASE/$acc/@@download/$acc.bed.gz" -O "$out"
  else
    echo "  Skipping $cell (already downloaded)"
  fi
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Decompress and merge NaiveB donors; prepare BED files
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 2: Preparing BED files..."

for cell in K562 GM12878 HepG2 Liver; do
  bed="$OUT_DIR/peaks/${cell}.bed"
  if [[ ! -f "$bed" ]]; then
    zcat "$OUT_DIR/peaks/${cell}.narrowPeak.gz" \
      | cut -f1-3 \
      | sort -k1,1 -k2,2n \
      > "$bed"
    echo "  $cell: $(wc -l < "$bed") peaks"
  fi
done

# Merge two NaiveB donors
naiveb_bed="$OUT_DIR/peaks/NaiveB.bed"
if [[ ! -f "$naiveb_bed" ]]; then
  zcat "$OUT_DIR/peaks/NaiveB_F.narrowPeak.gz" \
       "$OUT_DIR/peaks/NaiveB_M.narrowPeak.gz" \
    | cut -f1-3 \
    | sort -k1,1 -k2,2n \
    | bedtools merge -i stdin \
    > "$naiveb_bed"
  echo "  NaiveB (merged 2 donors): $(wc -l < "$naiveb_bed") peaks"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Compute condition-specific peaks for each pair
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 3: Computing condition-specific peaks..."

# Liver pair: HepG2-specific vs Liver-specific
hepg2_specific="$OUT_DIR/peaks/HepG2_specific.bed"
liver_specific="$OUT_DIR/peaks/Liver_specific.bed"
if [[ ! -f "$hepg2_specific" ]]; then
  bedtools subtract -a "$OUT_DIR/peaks/HepG2.bed" \
                    -b "$OUT_DIR/peaks/Liver.bed" \
    > "$hepg2_specific"
  echo "  HepG2-specific: $(wc -l < "$hepg2_specific") peaks"
fi
if [[ ! -f "$liver_specific" ]]; then
  bedtools subtract -a "$OUT_DIR/peaks/Liver.bed" \
                    -b "$OUT_DIR/peaks/HepG2.bed" \
    > "$liver_specific"
  echo "  Liver-specific: $(wc -l < "$liver_specific") peaks"
fi

# Lymph pair: GM12878-specific vs NaiveB-specific
gm_specific="$OUT_DIR/peaks/GM12878_specific.bed"
naiveb_specific="$OUT_DIR/peaks/NaiveB_specific.bed"
if [[ ! -f "$gm_specific" ]]; then
  bedtools subtract -a "$OUT_DIR/peaks/GM12878.bed" \
                    -b "$OUT_DIR/peaks/NaiveB.bed" \
    > "$gm_specific"
  echo "  GM12878-specific: $(wc -l < "$gm_specific") peaks"
fi
if [[ ! -f "$naiveb_specific" ]]; then
  bedtools subtract -a "$OUT_DIR/peaks/NaiveB.bed" \
                    -b "$OUT_DIR/peaks/GM12878.bed" \
    > "$naiveb_specific"
  echo "  NaiveB-specific: $(wc -l < "$naiveb_specific") peaks"
fi

# Blood pair: K562 all peaks (HSC not available on ENCODE)
echo "  NOTE: HSC ATAC-seq not available on ENCODE. Running K562 vs genome only."

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Run HOMER genome-wide on each condition-specific peak set
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 4: Running HOMER genome-wide..."

declare -A HOMER_INPUTS=(
  ["K562_vitro_blood"]="$OUT_DIR/peaks/K562.bed"
  ["HepG2_vitro_liver"]="$OUT_DIR/peaks/HepG2_specific.bed"
  ["Liver_vivo_liver"]="$OUT_DIR/peaks/Liver_specific.bed"
  ["GM12878_vitro_lymph"]="$OUT_DIR/peaks/GM12878_specific.bed"
  ["NaiveB_vivo_lymph"]="$OUT_DIR/peaks/NaiveB_specific.bed"
)

for tag in "${!HOMER_INPUTS[@]}"; do
  input="${HOMER_INPUTS[$tag]}"
  homer_out="$OUT_DIR/homer/$tag"

  n_peaks=$(wc -l < "$input")
  if [[ $n_peaks -lt 10 ]]; then
    echo "  SKIPPING $tag: only $n_peaks peaks (too few)"
    continue
  fi

  if [[ -d "$homer_out/homerResults" ]]; then
    echo "  Skipping HOMER $tag (already done)"
    continue
  fi

  echo "  Running HOMER: $tag ($n_peaks peaks)..."
  mkdir -p "$homer_out"
  findMotifsGenome.pl \
    "$input" \
    "$GENOME" \
    "$homer_out" \
    -size 200 \
    -mask \
    -genomeBg \
    -p 8 \
    2>"$homer_out/homer.log" \
  && echo "  Done: $tag" \
  || echo "  WARNING: HOMER failed for $tag (see $homer_out/homer.log)"
done

echo ""
echo "============================================================"
echo "All ENCODE ATAC genome-wide HOMER runs complete."
echo "Results in: $OUT_DIR/homer/"
echo "Cell types: K562 (blood vitro), HepG2 vs Liver (liver pair),"
echo "            GM12878 vs NaiveB (lymph pair)"
echo "Note: HSC (blood vivo) not available on ENCODE — skipped."
echo "============================================================"
