#!/usr/bin/env bash
# =============================================================================
# run_annotation.sh
# Full bio-annotation pipeline for EpiBERT SAE top features.
# Runs HOMER motif enrichment, ChromHMM state annotation, and GO enrichment,
# then generates figures and pushes everything to GitHub.
#
# Prerequisites (all available on the H100 server):
#   - HOMER installed and in PATH  (homer.ucsd.edu/homer)
#   - bedtools installed           (bedtools.readthedocs.io)
#   - Python >= 3.9 with: matplotlib, numpy, pandas, scipy, requests
#   - R with: rGREAT, ggplot2, dplyr  (for GO enrichment)
#   - hg38 genome configured in HOMER (perl configureHomer.pl -install hg38)
#   - git configured with push access to the repo
#
# Usage:
#   cd /workspace
#   bash run_annotation.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TOP_FEATURES_DIR="$REPO_ROOT/outputs/top_features"
ANNOTATION_DIR="$REPO_ROOT/outputs/annotation"
FIGURES_DIR="$REPO_ROOT/results/figures"
GENOME="hg38"
N_TOP=50          # features per layer to annotate
LAYERS="early mid late"

echo "============================================================"
echo "EpiBERT SAE Annotation Pipeline"
echo "Repo root : $REPO_ROOT"
echo "Top feats : $TOP_FEATURES_DIR"
echo "Output    : $ANNOTATION_DIR"
echo "============================================================"

mkdir -p "$ANNOTATION_DIR/homer"
mkdir -p "$ANNOTATION_DIR/chromhmm"
mkdir -p "$ANNOTATION_DIR/go"
mkdir -p "$FIGURES_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Download ENCODE ChromHMM tracks for cell lines
# 15-state ChromHMM model (Roadmap Epigenomics) for each cell line
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 1: Downloading ChromHMM reference tracks..."

CHROMHMM_DIR="$REPO_ROOT/data/chromhmm"
mkdir -p "$CHROMHMM_DIR"

declare -A CHROMHMM_URLS=(
  # ENCODE 15-state ChromHMM, hg38, from UCSC / Roadmap
  ["K562"]="https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/E123_15_coreMarks_hg38lift_stateno.bed.gz"
  ["HepG2"]="https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/E118_15_coreMarks_hg38lift_stateno.bed.gz"
  ["GM12878"]="https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/E116_15_coreMarks_hg38lift_stateno.bed.gz"
)

declare -A CHROMHMM_STATE_NAMES=(
  ["1"]="TssA"
  ["2"]="TssAFlnk"
  ["3"]="TxFlnk"
  ["4"]="Tx"
  ["5"]="TxWk"
  ["6"]="EnhG"
  ["7"]="Enh"
  ["8"]="ZNF/Rpts"
  ["9"]="Het"
  ["10"]="TssBiv"
  ["11"]="BivFlnk"
  ["12"]="EnhBiv"
  ["13"]="ReprPC"
  ["14"]="ReprPCWk"
  ["15"]="Quies"
)

for cell in "${!CHROMHMM_URLS[@]}"; do
  OUTFILE="$CHROMHMM_DIR/${cell}_chromhmm.bed"
  if [[ ! -f "$OUTFILE" ]]; then
    echo "  Downloading ChromHMM for $cell..."
    wget -q "${CHROMHMM_URLS[$cell]}" -O "$OUTFILE.gz"
    gunzip "$OUTFILE.gz"
    echo "  Done: $OUTFILE"
  else
    echo "  Skipping $cell (already downloaded)"
  fi
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — HOMER motif enrichment on top-feature BED files
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 2: Running HOMER motif enrichment..."

# Build a background BED file from all 200 evaluation windows
# (all conditions pooled, to provide a matched genomic background)
BACKGROUND_BED="$ANNOTATION_DIR/homer/background_all_windows.bed"
if [[ ! -f "$BACKGROUND_BED" ]]; then
  echo "  Building background BED from all top-feature windows..."
  { cat "$TOP_FEATURES_DIR"/*/*/top_windows_*.bed \
    | sort -k1,1 -k2,2n \
    | bedtools merge -i stdin \
    > "$BACKGROUND_BED"; } || true
  echo "  Background: $(wc -l < "$BACKGROUND_BED") merged windows"
fi

for layer in $LAYERS; do
  for side in vitro vivo; do
    for pair in blood liver lymph; do
      # Collect BED files for all top-N features matching layer/side/pair
      MERGED_BED="$ANNOTATION_DIR/homer/${layer}_${side}_${pair}.bed"
      if [[ ! -f "$MERGED_BED" ]]; then
        # Pool top windows across all features for this layer/side/pair
        # (pipefail suppressed here: head exits early and causes SIGPIPE in sort)
        { find "$TOP_FEATURES_DIR/layer_${layer}" -name "top_windows_${side}_${pair}.bed" \
          | head -n "$N_TOP" \
          | xargs cat \
          | sort -k1,1 -k2,2n \
          | bedtools merge -i stdin \
          > "$MERGED_BED"; } || true
        echo "  Merged BED for ${layer}/${side}/${pair}: $(wc -l < "$MERGED_BED") windows"
      fi

      HOMER_OUT="$ANNOTATION_DIR/homer/${layer}_${side}_${pair}"
      if [[ ! -d "$HOMER_OUT/homerResults" ]]; then
        echo "  Running HOMER: ${layer} ${side} ${pair}..."
        mkdir -p "$HOMER_OUT"
        findMotifsGenome.pl \
          "$MERGED_BED" \
          "$GENOME" \
          "$HOMER_OUT" \
          -size 200 \
          -mask \
          -bg "$BACKGROUND_BED" \
          -p 8 \
          2>"$HOMER_OUT/homer.log"
        echo "  HOMER done: $HOMER_OUT"
      else
        echo "  Skipping HOMER ${layer}/${side}/${pair} (already done)"
      fi
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — ChromHMM state annotation via bedtools intersect
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 3: ChromHMM state annotation..."

declare -A PAIR_VITRO_CELL=(
  ["blood"]="K562"
  ["liver"]="HepG2"
  ["lymph"]="GM12878"
)

for pair in blood liver lymph; do
  cell="${PAIR_VITRO_CELL[$pair]}"
  CHROMHMM_BED="$CHROMHMM_DIR/${cell}_chromhmm.bed"

  for layer in $LAYERS; do
    for side in vitro vivo; do
      INPUT_BED="$ANNOTATION_DIR/homer/${layer}_${side}_${pair}.bed"
      OUT_TSV="$ANNOTATION_DIR/chromhmm/${layer}_${side}_${pair}_states.tsv"

      if [[ ! -f "$OUT_TSV" ]]; then
        echo "  ChromHMM intersect: ${layer}/${side}/${pair} vs ${cell}..."
        # Count state overlaps, output state -> count -> fraction
        TOTAL=$(wc -l < "$INPUT_BED")
        bedtools intersect \
          -a "$INPUT_BED" \
          -b "$CHROMHMM_BED" \
          -wa -wb \
        | awk '{print $NF}' \
        | sort \
        | uniq -c \
        | sort -rn \
        | awk -v total="$TOTAL" \
          'BEGIN{print "state\tcount\tfraction"}
           {printf "%s\t%d\t%.4f\n", $2, $1, $1/total}' \
        > "$OUT_TSV"
        echo "  Done: $OUT_TSV"
      else
        echo "  Skipping ChromHMM ${layer}/${side}/${pair} (already done)"
      fi
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — GO enrichment via rGREAT (R script)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 4: GO enrichment (rGREAT)..."

Rscript "$REPO_ROOT/src/run_great.R" \
  --top_features_dir "$TOP_FEATURES_DIR" \
  --annotation_dir   "$ANNOTATION_DIR/go" \
  --layers           "$LAYERS" \
  --n_top            "$N_TOP" \
  --genome           "$GENOME" \
  2>&1 | tee "$ANNOTATION_DIR/go/great.log"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Parse results and generate figures
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 5: Generating annotation figures..."

python3 "$REPO_ROOT/src/plot_annotation.py" \
  --annotation_dir "$ANNOTATION_DIR" \
  --figures_dir    "$FIGURES_DIR" \
  --layers         "$LAYERS" \
  --n_top          "$N_TOP"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Generate Phase 8 report
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 6: Writing Phase 8 report..."

python3 "$REPO_ROOT/src/write_phase8_report.py" \
  --annotation_dir "$ANNOTATION_DIR" \
  --output         "$REPO_ROOT/PHASE_8_REPORT.md"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Commit and push to GitHub
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 7: Committing to GitHub..."

cd "$REPO_ROOT"
git add \
  outputs/annotation/ \
  results/figures/fig3_annotation_heatmap.pdf \
  results/figures/fig3_annotation_heatmap.png \
  results/figures/fig6_case_studies.pdf \
  results/figures/fig6_case_studies.png \
  PHASE_8_REPORT.md \
  src/run_great.R \
  src/plot_annotation.py \
  src/write_phase8_report.py \
  run_annotation.sh

git commit -m "Phase 8: HOMER motif, ChromHMM, and GO annotation of top SAE features

- HOMER motif enrichment on top-$N_TOP features per layer/side/pair
- ChromHMM state annotation via bedtools intersect (K562, HepG2, GM12878)
- GO:BP enrichment via rGREAT for each feature set
- Figures: fig3_annotation_heatmap, fig6_case_studies
- Phase 8 report: PHASE_8_REPORT.md"

git push origin main
echo ""
echo "============================================================"
echo "Annotation pipeline complete. Results in outputs/annotation/"
echo "Figures in results/figures/"
echo "Pushed to GitHub."
echo "============================================================"
