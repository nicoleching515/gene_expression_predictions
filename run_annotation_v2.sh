#!/usr/bin/env bash
# =============================================================================
# run_annotation.sh  (v2 -- fixed layer-specific BED lookup)
#
# Key fixes from v1:
#   - find now scopes to per-layer directories so early/mid/late each get
#     their own distinct BED files, HOMER runs, and ChromHMM results
#   - plot_homer_go_figures.py called after HOMER + GO to generate
#     standalone figures: fig8_homer_motifs.pdf and fig9_go_enrichment.pdf
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
N_TOP=50
LAYERS="early mid late"

echo "============================================================"
echo "EpiBERT SAE Annotation Pipeline  (v2)"
echo "Repo root : $REPO_ROOT"
echo "============================================================"

mkdir -p "$ANNOTATION_DIR/homer"
mkdir -p "$ANNOTATION_DIR/chromhmm"
mkdir -p "$ANNOTATION_DIR/go"
mkdir -p "$FIGURES_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Download ChromHMM tracks
# ─────────────────────────────────────────────────────────────────────────────
echo ">>> STEP 1: Downloading ChromHMM reference tracks..."

CHROMHMM_DIR="$REPO_ROOT/data/chromhmm"
mkdir -p "$CHROMHMM_DIR"

declare -A CHROMHMM_URLS=(
  ["K562"]="https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/E123_15_coreMarks_hg38lift_stateno.bed.gz"
  ["HepG2"]="https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/E118_15_coreMarks_hg38lift_stateno.bed.gz"
  ["GM12878"]="https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/E116_15_coreMarks_hg38lift_stateno.bed.gz"
)

for cell in "${!CHROMHMM_URLS[@]}"; do
  OUTFILE="$CHROMHMM_DIR/${cell}_chromhmm.bed"
  if [[ ! -f "$OUTFILE" ]]; then
    echo "  Downloading ChromHMM for $cell..."
    wget -q "${CHROMHMM_URLS[$cell]}" -O "$OUTFILE.gz"
    gunzip "$OUTFILE.gz"
  else
    echo "  Skipping $cell (already downloaded)"
  fi
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build background BED (all evaluation windows, pooled)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 2: Building genomic background BED..."

BACKGROUND_BED="$ANNOTATION_DIR/homer/background_all_windows.bed"
if [[ ! -f "$BACKGROUND_BED" ]]; then
  find "$TOP_FEATURES_DIR" -name "top_windows_*.bed" \
    | xargs cat \
    | sort -k1,1 -k2,2n \
    | bedtools merge -i stdin \
    > "$BACKGROUND_BED"
  echo "  Background: $(wc -l < "$BACKGROUND_BED") merged windows"
else
  echo "  Skipping background (already built)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — HOMER motif enrichment  (FIX: scoped to per-layer directory)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 3: Running HOMER motif enrichment (layer-specific)..."

for layer in $LAYERS; do
  for side in vitro vivo; do
    for pair in blood liver lymph; do
      TAG="${layer}_${side}_${pair}"
      MERGED_BED="$ANNOTATION_DIR/homer/${TAG}.bed"

      # ── KEY FIX: scope find to the per-layer top-features directory ─────
      # Expected structure:
      #   outputs/top_features/{layer}/{feature_id}/top_windows_{side}_{pair}.bed
      LAYER_DIR="$TOP_FEATURES_DIR/$layer"

      if [[ ! -f "$MERGED_BED" ]]; then
        if [[ -d "$LAYER_DIR" ]]; then
          find "$LAYER_DIR" -name "top_windows_${side}_${pair}.bed" \
            | head -n "$N_TOP" \
            | xargs cat \
            | sort -k1,1 -k2,2n \
            | bedtools merge -i stdin \
            > "$MERGED_BED"
          echo "  Merged BED ${TAG}: $(wc -l < "$MERGED_BED") windows"
        else
          # Fallback: pool all layers (emits a warning)
          echo "  WARNING: $LAYER_DIR not found; pooling all layers for $TAG"
          find "$TOP_FEATURES_DIR" -name "top_windows_${side}_${pair}.bed" \
            | head -n "$N_TOP" \
            | xargs cat \
            | sort -k1,1 -k2,2n \
            | bedtools merge -i stdin \
            > "$MERGED_BED"
        fi
      fi

      HOMER_OUT="$ANNOTATION_DIR/homer/${TAG}"
      if [[ ! -d "$HOMER_OUT/homerResults" ]]; then
        echo "  Running HOMER: $TAG ..."
        mkdir -p "$HOMER_OUT"
        findMotifsGenome.pl \
          "$MERGED_BED" \
          "$GENOME" \
          "$HOMER_OUT" \
          -size 200 \
          -mask \
          -bg "$BACKGROUND_BED" \
          -p 8 \
          2>"$HOMER_OUT/homer.log" \
        && echo "  HOMER done: $HOMER_OUT" \
        || echo "  WARNING: HOMER failed for $TAG (see $HOMER_OUT/homer.log)"
      else
        echo "  Skipping HOMER $TAG (already done)"
      fi
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — ChromHMM state annotation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 4: ChromHMM state annotation..."

declare -A PAIR_VITRO_CELL=(["blood"]="K562" ["liver"]="HepG2" ["lymph"]="GM12878")

for pair in blood liver lymph; do
  cell="${PAIR_VITRO_CELL[$pair]}"
  CHROMHMM_BED="$CHROMHMM_DIR/${cell}_chromhmm.bed"

  for layer in $LAYERS; do
    for side in vitro vivo; do
      TAG="${layer}_${side}_${pair}"
      INPUT_BED="$ANNOTATION_DIR/homer/${TAG}.bed"
      OUT_TSV="$ANNOTATION_DIR/chromhmm/${TAG}_states.tsv"

      if [[ ! -f "$OUT_TSV" ]]; then
        echo "  ChromHMM: $TAG vs $cell..."
        TOTAL=$(wc -l < "$INPUT_BED")
        bedtools intersect \
          -a "$INPUT_BED" \
          -b "$CHROMHMM_BED" \
          -wa -wb \
        | awk '{print $NF}' \
        | sort | uniq -c | sort -rn \
        | awk -v total="$TOTAL" \
            'BEGIN{print "state\tcount\tfraction"}
             {printf "%s\t%d\t%.4f\n", $2, $1, $1/total}' \
        > "$OUT_TSV"
      else
        echo "  Skipping ChromHMM $TAG (already done)"
      fi
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — GO enrichment via rGREAT
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 5: GO:BP enrichment (rGREAT)..."

Rscript "$REPO_ROOT/src/run_great.R" \
  --top_features_dir "$TOP_FEATURES_DIR" \
  --annotation_dir   "$ANNOTATION_DIR/go" \
  --layers           "$LAYERS" \
  --n_top            "$N_TOP" \
  --genome           "$GENOME" \
  2>&1 | tee "$ANNOTATION_DIR/go/great.log"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — ChromHMM figures (original)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 6: Generating ChromHMM figures..."

python3 "$REPO_ROOT/src/plot_annotation.py" \
  --annotation_dir "$ANNOTATION_DIR" \
  --figures_dir    "$FIGURES_DIR" \
  --layers         "$LAYERS" \
  --n_top          "$N_TOP"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — HOMER motif + GO enrichment figures  (NEW)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 7: Generating HOMER + GO figures..."

python3 "$REPO_ROOT/src/plot_homer_go_figures.py" \
  --annotation_dir  "$ANNOTATION_DIR" \
  --figures_dir     "$FIGURES_DIR" \
  --layers          "$LAYERS" \
  --n_top_motifs    10 \
  --n_top_go        10

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Phase 8 report
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 8: Writing Phase 8 report..."

python3 "$REPO_ROOT/src/write_phase8_report.py" \
  --annotation_dir "$ANNOTATION_DIR" \
  --output         "$REPO_ROOT/PHASE_8_REPORT.md"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Commit and push
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 9: Committing to GitHub..."

cd "$REPO_ROOT"
git add \
  outputs/annotation/ \
  results/figures/fig3_annotation_heatmap.pdf \
  results/figures/fig3_annotation_heatmap.png \
  results/figures/fig6_case_studies.pdf \
  results/figures/fig6_case_studies.png \
  results/figures/fig8_homer_motifs.pdf \
  results/figures/fig8_homer_motifs.png \
  results/figures/fig9_go_enrichment.pdf \
  results/figures/fig9_go_enrichment.png \
  PHASE_8_REPORT.md \
  src/plot_homer_go_figures.py \
  run_annotation.sh

git commit -m "Phase 8 (v2): fix layer-specific BED lookup; add HOMER + GO figures

- Fixed run_annotation.sh: find scoped to per-layer directory so
  early/mid/late get distinct BED files and distinct HOMER results
- Added src/plot_homer_go_figures.py: standalone HOMER motif heatmap
  (fig8_homer_motifs) and GO:BP dot-plot (fig9_go_enrichment)
- All three annotations now have dedicated publication figures"

git push origin main

echo ""
echo "============================================================"
echo "Pipeline complete."
echo "New figures:"
echo "  $FIGURES_DIR/fig8_homer_motifs.{pdf,png}"
echo "  $FIGURES_DIR/fig9_go_enrichment.{pdf,png}"
echo "============================================================"
