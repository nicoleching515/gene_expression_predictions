#!/usr/bin/env bash
# =============================================================================
# run_homer_genomewide.sh
#
# Runs HOMER findMotifsGenome.pl genome-wide for all 18 layer/context/pair
# conditions using HOMER's built-in -genomeBg background (random hg38
# sequences matched by GC content) instead of the chr8/chr9-only custom
# background that was used previously.
#
# Deletes existing HOMER output directories before re-running so results
# are fresh.
#
# Usage:
#   cd /workspace/gene_expression_predictions
#   bash run_homer_genomewide.sh
# =============================================================================

set -euo pipefail

export PATH="/workspace/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TOP_FEATURES_DIR="$REPO_ROOT/outputs/top_features"
ANNOTATION_DIR="$REPO_ROOT/outputs/annotation"
GENOME="hg38"
N_TOP=50
LAYERS="early mid late"

echo "============================================================"
echo "HOMER Genome-Wide Motif Analysis"
echo "Genome   : $GENOME"
echo "Background: -genomeBg (HOMER random genome-wide, GC-matched)"
echo "Repo root: $REPO_ROOT"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Ensure merged BED files exist (reuse or rebuild)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 1: Checking merged BED files..."

for layer in $LAYERS; do
  for side in vitro vivo; do
    for pair in blood liver lymph; do
      TAG="${layer}_${side}_${pair}"
      MERGED_BED="$ANNOTATION_DIR/homer/${TAG}.bed"
      LAYER_DIR="$TOP_FEATURES_DIR/$layer"

      if [[ ! -f "$MERGED_BED" ]]; then
        echo "  Building merged BED for $TAG..."
        if [[ -d "$LAYER_DIR" ]]; then
          find "$LAYER_DIR" -name "top_windows_${side}_${pair}.bed" \
            | head -n "$N_TOP" \
            | xargs cat \
            | sort -k1,1 -k2,2n \
            | bedtools merge -i stdin \
            > "$MERGED_BED"
        else
          echo "  WARNING: $LAYER_DIR not found; pooling all layers for $TAG"
          find "$TOP_FEATURES_DIR" -name "top_windows_${side}_${pair}.bed" \
            | head -n "$N_TOP" \
            | xargs cat \
            | sort -k1,1 -k2,2n \
            | bedtools merge -i stdin \
            > "$MERGED_BED"
        fi
        echo "  Merged BED ${TAG}: $(wc -l < "$MERGED_BED") windows"
      else
        echo "  Merged BED exists: $TAG ($(wc -l < "$MERGED_BED") windows)"
      fi
    done
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Remove stale HOMER output dirs, then run genome-wide HOMER
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 2: Running genome-wide HOMER (removing stale results first)..."

for layer in $LAYERS; do
  for side in vitro vivo; do
    for pair in blood liver lymph; do
      TAG="${layer}_${side}_${pair}"
      MERGED_BED="$ANNOTATION_DIR/homer/${TAG}.bed"
      HOMER_OUT="$ANNOTATION_DIR/homer/${TAG}"

      # Remove stale output directory so HOMER re-runs fresh
      if [[ -d "$HOMER_OUT" ]]; then
        echo "  Removing stale output: $HOMER_OUT"
        rm -rf "$HOMER_OUT"
      fi

      mkdir -p "$HOMER_OUT"

      echo "  Running HOMER genome-wide: $TAG ..."
      findMotifsGenome.pl \
        "$MERGED_BED" \
        "$GENOME" \
        "$HOMER_OUT" \
        -size 200 \
        -mask \
        -genomeBg \
        -p 8 \
        2>"$HOMER_OUT/homer.log" \
      && echo "  Done: $TAG" \
      || echo "  WARNING: HOMER failed for $TAG (see $HOMER_OUT/homer.log)"

    done
  done
done

echo ""
echo "============================================================"
echo "All HOMER genome-wide runs complete."
echo "Results in: $ANNOTATION_DIR/homer/"
echo "============================================================"
