#!/usr/bin/env bash
# scripts/run_cross_model_pipeline.sh
# ------------------------------------
# Wait for all four models' activations to be complete, then run:
#   1. Cross-model SAE analysis
#   2. Cross-model figure generation (Figs 12–14)
#
# Usage:
#   cd /workspace/gene_expression_predictions
#   bash scripts/run_cross_model_pipeline.sh [--smoke-test]
#
# Exits 0 on success.

set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=${1:-""}

MODELS=(epibert enformer hyenadna nucleotide_transformer)
LAYERS=(early mid late)
PAIRS=(blood liver lymph)
PAIR_CONDS=(
  "blood:K562:HSC"
  "liver:HepG2:Liver"
  "lymph:GM12878:NaiveB"
)

echo "=== Cross-model pipeline ==="
echo "Checking activation completeness..."

ALL_OK=true
for model in "${MODELS[@]}"; do
  for pair_str in "${PAIR_CONDS[@]}"; do
    pair=$(echo "$pair_str" | cut -d: -f1)
    vitro=$(echo "$pair_str" | cut -d: -f2)
    vivo=$(echo "$pair_str"  | cut -d: -f3)
    for cond in "$vitro" "$vivo"; do
      for layer in "${LAYERS[@]}"; do
        f="activations/${model}/${pair}/${cond}/${layer}.pt"
        if [[ ! -f "$f" ]]; then
          echo "  MISSING: $f"
          ALL_OK=false
        fi
      done
    done
  done
done

if [[ "$ALL_OK" == "false" ]]; then
  echo ""
  echo "WARNING: Some activation files are missing."
  echo "Run collect_activations_multimodel.py for the affected models first."
  echo "Continuing with available activations..."
fi

echo ""
echo "Step 1 — Cross-model SAE training + CDS analysis..."
CMD="PYTHONPATH=src python src/cross_model_analysis.py"
if [[ "$SMOKE" == "--smoke-test" ]]; then
  CMD="$CMD --smoke-test"
fi
eval "$CMD" 2>&1 | tee logs/cross_model_analysis.log

echo ""
echo "Step 2 — Generating Figs 12–14..."
PYTHONPATH=src python src/plot_cross_model_figures.py \
    --results_dir results/cross_model \
    --figures_dir results/figures \
    2>&1 | tee logs/cross_model_figures.log

echo ""
echo "Pipeline complete."
echo "Figures: results/figures/fig1{2,3,4}_cross_model*.{pdf,png}"
echo "Results: results/cross_model/"
