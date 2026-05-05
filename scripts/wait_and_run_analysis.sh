#!/usr/bin/env bash
# scripts/wait_and_run_analysis.sh
# ----------------------------------------
# Polls for completion of all activation files,
# then runs the cross-model SAE analysis and figures.
#
# Usage (background):
#   cd /workspace/gene_expression_predictions
#   bash scripts/wait_and_run_analysis.sh &

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(enformer hyenadna nucleotide_transformer)
LAYERS=(early mid late)
PAIR_CONDS=(
  "blood:K562:HSC"
  "liver:HepG2:Liver"
  "lymph:GM12878:NaiveB"
)
POLL_SECS=120

echo "[wait_and_run] Waiting for activations (polling every ${POLL_SECS}s)..."

while true; do
  MISSING=0
  for model in "${MODELS[@]}"; do
    for pair_str in "${PAIR_CONDS[@]}"; do
      pair=$(echo "$pair_str" | cut -d: -f1)
      vitro=$(echo "$pair_str" | cut -d: -f2)
      vivo=$(echo "$pair_str"  | cut -d: -f3)
      for cond in "$vitro" "$vivo"; do
        for layer in "${LAYERS[@]}"; do
          f="activations/${model}/${pair}/${cond}/${layer}.pt"
          if [[ ! -f "$f" ]]; then
            MISSING=$((MISSING + 1))
          fi
        done
      done
    done
  done

  if [[ $MISSING -eq 0 ]]; then
    echo "[wait_and_run] All activations present. Starting analysis at $(date)."
    break
  fi

  TOTAL=$((${#MODELS[@]} * 6 * 3))
  DONE=$((TOTAL - MISSING))
  echo "[wait_and_run] $(date '+%H:%M') — ${DONE}/${TOTAL} files ready ($MISSING missing)."
  sleep $POLL_SECS
done

echo ""
echo "[wait_and_run] Step 1 — Cross-model SAE analysis..."
PYTHONPATH=src python src/cross_model_analysis.py 2>&1 | tee logs/cross_model_analysis.log

echo ""
echo "[wait_and_run] Step 2 — Figs 12–14..."
PYTHONPATH=src python src/plot_cross_model_figures.py \
    --results_dir results/cross_model \
    --figures_dir results/figures \
    2>&1 | tee logs/cross_model_figures.log

echo ""
echo "[wait_and_run] Pipeline complete at $(date)."
echo "Figures: results/figures/fig1{2,3,4}_cross_model*.{pdf,png}"
