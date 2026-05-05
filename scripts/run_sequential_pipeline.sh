#!/usr/bin/env bash
# Optimal execution order for A100 80GB:
#   Phase 1: HyenaDNA + NT in parallel (fast models, low memory)
#   Phase 2: Enformer alone (heavy model, needs full GPU compute)
#   Phase 3: EpiBERT (last, requires ATAC BAM files)
#   Phase 4: Cross-model SAE analysis + figures

set -euo pipefail
REPO=/workspace/gene_expression_predictions
LOG=$REPO/logs
mkdir -p "$LOG"

START_TOTAL=$(date +%s)
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG/sequential_pipeline.log"; }

# ── Phase 1: HyenaDNA + NT in parallel ───────────────────────────────────────
log "=== PHASE 1: HyenaDNA + NT (parallel) ==="
cd "$REPO"
nohup python src/collect_activations_multimodel.py \
    --model hyenadna --use-genome --skip-existing \
    > "$LOG/collect_hyenadna.log" 2>&1 &
PID_HYENA=$!

nohup python src/collect_activations_multimodel.py \
    --model nucleotide_transformer --use-genome --skip-existing \
    > "$LOG/collect_nt.log" 2>&1 &
PID_NT=$!

log "HyenaDNA PID=$PID_HYENA  |  NT PID=$PID_NT"
log "Waiting for Phase 1 to complete…"

wait $PID_HYENA && log "HyenaDNA DONE ✓" || { log "HyenaDNA FAILED"; exit 1; }
wait $PID_NT    && log "NT DONE ✓"       || { log "NT FAILED"; exit 1; }

PHASE1_ELAPSED=$(( $(date +%s) - START_TOTAL ))
log "Phase 1 complete in $((PHASE1_ELAPSED/60)) min $((PHASE1_ELAPSED%60)) s"

# ── Phase 2: Enformer alone ───────────────────────────────────────────────────
log "=== PHASE 2: Enformer (solo) ==="
ENFORMER_START=$(date +%s)
python src/collect_activations_multimodel.py \
    --model enformer --use-genome --skip-existing \
    > "$LOG/collect_enformer.log" 2>&1 \
    && log "Enformer DONE ✓" || { log "Enformer FAILED"; exit 1; }

ENFORMER_ELAPSED=$(( $(date +%s) - ENFORMER_START ))
log "Enformer complete in $((ENFORMER_ELAPSED/60)) min $((ENFORMER_ELAPSED%60)) s"

# ── Phase 3: Cross-model analysis ────────────────────────────────────────────
log "=== PHASE 3: SAE analysis + figures ==="
python src/cross_model_analysis.py \
    > "$LOG/cross_model_analysis.log" 2>&1 \
    && log "SAE analysis DONE ✓" || { log "SAE analysis FAILED"; exit 1; }

python src/plot_cross_model_figures.py \
    > "$LOG/plot_cross_model.log" 2>&1 \
    && log "Figures DONE ✓" || { log "Figures FAILED"; }

TOTAL=$(( $(date +%s) - START_TOTAL ))
log "=== ALL DONE in $((TOTAL/3600))h $(( (TOTAL%3600)/60 ))m ==="
