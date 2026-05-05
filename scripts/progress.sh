#!/usr/bin/env bash
# Live progress bar for activation collection.
# Usage: bash scripts/progress.sh [interval_seconds]
REPO=/workspace/gene_expression_predictions
INTERVAL=${1:-30}
CONDITIONS=(K562 HSC HepG2 Liver GM12878 NaiveB)
DEPTHS=(early mid late)
TOTAL_PER_MODEL=$(( ${#CONDITIONS[@]} * ${#DEPTHS[@]} ))  # 18

bar() {
    local done=$1 total=$2 width=40
    local filled=$(( done * width / total ))
    local empty=$(( width - filled ))
    printf "[%s%s] %d/%d" "$(printf '#%.0s' $(seq 1 $filled 2>/dev/null || true))" \
           "$(printf '.%.0s' $(seq 1 $empty  2>/dev/null || true))" "$done" "$total"
}

# Approximate seconds-per-file based on model type (calibrated for A100)
# HyenaDNA@bs16: ~6 files/min  NT@bs16: ~4 files/min  Enformer@bs4: ~2 files/min
secs_per_file() {
    case $1 in
        hyenadna)             echo 10 ;;   # ~6 conditions × 3 depths / 36 min
        nucleotide_transformer) echo 15 ;; # slower per file at 512 tokens but similar overall
        enformer)             echo 25 ;;   # heavy model, even at batch_size=4
        *)                    echo 20 ;;
    esac
}

while true; do
    clear
    echo "══════════════════════════════════════════════════════════════════"
    echo "  Activation Collection Progress  —  $(date '+%H:%M:%S')"
    echo "══════════════════════════════════════════════════════════════════"
    echo ""

    TOTAL_DONE=0
    TOTAL_ALL=0
    ALL_DONE=true

    for MODEL in hyenadna nucleotide_transformer enformer epibert; do
        DIR="$REPO/activations/$MODEL"
        DONE=$(find "$DIR" -name "*.pt" ! -name "*_partial.pt" 2>/dev/null | wc -l)
        TOTAL=$TOTAL_PER_MODEL

        TOTAL_DONE=$(( TOTAL_DONE + DONE ))
        TOTAL_ALL=$(( TOTAL_ALL + TOTAL ))

        # ETA
        if [[ $DONE -lt $TOTAL ]]; then
            ALL_DONE=false
            SPF=$(secs_per_file $MODEL)
            REMAINING=$(( (TOTAL - DONE) * SPF ))
            ETA_STR="${REMAINING}s (~$(( REMAINING/60 ))m)"
        else
            ETA_STR="DONE ✓"
        fi

        # Running status
        if pgrep -f "collect_activations_multimodel.py --model $MODEL" > /dev/null 2>&1; then
            STATUS="running"
        elif [[ $DONE -ge $TOTAL ]]; then
            STATUS="complete"
        else
            STATUS="waiting"
        fi

        printf "  %-25s  " "$MODEL ($STATUS)"
        bar $DONE $TOTAL
        printf "  ETA: %s\n" "$ETA_STR"
    done

    echo ""
    echo "  ──────────────────────────────────────────────────────────────"
    printf "  %-25s  " "OVERALL"
    bar $TOTAL_DONE $TOTAL_ALL
    echo ""
    echo ""

    # Pipeline phase
    PHASE_LOG="$REPO/logs/sequential_pipeline.log"
    if [[ -f "$PHASE_LOG" ]]; then
        echo "  Pipeline log (last 4 lines):"
        tail -4 "$PHASE_LOG" | sed 's/^/    /'
    fi

    echo ""
    if $ALL_DONE; then
        echo "  All activations complete! SAE analysis should be running."
        break
    fi

    echo "  Refreshing every ${INTERVAL}s  (Ctrl-C to stop)"
    sleep "$INTERVAL"
done
