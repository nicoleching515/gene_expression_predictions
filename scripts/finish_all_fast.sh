#!/usr/bin/env bash
# Fast parallel BAM refresh + EpiBERT activations + cross-model analysis + figures.
# Logs: /workspace/atac_bams/logs/master_pipeline.log (redirect nohup here)
set -euo pipefail

BAMS=/workspace/atac_bams
PROJ=/workspace/gene_expression_predictions
LOG=${BAMS}/logs
mkdir -p "$LOG"

echo "=== $(date -u) master pipeline start ==="

verify_bam () {
  local f="$1" exp="$2" label="$3"
  [[ -f "$f" ]] || { echo "MISSING $label"; return 1; }
  python3 - "$f" "$exp" "$label" <<'PY'
import sys, pathlib
path, exp, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
p = pathlib.Path(path)
sz = p.stat().st_size
with p.open("rb") as fh:
    magic = fh.read(3)
ok_gz = magic[:2] == b"\x1f\x8b"
if not ok_gz or abs(sz - exp) > max(50_000_000, exp // 50):
    print(f"BAD {label}: gzip={ok_gz} size={sz} expected~{exp}")
    sys.exit(1)
print(f"OK {label}: {sz} bytes")
PY
}

echo "--- parallel wget (resume) ---"
DL_GM="${LOG}/wget_GM12878.log"
DL_NB="${LOG}/wget_NaiveB.log"
DL_LV="${LOG}/wget_Liver.log"

(
  wget --retry-connrefused --tries=0 --waitretry=10 --read-timeout=120 --timeout=120 \
       --continue --progress=dot:giga \
       -O "${BAMS}/GM12878-ATAC.bam" \
       "https://www.encodeproject.org/files/ENCFF415FEC/@@download/ENCFF415FEC.bam" \
       2>&1 | tee "$DL_GM"
) &
PID_GM=$!

(
  wget --retry-connrefused --tries=0 --waitretry=10 --read-timeout=120 --timeout=120 \
       --continue --progress=dot:giga \
       -O "${BAMS}/NaiveB-ATAC.bam" \
       "https://www.encodeproject.org/files/ENCFF298OEW/@@download/ENCFF298OEW.bam" \
       2>&1 | tee "$DL_NB"
) &
PID_NB=$!

(
  wget --retry-connrefused --tries=0 --waitretry=10 --read-timeout=120 --timeout=120 \
       --continue --progress=dot:giga \
       -O "${BAMS}/Liver-ATAC.bam" \
       "https://www.encodeproject.org/files/ENCFF974NEA/@@download/ENCFF974NEA.bam" \
       2>&1 | tee "$DL_LV"
) &
PID_LV=$!

wait $PID_GM $PID_NB $PID_LV || { echo "One or more wget jobs failed"; exit 1; }

# ENCODE byte sizes (GRCh38 ATAC unfiltered alignments)
verify_bam "${BAMS}/GM12878-ATAC.bam" 2515550487 GM12878
verify_bam "${BAMS}/NaiveB-ATAC.bam" 5565253456 NaiveB
verify_bam "${BAMS}/Liver-ATAC.bam" 37441140977 Liver
verify_bam "${BAMS}/K562-ATAC.bam" 21952014426 K562
verify_bam "${BAMS}/HepG2-ATAC.bam" 17328117861 HepG2
verify_bam "${BAMS}/HSC-ATAC.bam" 558106280 HSC

echo "--- parallel BAM indexes (pysam; rebuilds each .bai) ---"
python3 << 'PY'
import multiprocessing as mp
from pathlib import Path
import pysam

def index_one(bam: str) -> str:
    bam = Path(bam)
    pysam.index(str(bam))
    return f"indexed {bam.name}"

bams = [
    "/workspace/atac_bams/K562-ATAC.bam",
    "/workspace/atac_bams/HepG2-ATAC.bam",
    "/workspace/atac_bams/GM12878-ATAC.bam",
    "/workspace/atac_bams/HSC-ATAC.bam",
    "/workspace/atac_bams/Liver-ATAC.bam",
    "/workspace/atac_bams/NaiveB-ATAC.bam",
]
# Parallel indexing — separate files, minimal lock contention.
with mp.Pool(min(3, len(bams))) as pool:
    for msg in pool.map(index_one, bams):
        print(msg)
PY

echo "--- EpiBERT activation collection (GPU) ---"
cd "$PROJ"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PYTHONPATH=src python3 src/collect_activations_multimodel.py \
  --model epibert \
  --use-genome \
  --batch-size 10 \
  2>&1 | tee "${LOG}/collect_epibert.log"

echo "--- Cross-model SAE + CDS (loads existing SAE checkpoints; trains epibert only) ---"
PYTHONPATH=src python3 src/cross_model_analysis.py \
  2>&1 | tee "${LOG}/cross_model_analysis.log"

echo "--- Cross-model figures ---"
PYTHONPATH=src python3 src/plot_cross_model_figures.py \
  --results_dir results/cross_model \
  --figures_dir results/figures \
  2>&1 | tee "${LOG}/plot_cross_model.log"

echo "=== $(date -u) master pipeline DONE ==="
