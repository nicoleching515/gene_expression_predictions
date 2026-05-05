#!/usr/bin/env bash
# =============================================================================
# run_homer_with_denovo.sh
# =============================================================================
# Run HOMER findMotifsGenome.pl for all 18 SAE-derived conditions and for the
# ENCODE ATAC peak sets, preserving BOTH de novo (homerResults.html) and
# known-motif (knownResults.txt) output files.
#
# Key changes vs. earlier scripts (run_homer_genomewide.sh):
#   1. Does NOT add -nomotif → de novo motif finding is enabled by default.
#   2. Explicitly copies / renames critical output files so they persist after
#      subsequent re-runs (HOMER silently overwrites on re-run).
#   3. Saves a run-manifest (homer_manifest.tsv) listing every output path.
#
# Prerequisites:
#   - HOMER ≥ 4.11 installed; `findMotifsGenome.pl` on PATH
#   - hg38 genome installed in HOMER   (run: perl configureHomer.pl -install hg38)
#   - BED files already in outputs/annotation/homer/<tag>.bed
#
# Usage:
#   cd /workspace/gene_expression_predictions
#   bash run_homer_with_denovo.sh [--threads 8] [--n-denovo 25]
#
#   --threads N    HOMER -p threads (default: 8)
#   --n-denovo N   Number of de novo motifs to search for, -S N (default: 25)
#   --dry-run      Print commands but do not execute
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
ANNO_DIR="${REPO_ROOT}/outputs/annotation/homer"
BED_DIR="${ANNO_DIR}"
MANIFEST="${ANNO_DIR}/homer_manifest.tsv"

THREADS=8
N_DENOVO=25
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads)  THREADS="$2";  shift 2 ;;
        --n-denovo) N_DENOVO="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true;  shift   ;;
        *) echo "Unknown arg: $1"; exit 1  ;;
    esac
done

LAYERS=(early mid late)
SIDES=(vitro vivo)
PAIRS=(blood liver lymph)

echo "HOMER de-novo + known motif run"
echo "  ANNO_DIR  : ${ANNO_DIR}"
echo "  Threads   : ${THREADS}"
echo "  De novo N : ${N_DENOVO}"
echo "  Dry-run   : ${DRY_RUN}"
echo ""

# Write manifest header
echo -e "tag\tbed\thomer_dir\tknown_results\tdenovo_html\tdenovo_motifs" > "${MANIFEST}"

_run_homer() {
    local tag="$1"
    local bed="${BED_DIR}/${tag}.bed"
    local outdir="${ANNO_DIR}/${tag}"

    if [[ ! -f "${bed}" ]]; then
        echo "  [SKIP] BED not found: ${bed}"
        return
    fi

    echo "── ${tag} ──────────────────────────────"

    # Clear stale HOMER output so cached (invalid) results are not reused
    rm -rf "${outdir}"
    mkdir -p "${outdir}"

    local cmd=(
        findMotifsGenome.pl
        "${bed}" hg38 "${outdir}"
        -size 200
        -mask
        -genomeBg
        -S "${N_DENOVO}"
        -p "${THREADS}"
    )

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "  [DRY-RUN] ${cmd[*]}"
    else
        "${cmd[@]}" 2>&1 | tee "${outdir}/homer.log"
    fi

    # ── Post-process: snapshot key files ───────────────────────────────────
    local known_results="${outdir}/knownResults.txt"
    local denovo_html="${outdir}/homerResults.html"
    local denovo_motifs="${outdir}/homerMotifs.motifs"

    # Explicitly collect all individual de novo .motif files → archive
    if [[ "${DRY_RUN}" == "false" && -d "${outdir}" ]]; then
        local motif_archive="${outdir}/denovo_motifs.tar.gz"
        find "${outdir}" -maxdepth 1 -name "motif*.motif" \
            | tar czf "${motif_archive}" -T - 2>/dev/null || true
        echo "  Archived per-motif files → ${motif_archive}"
    fi

    # Append to manifest
    echo -e "${tag}\t${bed}\t${outdir}\t${known_results}\t${denovo_html}\t${denovo_motifs}" \
        >> "${MANIFEST}"
}

# ── SAE-derived conditions (18 = 3 layers × 2 sides × 3 pairs) ──────────────
for layer in "${LAYERS[@]}"; do
    for side in "${SIDES[@]}"; do
        for pair in "${PAIRS[@]}"; do
            _run_homer "${layer}_${side}_${pair}"
        done
    done
done

echo ""
echo "Run manifest written to: ${MANIFEST}"
echo ""
echo "To regenerate figures using de novo results:"
echo "  python src/plot_homer_go_figures.py \\"
echo "    --annotation_dir outputs/annotation \\"
echo "    --figures_dir results/figures \\"
echo "    --homer-source denovo"
echo ""
echo "Done."
