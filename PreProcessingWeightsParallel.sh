#!/bin/bash
# =============================================================================
# PreProcessingWeightsParallel.sh — ECC-CODE-Engine parallel-per-arch submission
#
# Submits one INDEPENDENT 3-stage chain per architecture in $ARCHS, run
# concurrently instead of looping over all archs in a single chain:
#
#   for each ARC in $ARCHS:
#     1-Quantization(ARC)  →  2-Sensitivity(ARC)  →  3-PatternFinder(ARC)
#
# Each arch's chain runs on its own GPU, fully independent of the others —
# arch B's jobs never wait on arch A. No env vars need to be set per arch;
# per-arch BATCH_SIZE/QAT_SKIP_FIRST_LAST overrides (env.sh) are resolved
# automatically inside each chain exactly as they are for a single-arch run.
#
# Usage:
#   bash PreProcessingWeightsParallel.sh                # one chain per arch in env.sh's ARCHS default
#   ARCHS="densenet121 xception" bash PreProcessingWeightsParallel.sh   # only those two, in parallel
# =============================================================================

set -euo pipefail

# SLURM copies scripts to /var/spool/slurmd/... so BASH_SOURCE[0] is wrong when
# sbatch'd — SLURM_SUBMIT_DIR always reflects where sbatch/bash was invoked from.
PROJ_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Resolve $ARCHS (and other overridable defaults) the same way every run.sh does.
source "${PROJ_DIR}/env.sh"

echo "=============================================="
echo " ECC-CODE-Engine — Parallel Per-Arch Submission"
echo " PROJ_DIR: ${PROJ_DIR}"
echo " ARCHS:    ${ARCHS}"
echo "=============================================="
echo ""

ALL_JOB_IDS=()

for ARC in $ARCHS; do
    echo "---- ${ARC} ----"

    job1_out=$(cd "${PROJ_DIR}/1-Quantization" && \
        sbatch --job-name="1-ecc-quantize-${ARC}" --export=ALL,ARCHS="${ARC}" run.sh)
    job1=$(echo "${job1_out}" | awk '{print $4}')
    echo "  1-Quantization  submitted → job ${job1}"

    job2_out=$(cd "${PROJ_DIR}/2-Sensitivity" && \
        sbatch --job-name="2-ecc-sensitivity-${ARC}" --export=ALL,ARCHS="${ARC}" \
               --dependency=afterok:${job1} run.sh)
    job2=$(echo "${job2_out}" | awk '{print $4}')
    echo "  2-Sensitivity   submitted → job ${job2}  (after ${job1})"

    job3_out=$(cd "${PROJ_DIR}/3-PatternFinder" && \
        sbatch --job-name="3-ecc-patterns-${ARC}" --export=ALL,ARCHS="${ARC}" \
               --dependency=afterok:${job2} run.sh)
    job3=$(echo "${job3_out}" | awk '{print $4}')
    echo "  3-PatternFinder submitted → job ${job3}  (after ${job2})"

    ALL_JOB_IDS+=("${job1}" "${job2}" "${job3}")
    echo ""
done

echo "=============================================="
echo " All chains queued: $(echo "$ARCHS" | wc -w) archs × 3 stages = ${#ALL_JOB_IDS[@]} jobs"
echo "=============================================="
echo ""
echo "Monitor : squeue -u \$USER"
echo "Cancel all: scancel ${ALL_JOB_IDS[*]}"
