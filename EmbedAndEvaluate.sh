#!/bin/bash
# =============================================================================
# EmbedAndEvaluate.sh — ECC-CODE-Engine pipeline chain submission
#
# Submits the four post-processing stages as dependent SLURM jobs:
#   4-EmbeddingECC  →  5-EmbeddingsMerging  →  6-BaseAccuracyTesting  (parallel)
#                                            →  7-SensitiveAccuracyTesting
#
# If EMBED_SKIP_PROCESS=true (default in env.sh), stage 4 is skipped and
# stage 5 starts immediately (ECC chunks from a prior run are reused).
#
# Stages 6 and 7 are submitted in parallel — both depend on stage 5 but
# are independent of each other.
#
# Usage:
#   bash EmbedAndEvaluate.sh
#
# Environment overrides (set before running):
#   EMBED_SKIP_PROCESS=false    re-run the ECC embedding (step 4) from scratch
#   EMBED_DATASETS="IMAGENET"   restrict to one dataset
#   EMBED_ARCHS="mobilenet_v2"  restrict to one architecture
#   EMBED_APPROACH="greedy"     ECC approach (greedy | search3 | no | parfix | ...)
#   EMBED_CODEWORD="63"         codeword length (63 | 127 | 255)
#   EMBED_WORKERS="16"          parallel workers for step 4
#   EMBED_RUN_CPP=true          use C++ runner for step 4 (requires ecc_cpp.sif)
#
# Examples:
#   # Merge + evaluate only (step 4 already done — default):
#   bash EmbedAndEvaluate.sh
#
#   # Full run including re-embedding:
#   EMBED_SKIP_PROCESS=false bash EmbedAndEvaluate.sh
#
#   # Quick smoke test — single dataset/arch, skip embedding:
#   EMBED_SKIP_PROCESS=true EMBED_DATASETS=IMAGENET EMBED_ARCHS=mobilenet_v2 \
#       bash EmbedAndEvaluate.sh
# =============================================================================

set -euo pipefail

# When submitted via sbatch, SLURM copies this script to /var/spool/slurmd/... so
# BASH_SOURCE[0] is wrong.  SLURM_SUBMIT_DIR always reflects where sbatch was called —
# use the same pattern as every run.sh in this project.
PROJ_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

source "${PROJ_DIR}/env.sh"

echo "============================================================"
echo " ECC-CODE-Engine  —  Embed + Evaluate Chain Submission"
echo " PROJ_DIR           : ${PROJ_DIR}"
echo " EMBED_DATASETS     : ${EMBED_DATASETS}"
echo " EMBED_ARCHS        : ${EMBED_ARCHS}"
echo " EMBED_APPROACH     : ${EMBED_APPROACH}  codeword=${EMBED_CODEWORD}"
echo " EMBED_RUN_CPP      : ${EMBED_RUN_CPP}"
echo " EMBED_SKIP_PROCESS : ${EMBED_SKIP_PROCESS}"
echo "============================================================"
echo ""

# ---- Stage 4: 4-EmbeddingECC (skipped when EMBED_SKIP_PROCESS=true) ----
dep5=""
if [ "${EMBED_SKIP_PROCESS}" = "true" ]; then
    echo "4-EmbeddingECC  SKIPPED  (EMBED_SKIP_PROCESS=true)"
    echo "  → Using ECC chunks from a prior run in ${EMBEDDED_ECC_CHUNKS_DIR}"
else
    job4_out=$(cd "${PROJ_DIR}/4-EmbeddingECC" && sbatch run.sh)
    job4=$(echo "${job4_out}" | awk '{print $4}')
    echo "4-EmbeddingECC  submitted → job ${job4}"
    dep5="--dependency=afterok:${job4}"
fi

# ---- Stage 5: 5-EmbeddingsMerging ----
if [ -n "${dep5}" ]; then
    job5_out=$(cd "${PROJ_DIR}/5-EmbeddingsMerging" && sbatch "${dep5}" run.sh)
else
    job5_out=$(cd "${PROJ_DIR}/5-EmbeddingsMerging" && sbatch run.sh)
fi
job5=$(echo "${job5_out}" | awk '{print $4}')
if [ -n "${dep5}" ]; then
    echo "5-EmbeddingsMerging     submitted → job ${job5}  (after ${job4})"
else
    echo "5-EmbeddingsMerging     submitted → job ${job5}"
fi

# ---- Stage 6: 6-BaseAccuracyTesting (depends on Stage 5) ----
job6_out=$(cd "${PROJ_DIR}/6-BaseAccuracyTesting" && sbatch --dependency=afterok:${job5} run.sh)
job6=$(echo "${job6_out}" | awk '{print $4}')
echo "6-BaseAccuracy          submitted → job ${job6}  (after ${job5})"

# ---- Stage 7: 7-SensitiveAccuracyTesting (depends on Stage 5, parallel with 6) ----
job7_out=$(cd "${PROJ_DIR}/7-SensitiveAccuracyTesting" && sbatch --dependency=afterok:${job5} run.sh)
job7=$(echo "${job7_out}" | awk '{print $4}')
echo "7-SensitiveAccuracy     submitted → job ${job7}  (after ${job5})"

echo ""
echo "Pipeline queued:"
if [ "${EMBED_SKIP_PROCESS}" != "true" ]; then
    echo "  4-EmbeddingECC           : ${job4}"
fi
echo "  5-EmbeddingsMerging      : ${job5}"
echo "  6-BaseAccuracyTesting    : ${job6}"
echo "  7-SensitiveAccuracyTesting: ${job7}"
echo ""
echo "Monitor : squeue -u \$USER"
echo "Logs    : tail -f ${PROJ_DIR}/5-EmbeddingsMerging/logs/ecc-merge.${job5}_*.out"
if [ "${EMBED_SKIP_PROCESS}" != "true" ]; then
    echo "Cancel  : scancel ${job4} ${job5} ${job6} ${job7}"
else
    echo "Cancel  : scancel ${job5} ${job6} ${job7}"
fi
