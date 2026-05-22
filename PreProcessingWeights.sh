#!/bin/bash
# =============================================================================
# PreProcessingWeights.sh — ECC-CODE-Engine pipeline chain submission
#
# Submits the three preprocessing stages as dependent SLURM jobs:
#   1-Quantization  →  2-Sensitivity  →  3-PatternFinder
#
# Each stage starts only after the previous one exits successfully (afterok).
#
# Usage:
#   bash PreProcessingWeights.sh
#
# Environment overrides (set before running):
#   SKIP_TRAIN=true           skip training if float32 checkpoints already exist
#   DATASETS="CIFAR10"        restrict to one dataset
#   ARCHS="resnet18"          restrict to one architecture
#   QUANT_LEVELS="32 8"       restrict to float32 and 8-bit levels
#   QUANTIZE_BITS="8"         restrict PTQ to 8-bit only
#
# Examples:
#   # Full pipeline from scratch:
#   bash PreProcessingWeights.sh
#
#   # Skip re-training (models already exist), run all quant levels:
#   SKIP_TRAIN=true bash PreProcessingWeights.sh
#
#   # Quick smoke test — CIFAR10, resnet18, float32 + 8-bit only:
#   SKIP_TRAIN=true DATASETS=CIFAR10 ARCHS=resnet18 QUANT_LEVELS="32 8" QUANTIZE_BITS=8 \
#       bash PreProcessingWeights.sh
# =============================================================================

set -euo pipefail

# When submitted via sbatch, SLURM copies this script to /var/spool/slurmd/... so
# BASH_SOURCE[0] is wrong.  SLURM_SUBMIT_DIR always reflects where sbatch was called —
# use the same pattern as every run.sh in this project.
# Usage: sbatch PreProcessingWeights.sh   (from the project root)
#        bash   PreProcessingWeights.sh   (from anywhere)
PROJ_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

echo "=============================================="
echo " ECC-CODE-Engine  —  Pipeline Chain Submission"
echo " PROJ_DIR: ${PROJ_DIR}"
echo "=============================================="
echo ""

# ---- Stage 1: 1-Quantization ----
job1_out=$(cd "${PROJ_DIR}/1-Quantization" && sbatch run.sh)
job1=$(echo "${job1_out}" | awk '{print $4}')
echo "1-Quantization  submitted → job ${job1}"

# ---- Stage 2: 2-Sensitivity (depends on Stage 1) ----
job2_out=$(cd "${PROJ_DIR}/2-Sensitivity" && sbatch --dependency=afterok:${job1} run.sh)
job2=$(echo "${job2_out}" | awk '{print $4}')
echo "2-Sensitivity   submitted → job ${job2}  (after ${job1})"

# ---- Stage 3: 3-PatternFinder (depends on Stage 2) ----
job3_out=$(cd "${PROJ_DIR}/3-PatternFinder" && sbatch --dependency=afterok:${job2} run.sh)
job3=$(echo "${job3_out}" | awk '{print $4}')
echo "3-PatternFinder submitted → job ${job3}  (after ${job2})"

echo ""
echo "Pipeline queued:"
echo "  1-Quantization  : ${job1}"
echo "  2-Sensitivity   : ${job2}"
echo "  3-PatternFinder : ${job3}"
echo ""
echo "Monitor : squeue -u \$USER"
echo "Logs    : tail -f ${PROJ_DIR}/1-Quantization/logs/ecc-quantize.${job1}.out"
echo "Cancel  : scancel ${job1} ${job2} ${job3}"
