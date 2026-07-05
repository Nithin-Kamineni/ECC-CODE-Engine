#!/bin/bash
# =============================================================================
# run_weight_distortion.sh — Ablation: weight distortion vs BER, accuracy vs BER
#
# Works as both a plain bash script and an sbatch job:
#   bash   run_weight_distortion.sh
#   sbatch run_weight_distortion.sh          # call from the script's directory
#
# Override which combination to plot (defaults shown):
#   PLOT_DATASET="IMAGENET"                          bash/sbatch run_weight_distortion.sh
#   PLOT_ARCH="mobilenet_v2"
#   PLOT_QMODE="QAT"                                 # PTQ | QAT
#   PLOT_QUANT_BITS="8"                              # 4 | 6 | 8 | 16
#   PLOT_APPROACHES="search3 greedy parfix replace"  # space-separated list
#   PLOT_CODEWORD="63"
#
# Example — subset of methods:
#   PLOT_APPROACHES="search3 greedy" bash run_weight_distortion.sh
# =============================================================================

#SBATCH --job-name=weight_distortion_plot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/Ablasian-Studies/Weight-distortion/logs/slurm_%j.out
#SBATCH --error=/blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/Ablasian-Studies/Weight-distortion/logs/slurm_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve the real script directory.
# Under sbatch, BASH_SOURCE[0] points to the Slurm spool copy
# (/var/spool/slurmd/.../slurm_script), not the original file.
# SLURM_SUBMIT_DIR is set by Slurm to the directory where 'sbatch' was called,
# so run sbatch from the same directory as this script.
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset — run sbatch from the script directory}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Ensure the logs directory exists before Slurm tries to open --output/--error
mkdir -p "${SCRIPT_DIR}/logs"

source "${SCRIPT_DIR}/../../env.sh"

PLOT_DATASET="${PLOT_DATASET:-IMAGENET}"
PLOT_ARCH="${PLOT_ARCH:-efficientnet_b0}"
PLOT_QMODE="${PLOT_QMODE:-QAT}"
PLOT_QUANT_BITS="${PLOT_QUANT_BITS:-8}"
PLOT_APPROACHES="${PLOT_APPROACHES:-search3 greedy parfix replace}"
PLOT_CODEWORD="${PLOT_CODEWORD:-63}"

echo "[run_weight_distortion.sh] PLOT_DATASET=${PLOT_DATASET}  PLOT_ARCH=${PLOT_ARCH}"
echo "[run_weight_distortion.sh] PLOT_QMODE=${PLOT_QMODE}  PLOT_QUANT_BITS=${PLOT_QUANT_BITS}"
echo "[run_weight_distortion.sh] PLOT_APPROACHES=${PLOT_APPROACHES}  PLOT_CODEWORD=${PLOT_CODEWORD}"

# shellcheck disable=SC2086  # word-split of PLOT_APPROACHES is intentional
singularity exec \
    --bind /blue \
    "${SIF}" \
    python3 "${SCRIPT_DIR}/plot_weight_distortion.py" \
        --dataset    "${PLOT_DATASET}" \
        --arch       "${PLOT_ARCH}" \
        --qmode      "${PLOT_QMODE}" \
        --quant-bits "${PLOT_QUANT_BITS}" \
        --approaches ${PLOT_APPROACHES} \
        --codeword   "${PLOT_CODEWORD}"
