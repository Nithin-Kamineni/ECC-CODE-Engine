#!/bin/bash
# =============================================================================
# SLURM submit script — 2-Sensitivity
#
# Usage:
#   cd 2-Sensitivity && sbatch run.sh
#
# What it does:
#   For each combination of $DATASETS × $ARCHS × $QUANT_LEVELS, runs:
#     Step 1 — Per-weight sensitivity analysis   (sensitivity.py)
#     Step 2 — Layer-then-weight selection       (layer_then_weight.py)
#
#   QUANT_LEVELS controls which bit-widths are analysed (default: 32 16 8 4).
#   32 = float32 baseline (no quantization applied).
#   16 / 8 / 4 = INT16 / INT8 / INT4 PTQ (applied in-memory via test-Quantizer).
#
#   Output directory per (dataset, arch, level):
#     0-Data/artifacts/sensitivity/{dataset}/{arch}/PTQ/float32/   (for level 32)
#     0-Data/artifacts/sensitivity/{dataset}/{arch}/PTQ/16-bit/    (for level 16)
#     0-Data/artifacts/sensitivity/{dataset}/{arch}/PTQ/8-bit/     (for level  8)
#     0-Data/artifacts/sensitivity/{dataset}/{arch}/PTQ/4-bit/     (for level  4)
#
#   CIFAR10 / CIFAR100: loads trained float32 checkpoint from 0-Data/artifacts/models/.
#   IMAGENET:           uses pretrained torchvision weights (--use-pretrained 1).
#
# Overrides (set before sbatch):
#   DATASETS="CIFAR10"          # run a single dataset
#   ARCHS="resnet18"            # run a single architecture
#   QUANT_LEVELS="32 8"         # run only float32 and 8-bit
# =============================================================================

#SBATCH --job-name=ecc-sensitivity
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# ---- Banner ----
date; hostname; pwd
mkdir -p logs

# ---- Load global environment ----
# Must submit from the script's own folder: cd 2-Sensitivity && sbatch run.sh
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/../env.sh"

# ---- Load Singularity ----
module load singularity

echo "[2-Sensitivity/run.sh] SIF=${SIF}"
echo "[2-Sensitivity/run.sh] DATASETS=${DATASETS}"
echo "[2-Sensitivity/run.sh] ARCHS=${ARCHS}"
echo "[2-Sensitivity/run.sh] QUANT_LEVELS=${QUANT_LEVELS}"
echo "[2-Sensitivity/run.sh] SENSITIVITY_DIR=${SENSITIVITY_DIR}"

# ---- Helper: run python inside container ----
run_py() {
    singularity exec --nv --bind /blue "${SIF}" \
        python3 "$@"
}

# ---- Iterate over every dataset × arch × quantization level ----
for DS in $DATASETS; do

    # Extra flags needed for ImageNet
    PRETRAINED_FLAG=""
    IMGNET_FLAG=""
    if [ "$DS" = "IMAGENET" ]; then
        PRETRAINED_FLAG="--use-pretrained 1"
        IMGNET_FLAG="--imagenet-root ${IMAGENET_ROOT}"
        MAX_B=4   # fewer batches for ImageNet (larger images)
    else
        MAX_B="${MAX_BATCHES}"
    fi

    for ARC in $ARCHS; do
        DS_LOWER="${DS,,}"   # CIFAR10 → cifar10

        echo "========================================================"
        echo "[2-Sensitivity] Dataset: ${DS}  Arch: ${ARC}"
        echo "========================================================"

        for BITS in $QUANT_LEVELS; do

            # Map quantization level to label, tag, and --quantize-bits flag
            if [ "${BITS}" = "32" ]; then
                LABEL="float32"
                TAG="float32"
                QBITS_FLAG=""      # no quantization — run float32 baseline only
            else
                LABEL="${BITS}-bit"
                TAG="int${BITS}"
                QBITS_FLAG="--quantize-bits ${BITS}"
            fi

            SENS_OUT="${SENSITIVITY_DIR}/${DS_LOWER}/${ARC}/PTQ/${LABEL}"
            mkdir -p "${SENS_OUT}"

            echo "  [Level ${BITS}] label=${LABEL}  out=${SENS_OUT}"

            # ------------------------------------------------------------------
            # Step 1: Per-weight sensitivity (sensitivity.py)
            # ------------------------------------------------------------------
            echo "  [Step 1] sensitivity.py  ds=${DS}  arch=${ARC}  level=${LABEL}"
            run_py "${SCRIPT_DIR}/sensitivity.py" \
                --dataset    "${DS}" \
                --arch       "${ARC}" \
                --data-root  "${DATASET_DIR}" \
                --out-dir    "${SENS_OUT}" \
                --methods magnitude grad_abs taylor fisher \
                --max-batches "${MAX_B}" \
                ${QBITS_FLAG} \
                ${PRETRAINED_FLAG} ${IMGNET_FLAG}

            # ------------------------------------------------------------------
            # Step 2: Layer-then-weight (layer_then_weight.py)
            #         Called per-arch so each arch routes to its own subdir.
            # ------------------------------------------------------------------
            echo "  [Step 2] layer_then_weight.py  ds=${DS}  arch=${ARC}  level=${LABEL}"
            run_py "${SCRIPT_DIR}/layer_then_weight.py" \
                --dataset       "${DS}" \
                --archs         "${ARC}" \
                --data-root     "${DATASET_DIR}" \
                --out-dir       "${SENS_OUT}" \
                --top-layers    "${TOP_LAYERS}" \
                --top-per-layer "${TOP_PER_LAYER}" \
                --layer-metric  "${LAYER_METRIC}" \
                --max-batches   "${MAX_B}" \
                ${QBITS_FLAG} \
                ${PRETRAINED_FLAG} ${IMGNET_FLAG}

        done  # BITS
    done  # ARC
done  # DS

echo "[2-Sensitivity/run.sh] All datasets done. Exit code $?"
