#!/bin/bash
# =============================================================================
# SLURM submit script — 1-Quantization (training + PTQ)
#
# Usage:
#   sbatch 1-Quantization/run.sh
#
# What it does (two passes):
#   Pass 1 — Train each architecture on CIFAR10 and CIFAR100.
#             ImageNet is skipped; pretrained torchvision weights are used instead.
#             If SKIP_TRAIN=true and model_float32.pth already exists, training
#             is skipped for that (dataset, arch) pair.
#
#   Pass 2 — Quantize every (dataset, arch) pair to each bit-width in
#             QUANTIZE_BITS (default: "8 4").
#             For CIFAR: loads the trained model_float32.pth and quantizes it.
#             For ImageNet: pulls pretrained torchvision weights (--use-pretrained 1),
#             saves model_float32.pth, then quantizes.
#
#   All model checkpoints are written to 0-Data/artifacts/models/.
#
# Overrides (pass via env before sbatch):
#   DATASETS="CIFAR10"              # restrict to one dataset
#   ARCHS="resnet18"                # restrict to one architecture
#   EPOCHS=50                       # shorten training
#   SKIP_TRAIN=true                 # skip training if float32 checkpoint exists
#   QUANTIZE_BITS="8"               # quantize to 8-bit only
# =============================================================================

#SBATCH --job-name=1-ecc-quantize
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# ---- Banner ----
date; hostname; pwd
mkdir -p logs

# ---- Load global environment ----
# SLURM copies this script to /var/spool/slurmd/... so BASH_SOURCE[0] is wrong.
# SLURM_SUBMIT_DIR is always the directory where sbatch was called — use that.
# Must submit from the script's own folder: cd 1-Quantization && sbatch run.sh
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/../env.sh"

# ---- Load Singularity ----
module load singularity

EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.1}"

echo "[1-Quantization/run.sh] SIF=${SIF}"
echo "[1-Quantization/run.sh] DATASETS=${DATASETS}"
echo "[1-Quantization/run.sh] ARCHS=${ARCHS}"
echo "[1-Quantization/run.sh] ARTIFACTS_DIR=${ARTIFACTS_DIR}"
echo "[1-Quantization/run.sh] QUANTIZE_BITS=${QUANTIZE_BITS}"
echo "[1-Quantization/run.sh] SKIP_TRAIN=${SKIP_TRAIN}"
echo "[1-Quantization/run.sh] QAT_ENABLED=${QAT_ENABLED}"

# =============================================================================
# Pass 1 — Train each arch on each trainable dataset (CIFAR10, CIFAR100)
# =============================================================================
echo ""
echo "=== Pass 1: Training ==="
for DS in $DATASETS; do
    if [ "$DS" = "IMAGENET" ]; then
        echo "[1-Quantization] Skipping training for IMAGENET (pretrained torchvision weights used)"
        continue
    fi
    for ARC in $ARCHS; do
        FLOAT32_PATH="${MODELS_DIR}/${DS,,}/${ARC,,}/model_float32.pth"
        if [ "${SKIP_TRAIN}" = "true" ] && [ -f "${FLOAT32_PATH}" ]; then
            echo "[1-Quantization] SKIP_TRAIN=true: ${ARC} on ${DS} already trained (${FLOAT32_PATH}), skipping."
        else
            echo "[1-Quantization] Training ${ARC} on ${DS} ..."
            singularity exec \
                --nv \
                --bind /blue \
                "${SIF}" \
                python3 "${SCRIPT_DIR}/test-Quantizer.py" \
                    --data-root       "${DATASET_DIR}" \
                    --artifacts-root  "${ARTIFACTS_DIR}" \
                    --workers         "${WORKERS}" \
                    train \
                    --dataset         "${DS}" \
                    --arch            "${ARC}" \
                    --epochs          "${EPOCHS}" \
                    --lr              "${LR}" \
                    --optim sgd --momentum 0.9 \
                    --weight-decay 5e-4 --scheduler cosine --warmup-epochs 5 \
                    --label-smoothing 0.1
        fi
    done
done

# =============================================================================
# Pass 2 — PTQ: Quantize every (dataset, arch) pair to each bit-width
#           Runs when QAT_ENABLED=false (default).
#           CIFAR: loads model_float32.pth trained above
#           ImageNet: pulls pretrained torchvision weights, saves float32 + quantized
#
# Pass 3 — QAT: Fine-tune with fake-quantization for every (dataset, arch, bits).
#           Runs when QAT_ENABLED=true.
#           Outputs: 0-Data/artifacts/models/{dataset}/{arch}/QAT/model_int{N}_qat.pth
# =============================================================================
echo ""
if [ "${QAT_ENABLED}" = "true" ]; then
    echo "=== Pass 3: QAT (QAT_ENABLED=true) ==="
    echo "[1-Quantization/run.sh] QAT_EPOCHS=${QAT_EPOCHS}  QAT_LR=${QAT_LR}  QAT_SKIP_FIRST_LAST=${QAT_SKIP_FIRST_LAST}"
    for DS in $DATASETS; do
        for ARC in $ARCHS; do
            # Per-arch overrides (env.sh) fall back to the global BATCH_SIZE /
            # QAT_SKIP_FIRST_LAST above for any arch not listed there.
            ARC_BATCH_SIZE="${ARCH_BATCH_SIZE[$ARC]:-$BATCH_SIZE}"
            ARC_SKIP_FIRST_LAST="${ARCH_SKIP_FIRST_LAST[$ARC]:-$QAT_SKIP_FIRST_LAST}"
            for BITS in $QUANTIZE_BITS; do
                echo "[1-Quantization] QAT fine-tuning ${ARC} on ${DS} @ ${BITS}-bit (batch_size=${ARC_BATCH_SIZE}, skip_first_last=${ARC_SKIP_FIRST_LAST}) ..."
                singularity exec \
                    --nv \
                    --bind /blue \
                    "${SIF}" \
                    python3 "${SCRIPT_DIR}/qat_train.py" \
                        --data-root       "${DATASET_DIR}" \
                        --artifacts-root  "${ARTIFACTS_DIR}" \
                        --imagenet-root   "${IMAGENET_ROOT}" \
                        --use-pretrained  "$([ "$DS" = "IMAGENET" ] && echo 1 || echo 0)" \
                        --dataset         "${DS}" \
                        --arch            "${ARC}" \
                        --bits            "${BITS}" \
                        --epochs          "${QAT_EPOCHS}" \
                        --lr              "${QAT_LR}" \
                        --skip-first-last "${ARC_SKIP_FIRST_LAST}" \
                        --batch-size      "${ARC_BATCH_SIZE}" \
                        --workers         "${WORKERS}"
            done
        done
    done
else
    echo "=== Pass 2: PTQ (QAT_ENABLED=false) ==="
    for DS in $DATASETS; do
        for ARC in $ARCHS; do
            for BITS in $QUANTIZE_BITS; do
                echo "[1-Quantization] Quantizing ${ARC} on ${DS} @ ${BITS}-bit ..."
                if [ "$DS" = "IMAGENET" ]; then
                    singularity exec \
                        --nv \
                        --bind /blue \
                        "${SIF}" \
                        python3 "${SCRIPT_DIR}/test-Quantizer.py" \
                            --data-root       "${DATASET_DIR}" \
                            --artifacts-root  "${ARTIFACTS_DIR}" \
                            --imagenet-root   "${IMAGENET_ROOT}" \
                            --use-pretrained  1 \
                            quantize \
                            --dataset         "${DS}" \
                            --arch            "${ARC}" \
                            --bits            "${BITS}"
                else
                    singularity exec \
                        --nv \
                        --bind /blue \
                        "${SIF}" \
                        python3 "${SCRIPT_DIR}/test-Quantizer.py" \
                            --data-root       "${DATASET_DIR}" \
                            --artifacts-root  "${ARTIFACTS_DIR}" \
                            quantize \
                            --dataset         "${DS}" \
                            --arch            "${ARC}" \
                            --bits            "${BITS}"
                fi
            done
        done
    done
fi

echo ""
echo "[1-Quantization/run.sh] Job finished with exit code $?"
