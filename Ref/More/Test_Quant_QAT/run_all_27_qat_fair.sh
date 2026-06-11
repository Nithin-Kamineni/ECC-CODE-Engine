#!/bin/bash
# run_all_27_qat_fair.sh
# ========================
# Run all 9 archs x 3 bit-widths = 27 QAT jobs with FAIR Float32 baseline
# (i.e., the Float32 baseline is also fine-tuned on the same 40K subset).
# Splits the 27 jobs across 2 GPUs (CUDA_VISIBLE_DEVICES=0 and 1).
#
# Usage:
#   chmod +x run_all_27_qat_fair.sh
#   ./run_all_27_qat_fair.sh
#
# After this finishes, run bit-flip validations with --qtag qat.

set -u  # error on undefined variables (but NOT on command failures: we want chain to continue)

# --------------------------- Common config ---------------------------
DATASET="IMAGENET"
IMAGENET_ROOT="./imagenet-val"
QAT_EPOCHS=3
FP32_FT_EPOCHS=3
LR=1e-4
WARMUP=1

# --------------------------- Helper to run one job ---------------------------
# Args: gpu, arch, bits, batch_size, extra_flags
run_one() {
    local GPU=$1
    local ARCH=$2
    local BITS=$3
    local BS=$4
    local EXTRA=${5:-""}
    local TAG=$(echo "$ARCH" | tr '[:upper:]' '[:lower:]')
    local LOG="qat_${TAG}_int${BITS}_fair.log"

    echo ">>> [GPU ${GPU}] starting ${ARCH} INT${BITS}  (log: ${LOG})"
    CUDA_VISIBLE_DEVICES=${GPU} python qat_train.py \
        --dataset ${DATASET} \
        --imagenet-root ${IMAGENET_ROOT} \
        --arch ${ARCH} \
        --bits ${BITS} \
        --epochs ${QAT_EPOCHS} \
        --float32-finetune-epochs ${FP32_FT_EPOCHS} \
        --lr ${LR} \
        --batch-size ${BS} \
        --use-pretrained 1 \
        --warmup-epochs ${WARMUP} \
        ${EXTRA} \
        2>&1 | tee "${LOG}"
    echo ">>> [GPU ${GPU}] done ${ARCH} INT${BITS}"
}

# --------------------------- GPU 0 chain (5 archs) ---------------------------
gpu0_chain() {
    run_one 0 resnet18      8 128 ""
    run_one 0 resnet18      6 128 ""
    run_one 0 resnet18      4 128 ""
    run_one 0 resnet50      8 128 ""
    run_one 0 resnet50      6 128 ""
    run_one 0 resnet50      4 128 ""
    run_one 0 resnet152     8  64 ""
    run_one 0 resnet152     6  64 ""
    run_one 0 resnet152     4  64 ""
    run_one 0 densenet121   8 128 ""
    run_one 0 densenet121   6 128 ""
    run_one 0 densenet121   4 128 ""
    run_one 0 squeezenet1_1 8 128 ""
    run_one 0 squeezenet1_1 6 128 ""
    run_one 0 squeezenet1_1 4 128 ""
}

# --------------------------- GPU 1 chain (4 archs) ---------------------------
gpu1_chain() {
    run_one 1 mobilenet_v2    8 128 ""
    run_one 1 mobilenet_v2    6 128 ""
    run_one 1 mobilenet_v2    4 128 ""
    run_one 1 efficientnet_b0 8 128 ""
    run_one 1 efficientnet_b0 6 128 ""
    run_one 1 efficientnet_b0 4 128 ""
    run_one 1 convnext_tiny   8  64 ""
    run_one 1 convnext_tiny   6  64 ""
    run_one 1 convnext_tiny   4  64 ""
    run_one 1 xception        8  64 "--skip-first-last 1"
    run_one 1 xception        6  64 "--skip-first-last 1"
    run_one 1 xception        4  64 "--skip-first-last 1"
}

# --------------------------- Launch in parallel ---------------------------
echo "============================================================"
echo " Launching 27 fair-baseline QAT runs (GPU0: 15 jobs, GPU1: 12 jobs)"
echo " Estimated total wall time: ~1.5 - 2 hours"
echo "============================================================"

gpu0_chain &
PID0=$!
echo "GPU 0 chain started, PID=${PID0}"

gpu1_chain &
PID1=$!
echo "GPU 1 chain started, PID=${PID1}"

wait ${PID0}
echo "*** GPU 0 chain FINISHED ***"

wait ${PID1}
echo "*** GPU 1 chain FINISHED ***"

echo ""
echo "============================================================"
echo " ALL 27 FAIR-BASELINE QAT JOBS DONE"
echo "============================================================"
echo "Next step: run bit-flip validations with --qtag qat"
