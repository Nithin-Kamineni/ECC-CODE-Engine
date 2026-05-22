#!/bin/bash
# =============================================================================
# ECC Resilience — full sensitivity pipeline over all datasets & architectures.
#
# THIS SCRIPT RUNS INSIDE THE SINGULARITY CONTAINER launched by run.sh:
#
#     singularity exec --nv $SIF bash run_all.sh
#
# Every python3 call uses the container's PyTorch/CUDA automatically.
#
# For each dataset in $DATASETS it runs:
#   1. Layer-then-weight selection  (layer_then_weight.py — all archs at once)
#   2. Float-side bit-flip          (validate_bitflip.py  — all archs at once)
#   3. Integer-side bit-flip        (validate_bitflip_int.py — all archs at once)
#
# All outputs → $SENSITIVITY_DIR  (0-Data/artifacts/sensitivity/)
# Trained models read from → $MODELS_DIR  (0-Data/artifacts/models/)
# =============================================================================

set -e
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/../env.sh"

echo "[run_all] starting at $(date)"
echo "[run_all] DATASETS:        ${DATASETS}"
echo "[run_all] ARCHS:           ${ARCHS}"
echo "[run_all] SENSITIVITY_DIR: ${SENSITIVITY_DIR}"
python3 -c "import torch; print(f'[run_all] torch {torch.__version__} | cuda {torch.cuda.is_available()} | devices {torch.cuda.device_count()}')"

for DS in $DATASETS; do

    if [ "$DS" = "IMAGENET" ]; then
        PRETRAINED_FLAG="--use-pretrained 1"
        IMGNET_FLAG="--imagenet-root ${IMAGENET_ROOT}"
        MAX_B=4
    else
        PRETRAINED_FLAG=""
        IMGNET_FLAG=""
        MAX_B="${MAX_BATCHES}"
    fi

    echo "============================================================"
    echo "[run_all] Dataset: ${DS}"
    echo "============================================================"

    # ---- Layer-then-weight ----
    python3 "${SCRIPT_DIR}/layer_then_weight.py" \
        --dataset       "${DS}" \
        --archs         ${ARCHS} \
        --data-root     "${DATASET_DIR}" \
        --out-dir       "${SENSITIVITY_DIR}" \
        --quantize-bits ${QUANTIZE_BITS} \
        --top-layers    "${TOP_LAYERS}" \
        --top-per-layer "${TOP_PER_LAYER}" \
        --layer-metric  "${LAYER_METRIC}" \
        --max-batches   "${MAX_B}" \
        ${PRETRAINED_FLAG} ${IMGNET_FLAG}

    # ---- Float-side bit-flip ----
    python3 "${SCRIPT_DIR}/validate_bitflip.py" \
        --dataset       "${DS}" \
        --archs         ${ARCHS} \
        --data-root     "${DATASET_DIR}" \
        --in-dir        "${SENSITIVITY_DIR}" \
        --out-dir       "${SENSITIVITY_DIR}" \
        --formats float32 int8 int4 \
        --top-layers    "${TOP_LAYERS}" \
        --top-per-layer "${TOP_PER_LAYER}" \
        --layer-metric  "${LAYER_METRIC}" \
        --k-list 1 5 10 50 100 500 1000 \
        --bit-position sign --random-trials 3 \
        ${PRETRAINED_FLAG} ${IMGNET_FLAG}

    # ---- Integer-side bit-flip ----
    python3 "${SCRIPT_DIR}/validate_bitflip_int.py" \
        --dataset       "${DS}" \
        --archs         ${ARCHS} \
        --data-root     "${DATASET_DIR}" \
        --in-dir        "${SENSITIVITY_DIR}" \
        --out-dir       "${SENSITIVITY_DIR}" \
        --formats float32 int8 int4 \
        --top-layers    "${TOP_LAYERS}" \
        --top-per-layer "${TOP_PER_LAYER}" \
        --layer-metric  "${LAYER_METRIC}" \
        --k-list 1 5 10 50 100 500 1000 \
        --bit-position sign --random-trials 3 \
        ${PRETRAINED_FLAG} ${IMGNET_FLAG} \
        2>&1 | tee "${SENSITIVITY_DIR}/int_${DS,,}.log"

done

echo
echo "============================================================"
echo "  ALL DONE.  Results in ${SENSITIVITY_DIR}"
echo "  Finished at $(date)"
echo "============================================================"
