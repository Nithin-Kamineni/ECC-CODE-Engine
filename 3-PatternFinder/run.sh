#!/bin/bash
# =============================================================================
# SLURM submit script — 3-PatternFinder
#
# Usage:
#   cd 3-PatternFinder && sbatch run.sh
#
# What it does:
#   For every combination of $DATASETS × $ARCHS × $QUANT_LEVELS, finds the
#   top-${TOP_PATTERNS} hardware-friendly interleaver patterns and saves, per
#   layer and per rank (0..K-1):
#     - {layer}_rank{i}_perm.npy        : permutation indices
#     - {layer}_rank{i}_inv_perm.npy    : inverse permutation (for weight recovery)
#     - {layer}_rank{i}_weights_perm.npy: actual weights in pattern order (if checkpoint available)
#     - pattern_manifest.json      : rank-0 metadata for all layers (backward compat)
#     - top_patterns_manifest.json : top-K candidates per layer (used by 4-EmbeddingECC)
#     - pattern_search_summary.csv
#
#   If DISABLE_PATTERN_FIND=true, only rank 0 (identity permutation) is saved.
#
#   Output directories are separated by dataset / arch / {PTQ,QAT} / quantization level:
#     0-Data/artifacts/patterns/{dataset}/{arch}/PTQ/float32/
#     0-Data/artifacts/patterns/{dataset}/{arch}/{PTQ,QAT}/16-bit/
#     0-Data/artifacts/patterns/{dataset}/{arch}/{PTQ,QAT}/8-bit/
#     0-Data/artifacts/patterns/{dataset}/{arch}/{PTQ,QAT}/4-bit/
#   PTQ vs QAT is selected by QAT_ENABLED (mirrors 2-Sensitivity/run.sh).
#
#   The sensitivity CSV read for each level must already exist (produced by 2-Sensitivity).
#   Missing CSVs are skipped gracefully.
#
#   For weight permutation, the matching checkpoint is used:
#     - float32 level : models/{ds}/{arch}/model_float32.pth
#     - N-bit  level  : models/{ds}/{arch}/PTQ/model_intN_ptq.pth (PTQ)
#                       models/{ds}/{arch}/QAT/model_intN_qat.pth (QAT)
#   All datasets (including IMAGENET) now have saved checkpoints from 1-Quantization.
#   If the checkpoint file is absent, perm/inv_perm are still computed; weights skipped.
#
# Overrides (set before sbatch):
#   DATASETS="CIFAR10"      # restrict to one dataset
#   ARCHS="resnet18"        # restrict to one architecture
#   QUANT_LEVELS="32 8"     # restrict to float32 and 8-bit levels
# =============================================================================

#SBATCH --job-name=3-ecc-patterns
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# ---- Banner ----
date; hostname; pwd
mkdir -p logs

# ---- Load global environment ----
# Must submit from the script's own folder: cd 3-PatternFinder && sbatch run.sh
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/../env.sh"

# ---- Load Singularity ----
module load singularity

QMODE="$( [ "${QAT_ENABLED}" = "true" ] && echo QAT || echo PTQ )"

echo "[3-PatternFinder/run.sh] SIF=${SIF}"
echo "[3-PatternFinder/run.sh] DATASETS=${DATASETS}"
echo "[3-PatternFinder/run.sh] ARCHS=${ARCHS}"
echo "[3-PatternFinder/run.sh] QUANT_LEVELS=${QUANT_LEVELS}"
echo "[3-PatternFinder/run.sh] GROUP_SIZE=${GROUP_SIZE}  MAX_SENS=${MAX_SENS}  TOP_SENSITIVE=${TOP_SENSITIVE}  SENS_THRESHOLD=${SENS_THRESHOLD}  MAX_STRIDE=${MAX_STRIDE}"
echo "[3-PatternFinder/run.sh] DISABLE_PATTERN_FIND=${DISABLE_PATTERN_FIND}  RANDOM_PATTERN_FIND=${RANDOM_PATTERN_FIND:-false}"
echo "[3-PatternFinder/run.sh] PATTERNS_DIR=${PATTERNS_DIR}"
echo "[3-PatternFinder/run.sh] QAT_ENABLED=${QAT_ENABLED}  QMODE=${QMODE}  TOP_PATTERNS=${TOP_PATTERNS}"

# ---- Loop over all dataset × arch × quantization level combinations ----
for DS in $DATASETS; do
    for ARC in $ARCHS; do
        DS_LOWER="${DS,,}"   # CIFAR10 → cifar10

        for BITS in $QUANT_LEVELS; do

            # Map level → label (directory name), tag (CSV filename fragment), model path
            if [ "${BITS}" = "32" ]; then
                LABEL="float32"
                TAG="float32"
                MODEL_PATH="${MODELS_DIR}/${DS_LOWER}/${ARC}/model_float32.pth"
            elif [ "${QMODE}" = "QAT" ]; then
                LABEL="${BITS}-bit"
                TAG="int${BITS}"
                MODEL_PATH="${MODELS_DIR}/${DS_LOWER}/${ARC}/QAT/model_int${BITS}_qat.pth"
            else
                LABEL="${BITS}-bit"
                TAG="int${BITS}"
                MODEL_PATH="${MODELS_DIR}/${DS_LOWER}/${ARC}/PTQ/model_int${BITS}_ptq.pth"
            fi

            # Sensitivity CSV produced by 2-Sensitivity for this level
            if [ "${BITS}" = "32" ]; then
                CSV="${SENSITIVITY_DIR}/${DS_LOWER}/${ARC}/PTQ/${LABEL}/layer_then_weight_${DS_LOWER}_${ARC}_${TAG}_L${TOP_LAYERS}xN${TOP_PER_LAYER}_${LAYER_METRIC}.csv"
            else
                CSV="${SENSITIVITY_DIR}/${DS_LOWER}/${ARC}/${QMODE}/${LABEL}/layer_then_weight_${DS_LOWER}_${ARC}_${TAG}_L${TOP_LAYERS}xN${TOP_PER_LAYER}_${LAYER_METRIC}.csv"
            fi

            # Skip if sensitivity CSV hasn't been produced yet
            if [ ! -f "${CSV}" ]; then
                echo "[skip] CSV not found: ${CSV}"
                continue
            fi

            # Per-combo output directory
            if [ "${BITS}" = "32" ]; then
                OUT_DIR="${PATTERNS_DIR}/${DS_LOWER}/${ARC}/PTQ/${LABEL}"
            else
                OUT_DIR="${PATTERNS_DIR}/${DS_LOWER}/${ARC}/${QMODE}/${LABEL}"
            fi
            mkdir -p "${OUT_DIR}"

            # Model checkpoint — use matching checkpoint if available, otherwise skip weights
            MODEL_FLAG=""
            if [ -f "${MODEL_PATH}" ]; then
                MODEL_FLAG="--model-path ${MODEL_PATH}"
            else
                echo "[warn] checkpoint missing: ${MODEL_PATH} — perm files saved, weights skipped"
            fi

            echo "========================================================"
            echo "[3-PatternFinder] ${DS} / ${ARC} / ${LABEL}"
            echo "  CSV:       ${CSV}"
            echo "  OUT_DIR:   ${OUT_DIR}"
            echo "  MODEL:     ${MODEL_FLAG:-<none>}"
            echo "========================================================"

            # Choose search mode: identity bypass, random stride, or full search
            if [ "${DISABLE_PATTERN_FIND}" = "true" ]; then
                SEARCH_FLAG="--identity-perm"
            elif [ "${RANDOM_PATTERN_FIND:-false}" = "true" ]; then
                SEARCH_FLAG="--random-stride"
            else
                SEARCH_FLAG="--run-search"
            fi

            # float32 checkpoints are not quantized — _load_raw_sd_from_qat_ckpt
            # requires meta.scales, so always use --qmode PTQ for the float32 level.
            CKPT_QMODE="$( [ "${BITS}" = "32" ] && echo PTQ || echo "${QMODE}" )"

            singularity exec --nv --bind /blue "${SIF}" \
                python3 "${SCRIPT_DIR}/prepare_patterns.py" \
                    --csv           "${CSV}" \
                    --arch          "${ARC}" \
                    --out-dir       "${OUT_DIR}" \
                    ${MODEL_FLAG} \
                    ${SEARCH_FLAG} \
                    --group-size    "${GROUP_SIZE}" \
                    --max-sens      "${MAX_SENS}" \
                    --top-sensitive "${TOP_SENSITIVE}" \
                    --threshold     "${SENS_THRESHOLD}" \
                    --max-stride    "${MAX_STRIDE}" \
                    --top-k         "${TOP_PATTERNS}" \
                    --qmode         "${CKPT_QMODE}"

            echo "[3-PatternFinder] ${DS}/${ARC}/${LABEL} done (exit $?)"

        done  # BITS
    done  # ARC
done  # DS

echo "[3-PatternFinder/run.sh] All combinations complete."
