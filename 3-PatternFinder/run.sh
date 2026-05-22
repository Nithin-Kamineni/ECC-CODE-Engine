#!/bin/bash
# =============================================================================
# SLURM submit script — 3-PatternFinder
#
# Usage:
#   cd 3-PatternFinder && sbatch run.sh
#
# What it does:
#   For every combination of $DATASETS × $ARCHS × $QUANT_LEVELS, finds the
#   best hardware-friendly interleaver pattern and saves:
#     - {layer}_perm.npy        : permutation indices
#     - {layer}_inv_perm.npy    : inverse permutation (for weight recovery)
#     - {layer}_weights_perm.npy: actual weights in pattern order (if checkpoint available)
#     - pattern_manifest.json   : full metadata for all layers
#     - pattern_search_summary.csv
#
#   Output directories are separated by dataset / arch / PTQ / quantization level:
#     0-Data/artifacts/patterns/{dataset}/{arch}/PTQ/float32/
#     0-Data/artifacts/patterns/{dataset}/{arch}/PTQ/16-bit/
#     0-Data/artifacts/patterns/{dataset}/{arch}/PTQ/8-bit/
#     0-Data/artifacts/patterns/{dataset}/{arch}/PTQ/4-bit/
#
#   The sensitivity CSV read for each level must already exist (produced by 2-Sensitivity).
#   Missing CSVs are skipped gracefully.
#
#   For weight permutation, the matching checkpoint is used:
#     - float32 level : models/{ds}/{arch}/model_float32.pth
#     - N-bit  level  : models/{ds}/{arch}/PTQ/model_intN_ptq.pth
#   All datasets (including IMAGENET) now have saved checkpoints from 1-Quantization.
#   If the checkpoint file is absent, perm/inv_perm are still computed; weights skipped.
#
# Overrides (set before sbatch):
#   DATASETS="CIFAR10"      # restrict to one dataset
#   ARCHS="resnet18"        # restrict to one architecture
#   QUANT_LEVELS="32 8"     # restrict to float32 and 8-bit levels
# =============================================================================

#SBATCH --job-name=ecc-patterns
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

echo "[3-PatternFinder/run.sh] SIF=${SIF}"
echo "[3-PatternFinder/run.sh] DATASETS=${DATASETS}"
echo "[3-PatternFinder/run.sh] ARCHS=${ARCHS}"
echo "[3-PatternFinder/run.sh] QUANT_LEVELS=${QUANT_LEVELS}"
echo "[3-PatternFinder/run.sh] GROUP_SIZE=${GROUP_SIZE}  MAX_SENS=${MAX_SENS}  TOP_SENSITIVE=${TOP_SENSITIVE}  SENS_THRESHOLD=${SENS_THRESHOLD}"
echo "[3-PatternFinder/run.sh] PATTERNS_DIR=${PATTERNS_DIR}"

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
            else
                LABEL="${BITS}-bit"
                TAG="int${BITS}"
                MODEL_PATH="${MODELS_DIR}/${DS_LOWER}/${ARC}/PTQ/model_int${BITS}_ptq.pth"
            fi

            # Sensitivity CSV produced by 2-Sensitivity for this level
            CSV="${SENSITIVITY_DIR}/${DS_LOWER}/${ARC}/PTQ/${LABEL}/layer_then_weight_${DS_LOWER}_${ARC}_${TAG}_L${TOP_LAYERS}xN${TOP_PER_LAYER}_${LAYER_METRIC}.csv"

            # Skip if sensitivity CSV hasn't been produced yet
            if [ ! -f "${CSV}" ]; then
                echo "[skip] CSV not found: ${CSV}"
                continue
            fi

            # Per-combo output directory
            OUT_DIR="${PATTERNS_DIR}/${DS_LOWER}/${ARC}/PTQ/${LABEL}"
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

            # Choose full interleaver search or identity-permutation bypass
            if [ "${DISABLE_PATTERN_FIND}" = "true" ]; then
                SEARCH_FLAG="--identity-perm"
            else
                SEARCH_FLAG="--run-search"
            fi

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
                    --threshold     "${SENS_THRESHOLD}"

            echo "[3-PatternFinder] ${DS}/${ARC}/${LABEL} done (exit $?)"

        done  # BITS
    done  # ARC
done  # DS

echo "[3-PatternFinder/run.sh] All combinations complete."
