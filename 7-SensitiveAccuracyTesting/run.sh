#!/bin/bash
# =============================================================================
# SLURM submit script — 7-SensitiveAccuracyTesting
#
# Usage:
#   cd 7-SensitiveAccuracyTesting && sbatch run.sh
#
# What it does:
#   Same as stage 6, but before inference selectively replaces ECC-embedded
#   weights with original int8 weights for layers that are small AND highly
#   sensitive (see sensitive_test_accuracy.py for the selection logic).
#   Writes:
#
#     accuracy_results/{ds}/{arch}/PTQ/{bit}/M{cw}_t{tval}/{approach}/
#         sensitive_accuracy.json
#
#   --array=1,2,4,6 creates 4 simultaneous jobs, one per t-value.
#
# Tunable protection parameters (override via env before sbatch):
#   NUMEL_THRESHOLD   max numel for a layer to be eligible (default: 5000)
#   SCORE_PERCENTILE  grad_norm percentile threshold, 0-1 (default: 0.90)
#   MAX_PROTECT       max fraction of total weights to replace (default: 0.02)
#
# Run AFTER 5-EmbeddingsMerging has finished.
# =============================================================================

#SBATCH --job-name=ecc-sens-acc
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --array=1,2,4,6
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err

# ---- Banner ----
date; hostname; pwd
mkdir -p logs

# ---- Load global environment ----
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/../env.sh"

# ---- Load Singularity ----
module load singularity

# ---- T-value from SLURM array task ID ----
T_VALUE="${SLURM_ARRAY_TASK_ID}"

# ---- Output directory for accuracy results ----
RESULTS_DIR="${ARTIFACTS_DIR}/accuracy_results"

# ---- Protection parameters (tunable) ----
NUMEL_THRESHOLD="${NUMEL_THRESHOLD:-5000}"
SCORE_PERCENTILE="${SCORE_PERCENTILE:-0.90}"
MAX_PROTECT="${MAX_PROTECT:-0.02}"

echo "[7-SensitiveAccuracyTesting/run.sh] SIF=${SIF}"
echo "[7-SensitiveAccuracyTesting/run.sh] T_VALUE=${T_VALUE}  (SLURM array task)"
echo "[7-SensitiveAccuracyTesting/run.sh] EMBED_DATASETS=${EMBED_DATASETS}"
echo "[7-SensitiveAccuracyTesting/run.sh] EMBED_ARCHS=${EMBED_ARCHS}"
echo "[7-SensitiveAccuracyTesting/run.sh] EMBED_QUANT_BITS=${EMBED_QUANT_BITS}"
echo "[7-SensitiveAccuracyTesting/run.sh] EMBED_APPROACH=${EMBED_APPROACH}  EMBED_CODEWORD=${EMBED_CODEWORD}"
echo "[7-SensitiveAccuracyTesting/run.sh] NUMEL_THRESHOLD=${NUMEL_THRESHOLD}  SCORE_PERCENTILE=${SCORE_PERCENTILE}  MAX_PROTECT=${MAX_PROTECT}"
echo "[7-SensitiveAccuracyTesting/run.sh] RESULTS_DIR=${RESULTS_DIR}"

# ---- Loop over all dataset × arch × quant-bits combinations ----
for DS in $EMBED_DATASETS; do
    for ARC in $EMBED_ARCHS; do
        for BITS in $EMBED_QUANT_BITS; do
            DS_LOWER="${DS,,}"
            BIT_LABEL="${BITS}-bit"
            ECC_MODEL="${EMBEDDED_ECC_DIR}/${DS_LOWER}/${ARC}/PTQ/${BIT_LABEL}/M${EMBED_CODEWORD}_t${T_VALUE}/${EMBED_APPROACH}/ECC_Embedded_model.pth"

            if [ ! -f "${ECC_MODEL}" ]; then
                echo "[skip] ECC model not found: ${ECC_MODEL}"
                continue
            fi

            echo "========================================================"
            echo "[7-SensitiveAccuracyTesting] ${DS}/${ARC}/${BIT_LABEL}/t=${T_VALUE}"
            echo "========================================================"

            IMAGENET_ARG=""
            if [ "${DS}" = "IMAGENET" ]; then
                IMAGENET_ARG="--imagenet-root ${IMAGENET_ROOT}"
            fi

            srun --cpu-bind=cores --mem-bind=local \
                singularity exec \
                    --nv \
                    --bind /blue \
                    "${SIF}" \
                    python3 "${SCRIPT_DIR}/sensitive_test_accuracy.py" \
                        --dataset           "${DS}" \
                        --arch              "${ARC}" \
                        --quant-bits        "${BITS}" \
                        --t-value           "${T_VALUE}" \
                        --approach          "${EMBED_APPROACH}" \
                        --codeword          "${EMBED_CODEWORD}" \
                        --ecc-dir           "${EMBEDDED_ECC_DIR}" \
                        --models-dir        "${MODELS_DIR}" \
                        --sensitivity-dir   "${SENSITIVITY_DIR}" \
                        --results-dir       "${RESULTS_DIR}" \
                        --numel-threshold   "${NUMEL_THRESHOLD}" \
                        --score-percentile  "${SCORE_PERCENTILE}" \
                        --max-protect       "${MAX_PROTECT}" \
                        ${IMAGENET_ARG}

            echo "[7-SensitiveAccuracyTesting] ${DS}/${ARC}/${BIT_LABEL}/t=${T_VALUE} done (exit $?)"
        done
    done
done

echo "[7-SensitiveAccuracyTesting/run.sh] Array task ${T_VALUE} complete."
