#!/bin/bash
# =============================================================================
# SLURM submit script — 6-BaseAccuracyTesting
#
# Usage:
#   cd 6-BaseAccuracyTesting && sbatch run.sh
#
# What it does:
#   For every combination of EMBED_DATASETS × EMBED_ARCHS × EMBED_QUANT_BITS,
#   evaluates the Top-1/Top-5 accuracy of the ECC-embedded model and writes:
#
#     accuracy_results/{ds}/{arch}/PTQ/{bit}/M{cw}_t{tval}/{approach}/
#         accuracy.json
#
#   --array=1,2,4,6 creates 4 simultaneous jobs, one per t-value.
#   Each job loops over all dataset/arch/quant-bits combinations.
#
# Run AFTER 5-EmbeddingsMerging has finished (ECC_Embedded_model.pth present).
#
# Overrides (same as stages 4 and 5):
#   EMBED_DATASETS / EMBED_ARCHS / EMBED_QUANT_BITS
#   EMBED_APPROACH / EMBED_CODEWORD
# =============================================================================

#SBATCH --job-name=6-ecc-base-acc
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

echo "[6-BaseAccuracyTesting/run.sh] SIF=${SIF}"
echo "[6-BaseAccuracyTesting/run.sh] T_VALUE=${T_VALUE}  (SLURM array task)"
echo "[6-BaseAccuracyTesting/run.sh] EMBED_DATASETS=${EMBED_DATASETS}"
echo "[6-BaseAccuracyTesting/run.sh] EMBED_ARCHS=${EMBED_ARCHS}"
echo "[6-BaseAccuracyTesting/run.sh] EMBED_QUANT_BITS=${EMBED_QUANT_BITS}"
echo "[6-BaseAccuracyTesting/run.sh] EMBED_APPROACH=${EMBED_APPROACH}  EMBED_CODEWORD=${EMBED_CODEWORD}"
echo "[6-BaseAccuracyTesting/run.sh] EMBEDDED_ECC_DIR=${EMBEDDED_ECC_DIR}"
echo "[6-BaseAccuracyTesting/run.sh] RESULTS_DIR=${RESULTS_DIR}"

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
            echo "[6-BaseAccuracyTesting] ${DS}/${ARC}/${BIT_LABEL}/t=${T_VALUE}"
            echo "========================================================"

            IMAGENET_ARG=""
            if [ "${DS}" = "IMAGENET" ]; then
                IMAGENET_ARG="--imagenet-root ${IMAGENET_ROOT}"
            fi

            singularity exec \
                --nv \
                --bind /blue \
                "${SIF}" \
                python3 "${SCRIPT_DIR}/test_accuracy.py" \
                    --dataset       "${DS}" \
                    --arch          "${ARC}" \
                    --quant-bits    "${BITS}" \
                    --t-value       "${T_VALUE}" \
                    --approach      "${EMBED_APPROACH}" \
                    --codeword      "${EMBED_CODEWORD}" \
                    --ecc-dir       "${EMBEDDED_ECC_DIR}" \
                    --results-dir   "${RESULTS_DIR}" \
                    ${IMAGENET_ARG}

            echo "[6-BaseAccuracyTesting] ${DS}/${ARC}/${BIT_LABEL}/t=${T_VALUE} done (exit $?)"
        done
    done
done

echo "[6-BaseAccuracyTesting/run.sh] Array task ${T_VALUE} complete."
