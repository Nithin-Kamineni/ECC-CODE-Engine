#!/bin/bash
# =============================================================================
# SLURM submit script — 5-EmbeddingsMerging
#
# Usage:
#   cd 5-EmbeddingsMerging && sbatch run.sh
#
# What it does:
#   For every combination of EMBED_DATASETS × EMBED_ARCHS × EMBED_QUANT_BITS
#   × t-values, merges the per-worker JSONL chunk files written by
#   4-EmbeddingECC, fills any gaps by re-encoding missing ranges, applies
#   inv_perm to restore original weight ordering, and saves:
#
#     embeddedECC/{ds}/{arch}/PTQ/{bit}/M{codeword}_t{tval}/{approach}/
#         ECC_Embedded_model.pth
#
# Run AFTER 4-EmbeddingECC has finished (all array tasks complete).
#
# Overrides:
#   EMBED_DATASETS / EMBED_ARCHS / EMBED_QUANT_BITS — same as 4-EmbeddingECC
#   EMBED_T_VALUES — space-separated list (default: "1 2 4 6")
#   EMBED_APPROACH / EMBED_CODEWORD — must match what was used in 4-EmbeddingECC
# =============================================================================

#SBATCH --job-name=5-ecc-merge
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
# Must submit from the script's own folder: cd 5-EmbeddingsMerging && sbatch run.sh
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/../env.sh"

# ---- Load Singularity ----
module load singularity

# ---- T-value from SLURM array task ID ----
T_VALUE="${SLURM_ARRAY_TASK_ID}"

# ---- Path to 4-EmbeddingECC for ECC tool imports ----
ECC_SOURCE="${SCRIPT_DIR}/../4-EmbeddingECC"

echo "[5-EmbeddingsMerging/run.sh] SIF=${SIF}"
echo "[5-EmbeddingsMerging/run.sh] T_VALUE=${T_VALUE}  (SLURM array task)"
echo "[5-EmbeddingsMerging/run.sh] EMBED_DATASETS=${EMBED_DATASETS}"
echo "[5-EmbeddingsMerging/run.sh] EMBED_ARCHS=${EMBED_ARCHS}"
echo "[5-EmbeddingsMerging/run.sh] EMBED_QUANT_BITS=${EMBED_QUANT_BITS}"
echo "[5-EmbeddingsMerging/run.sh] EMBED_APPROACH=${EMBED_APPROACH}  EMBED_CODEWORD=${EMBED_CODEWORD}"
echo "[5-EmbeddingsMerging/run.sh] EMBEDDED_ECC_DIR=${EMBEDDED_ECC_DIR}"
echo "[5-EmbeddingsMerging/run.sh] EMBEDDED_ECC_CHUNKS_DIR=${EMBEDDED_ECC_CHUNKS_DIR}"

# ---- Loop over all dataset × arch × quant-bits combinations ----
for DS in $EMBED_DATASETS; do
    for ARC in $EMBED_ARCHS; do
        for BITS in $EMBED_QUANT_BITS; do
            DS_LOWER="${DS,,}"
            BIT_LABEL="${BITS}-bit"
            MANIFEST="${PATTERNS_DIR}/${DS_LOWER}/${ARC}/PTQ/${BIT_LABEL}/pattern_manifest.json"

            if [ ! -f "${MANIFEST}" ]; then
                echo "[skip] No manifest: ${MANIFEST}"
                continue
            fi

            echo "========================================================"
            echo "[5-EmbeddingsMerging] ${DS}/${ARC}/${BIT_LABEL}/t=${T_VALUE}"
            echo "========================================================"

            singularity exec \
                --nv \
                --bind /blue \
                "${SIF}" \
                python3 "${SCRIPT_DIR}/merge_ecc.py" \
                    --dataset       "${DS}" \
                    --arch          "${ARC}" \
                    --quant-bits    "${BITS}" \
                    --t-value       "${T_VALUE}" \
                    --approach      "${EMBED_APPROACH}" \
                    --codeword      "${EMBED_CODEWORD}" \
                    --patterns-dir  "${PATTERNS_DIR}" \
                    --chunks-dir    "${EMBEDDED_ECC_CHUNKS_DIR}" \
                    --ecc-dir       "${EMBEDDED_ECC_DIR}" \
                    --models-dir    "${MODELS_DIR}" \
                    --ecc-source    "${ECC_SOURCE}"

            echo "[5-EmbeddingsMerging] ${DS}/${ARC}/${BIT_LABEL}/t=${T_VALUE} done (exit $?)"
        done
    done
done

echo "[5-EmbeddingsMerging/run.sh] Array task ${T_VALUE} complete."
