#!/bin/bash
# =============================================================================
# SLURM submit script — 4-EmbeddingECC
#
# Usage:
#   cd 4-EmbeddingECC && sbatch run.sh
#
# What it does:
#   Runs ECC embedding for all EMBED_DATASETS × EMBED_ARCHS × EMBED_QUANT_BITS
#   combinations for the t-value assigned to this SLURM array task.
#
#   --array=1,2,4,6 creates 4 simultaneous jobs, one per t-value.
#   Each job loops over all dataset/arch/quant-bits combinations.
#
#   Input:  0-Data/artifacts/patterns/{ds}/{arch}/PTQ/{bit}/pattern_manifest.json
#           0-Data/artifacts/patterns/{ds}/{arch}/PTQ/{bit}/{layer}_weights_perm.npy
#
#   Output: 0-Data/artifacts/embeddedECC_Chunks/{ds}/{arch}/PTQ/{bit}/
#               M{codeword}_t{tval}/{approach}/{layer}/chunks_p{p}.jsonl
#
# Overrides (pass via env before sbatch):
#   EMBED_DATASETS="CIFAR10"        restrict to one dataset
#   EMBED_ARCHS="resnet18"          restrict to one architecture
#   EMBED_QUANT_BITS="8"            restrict to one quant level
#   EMBED_APPROACH="parfit"         change ECC approach
#   EMBED_CODEWORD="63"             change codeword length (M value)
#   EMBED_WORKERS="16"              number of parallel workers
# =============================================================================

#SBATCH --job-name=ecc-embed
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --array=1,2,4,6
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err

# ---- Banner ----
date; hostname; pwd
mkdir -p logs

# ---- Load global environment ----
# Must submit from the script's own folder: cd 4-EmbeddingECC && sbatch run.sh
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/../env.sh"

# ---- Load Singularity ----
module load singularity

# ---- T-value from SLURM array task ID ----
T_VALUE="${SLURM_ARRAY_TASK_ID}"

echo "[4-EmbeddingECC/run.sh] SIF=${SIF}"
echo "[4-EmbeddingECC/run.sh] T_VALUE=${T_VALUE}  (SLURM array task)"
echo "[4-EmbeddingECC/run.sh] EMBED_DATASETS=${EMBED_DATASETS}"
echo "[4-EmbeddingECC/run.sh] EMBED_ARCHS=${EMBED_ARCHS}"
echo "[4-EmbeddingECC/run.sh] EMBED_QUANT_BITS=${EMBED_QUANT_BITS}"
echo "[4-EmbeddingECC/run.sh] EMBED_APPROACH=${EMBED_APPROACH}  EMBED_CODEWORD=${EMBED_CODEWORD}"
echo "[4-EmbeddingECC/run.sh] EMBED_RUN_CPP=${EMBED_RUN_CPP}"
echo "[4-EmbeddingECC/run.sh] EMBEDDED_ECC_CHUNKS_DIR=${EMBEDDED_ECC_CHUNKS_DIR}"

# ---- Decide: Python or C++ runner ----
# C++ is used when EMBED_RUN_CPP=true AND approach is search3, greedy, or no.
CPP_DIR="${SCRIPT_DIR}/c++"
CPP_BINARY="${CPP_DIR}/ecc_embed_cpp"
CPP_SIF="${CPP_DIR}/ecc_cpp.sif"

_CPP_APPROACH=false
if [ "${EMBED_APPROACH}" = "search3" ] || [ "${EMBED_APPROACH}" = "greedy" ] || \
   [ "${EMBED_APPROACH}" = "no" ]; then
    _CPP_APPROACH=true
fi

USE_CPP=false
if [ "${EMBED_RUN_CPP}" = "true" ] && [ "${_CPP_APPROACH}" = "true" ]; then
    if [ ! -f "${CPP_SIF}" ]; then
        echo "[4-EmbeddingECC] ERROR: C++ SIF not found at ${CPP_SIF}"
        echo "  Build it first: singularity build --fakeroot ${CPP_SIF} ${CPP_DIR}/ecc_embed.def"
        exit 1
    fi
    # Auto-compile if binary is missing
    if [ ! -f "${CPP_BINARY}" ]; then
        echo "[4-EmbeddingECC] Compiling C++ binary (first run) ..."
        singularity exec --bind /blue "${CPP_SIF}" make -C "${CPP_DIR}" -j4
        if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
    fi
    USE_CPP=true
    echo "[4-EmbeddingECC] Using C++ runner: ${CPP_BINARY}"
else
    echo "[4-EmbeddingECC] Using Python runner: ecc_embed.py"
fi

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
            echo "[4-EmbeddingECC] ${DS} / ${ARC} / ${BIT_LABEL} / t=${T_VALUE} ($([ "${USE_CPP}" = "true" ] && echo "C++" || echo "Python"))"
            echo "========================================================"

            if [ "${USE_CPP}" = "true" ]; then
                # ---- C++ runner ----
                srun --cpu-bind=cores --mem-bind=local \
                    singularity exec \
                        --bind /blue \
                        "${CPP_SIF}" \
                        "${CPP_BINARY}" \
                            --dataset       "${DS}" \
                            --arch          "${ARC}" \
                            --quant-bits    "${BITS}" \
                            --t-value       "${T_VALUE}" \
                            --approach      "${EMBED_APPROACH}" \
                            --codeword      "${EMBED_CODEWORD}" \
                            --workers       "${EMBED_WORKERS}" \
                            --patterns-dir  "${PATTERNS_DIR}" \
                            --chunks-dir    "${EMBEDDED_ECC_CHUNKS_DIR}" \
                            --sensitivity-dir "${SENSITIVITY_DIR}"
            else
                # ---- Python runner ----
                srun --cpu-bind=cores --mem-bind=local \
                    singularity exec \
                        --nv \
                        --bind /blue \
                        "${SIF}" \
                        python3 "${SCRIPT_DIR}/ecc_embed.py" \
                            --dataset       "${DS}" \
                            --arch          "${ARC}" \
                            --quant-bits    "${BITS}" \
                            --t-value       "${T_VALUE}" \
                            --approach      "${EMBED_APPROACH}" \
                            --codeword      "${EMBED_CODEWORD}" \
                            --workers       "${EMBED_WORKERS}" \
                            --patterns-dir  "${PATTERNS_DIR}" \
                            --chunks-dir    "${EMBEDDED_ECC_CHUNKS_DIR}" \
                            --sensitivity-dir "${SENSITIVITY_DIR}"
            fi

            echo "[4-EmbeddingECC] ${DS}/${ARC}/${BIT_LABEL}/t=${T_VALUE} done (exit $?)"
        done
    done
done

echo "[4-EmbeddingECC/run.sh] Array task ${T_VALUE} complete."
