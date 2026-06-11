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
#   PTQ vs QAT (QMODE) is selected by QAT_ENABLED, mirroring 2/3.
#
#   For EMBED_APPROACH=search3 (default), each combination runs TWO phases
#   using the top-${TOP_PATTERNS} candidate patterns from 3-PatternFinder:
#     Phase 1 (greedy): scores all candidate patterns per layer by total
#                        greedy distortion, writes best_pattern_selection.json
#     Phase 2 (search3): embeds using each layer's winning pattern
#   Other approaches use the legacy single-pattern (rank 0) flow.
#
#   Input:  0-Data/artifacts/patterns/{ds}/{arch}/{PTQ,QAT}/{bit}/top_patterns_manifest.json
#           0-Data/artifacts/patterns/{ds}/{arch}/{PTQ,QAT}/{bit}/{layer}_rank{i}_weights_perm.npy
#
#   Output: 0-Data/artifacts/best_patterns/{ds}/{arch}/{PTQ,QAT}/{bit}/M{codeword}_t{tval}/
#               best_pattern_selection.json
#           0-Data/artifacts/embeddedECC_Chunks/{ds}/{arch}/{PTQ,QAT}/{bit}/
#               M{codeword}_t{tval}/{approach}/{layer}/chunks_p{p}.jsonl
#
# Overrides (pass via env before sbatch):
#   EMBED_DATASETS="CIFAR10"        restrict to one dataset
#   EMBED_ARCHS="resnet18"          restrict to one architecture
#   EMBED_QUANT_BITS="8"            restrict to one quant level
#   EMBED_APPROACH="parfit"         change ECC approach
#   EMBED_CODEWORD="63"             change codeword length (M value)
#   EMBED_WORKERS="8"              number of parallel workers
# =============================================================================

#SBATCH --job-name=4-ecc-embed
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
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

# ---- PTQ vs QAT (mirrors 2-Sensitivity/run.sh and 3-PatternFinder/run.sh) ----
QMODE="$( [ "${QAT_ENABLED}" = "true" ] && echo QAT || echo PTQ )"

echo "[4-EmbeddingECC/run.sh] SIF=${SIF}"
echo "[4-EmbeddingECC/run.sh] T_VALUE=${T_VALUE}  (SLURM array task)"
echo "[4-EmbeddingECC/run.sh] EMBED_DATASETS=${EMBED_DATASETS}"
echo "[4-EmbeddingECC/run.sh] EMBED_ARCHS=${EMBED_ARCHS}"
echo "[4-EmbeddingECC/run.sh] EMBED_QUANT_BITS=${EMBED_QUANT_BITS}"
echo "[4-EmbeddingECC/run.sh] EMBED_APPROACH=${EMBED_APPROACH}  EMBED_CODEWORD=${EMBED_CODEWORD}"
echo "[4-EmbeddingECC/run.sh] EMBED_RUN_CPP=${EMBED_RUN_CPP}"
echo "[4-EmbeddingECC/run.sh] QAT_ENABLED=${QAT_ENABLED}  QMODE=${QMODE}"
echo "[4-EmbeddingECC/run.sh] TOP_PATTERNS=${TOP_PATTERNS}  BEST_PATTERNS_DIR=${BEST_PATTERNS_DIR}"
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

# ---- If using C++, export the BCH parity matrix from Python (galois.BCH) ----
# The C++ scratch implementation uses a different primitive polynomial than
# Python's galois library, producing an incompatible parity matrix.  We export
# the galois.BCH matrix once per (codeword, t) pair and pass it to the binary.
BCH_PARITY_FILE=""
if [ "${USE_CPP}" = "true" ]; then
    BCH_PARITY_FILE="/tmp/bch_P_${EMBED_CODEWORD}_t${T_VALUE}.npy"
    if [ ! -f "${BCH_PARITY_FILE}" ]; then
        echo "[4-EmbeddingECC] Exporting BCH(${EMBED_CODEWORD}, t=${T_VALUE}) parity matrix from Python ..."
        singularity exec \
            --nv \
            --bind /blue \
            "${SIF}" \
            python3 "${CPP_DIR}/export_parity_matrix.py" \
                --n      "${EMBED_CODEWORD}" \
                --t      "${T_VALUE}" \
                --output "${BCH_PARITY_FILE}"
        if [ $? -ne 0 ]; then
            echo "[4-EmbeddingECC] ERROR: Failed to export BCH parity matrix"
            exit 1
        fi
    else
        echo "[4-EmbeddingECC] Reusing cached parity matrix: ${BCH_PARITY_FILE}"
    fi
fi

# ---- Sensitivity kill-switch (EMBED_SENSITIVITY=false → all weights = 1.0) ----
SENS_FLAG=""
if [ "${EMBED_SENSITIVITY}" != "true" ]; then
    SENS_FLAG="--no-sensitivity"
    echo "[4-EmbeddingECC] EMBED_SENSITIVITY=${EMBED_SENSITIVITY} → sensitivity disabled (all weights = 1.0)"
else
    echo "[4-EmbeddingECC] EMBED_SENSITIVITY=true → using sensitivity weights"
fi

# ---- If using C++ with sensitivity enabled, export Python sensitivity arrays ----
# Produces <layer>_sens_py.npy files (continuous Taylor scores, normalized to [0.5,1.0])
# so C++ uses the same values as Python instead of the binary {0.5,1.0} from _sens.npy.
# Files are cached — only re-exported if missing (or delete them to force re-export).
if [ "${USE_CPP}" = "true" ] && [ "${EMBED_SENSITIVITY}" = "true" ]; then
    echo "[4-EmbeddingECC] Exporting Python sensitivity arrays for C++ ..."
    for DS in $EMBED_DATASETS; do
        for ARC in $EMBED_ARCHS; do
            for BITS in $EMBED_QUANT_BITS; do
                singularity exec \
                    --nv \
                    --bind /blue \
                    "${SIF}" \
                    python3 "${CPP_DIR}/export_sensitivity.py" \
                        --dataset         "${DS}" \
                        --arch            "${ARC}" \
                        --quant-bits      "${BITS}" \
                        --patterns-dir    "${PATTERNS_DIR}" \
                        --sensitivity-dir "${SENSITIVITY_DIR}" \
                        --qmode           "${QMODE}"
                if [ $? -ne 0 ]; then
                    echo "[4-EmbeddingECC] ERROR: export_sensitivity.py failed for ${DS}/${ARC}/${BITS}-bit"
                    exit 1
                fi
            done
        done
    done
    echo "[4-EmbeddingECC] Sensitivity export complete."
fi

# ---- Helper: run one ecc_embed invocation (C++ runner) ----
# $1 = approach; remaining args are appended verbatim (e.g. top-K pattern flags)
run_embed_cpp() {
    local approach="$1"; shift
    singularity exec \
        --bind /blue \
        "${CPP_SIF}" \
        "${CPP_BINARY}" \
            --dataset         "${DS}" \
            --arch            "${ARC}" \
            --quant-bits      "${BITS}" \
            --t-value         "${T_VALUE}" \
            --approach        "${approach}" \
            --codeword        "${EMBED_CODEWORD}" \
            --workers         "${EMBED_WORKERS}" \
            --patterns-dir    "${PATTERNS_DIR}" \
            --chunks-dir      "${EMBEDDED_ECC_CHUNKS_DIR}" \
            --sensitivity-dir "${SENSITIVITY_DIR}" \
            --parity-matrix   "${BCH_PARITY_FILE}" \
            --qmode           "${QMODE}" \
            ${SENS_FLAG} \
            "$@"
}

# ---- Helper: run one ecc_embed invocation (Python runner) ----
run_embed_py() {
    local approach="$1"; shift
    singularity exec \
        --nv \
        --bind /blue \
        "${SIF}" \
        python3 "${SCRIPT_DIR}/ecc_embed.py" \
            --dataset       "${DS}" \
            --arch          "${ARC}" \
            --quant-bits    "${BITS}" \
            --t-value       "${T_VALUE}" \
            --approach      "${approach}" \
            --codeword      "${EMBED_CODEWORD}" \
            --workers       "${EMBED_WORKERS}" \
            --patterns-dir  "${PATTERNS_DIR}" \
            --chunks-dir    "${EMBEDDED_ECC_CHUNKS_DIR}" \
            --sensitivity-dir "${SENSITIVITY_DIR}" \
            --qmode         "${QMODE}" \
            ${SENS_FLAG} \
            "$@"
}

# ---- Loop over all dataset × arch × quant-bits combinations ----
for DS in $EMBED_DATASETS; do
    for ARC in $EMBED_ARCHS; do
        for BITS in $EMBED_QUANT_BITS; do
            DS_LOWER="${DS,,}"
            BIT_LABEL="${BITS}-bit"

            # search3 uses the top-K pattern flow (top_patterns_manifest.json);
            # all other approaches use the legacy single-pattern manifest.
            if [ "${EMBED_APPROACH}" = "search3" ]; then
                MANIFEST="${PATTERNS_DIR}/${DS_LOWER}/${ARC}/${QMODE}/${BIT_LABEL}/top_patterns_manifest.json"
            else
                MANIFEST="${PATTERNS_DIR}/${DS_LOWER}/${ARC}/${QMODE}/${BIT_LABEL}/pattern_manifest.json"
            fi

            if [ ! -f "${MANIFEST}" ]; then
                echo "[skip] No manifest: ${MANIFEST}"
                continue
            fi

            echo "========================================================"
            echo "[4-EmbeddingECC] ${DS} / ${ARC} / ${QMODE} / ${BIT_LABEL} / t=${T_VALUE} ($([ "${USE_CPP}" = "true" ] && echo "C++" || echo "Python"))"
            echo "========================================================"

            TOPK_ARGS=(--top-patterns-dir "${PATTERNS_DIR}" --best-patterns-dir "${BEST_PATTERNS_DIR}")

            if [ "${USE_CPP}" = "true" ]; then
                # ---- C++ runner ----
                if [ "${EMBED_APPROACH}" = "search3" ]; then
                    echo "[4-EmbeddingECC] Phase 1/2: greedy selection across top-${TOP_PATTERNS} patterns"
                    run_embed_cpp greedy "${TOPK_ARGS[@]}"
                    echo "[4-EmbeddingECC] Phase 2/2: search3 with best pattern"
                    run_embed_cpp search3 "${TOPK_ARGS[@]}"
                else
                    run_embed_cpp "${EMBED_APPROACH}"
                fi
            else
                # ---- Python runner ----
                if [ "${EMBED_APPROACH}" = "search3" ]; then
                    echo "[4-EmbeddingECC] Phase 1/2: greedy selection across top-${TOP_PATTERNS} patterns"
                    run_embed_py greedy "${TOPK_ARGS[@]}"
                    echo "[4-EmbeddingECC] Phase 2/2: search3 with best pattern"
                    run_embed_py search3 "${TOPK_ARGS[@]}"
                else
                    run_embed_py "${EMBED_APPROACH}"
                fi
            fi

            echo "[4-EmbeddingECC] ${DS}/${ARC}/${QMODE}/${BIT_LABEL}/t=${T_VALUE} done (exit $?)"
        done
    done
done

echo "[4-EmbeddingECC/run.sh] Array task ${T_VALUE} complete."
