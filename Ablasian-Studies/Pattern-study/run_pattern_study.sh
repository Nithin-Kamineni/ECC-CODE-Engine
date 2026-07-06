#!/bin/bash
# =============================================================================
# run_pattern_study.sh — Ablation: permutation strategy comparison (multi-model)
#
# Compares three weight permutation strategies for ECC embedding across
# multiple architectures, measuring top-1/top-5 accuracy after full ECC
# embedding (stages 3-7).
#
# Strategies:
#   contiguous — identity permutation (DISABLE_PATTERN_FIND=true)
#   random     — one randomly chosen coprime stride per layer (RANDOM_PATTERN_FIND=true)
#   strided    — exhaustive search for best coprime stride (default)
#
# Works as both a plain bash script and an sbatch job:
#   bash   run_pattern_study.sh
#   sbatch run_pattern_study.sh          (run from this directory)
#
# Override study parameters:
#   STUDY_ARCHS="resnet18"       bash run_pattern_study.sh   # single arch
#   STUDY_T=2                    bash run_pattern_study.sh
#   STUDY_QUANT_BITS=6           bash run_pattern_study.sh
# =============================================================================

#SBATCH --job-name=pattern-study
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/Ablasian-Studies/Pattern-study/logs/slurm_%j.out
#SBATCH --error=/blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/Ablasian-Studies/Pattern-study/logs/slurm_%j.err

set -eo pipefail

# ---------------------------------------------------------------------------
# Resolve real script directory — works for both bash and sbatch
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset — run sbatch from the script directory}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

mkdir -p "${SCRIPT_DIR}/logs"

source "${SCRIPT_DIR}/../../env.sh"

# ---------------------------------------------------------------------------
# Study parameters — all overridable via environment
# ---------------------------------------------------------------------------
STUDY_ARCHS="${STUDY_ARCHS:-efficientnet_b0 mobilenet_v2 resnet18}"
STUDY_DATASET="${STUDY_DATASET:-IMAGENET}"
STUDY_QUANT_BITS="${STUDY_QUANT_BITS:-8}"
STUDY_QMODE_QAT="${STUDY_QMODE_QAT:-true}"
STUDY_T="${STUDY_T:-4}"
STUDY_APPROACH="${STUDY_APPROACH:-search3}"
STUDY_CODEWORD="${STUDY_CODEWORD:-63}"

BASE_RESULTS_DIR="${SCRIPT_DIR}/results"

STAGE3_DIR="${SCRIPT_DIR}/../../3-PatternFinder"
STAGE4_DIR="${SCRIPT_DIR}/../../4-EmbeddingECC"
STAGE5_DIR="${SCRIPT_DIR}/../../5-EmbeddingsMerging"
STAGE6_DIR="${SCRIPT_DIR}/../../6-BaseAccuracyTesting"
STAGE7_DIR="${SCRIPT_DIR}/../../7-SensitiveAccuracyTesting"

# Derived values (same for all archs)
DS_LOWER=$(echo "${STUDY_DATASET}" | tr '[:upper:]' '[:lower:]')
BIT_LABEL="${STUDY_QUANT_BITS}-bit"
QMODE="$([ "${STUDY_QMODE_QAT}" = "true" ] && echo QAT || echo PTQ)"

echo "[run_pattern_study.sh] ================================================="
echo "[run_pattern_study.sh] STUDY_ARCHS=${STUDY_ARCHS}"
echo "[run_pattern_study.sh] STUDY_DATASET=${STUDY_DATASET}  STUDY_QMODE=${QMODE}"
echo "[run_pattern_study.sh] STUDY_QUANT_BITS=${STUDY_QUANT_BITS}"
echo "[run_pattern_study.sh] STUDY_T=${STUDY_T}  STUDY_APPROACH=${STUDY_APPROACH}"
echo "[run_pattern_study.sh] ================================================="

# ---------------------------------------------------------------------------
# Outer loop — one iteration per architecture
# ---------------------------------------------------------------------------
for ARCH in ${STUDY_ARCHS}; do
    STUDY_ARCH="${ARCH}"
    RESULTS_DIR="${BASE_RESULTS_DIR}/${ARCH}"
    mkdir -p "${RESULTS_DIR}"

    echo ""
    echo "[run_pattern_study.sh] ======== arch: ${ARCH} ========"

    # -------------------------------------------------------------------------
    # Per-condition loop
    # -------------------------------------------------------------------------
    for CONDITION in contiguous random strided; do
        SNAP_DIR="${RESULTS_DIR}/${CONDITION}"

        # Idempotent — skip if accuracy already staged for this condition
        if [[ -f "${SNAP_DIR}/accuracy.json" ]]; then
            echo "[skip] ${ARCH}/${CONDITION} already staged"
            continue
        fi

        echo ""
        echo "[run_pattern_study.sh] ---- arch: ${ARCH}  condition: ${CONDITION} ----"

        # Set permutation flags for stage 3
        case "${CONDITION}" in
            contiguous)
                export DISABLE_PATTERN_FIND=true
                export RANDOM_PATTERN_FIND=false
                ;;
            random)
                export DISABLE_PATTERN_FIND=false
                export RANDOM_PATTERN_FIND=true
                ;;
            strided)
                export DISABLE_PATTERN_FIND=false
                export RANDOM_PATTERN_FIND=false
                ;;
        esac

        # Common stage overrides (respected by all stage run.sh scripts via :-default)
        export SLURM_ARRAY_TASK_ID="${STUDY_T}"
        export EMBED_DATASETS="${STUDY_DATASET}"
        export EMBED_ARCHS="${STUDY_ARCH}"
        export EMBED_QUANT_BITS="${STUDY_QUANT_BITS}"
        export EMBED_APPROACH="${STUDY_APPROACH}"
        export EMBED_CODEWORD="${STUDY_CODEWORD}"
        export QAT_ENABLED="${STUDY_QMODE_QAT}"
        export EMBED_SENSITIVITY="true"
        # Stage 3 variables
        export DATASETS="${STUDY_DATASET}"
        export ARCHS="${STUDY_ARCH}"
        export QUANT_LEVELS="${STUDY_QUANT_BITS}"

        # Run stage 3 — pattern finding (unset SLURM_SUBMIT_DIR so stage uses its own path)
        echo "[run_pattern_study.sh] Stage 3 — PatternFinder (${CONDITION})"
        ( unset SLURM_SUBMIT_DIR; bash "${STAGE3_DIR}/run.sh" )

        # Capture histogram BEFORE stage 4 can overwrite the patterns directory
        mkdir -p "${SNAP_DIR}"
        HIST_SRC="${PATTERNS_DIR}/${DS_LOWER}/${STUDY_ARCH}/${QMODE}/${BIT_LABEL}/codeword_histogram.json"
        if [[ -f "${HIST_SRC}" ]]; then
            cp "${HIST_SRC}" "${SNAP_DIR}/codeword_histogram.json"
            echo "[staged] codeword_histogram.json → ${SNAP_DIR}/"
        else
            echo "[warn] codeword_histogram.json not found at ${HIST_SRC}" \
                | tee -a "${SCRIPT_DIR}/logs/missing_data.log"
        fi

        # Run stages 4-7
        echo "[run_pattern_study.sh] Stage 4 — EmbeddingECC"
        ( unset SLURM_SUBMIT_DIR; bash "${STAGE4_DIR}/run.sh" )

        echo "[run_pattern_study.sh] Stage 5 — EmbeddingsMerging"
        ( unset SLURM_SUBMIT_DIR; bash "${STAGE5_DIR}/run.sh" )

        echo "[run_pattern_study.sh] Stage 6 — BaseAccuracyTesting"
        ( unset SLURM_SUBMIT_DIR; bash "${STAGE6_DIR}/run.sh" )

        echo "[run_pattern_study.sh] Stage 7 — SensitiveAccuracyTesting"
        ( unset SLURM_SUBMIT_DIR; bash "${STAGE7_DIR}/run.sh" )

        # Capture accuracy results
        ACC_SRC="${ARTIFACTS_DIR}/accuracy_results/${DS_LOWER}/${STUDY_ARCH}/${QMODE}/${BIT_LABEL}/M${STUDY_CODEWORD}_t${STUDY_T}/${STUDY_APPROACH}"

        if [[ -f "${ACC_SRC}/accuracy.json" ]]; then
            cp "${ACC_SRC}/accuracy.json" "${SNAP_DIR}/accuracy.json"
            echo "[staged] accuracy.json → ${SNAP_DIR}/"
        else
            echo "[warn] accuracy.json not found at ${ACC_SRC}" \
                | tee -a "${SCRIPT_DIR}/logs/missing_data.log"
        fi

        if [[ -f "${ACC_SRC}/sensitive_accuracy.json" ]]; then
            cp "${ACC_SRC}/sensitive_accuracy.json" "${SNAP_DIR}/sensitive_accuracy.json"
            echo "[staged] sensitive_accuracy.json → ${SNAP_DIR}/"
        fi

    done  # condition

    echo ""
    echo "[run_pattern_study.sh] arch ${ARCH} complete."

done  # arch

echo ""
echo "[run_pattern_study.sh] All archs complete. Aggregating accuracy table..."

singularity exec \
    --bind /blue \
    "${SIF}" \
    python3 "${SCRIPT_DIR}/aggregate_accuracy_table.py" \
        --results-dir "${BASE_RESULTS_DIR}" \
        --archs       ${STUDY_ARCHS} \
        --t           "${STUDY_T}" \
        --output      "${BASE_RESULTS_DIR}/all_models_accuracy_table_Patterns.json"

echo "[run_pattern_study.sh] Done. Table: ${BASE_RESULTS_DIR}/all_models_accuracy_table_Patterns.json"
