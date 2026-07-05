#!/bin/bash
# =============================================================================
# run_sensitivity_sweep.sh — Sensitivity study Part 2: accuracy vs SENS_NORM_MIN
#
# Sweeps SENS_NORM_MIN from 0.0 to 1.0 (or any subset), running stages 4-7 for
# each value and staging the accuracy results before the next iteration
# overwrites the artifact paths. After the sweep, plots both Top-1 accuracy
# lines (base and sensitive) against SENS_NORM_MIN.
#
# Works as both a plain bash script and an sbatch job:
#   bash   run_sensitivity_sweep.sh                  (run from this directory)
#   sbatch run_sensitivity_sweep.sh                  (run from this directory!)
#
# Override study parameters:
#   STUDY_T=4                                        bash run_sensitivity_sweep.sh
#   STUDY_ARCHS="mobilenet_v2 resnet18"              bash run_sensitivity_sweep.sh
#   SENS_NORM_VALUES="0.0 0.5 1.0" bash run_sensitivity_sweep.sh   (subset/test)
#
# NOTE: always call sbatch from THIS directory so SLURM_SUBMIT_DIR is correct.
# =============================================================================

#SBATCH --job-name=sens-norm-sweep
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/Ablasian-Studies/Sensitivity-study/2-graph/logs/slurm_%j.out
#SBATCH --error=/blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/Ablasian-Studies/Sensitivity-study/2-graph/logs/slurm_%j.err

set -eo pipefail

# ---------------------------------------------------------------------------
# Resolve real script directory — works for both bash and sbatch
# Under sbatch, BASH_SOURCE[0] is the spool copy; use SLURM_SUBMIT_DIR instead.
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset — run sbatch from the script directory}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

mkdir -p "${SCRIPT_DIR}/logs"

source "${SCRIPT_DIR}/../../../env.sh"

# ---------------------------------------------------------------------------
# Study parameters — all overridable via environment
# ---------------------------------------------------------------------------
STUDY_T="${STUDY_T:-4}"                              # BCH t-value (BER level)
# Space-separated list of architectures to sweep (each gets its own plot)
STUDY_ARCHS="${STUDY_ARCHS:-efficientnet_b0 mobilenet_v2 resnet18}"
STUDY_DATASET="${STUDY_DATASET:-IMAGENET}"
STUDY_QMODE_QAT="${STUDY_QMODE_QAT:-true}"          # true=QAT, false=PTQ
STUDY_QUANT_BITS="${STUDY_QUANT_BITS:-8}"
STUDY_APPROACH="${STUDY_APPROACH:-search3}"
STUDY_CODEWORD="${STUDY_CODEWORD:-63}"
# Space-separated list of SENS_NORM_MIN values to sweep
SENS_NORM_VALUES="${SENS_NORM_VALUES:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
# SENS_NORM_VALUES="${SENS_NORM_VALUES:-0.0 0.5}"

RESULTS_DIR="${SCRIPT_DIR}/results"
STAGE4_DIR="${SCRIPT_DIR}/../../../4-EmbeddingECC"
STAGE5_DIR="${SCRIPT_DIR}/../../../5-EmbeddingsMerging"
STAGE6_DIR="${SCRIPT_DIR}/../../../6-BaseAccuracyTesting"
STAGE7_DIR="${SCRIPT_DIR}/../../../7-SensitiveAccuracyTesting"

# Derived values
DS_LOWER=$(echo "${STUDY_DATASET}" | tr '[:upper:]' '[:lower:]')
BIT_LABEL="${STUDY_QUANT_BITS}-bit"
QMODE="$([ "${STUDY_QMODE_QAT}" = "true" ] && echo QAT || echo PTQ)"

echo "[run_sensitivity_sweep.sh] ============================================="
echo "[run_sensitivity_sweep.sh] STUDY_DATASET=${STUDY_DATASET}  STUDY_ARCHS=${STUDY_ARCHS}"
echo "[run_sensitivity_sweep.sh] STUDY_QMODE=${QMODE}  STUDY_QUANT_BITS=${STUDY_QUANT_BITS}"
echo "[run_sensitivity_sweep.sh] STUDY_T=${STUDY_T}  STUDY_APPROACH=${STUDY_APPROACH}"
echo "[run_sensitivity_sweep.sh] SENS_NORM_VALUES=${SENS_NORM_VALUES}"
echo "[run_sensitivity_sweep.sh] ============================================="

mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------
for SNS_VAL in ${SENS_NORM_VALUES}; do
    SNS_KEY=$(echo "${SNS_VAL}" | tr '.' '_')   # "0.1" → "0_1" (filesystem-safe)
    SNAPSHOT_DIR="${RESULTS_DIR}/sens_norm_${SNS_KEY}"

    # Idempotent — skip if every arch already has a staged accuracy.json for this SNS_VAL
    _all_staged=true
    for _arc in ${STUDY_ARCHS}; do
        if [[ ! -f "${SNAPSHOT_DIR}/${_arc}/accuracy.json" ]]; then
            _all_staged=false
            break
        fi
    done
    if [[ "${_all_staged}" = "true" ]]; then
        echo "[skip] SENS_NORM_MIN=${SNS_VAL} — all archs already staged in ${SNAPSHOT_DIR}"
        continue
    fi

    echo ""
    echo "[run_sensitivity_sweep.sh] ---- SENS_NORM_MIN=${SNS_VAL} ----"

    # Export all overrides before calling each stage's run.sh as bash.
    # The :-default syntax in env.sh respects these exports, so they take
    # precedence over env.sh defaults without modifying env.sh itself.
    export SLURM_ARRAY_TASK_ID="${STUDY_T}"   # stage scripts: T_VALUE="${SLURM_ARRAY_TASK_ID}"
    export EMBED_DATASETS="${STUDY_DATASET}"
    export EMBED_ARCHS="${STUDY_ARCHS}"
    export EMBED_QUANT_BITS="${STUDY_QUANT_BITS}"
    export EMBED_APPROACH="${STUDY_APPROACH}"
    export EMBED_CODEWORD="${STUDY_CODEWORD}"
    export QAT_ENABLED="${STUDY_QMODE_QAT}"
    export EMBED_SENSITIVITY="true"           # must be true for SENS_NORM_MIN to matter
    export SENS_NORM_MIN="${SNS_VAL}"
    # Force export_sensitivity.py to re-export _sens_py.npy / _sens_dense_py.npy
    # on every iteration — the default caches them and would reuse stale values
    # from the first SENS_NORM_MIN run for all subsequent iterations.
    export SENS_FORCE_EXPORT="true"

    # Run each stage as plain bash.  We unset SLURM_SUBMIT_DIR in a subshell
    # so that stage scripts which use the pattern
    #   SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
    # fall back to BASH_SOURCE[0] (their own real path) instead of inheriting
    # the sweep job's SLURM_SUBMIT_DIR (which points to 2-graph/, not the stage dir).
    # All exported variables (EMBED_ARCHS, SENS_NORM_MIN, etc.) are still visible
    # inside the subshell because they are in the environment, not just local vars.

    echo "[run_sensitivity_sweep.sh] Stage 4 — EmbeddingECC"
    ( unset SLURM_SUBMIT_DIR; bash "${STAGE4_DIR}/run.sh" )

    echo "[run_sensitivity_sweep.sh] Stage 5 — EmbeddingsMerging"
    ( unset SLURM_SUBMIT_DIR; bash "${STAGE5_DIR}/run.sh" )

    echo "[run_sensitivity_sweep.sh] Stage 6 — BaseAccuracyTesting"
    ( unset SLURM_SUBMIT_DIR; bash "${STAGE6_DIR}/run.sh" )

    echo "[run_sensitivity_sweep.sh] Stage 7 — SensitiveAccuracyTesting"
    ( unset SLURM_SUBMIT_DIR; bash "${STAGE7_DIR}/run.sh" )

    # Stage accuracy results for each arch before the next iteration overwrites them.
    # Each arch gets its own subdirectory: sens_norm_{key}/{arch}/
    for ARC in ${STUDY_ARCHS}; do
        ARC_SNAPSHOT="${SNAPSHOT_DIR}/${ARC}"
        mkdir -p "${ARC_SNAPSHOT}"
        ACC_SRC="${ARTIFACTS_DIR}/accuracy_results/${DS_LOWER}/${ARC}/${QMODE}/${BIT_LABEL}/M${STUDY_CODEWORD}_t${STUDY_T}/${STUDY_APPROACH}"

        if [[ -f "${ACC_SRC}/accuracy.json" ]]; then
            cp "${ACC_SRC}/accuracy.json" "${ARC_SNAPSHOT}/accuracy.json"
            echo "[staged] ${ARC} accuracy.json → ${ARC_SNAPSHOT}/"
        else
            echo "[warn] ${ARC} accuracy.json not found at ${ACC_SRC}" \
                | tee -a "${SCRIPT_DIR}/logs/missing_data.log"
        fi

        if [[ -f "${ACC_SRC}/sensitive_accuracy.json" ]]; then
            cp "${ACC_SRC}/sensitive_accuracy.json" "${ARC_SNAPSHOT}/sensitive_accuracy.json"
            echo "[staged] ${ARC} sensitive_accuracy.json → ${ARC_SNAPSHOT}/"
        else
            echo "[warn] ${ARC} sensitive_accuracy.json not found at ${ACC_SRC}" \
                | tee -a "${SCRIPT_DIR}/logs/missing_data.log"
        fi
    done

done

echo ""
echo "[run_sensitivity_sweep.sh] Sweep complete. Generating plot..."

# ---------------------------------------------------------------------------
# Plot — reads the staged results directory
# ---------------------------------------------------------------------------
# shellcheck disable=SC2086  # word-split of STUDY_ARCHS is intentional
singularity exec \
    --bind /blue \
    "${SIF}" \
    python3 "${SCRIPT_DIR}/plot_sensitivity_graph.py" \
        --results-dir "${RESULTS_DIR}" \
        --dataset     "${STUDY_DATASET}" \
        --archs       ${STUDY_ARCHS} \
        --qmode       "${QMODE}" \
        --quant-bits  "${STUDY_QUANT_BITS}" \
        --t           "${STUDY_T}" \
        --approach    "${STUDY_APPROACH}"

echo "[run_sensitivity_sweep.sh] Done."
