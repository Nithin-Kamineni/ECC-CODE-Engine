#!/bin/bash
# =============================================================================
# run_test.sh — Build and run the Python vs C++ ECC comparison test.
#
# Usage (from 4-EmbeddingECC/c++ directory):
#   bash run_test.sh                      # BCH(63,51,t=2), search3, seed=42
#   bash run_test.sh --t 4               # try t=4 where divergence is largest
#   bash run_test.sh --t 6 --verbose     # t=6 with verbose debug output
#   bash run_test.sh --approach greedy   # test greedy instead
#
# What it does:
#   1. Compile the C++ test binary (test_ecc) inside the C++ SIF
#   2. Run Python reference (inside Python SIF) → saves test_input.npy, P.npy,
#      and py_output.json to WORKDIR
#   3. Run C++ binary (inside C++ SIF) → saves cpp_output.json to WORKDIR
#   4. Compare outputs (inside Python SIF) → prints side-by-side diff
#
# The comparison shows EXACTLY where and how much Python and C++ diverge.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../env.sh"

T_VALUE=2
APPROACH=search3
VERBOSE=""
WORKDIR="/tmp/ecc_test_t${T_VALUE}_${APPROACH}"
N_VALS=63
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --t)        T_VALUE="$2"; shift 2;;
        --approach) APPROACH="$2"; shift 2;;
        --verbose)  VERBOSE="--verbose"; shift;;
        --workdir)  WORKDIR="$2"; shift 2;;
        --n-vals)   N_VALS="$2"; shift 2;;
        --seed)     SEED="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

WORKDIR="/tmp/ecc_test_t${T_VALUE}_${APPROACH}"

module load singularity 2>/dev/null || true

CPP_SIF="${SCRIPT_DIR}/ecc_cpp.sif"
PY_SIF="${SIF}"
CPP_BIN="${SCRIPT_DIR}/test_ecc"
PY_SCRIPT="${SCRIPT_DIR}/test_ecc_compare.py"

echo "============================================================"
echo "ECC Comparison Test: BCH(63, t=${T_VALUE}), approach=${APPROACH}"
echo "Workdir: ${WORKDIR}"
echo "============================================================"

# ── Step 1: compile C++ test binary ──────────────────────────────────────────
echo ""
echo "[1/4] Compiling C++ test binary (test_ecc) ..."
singularity exec --bind /blue "${CPP_SIF}" make -C "${SCRIPT_DIR}" test_ecc

# ── Step 2: run Python reference ─────────────────────────────────────────────
echo ""
echo "[2/4] Running Python ${APPROACH} reference ..."
singularity exec \
    --nv \
    --bind /blue \
    "${PY_SIF}" \
    python3 "${PY_SCRIPT}" \
        --mode generate \
        --workdir "${WORKDIR}" \
        --t "${T_VALUE}" \
        --approach "${APPROACH}" \
        --n-vals "${N_VALS}" \
        --seed "${SEED}" \
        ${VERBOSE}

# ── Step 3: run C++ binary ───────────────────────────────────────────────────
echo ""
echo "[3/4] Running C++ ${APPROACH} ..."
singularity exec \
    --bind /blue \
    "${CPP_SIF}" \
    "${CPP_BIN}" \
        --input        "${WORKDIR}/test_input.npy" \
        --parity-matrix "${WORKDIR}/test_P_t${T_VALUE}.npy" \
        --n 63 \
        --t "${T_VALUE}" \
        --approach "${APPROACH}" \
        --output   "${WORKDIR}/cpp_output.json" \
        ${VERBOSE:+--verbose}

# ── Step 4: compare outputs ───────────────────────────────────────────────────
echo ""
echo "[4/4] Comparing outputs ..."
singularity exec \
    --nv \
    --bind /blue \
    "${PY_SIF}" \
    python3 "${PY_SCRIPT}" \
        --mode compare \
        --workdir "${WORKDIR}" \
        --t "${T_VALUE}" \
        --approach "${APPROACH}" \
        --n-vals "${N_VALS}"

echo ""
echo "Done. Full test data saved in: ${WORKDIR}"
