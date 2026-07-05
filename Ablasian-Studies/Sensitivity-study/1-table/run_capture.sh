#!/bin/bash
# =============================================================================
# run_capture.sh — Ablation: sensitivity vs no-sensitivity ECC embedding
#
# Plain wrapper (no SBATCH, no singularity) — capture_and_build_table.py only
# reads accuracy.json files with the stdlib, no torch/matplotlib needed.
#
# See README.md in this directory for the full step-by-step procedure.
#
# Usage:
#   bash run_capture.sh --mode no-sensitivity
#   bash run_capture.sh --mode sensitivity --t-values 1 6
#
# All arguments are forwarded as-is to capture_and_build_table.py.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../env.sh"

python3 "${SCRIPT_DIR}/capture_and_build_table.py" "$@"
