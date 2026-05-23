#!/usr/bin/env bash
# =============================================================================
# Global environment for ECC-CODE-Engine
#
# Source this from every run.sh:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"
#   (adjust the number of ../ to match folder depth)
#
# Variables can be overridden before sourcing:
#   DATASET=CIFAR100 source ../env.sh
# =============================================================================

# ---- Singularity image (PyTorch + CUDA + torchvision) ----
SIF="/blue/rewetz/vkamineni/RECC_MIP_v15.sif"

# ---- Project root (auto-derived from this file's location) ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Data directories — ALL outputs go under 0-Data/ ----
DATA_ROOT="${PROJECT_ROOT}/0-Data"
DATASET_DIR="${DATA_ROOT}/data"
ARTIFACTS_DIR="${DATA_ROOT}/artifacts"
MODELS_DIR="${ARTIFACTS_DIR}/models"
SENSITIVITY_DIR="${ARTIFACTS_DIR}/sensitivity"
PATTERNS_DIR="${ARTIFACTS_DIR}/patterns"

# ---- ImageNet validation set ----
# Symlink once manually: ln -s /blue/<group>/imagenet/val "${DATA_ROOT}/imagenet-val"
IMAGENET_ROOT="${DATA_ROOT}/imagenet-val"

# ---- Datasets to process (space-separated, override via env before sourcing) ----
# Valid values: CIFAR10  CIFAR100  IMAGENET
DATASETS="${DATASETS:-CIFAR10 CIFAR100 IMAGENET}"

# ---- Architectures (space-separated, must match torchvision/model names exactly) ----
# Valid values: resnet18  resnet50  mobilenet_v2  efficientnet_b0  vgg16
ARCHS="${ARCHS:-resnet18 resnet50 mobilenet_v2 efficientnet_b0}"

# ---- Default runtime parameters ----
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda}"
QUANTIZE_BITS="${QUANTIZE_BITS:-16 8 4}"
# QUANT_LEVELS — full sweep used in 2-Sensitivity and 3-PatternFinder loops.
# 32 = float32 baseline (no quantization); 16/8/4 = PTQ levels.
# Scripts map 32 → float32 label internally. Modify freely.
QUANT_LEVELS="${QUANT_LEVELS:-32 16 8 4}"
TOP_LAYERS="${TOP_LAYERS:-999}"
TOP_PER_LAYER="${TOP_PER_LAYER:-30000}"
LAYER_METRIC="${LAYER_METRIC:-grad_norm}"
MAX_BATCHES="${MAX_BATCHES:-8}"

# ---- PatternFinder parameters ----
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_SENS="${MAX_SENS:-2}"
TOP_SENSITIVE="${TOP_SENSITIVE:-100}"
# SENS_THRESHOLD: Taylor score cutoff — weights above this are counted as sensitive.
# The final sensitive set = max(threshold_count, TOP_SENSITIVE).
SENS_THRESHOLD="${SENS_THRESHOLD:-0.001}"
# MAX_STRIDE: Maximum allowed stride s in perm[k]=(k*s) mod N.
# Set equal to your hardware's burst-fetch size (in number of weights).
# Strides s > MAX_STRIDE are never evaluated — keeps all accesses within one fetch tile.
MAX_STRIDE="${MAX_STRIDE:-256}"

# ---- Skip training if float32 checkpoint already exists ----
# Set SKIP_TRAIN=true to skip training and go directly to quantization
# when model_float32.pth already exists for a given dataset/arch pair.
SKIP_TRAIN="${SKIP_TRAIN:-false}"

# ---- Disable pattern search (identity permutation only) ----
# Set DISABLE_PATTERN_FIND=true to skip the interleaver search and save weights
# in their original order (identity permutation) for every layer.
DISABLE_PATTERN_FIND="${DISABLE_PATTERN_FIND:-false}"

# ---- EmbeddingECC — separate control lists (independent of 1-3 pipeline vars) ----

EMBED_RUN_CPP="${EMBED_RUN_CPP:-true}" # set to true to run the C++ embedding code (requires separate compile step)

# Modify these to subset the combinations you actually want to embed.
# EMBED_DATASETS="${EMBED_DATASETS:-CIFAR10 CIFAR100 IMAGENET}"
EMBED_DATASETS="${EMBED_DATASETS:-IMAGENET}"
# EMBED_ARCHS="${EMBED_ARCHS:-resnet18 resnet50 mobilenet_v2 efficientnet_b0}"
EMBED_ARCHS="${EMBED_ARCHS:-mobilenet_v2 efficientnet_b0}"
# EMBED_QUANT_BITS="${EMBED_QUANT_BITS:-8 4}"   # quantized levels only (not float32)
EMBED_QUANT_BITS="${EMBED_QUANT_BITS:-8}"   # quantized levels only (not float32)
EMBED_APPROACH="${EMBED_APPROACH:-search3}" # 'parfit', 'replace', 'no', 'parfix', 'search3', 'greedy'
EMBED_CODEWORD="${EMBED_CODEWORD:-63}"         # M value in M{codeword}_t{tval} path
EMBED_WORKERS="${EMBED_WORKERS:-16}"
EMBEDDED_ECC_DIR="${EMBEDDED_ECC_DIR:-${ARTIFACTS_DIR}/embeddedECC}"
EMBEDDED_ECC_CHUNKS_DIR="${EMBEDDED_ECC_CHUNKS_DIR:-${ARTIFACTS_DIR}/embeddedECC_Chunks}"

# ---- Ensure output directories exist ----
mkdir -p "${MODELS_DIR}" "${SENSITIVITY_DIR}" "${PATTERNS_DIR}" \
         "${EMBEDDED_ECC_DIR}" "${EMBEDDED_ECC_CHUNKS_DIR}"

export SIF PROJECT_ROOT DATA_ROOT DATASET_DIR ARTIFACTS_DIR \
       MODELS_DIR SENSITIVITY_DIR PATTERNS_DIR IMAGENET_ROOT \
       DATASETS ARCHS BATCH_SIZE DEVICE QUANTIZE_BITS QUANT_LEVELS \
       TOP_LAYERS TOP_PER_LAYER LAYER_METRIC MAX_BATCHES \
       GROUP_SIZE MAX_SENS TOP_SENSITIVE SENS_THRESHOLD MAX_STRIDE SKIP_TRAIN \
       DISABLE_PATTERN_FIND \
       EMBED_DATASETS EMBED_ARCHS EMBED_QUANT_BITS EMBED_APPROACH \
       EMBED_CODEWORD EMBED_WORKERS EMBEDDED_ECC_DIR EMBEDDED_ECC_CHUNKS_DIR
