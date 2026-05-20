#!/bin/bash
# ============================================================================
# SLURM submit script for the ECC Resilience pipeline on HiPerGator.
#
# Usage:
#   sbatch run.sh
#
# What it does:
#   1. Requests one GPU node, loads Singularity.
#   2. Launches a single `singularity exec --nv` that runs `bash run_all.sh`
#      INSIDE the container. Every python3 call in run_all.sh therefore uses
#      the container's PyTorch / CUDA without per-line wrapping.
#
# Adjust before submitting:
#   - SBATCH --time / --mem / --gres   (full pipeline is long; tune as needed)
#   - SIF        : path to your Singularity image
#   - WORKDIR    : directory holding run_all.sh, *.py, and artifacts/
# ============================================================================

#SBATCH --job-name=ecc-res
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# ---- Banner ----
date; hostname; pwd
mkdir -p logs

# ---- Load Singularity ----
module load singularity

# ---- Container image (PyTorch / CUDA / numpy / torchvision baked in) ----
export SIF="/blue/rewetz/vkamineni/RECC_MIP_v15.sif"

# ---- Working directory: where run_all.sh and the *.py files live ----
# Put your code on /blue (NOT /home -- /home has tight quotas on HiPerGator).
WORKDIR="/blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/Quantization"   # <-- EDIT THIS
cd "${WORKDIR}"

# ---- ImageNet validation set ----
# Steps 5, 8, 11 expect ./imagenet-val under the working dir.
# Symlink to a shared ImageNet copy on /blue rather than downloading.
# Example (do this ONCE, manually):
#   ln -s /blue/<group>/<shared>/imagenet/val ./imagenet-val
# If the link is missing, the ImageNet steps will fail with a clear error.

# ---- Run the entire pipeline inside the container ----
# --nv          : pass through NVIDIA driver / GPUs
# --bind /blue  : make /blue visible inside the container (auto-bound on
#                 HiPerGator by default, but explicit is safer)
# --pwd         : set the in-container working directory
srun --cpu-bind=cores --mem-bind=local \
    singularity exec \
        --nv \
        --bind /blue \
        --pwd "${WORKDIR}" \
        --env SIF="${SIF}" \
        "${SIF}" \
        bash run_all.sh

echo "[run.sh] Job finished with exit code $?"