#!/bin/bash
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x.%j.out

date;hostname;pwd


# Load Singularity
module load singularity

SIF="/blue/rewetz/vkamineni/RECC_MIP_v15.sif"
Execute_File="/blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/Sensitivity/Weights/plot_sensitivity.py"

# ARTIFACT_PATH=cifar10/resnet50/model_int8_ptq.pth
ARTIFACT_PATH=cifar10/resnet18/model_int8_ptq.pth
# ARTIFACT_PATH=cifar10/mobilenet_v2/model_int8_ptq.pth
# ARTIFACT_PATH=cifar10/efficientnet_b0/model_int8_ptq.pth

# Execute your Python script inside the container
srun --cpu-bind=cores --mem-bind=local \
singularity exec \
  --env ARTIFACT_PATH="${ARTIFACT_PATH}" \
  --nv "${SIF}" \
  python3 prepare_patterns.py \
    --csv /blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/0-Data/artifacts/sensitivity/layer_then_weight_cifar10_resnet18_float32_L5xN200_grad_norm.csv \
    --threshold 0.003 \
    --arch resnet18 \
    --run-search --group-size 8 --max-sens 2