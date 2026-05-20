#!/bin/bash
# ============================================================================
# ECC Resilience full pipeline.
#
# THIS SCRIPT IS MEANT TO BE EXECUTED INSIDE THE SINGULARITY CONTAINER
# launched by run.sh:
#
#     singularity exec --nv $SIF bash run_all.sh
#
# So every `python3` call here automatically uses the container's PyTorch
# and CUDA -- no `conda activate`, no per-command singularity wrapping.
#
# Steps:
#   1. Train CIFAR-10 ResNet-18
#   2. Train CIFAR-100 ResNet-18
#   3. Layer-then-weight: CIFAR-10
#   4. Layer-then-weight: CIFAR-100
#   5. Layer-then-weight: ImageNet (5 architectures, pretrained)
#   6. Float-side bit-flip: CIFAR-10
#   7. Float-side bit-flip: CIFAR-100
#   8. Float-side bit-flip: ImageNet
#   9. Integer-side bit-flip: CIFAR-10
#  10. Integer-side bit-flip: CIFAR-100
#  11. Integer-side bit-flip: ImageNet
#
# Outputs land in artifacts/sensitivity/ and artifacts/models/
# ============================================================================

set -e   # stop on first error

echo "[run_all] starting at $(date)"
echo "[run_all] working dir: $(pwd)"
echo "[run_all] python3:     $(which python3)"
python3 -c "import torch; print(f'[run_all] torch {torch.__version__} | cuda {torch.cuda.is_available()} | device count {torch.cuda.device_count()}')"

# ---- Step 1: Train CIFAR-10 ResNet-18 ----
python3 test-Quantizer.py train \
    --dataset CIFAR10 --arch resnet18 \
    --epochs 100 --lr 0.1 --optim sgd --momentum 0.9 \
    --weight-decay 5e-4 --scheduler cosine --warmup-epochs 5 \
    --label-smoothing 0.1

# ---- Step 2: Train CIFAR-100 ResNet-18 ----
python3 test-Quantizer.py train \
    --dataset CIFAR100 --arch resnet18 \
    --epochs 100 --lr 0.1 --optim sgd --momentum 0.9 \
    --weight-decay 5e-4 --scheduler cosine --warmup-epochs 5 \
    --label-smoothing 0.1

# ---- Step 3: Layer-then-weight on CIFAR-10 ----
python3 layer_then_weight.py \
    --dataset CIFAR10 --archs resnet18 \
    --quantize-bits 8 4 \
    --top-layers 5 --top-per-layer 200 \
    --layer-metric grad_norm --max-batches 8

# ---- Step 4: Layer-then-weight on CIFAR-100 ----
python3 layer_then_weight.py \
    --dataset CIFAR100 --archs resnet18 \
    --quantize-bits 8 4 \
    --top-layers 5 --top-per-layer 200 \
    --layer-metric grad_norm --max-batches 8

# ---- Step 5: Layer-then-weight on ImageNet (5 archs) ----
python3 layer_then_weight.py \
    --dataset IMAGENET --imagenet-root ./imagenet-val --use-pretrained 1 \
    --archs resnet18 resnet50 vgg16 mobilenet_v2 efficientnet_b0 \
    --quantize-bits 8 4 \
    --top-layers 5 --top-per-layer 200 \
    --layer-metric grad_norm --max-batches 4

# ---- Step 6: Float-side bit-flip on CIFAR-10 ----
python3 validate_bitflip.py \
    --dataset CIFAR10 --archs resnet18 \
    --formats float32 int8 int4 \
    --top-layers 5 --top-per-layer 200 --layer-metric grad_norm \
    --k-list 1 5 10 50 100 500 1000 \
    --bit-position sign --random-trials 3

# ---- Step 7: Float-side bit-flip on CIFAR-100 ----
python3 validate_bitflip.py \
    --dataset CIFAR100 --archs resnet18 \
    --formats float32 int8 int4 \
    --top-layers 5 --top-per-layer 200 --layer-metric grad_norm \
    --k-list 1 5 10 50 100 500 1000 \
    --bit-position sign --random-trials 3

# ---- Step 8: Float-side bit-flip on ImageNet ----
python3 validate_bitflip.py \
    --dataset IMAGENET --imagenet-root ./imagenet-val --use-pretrained 1 \
    --archs resnet18 resnet50 vgg16 mobilenet_v2 efficientnet_b0 \
    --formats float32 int8 int4 \
    --top-layers 5 --top-per-layer 200 --layer-metric grad_norm \
    --k-list 1 10 100 500 1000 \
    --bit-position sign --random-trials 3 --eval-max-batches 20

# ---- Step 9: Integer-side bit-flip on CIFAR-10 ----
python3 validate_bitflip_int.py \
    --dataset CIFAR10 --archs resnet18 \
    --formats float32 int8 int4 \
    --top-layers 5 --top-per-layer 200 --layer-metric grad_norm \
    --k-list 1 5 10 50 100 500 1000 \
    --bit-position sign --random-trials 3

# ---- Step 10: Integer-side bit-flip on CIFAR-100 ----
python3 validate_bitflip_int.py \
    --dataset CIFAR100 --archs resnet18 \
    --formats float32 int8 int4 \
    --top-layers 5 --top-per-layer 200 --layer-metric grad_norm \
    --k-list 1 5 10 50 100 500 1000 \
    --bit-position sign --random-trials 3

# ---- Step 11: Integer-side bit-flip on ImageNet ----
# Serialized version (1 GPU). The original script ran 3 jobs in parallel
# across CUDA_VISIBLE_DEVICES 1,2,3 -- under a single-GPU SLURM allocation
# we just run them one after another. If you bumped the SLURM script to
# --gres=gpu:a100:3, see the parallel block at the bottom of this file.

# Sub-step A: ResNet-18 + MobileNet-V2
python3 validate_bitflip_int.py \
    --dataset IMAGENET --imagenet-root ./imagenet-val --use-pretrained 1 \
    --archs resnet18 mobilenet_v2 \
    --formats float32 int8 int4 \
    --top-layers 5 --top-per-layer 200 --layer-metric grad_norm \
    --k-list 1 5 10 50 100 500 \
    --bit-position sign --random-trials 3 --eval-max-batches 20 \
    2>&1 | tee int_a.log

# Sub-step B: ResNet-50 + VGG-16
python3 validate_bitflip_int.py \
    --dataset IMAGENET --imagenet-root ./imagenet-val --use-pretrained 1 \
    --archs resnet50 vgg16 \
    --formats float32 int8 int4 \
    --top-layers 5 --top-per-layer 200 --layer-metric grad_norm \
    --k-list 1 5 10 50 100 500 \
    --bit-position sign --random-trials 3 --eval-max-batches 20 \
    2>&1 | tee int_b.log

# Sub-step C: EfficientNet-B0
python3 validate_bitflip_int.py \
    --dataset IMAGENET --imagenet-root ./imagenet-val --use-pretrained 1 \
    --archs efficientnet_b0 \
    --formats float32 int8 int4 \
    --top-layers 5 --top-per-layer 200 --layer-metric grad_norm \
    --k-list 1 5 10 50 100 500 \
    --bit-position sign --random-trials 3 --eval-max-batches 20 \
    2>&1 | tee int_c.log

echo
echo "============================================================"
echo "  ALL DONE.  Results in artifacts/sensitivity/"
echo "  Finished at $(date)"
echo "============================================================"

# ============================================================================
# OPTIONAL: Parallel Step 11 across 3 GPUs.
#
# If you change run.sh to request 3 GPUs (--gres=gpu:a100:3), comment out
# sub-steps A/B/C above and uncomment the block below. Note that under
# SLURM the allocated GPUs are ALWAYS visible as cuda:0, cuda:1, cuda:2
# regardless of their physical IDs -- so we use 0,1,2 here, NOT 1,2,3 like
# the original workstation script.
# ----------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0 python3 validate_bitflip_int.py \
#     --dataset IMAGENET --imagenet-root ./imagenet-val --use-pretrained 1 \
#     --archs resnet18 mobilenet_v2 \
#     --formats float32 int8 int4 \
#     --top-layers 5 --top-per-layer 200 --layer-metric grad_norm \
#     --k-list 1 5 10 50 100 500 \
#     --bit-position sign --random-trials 3 --eval-max-batches 20 \
#     2>&1 | tee int_a.log &
#
# CUDA_VISIBLE_DEVICES=1 python3 validate_bitflip_int.py \
#     --dataset IMAGENET --imagenet-root ./imagenet-val --use-pretrained 1 \
#     --archs resnet50 vgg16 \
#     --formats float32 int8 int4 \
#     --top-layers 5 --top-per-layer 200 --layer-metric grad_norm \
#     --k-list 1 5 10 50 100 500 \
#     --bit-position sign --random-trials 3 --eval-max-batches 20 \
#     2>&1 | tee int_b.log &
#
# CUDA_VISIBLE_DEVICES=2 python3 validate_bitflip_int.py \
#     --dataset IMAGENET --imagenet-root ./imagenet-val --use-pretrained 1 \
#     --archs efficientnet_b0 \
#     --formats float32 int8 int4 \
#     --top-layers 5 --top-per-layer 200 --layer-metric grad_norm \
#     --k-list 1 5 10 50 100 500 \
#     --bit-position sign --random-trials 3 --eval-max-batches 20 \
#     2>&1 | tee int_c.log &
#
# wait