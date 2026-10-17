#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
# One BLAS/OpenMP thread per rank. Two processes already share the host CPUs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Two T4s have no NVLink. P2P and cuMem SIGSEGV on the first DDP broadcast.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0
# Empty trains from scratch. Example: checkpoints/j76rest_AnchorGCN-AS_last.pth
RESUME="/kaggle/input/datasets/nqhuy23/checkpoints/signlanguage/checkpoints/j76rest_AnchorGCN-AS_last.pth"
torchrun --standalone --nproc_per_node=2 -m training.run \
  --root "/kaggle/input/datasets/huynguang/vsl400-keypoints-final/vsl400-keypoint" \
  --model AnchorGCN-AS \
  --frames 64 \
  --legs \
  --epochs 70 \
  --micro-batch 32 \
  --accumulate 2 \
  --parallel ddp \
  --device cuda \
  --workers 2 \
  --seed 0 \
  --warmup-epochs 5 \
  --label-smoothing 0.1 \
  --grad-clip 1.0 \
  --ema-decay 0.999 \
  --select-by macro_f1 \
  --checkpoint-dir checkpoints \
  --min-train-samples 20 \
  --crop-mode rest \
  --optimizer sgd \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 0.0004 \
  --tag j76rest \
  ${RESUME:+--resume "$RESUME"}
