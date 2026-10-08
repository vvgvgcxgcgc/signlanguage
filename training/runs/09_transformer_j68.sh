#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
# Empty trains from scratch. Example: checkpoints/j68_Transformer_last.pth
RESUME=""
torchrun --standalone --nproc_per_node=2 -m training.run \
  --root "/kaggle/input/datasets/huynguang/vsl400-keypoints-final/vsl400-keypoint" \
  --model Transformer \
  --frames 64 \
  --epochs 70 \
  --micro-batch 32 \
  --accumulate 2 \
  --parallel ddp \
  --workers 2 \
  --seed 0 \
  --warmup-epochs 5 \
  --label-smoothing 0.1 \
  --grad-clip 1.0 \
  --ema-decay 0.999 \
  --select-by macro_f1 \
  --checkpoint-dir checkpoints \
  --min-train-samples 20 \
  --optimizer adamw \
  --lr 0.001 \
  --momentum 0.9 \
  --weight-decay 0.01 \
  --tag j68 \
  ${RESUME:+--resume "$RESUME"}
