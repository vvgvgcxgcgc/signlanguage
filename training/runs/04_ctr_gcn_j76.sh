#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc_per_node=2 -m training.run \
  --root "/kaggle/input/datasets/huynguang/vsl400-keypoints-final/vsl400-keypoint" \
  --model CTR-GCN \
  --frames 64 \
  --legs \
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
  --optimizer sgd \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 0.0004 \
  --tag j76
