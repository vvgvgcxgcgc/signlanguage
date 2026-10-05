#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
python -m training.run \
  --root "/kaggle/input/datasets/huynguang/vsl400-keypoints-final/vsl400-keypoint" \
  --model ST-GCN \
  --frames 64 \
  --epochs 70 \
  --micro-batch 64 \
  --accumulate 2 \
  --parallel none \
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
  --tag j68
