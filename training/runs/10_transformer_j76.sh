#!/usr/bin/env bash
# Single process on CPU. DDP is CUDA-only and is not used here.
python -m training.run \
  --root "/Users/huynq/Projects/sign-language/vsl400-keypoint" \
  --model Transformer \
  --frames 64 \
  --legs \
  --epochs 70 \
  --micro-batch 32 \
  --accumulate 2 \
  --parallel none \
  --device cpu \
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
  --tag j76
