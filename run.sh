#!/bin/bash

datasets=("PEMS-BAY" "SD" "KnowAir" "BJAir")
mask_ratios=(0 0.1 0.3 0.5)

for dataset in "${datasets[@]}"; do
  for mask_ratio in "${mask_ratios[@]}"; do
    echo "dataset=$dataset mask_ratio=$mask_ratio"
    python experiments/monet/main.py \
      --device cuda:0 \
      --dataset "$dataset" \
      --bs 64 \
      --mask_ratio "$mask_ratio"
  done
done
