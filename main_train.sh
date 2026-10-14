#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

common_args=(
    --dataset porto
    --model ConvTraj
    --gpu_id 1
    --seed 0
    --batch_size 64
    --loss_type triplet
    --target_measure haus
    --emb_sim_metric euc
    --temperature 0.7
)

for neigh_num in 3 5 7; do
    echo "Starting training with neigh_num=${neigh_num}"
    python main_train_model.py "${common_args[@]}" --neighbor_num "$neigh_num"
done

echo "All training runs completed."
 