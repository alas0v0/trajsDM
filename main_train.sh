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
)

for temperature in 0.1 0.3 0.5; do
    echo "Starting training with temperature=${temperature}"
    python main_train_model.py "${common_args[@]}" --temperature "$temperature"
done

echo "All training runs completed."
