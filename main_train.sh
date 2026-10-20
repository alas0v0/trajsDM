#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# common_args=(
#     --dataset porto
#     --model ConvTraj
#     --gpu_id 0
#     --seed 0
#     --batch_size 64
#     --loss_type triplet
#     --target_measure haus
#     --emb_sim_metric euc
#     --temperature 0.7
#     --opt_method adam
#     --grid_mode soft
# )
common_args=(
    --dataset porto
    --model DualSIMformer
    --gpu_id 0
    --seed 0
    --batch_size 64
    --loss_type mse
    --target_measure dtw
    --emb_sim_metric cos
    --temperature 0.7
    --opt_method adam
    --grid_mode soft
)

python main_train_model.py "${common_args[@]}" 


echo "All training runs completed."
 