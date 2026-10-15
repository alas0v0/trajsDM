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
)

experiment_pairs=(
    "dtw:euc"
    "haus:euc"
    "fret:euc"
)

for pair in "${experiment_pairs[@]}"; do
    IFS=: read -r target_measure emb_sim_metric <<< "$pair"

    echo "Starting: measure=${target_measure}, metric=${emb_sim_metric}"

    python main_DM_test.py \
        "${common_args[@]}" \
        --target_measure "$target_measure" \
        --emb_sim_metric "$emb_sim_metric"
done

echo "All training runs completed."
