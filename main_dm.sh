#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# common_args=(
#     --dataset porto
#     --model ConvTraj
#     --gpu_id 0
#     --seed 42
#     --batch_size 32
#     --loss_type triplet
#     --temperature 0.7
#     --neighbor_num 3
#     --target_measure haus
#     --emb_sim_metric euc
#     --grid_mode hard
#     --use_embedding_label False
#     # --teacher_checkpoint /data/zhutairu/workplace/project/trajsDM/train_files/porto/dtw/euc/FullTrainingDataset/ConvTraj/opt_method_adam_lr_teacher0dot001/20261019-211144/best_model.pt
#     # --teacher_checkpoint /data/zhutairu/workplace/project/trajsDM/train_files/porto/haus/euc/FullTrainingDataset/ConvTraj/opt_method_adam_lr_teacher0dot001/20261019-225127/best_model.pt
# )
common_args=(
    --dataset porto
    --model DualSIMformer
    --gpu_id 0
    --seed 42
    --syn_num 300

    --batch_size 32
    --loss_type mse
    --temperature 0.7
    --neighbor_num 3
    --target_measure haus
    --emb_sim_metric chebyshev
    --grid_mode hard
    --use_embedding_label False
    --epoch_eval_train 100
    --opt_method adam
    # --teacher_checkpoint /data/zhutairu/workplace/project/trajsDM/soft_files_test/porto/dtw/cos/FullTrainingDataset/DualSIMformer/opt_method_adam_lr_teacher0dot0005/20261019-150658/best_model.pt
    # --teacher_checkpoint /data/zhutairu/workplace/project/trajsDM/train_files/porto/haus/chebyshev/FullTrainingDataset/DualSIMformer/opt_method_adam_lr_teacher0dot0005/20261019-225238/best_model.pt
)
    # --teacher_checkpoint 
# experiment_pairs=(
#     "dtw:euc"
#     "haus:euc"
#     "fret:euc"
# )
# experiment_pairs=(
#     "dtw:cos"
#     "haus:chebyshev"
#     "fret:chebyshev"
# )


python main_DM_test.py  "${common_args[@]}" 

echo "All training runs completed."
