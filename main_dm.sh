#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# common_args=(
#     --dataset porto
#     --model ConvTraj
#     --gpu_id 0
#     --seed 0
#     --batch_size 32
#     --loss_type triplet
#     --temperature 0.7
#     --neighbor_num 3
#     --target_measure fret
#     --emb_sim_metric euc
#     --grid_mode soft
#     --use_embedding_label True
#     # --teacher_checkpoint /home/zhutairu/workplace/learning/test/soft_files_test/porto/dtw/euc/FullTrainingDataset/ConvTraj/opt_method_adam_lr_teacher0dot001/20261015-221626/best_model.pt
#     # --teacher_checkpoint /home/zhutairu/workplace/learning/test/soft_files_test/porto/haus/euc/FullTrainingDataset/ConvTraj/opt_method_adam_lr_teacher0dot001/20261016-175422/best_model.pt
#     --teacher_checkpoint /home/zhutairu/workplace/learning/test/soft_files_test/porto/fret/euc/FullTrainingDataset/ConvTraj/opt_method_adam_lr_teacher0dot001/20261016-181549/best_model.pt
# )
common_args=(
    --dataset porto
    --model DualSIMformer
    --gpu_id 0
    --seed 0
    --syn_num 300
    --batch_size 64
    --loss_type mse
    --temperature 0.7
    --neighbor_num 3
    --target_measure dtw
    --emb_sim_metric cos
    --grid_mode soft
    --use_embedding_label True
    --epoch_eval_train 100
    --opt_method adam
    # --teacher_checkpoint /home/zhutairu/workplace/learning/test/soft_files_test/porto/fret/chebyshev/FullTrainingDataset/DualSIMformer/opt_method_adam_lr_teacher0dot0005/20261016-181627/best_model.pt
    # --teacher_checkpoint /home/zhutairu/workplace/learning/test/soft_files_test/porto/haus/chebyshev/FullTrainingDataset/DualSIMformer/opt_method_adam_lr_teacher0dot0005/20261016-173010/best_model.pt
    # --teacher_checkpoint /home/zhutairu/workplace/learning/test/soft_files_test/porto/dtw/cos/FullTrainingDataset/DualSIMformer/opt_method_adam_lr_teacher0dot0005/20261016-092434/best_model.pt
    # --teacher_checkpoint /home/zhutairu/workplace/learning/test/soft_files_test/porto/dtw/cos/FullTrainingDataset/DualSIMformer/opt_method_sgd_lr_teacher0dot5_mom0dot9_l20/20261018-152803/best_model.pt
    # --teacher_checkpoint /home/zhutairu/workplace/learning/test/soft_files_test/porto/haus/chebyshev/FullTrainingDataset/DualSIMformer/opt_method_sgd_lr_teacher0dot5_mom0dot9_l20/20261018-194535/best_model.pt
    --teacher_checkpoint /home/zhutairu/workplace/learning/test/soft_files_test/porto/dtw/cos/FullTrainingDataset/DualSIMformer/opt_method_adam_lr_teacher0dot0005/20261018-231113/best_model.pt
)

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
