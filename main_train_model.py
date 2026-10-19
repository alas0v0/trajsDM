# 用全部训练集训练模型
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import time  
import yaml
import torch 
import datetime

from utils.data_process import TrajData
from utils.tools import log_ini, log_and_print, set_seed 

from models import get_model
from utils.loss import initialize_loss
from dataloader import TrajSimLeaDataLoader
from parameter_setting import parse_args
from evaluate import test
from train import train_one_epoch_convtraj, train_one_epoch_trajcl, train_one_epoch_other, train_one_epoch_dualsim


args = parse_args()
with open(args.config_path) as f:
    config = yaml.safe_load(f)

# 储存实验结果的文件夹
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if args.opt_method == "adam":
    lr_mom_str = "opt_method_" + args.opt_method + "_lr_teacher" + str(config["model"][args.model]["lr_teacher"][args.opt_method]).replace(".", "dot")
elif args.opt_method == "sgd":
    lr_mom_str = "opt_method_" + args.opt_method + "_lr_teacher" + str(config["model"][args.model]["lr_teacher"][args.opt_method]).replace(".", "dot") + "_mom" + str(args.mom).replace(".", "dot") + "_l2" + str(args.l2).replace(".", "dot")

save_dir = os.path.join(args.file_output_path, args.dataset, args.target_measure, args.emb_sim_metric, "FullTrainingDataset", args.model, lr_mom_str, curr_time)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
yaml.dump(dict(vars(args)), open(os.path.join(save_dir, 'config.yaml'), 'w'))  # 将实验参数写入args.save_dir中的config.yaml文件

log_ini(save_dir)
set_seed(args.seed)
args.device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
grid_mode = args.grid_mode
print(grid_mode)
traj_data = TrajData(args.dataset, args.target_measure, config["datasets"], args.grid_size)
train_dataloader = TrajSimLeaDataLoader(traj_data=traj_data, args=args, mode="train",grid_mode=grid_mode)
eval_dataloader = TrajSimLeaDataLoader(traj_data=traj_data, args=args, mode="eval",grid_mode=grid_mode)
test_dataloader = TrajSimLeaDataLoader(traj_data=traj_data, args=args, mode="test",grid_mode=grid_mode)

model = get_model(args, config, traj_data)
criterion = initialize_loss(args).to(args.device)
#================new================
lr = config["model"][args.model]["lr_teacher"][args.opt_method]
print(lr)
if args.opt_method == "sgd": 
    optim = torch.optim.SGD( model.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
elif args.opt_method == "adam":
    optim = torch.optim.Adam( model.parameters(), lr=lr, weight_decay=args.l2 )

metric_num = 6 # 要测的指标的数量
best_eval_topk_acc = -torch.ones(metric_num)  # 储存最好的eval指标: top1, top5, top10, top20, top50, R10@50
training_time = 0  # 训练时长
eval_time = 0  # 验证时长
#初始化测试集最优指标
best_eval_topk_acc = -torch.ones(metric_num)
best_test_topk_acc = None
best_epoch = -1

start_time = time.time()
for e in range(args.train_epochs): 
    # 训练1个epoch
    if args.model == "ConvTraj":
        avg_batch_loss, epoch_training_time = train_one_epoch_convtraj(model, train_dataloader, optim, criterion, args.device, e)
    elif args.model == "TrajCL":
        avg_batch_loss, epoch_training_time = train_one_epoch_trajcl(model, train_dataloader, optim, criterion, args.device)
    elif args.model in ["SIMformer", "GRU", "LSTM", "Linear"]:
        avg_batch_loss, epoch_training_time = train_one_epoch_other(model, train_dataloader, optim, criterion, args.device)
    elif args.model == "DualSIMformer":
        avg_batch_loss, epoch_training_time = train_one_epoch_dualsim(model, train_dataloader, optim, criterion, args.device)
    else:
        raise ValueError("不支持该模型")
    log_and_print("Train:\tEpoch: {}\tTrain average batch loss: {:.6f}\tEpoch time: {}".format(e, avg_batch_loss, epoch_training_time))
    training_time += epoch_training_time

    # 验证集上测试
    eval_topk_acc, epoch_eval_time = test(model, eval_dataloader, args)
    log_and_print("Eval:\tEpoch: {}\t{}@Top-k Acc:{}\tEval time:{}".format(e, args.target_measure.upper(), eval_topk_acc, epoch_eval_time))
    eval_time += epoch_eval_time 

    # 若有一半以上的指标有提升, 则更新验证集效果指标, 并在测试集上测试模型效果
    cur_eval_topk_acc = torch.tensor(list(eval_topk_acc.values()))
    if (cur_eval_topk_acc > best_eval_topk_acc).sum() > (metric_num / 2):  # 有一半以上的指标有提升
        log_and_print(f"[Epoch {e}] Best Eval Score Updated, Test Model on Test Dataset...")
        # 更新验证集效果指标
        best_eval_topk_acc = cur_eval_topk_acc
        best_epoch = e
        # 在测试集上测试模型效果
        test_topk_acc_per_track, test_time = test(model, test_dataloader, args)
        best_test_topk_acc = test_topk_acc_per_track
        log_and_print("Test:\tEpoch: {}\t{}@Top-k Acc:{}\tTest time:{}\n".format(e, args.target_measure.upper(), test_topk_acc_per_track, test_time))
        #保存checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": e,
            "model": args.model,
            "dataset": args.dataset,
            "target_measure": args.target_measure,
            "hidden_dim": args.hidden_dim,
            "grid_size": args.grid_size,
            "grid_mode": args.grid_mode,
            "temperature": args.temperature,
            "neighbor_num": args.neighbor_num,
        }
        torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
final_test_topk_acc, final_test_time = test(model, test_dataloader, args)
log_and_print( "Final Test:\tEpoch: {}\t{}@Top-k Acc:{}\tTest time:{}\n".format( args.train_epochs - 1, args.target_measure.upper(), final_test_topk_acc, final_test_time ) )
end_time = time.time()
best_test_topk_acc = torch.tensor(list(best_test_topk_acc.values()))
log_and_print(f"\n\nbest eval topk acc: {best_eval_topk_acc}")
log_and_print(f"\n\nbest test topk acc: {best_test_topk_acc}")
log_and_print(f"best epoch: {best_epoch}")
log_and_print(f"training time: {training_time}")
log_and_print(f"evaluate time: {eval_time}")
log_and_print(f"end-to-end time: {end_time - start_time}")
