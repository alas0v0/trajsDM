import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import yaml
import time
import torch 
import datetime
import numpy as np
import torch.nn as nn

from utils.tools import log_ini, set_seed,log_and_print
from utils.data_process import TrajData

from parameter_setting import parse_args
from utils.syn_data_ini import k_medoids_sampling
from utils.loss import initialize_loss
from models import get_model
from dataloader import TrajSimLeaDataLoader
from evaluate import test, getEmbeddings, getEmbeddingsFromFull
from train import train_one_epoch_convtraj, train_one_epoch_trajcl, train_one_epoch_other, train_one_epoch_dualsim
from utils.similarity import emds2dist
#测试的时候
# 1. convtraj合成轨迹训练集，验证集和测试集都用离散dataloader, 模型不变
# 2. dualsimformer weights b,l,1 代表距离一个格子的权重，不等于1，再collate里面设置为1
args = parse_args()
with open(args.config_path) as f:
    config = yaml.safe_load(f)
# 储存实验结果的文件夹
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if args.syn_method == "DM":
    lr_mom_str = "" 
else:
    lr_mom_str = "opt_method_" + args.opt_method + "_lr_teacher" + str(config["model"][args.model]["lr_teacher"][args.opt_method]).replace(".", "dot") + "_mom" + str(args.mom).replace(".", "dot") + "_l2" + str(args.l2).replace(".", "dot")
save_dir = os.path.join(args.distill_output_path, args.dataset, args.target_measure, args.emb_sim_metric, args.syn_method, args.model, lr_mom_str, curr_time)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
yaml.dump(dict(vars(args)), open(os.path.join(save_dir, 'config.yaml'), 'w'))  # 将实验参数写入args.save_dir中的config.yaml文件

log_ini(save_dir)
set_seed(args.seed)
args.device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")

traj_data = TrajData(args.dataset, args.target_measure, config["datasets"], args.grid_size)
train_dataloader = TrajSimLeaDataLoader(traj_data=traj_data, args=args, mode="train", grid_mode=args.grid_mode)
eval_dataloader = TrajSimLeaDataLoader(traj_data=traj_data, args=args, mode="eval", grid_mode=args.grid_mode)
test_dataloader = TrajSimLeaDataLoader(traj_data=traj_data, args=args, mode="test", grid_mode=args.grid_mode)

# 储存evaluation的iteration number [0, 100, 200...]
eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
# model_eval_pool是模型名字的list, 测试合成数据集效果时, 都测试哪些模型的训练效果
model_eval_pool = [args.model]  # 用训练好的浓缩数据集训练这些模型,测试效果
metric_num = 6 # 要测的指标的数量
syn_num = args.syn_num  # 浓缩数据集的大小

# mean_x = torch.tensor([data_range["mean_lon"], data_range["mean_lat"]], dtype=torch.float, device=args.device).view(1, 1, -1)
# std_x = torch.tensor([data_range["std_lon"], data_range["std_lat"]], dtype=torch.float, device=args.device).view(1, 1, -1)
mean_x = torch.tensor([traj_data.mean_lon,traj_data.mean_lat],dtype=torch.float32,device=args.device).view(1, 1, -1)
std_x = torch.tensor([traj_data.std_lon,traj_data.std_lat],dtype=torch.float32,device=args.device).view(1, 1, -1)

train_dataset  = traj_data.train_data
# 初始化浓缩数据集
train_trajs, train_dist = traj_data.train_data[:2]

if args.syn_ini == "randomly_select":
    traj_idx = torch.randperm(len(train_trajs))[:args.syn_num]
    x_syn = [train_trajs[i].clone().detach().requires_grad_(True) for i in traj_idx]
    y_syn = train_dist[traj_idx, :][:, traj_idx].detach().clone().to(args.device)

elif args.syn_ini == "k-medoids":
    traj_idx = k_medoids_sampling(traj_data.train_data[1].numpy(), args.syn_num, max_iters=1000)
    traj_idx = torch.tensor(traj_idx)
    x_syn = [torch.tensor(train_trajs[i], dtype=torch.float, device=args.device) for i in traj_idx]
    y_syn = train_dist[traj_idx, :][:, traj_idx].detach().clone().to(args.device)

elif args.syn_ini == "randomly_generated":
    x_syn = [torch.normal(mean=mean_x.squeeze(), std=std_x.squeeze().expand(int(torch.randint(20, args.syn_len_max, ()).item()), -1)) for _ in range(args.syn_num)]
    A = torch.rand(args.syn_num, args.syn_num, dtype=torch.float, device=args.device)
    A = A * (train_dist.max() - train_dist.min()) + train_dist.min()
    y_syn = (A + A.T) / 2
    y_syn[torch.arange(args.syn_num), torch.arange(args.syn_num)] = 0.0

#判断是否使用emb label
if args.use_embedding_label == "True" and args.teacher_checkpoint is not None:
    log_and_print(f"use emb label, {args.teacher_checkpoint}")
    teacher = get_model(args, config, traj_data)
    checkpoint = torch.load(args.teacher_checkpoint, map_location=args.device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher_embeddings = getEmbeddingsFromFull(teacher, args.model, x_syn, args, traj_data)  
    # teacher_dist = emds2dist(teacher_embeddings,"euc")#TODO
    teacher_dist = emds2dist(teacher_embeddings, args.emb_sim_metric)
    y_syn = teacher_dist.detach().clone().to(args.device)
criterion = initialize_loss(args).to(args.device)

x_syn = nn.ParameterList(x_syn)  # 初始化后的合成轨迹
optimizer_data = torch.optim.SGD(x_syn, lr=args.lr_traj, momentum=0.5)  # 用于更新数据的优化器, 对应原始的optimizer_img


all_acc = []  # 所有测得的acc和training time, 最后可删除

best_acc: dict[str, torch.Tensor] = {}
best_acc_training_time: dict[str, float] = {}

for it in range(0, args.Iteration + 1):  # 第it次更新合成数据集
    log_and_print(f"第{it}次更新合成数据集")
    save_this_it = False
    if it in eval_it_pool or it == args.Iteration:  # 到了测试合成数据集效果的it
        # 得到数据
        eval_x_syn = [x.detach() for x in x_syn]
        eval_y_syn = y_syn.detach()

        cur_best_acc: dict[str, torch.Tensor] = {}
        cur_best_acc_training_time: dict[str, float] = {}

        # 测试用浓缩数据集训练model_eval_pool中模型的best_topk_acc和训练时长
        for model_eval in model_eval_pool:
            log_and_print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
            best_topk_acc_list = []
            training_time_list = []

            # 一个模型重复测args.num_eval次, 记录每次的best_topk_acc和训练时长
            for it_eval in range(args.num_eval):
                # 随机初始化一个模型
                net_eval = get_model(args, config, traj_data)
                # 得到data
                train_dataloader_eval = TrajSimLeaDataLoader(traj_data=traj_data, args=args, mode="syn", syn_data = (eval_x_syn, eval_y_syn), grid_mode=args.grid_mode)
                # 得到学习率
                lr_net = config["model"][args.model]["lr_teacher"][args.opt_method]  # 与真实数据相同

                # 得到优化器
                if args.opt_method == "sgd": 
                    eval_optim = torch.optim.SGD(net_eval.parameters(), lr=lr_net, momentum=args.mom, weight_decay=args.l2)
                elif args.opt_method == "adam":
                    eval_optim = torch.optim.Adam(net_eval.parameters(), lr=lr_net, weight_decay=args.l2)
                print(lr_net, args.opt_method)
                best_eval_topk_acc_per_eval = -torch.ones(metric_num)  # 储存最好的验证集指标: top1, top5, top10, top20, top50, R10@50
                training_time_per_eval_start = time.time()  # 训练时长
                # 用浓缩数据集训练模型, 记录最好的topk_acc
                for ep in range(args.epoch_eval_train):
                    # 训练一个epoch
                    if args.model == "ConvTraj":
                        avg_batch_loss, epoch_training_time = train_one_epoch_convtraj(net_eval, train_dataloader_eval, eval_optim, criterion, args.device, ep)
                    elif args.model == "TrajCL":
                        avg_batch_loss, epoch_training_time = train_one_epoch_trajcl(net_eval, train_dataloader_eval, eval_optim, criterion, args.device)
                    elif args.model == "DualSIMformer":
                        avg_batch_loss, epoch_training_time = train_one_epoch_dualsim(net_eval, train_dataloader_eval, eval_optim, criterion, args.device)
                    elif args.model in ["SIMformer", "GRU", "LSTM", "Linear"]:
                        avg_batch_loss, epoch_training_time = train_one_epoch_other(net_eval, train_dataloader_eval, eval_optim, criterion, args.device)
                    log_and_print("Eval_Syn (Train): It_eval: {}\tEpoch: {}\tTrain average batch loss: {:.6f}\tEpoch time: {}".format(it_eval, ep, avg_batch_loss, epoch_training_time))
                    # training_time_per_eval += epoch_training_time  # 加上当前epoch的训练时长
                    
                    # 验证集上测试
                    eval_topk_acc, epoch_eval_time = test(net_eval, eval_dataloader, args)
                    log_and_print("Eval_Syn (Eval): It_eval: {}\tEpoch: {}\t{}@Top-k Acc:{}\tEval time:{}".format(it_eval, ep, args.target_measure.upper(), eval_topk_acc, epoch_eval_time))
                    
                    # 若有一半以上的指标有提升, 则更新验证集效果指标, 并在测试集上测试模型效果
                    cur_eval_topk_acc = torch.tensor(list(eval_topk_acc.values()))
                    if (cur_eval_topk_acc > best_eval_topk_acc_per_eval).sum() > (metric_num / 2):  # 有一半以上的指标有提升
                        log_and_print(f"[Epoch {ep}] Best Eval Score Updated, Test Model on Test Dataset...")
                        # 更新验证集效果指标
                        best_eval_topk_acc_per_eval = cur_eval_topk_acc
                        
                        # 在测试集上测试模型效果 
                        test_topk_acc, test_time = test(net_eval, test_dataloader, args)
                        log_and_print("Eval_Syn (Test): It_eval: {}\tEpoch: {}\t{}@Top-k Acc:{}\tTest time:{}\n".format(it_eval, ep, args.target_measure.upper(), test_topk_acc, test_time))
                
                training_time_list.append(time.time() - training_time_per_eval_start)  # 该模型该次训练的端到端训练时长
                best_topk_acc_list.append(torch.tensor(list(test_topk_acc.values())))  # 该模型该次训练在测试集上最好的结果 

            cur_best_acc[model_eval] = torch.stack(best_topk_acc_list).mean(0)  # args.num_eval个模型最好效果的均值
            cur_best_acc_training_time[model_eval] = sum(training_time_list) / len(training_time_list)  # args.num_eval个模型训练时长的均值
            all_acc.append((cur_best_acc[model_eval], cur_best_acc_training_time[model_eval])) 

        # model_eval_pool中模型所有指标均值中，有一半以上有所提升, 则说明该浓缩数据集更好
        if not best_acc:
            save_this_it = True
        else: 
            save_this_it = (np.stack(list(cur_best_acc.values())).mean(0) > np.stack(list(best_acc.values())).mean(0)).sum() > (metric_num / 2)


        if save_this_it:  # 如果该合成数据集更好, 则储存该合成数据集, 并更新best_topk_acc
            best_acc.update(cur_best_acc)
            best_acc_training_time.update(cur_best_acc_training_time)
            with torch.no_grad():
                x_syn_save = [x_syn[i].detach().cpu() for i in range(syn_num)]
                y_syn_save = y_syn.detach().cpu()
                torch.save(x_syn_save, os.path.join(save_dir, "syn_trajs_best.pt"))
                torch.save(y_syn_save, os.path.join(save_dir, "syn_ys_best.pt"))
        log_and_print(f"浓缩数据集效果: 第{it}个循环\ncur_best_acc:{cur_best_acc}\ntraining_time:{cur_best_acc_training_time}")
        log_and_print(f"best_acc:{best_acc}\ntraining_time:{best_acc_training_time}")
    
    start_time = time.time()
    # 训练浓缩数据集
    net = get_model(args, config, traj_data)
    net.train()

    for param in list(net.parameters()):  # 将net的所有参数的requires_grad置为False
        param.requires_grad = False
    
    loss_avg = 0
    ''' update synthetic data '''
    loss = torch.tensor(0.0).to(args.device)
    # 合成轨迹的一个batch
    syn_minibatch_idx = torch.randperm(syn_num)[:args.use_syn_minibatch]  # 合成轨迹的idx
    x_syn_minibatch = [x_syn[i] for i in syn_minibatch_idx]  # 挑选一个batch的合成轨迹

    # 该batch的合成轨迹对应的真实轨迹
    real_minibatch_idx = []
    for idx in syn_minibatch_idx:
        real_minibatch_idx.extend(list(range(idx * args.use_num_real_per_syn, (idx + 1) * args.use_num_real_per_syn))[:args.use_num_real_per_syn])
    x_real_minibatch = [train_dataloader.dataset.traj_coor[i] for i in real_minibatch_idx]
    
    # x_real_minibatch_vec = getEmbeddings(net, args.model, x_real_minibatch, args.device, traj_data,"soft")
    x_real_minibatch_vec = getEmbeddings(net, args.model, x_real_minibatch, args.device, traj_data,args.grid_mode)
    x_real_minibatch_vec = nn.functional.normalize(x_real_minibatch_vec, dim=1)

    # x_syn_minibatch_vec = getEmbeddings(net, args.model, x_syn_minibatch, args.device, traj_data,"soft")
    x_syn_minibatch_vec = getEmbeddings(net, args.model, x_syn_minibatch, args.device, traj_data,args.grid_mode)
    x_syn_minibatch_vec = nn.functional.normalize(x_syn_minibatch_vec, dim=1)

    loss += torch.sum((torch.mean(x_real_minibatch_vec, dim=0) - torch.mean(x_syn_minibatch_vec, dim=0))**2)

    optimizer_data.zero_grad()
    loss.backward()
    optimizer_data.step()  # 更新数据
    loss_avg += loss.item()

    if it%10 == 0:
        log_and_print('%s iter = %04d, loss = %.4f' % (str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())), it, loss_avg))
    end_time = time.time()
    log_and_print(f"time cost: {end_time - start_time}")

log_and_print(f"最终浓缩数据集效果: 第{it}个循环\nbest_acc:{best_acc}\ntraining_time:{best_acc_training_time}")
log_and_print("每轮浓缩数据集效果：")
for acc in all_acc:
    log_and_print(acc)
