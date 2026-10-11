import argparse


def int_or_none(value: str):
    """将命令行字符串解析为整数或 None, 同时支持输入None、null或者int类型"""
    if value.lower() in {"none", "null"}:
        return None

    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{value!r} 必须是整数、None 或 null"
        ) from exc


def build_parser():
    parser = argparse.ArgumentParser()

    # experiment related
    parser.add_argument("--dataset", type=str, default="porto", choices=["porto","geolife","ais","tdrive"], help='Dataset.')
    parser.add_argument('--model', type=str, default='ConvTraj', choices=["ConvTraj", "TrajCL", "DualSIMformer", "SIMformer", "GRU", "LSTM", "Linear"], help='model')
    parser.add_argument("--config_path", type=str, default="./config.yaml", help="config file path")
    parser.add_argument('--gpu_id', type=int, default=2, help='gpu id used') 
    parser.add_argument('--seed', type=int, default=0, help='seed')

    # grid index related
    parser.add_argument('--grid_size', type=float, default=0.01, help='测试合成数据集效果、MTT训练training track和DC内层训练时的l2 regularization')
    parser.add_argument('--grid_emb_size', type=int, default=32, help="grid编码成向量的长度")
    parser.add_argument('--temperature', type=float, default=0.2 ,help="生成连续image用到的超参数")
    
    # model related
    parser.add_argument("--batch_size", type=int, default=32, help='batch size for training networks')
    parser.add_argument('--train_epochs', type=int, default=100, help="训练模型的轮数, training track的长度为train_epochs+1")
    parser.add_argument('--hidden_dim', type=int, default=128, help="轨迹编码成向量的长度")
    parser.add_argument("--opt_method", type=str, default="adam", choices=["adam", "sgd"], help="优化方法类型")
    parser.add_argument("--loss_type", type=str, default="triplet", choices=["triplet", "mse"], help="优化方法类型")
    parser.add_argument('--mom', type=float, default=0.9, help='测试合成数据集效果、MTT训练training track和DC内层训练时的momentum')
    parser.add_argument('--l2', type=float, default=0, help='测试合成数据集效果、MTT训练training track和DC内层训练时的l2 regularization')

    # Trajectory similarity measures related
    parser.add_argument("--sampling_num", type=int, default=1, help="为了效率做了修改, 本质上没有用到; how many target trajectories to select for anchor trajectory when training the trajectory encoding model") 
    parser.add_argument("--target_measure", type=str, default="dtw", choices=["dtw","haus","fret"], help="distance measures for ground truth similarity")
    parser.add_argument("--emb_sim_metric", type=str, default="euc", choices=["cos","euc","chebyshev"], help="representation simialrity function") 

    # dataset condensation related
    parser.add_argument('--distill_output_path', type=str, default='./distill_files_test', help='distill output dir')
    parser.add_argument('--file_output_path', type=str, default='./logs_files', help='distill output dir')
    parser.add_argument('--syn_num', type=int, default=100, help='合成数据集的大小')
    parser.add_argument('--Iteration', type=int, default=500, help='更新多少次浓缩数据集; how many distillation steps to perform')
    parser.add_argument('--lr_traj', type=float, default=0.1, help='learning rate for updating synthetic trajectories')
    parser.add_argument('--use_syn_minibatch', type=int, default=25, help='就是b_{t+n};每次更新浓缩数据集时, sample出的minibatch的数量, None表示用整个浓缩数据集来更新')
    parser.add_argument('--use_num_real_per_syn', type=int, default=10, help='每条合成轨迹对应多少条真实轨迹')
    parser.add_argument('--eval_it', type=int, default=50, help='每更新args.eval_it次浓缩数据集, 就测试一次浓缩数据集的效果;how often to evaluate')
    parser.add_argument('--num_eval', type=int, default=1, help='测试浓缩数据集效果时, 每个模型重复测几次; how many networks to evaluate on')
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='100测试浓缩数据集效果时, 要用浓缩数据集训练模型多少个epochs; epochs to train a model with synthetic data')
    parser.add_argument("--syn_ini", type=str, default="randomly_select", choices=["randomly_generated", "randomly_select", "k-medoids"], help="合成数据集更新时, 长度是否可变")
    parser.add_argument("--syn_method", type=str, default="DM", choices=["DM", "DC", "MTT"], help="合成数据集时所用的方法")
    
    # MTT buffer related
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--num_experts', type=int, default=10, help="一个buffer中training track的数量")
    parser.add_argument('--save_interval', type=int, default=10, help="每生成多少条training track就存一次文件")
    
    # MTT distill related
    parser.add_argument('--max_files', type=int_or_none, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int_or_none, default=None, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument('--lr_lr', type=float, default=1e-5, help='1e-2或1e-3 learning rate for updating... learning rate')
    parser.add_argument('--max_start_epoch', type=int, default=45, help='e_0的最大值; max epoch we can start at')
    parser.add_argument('--expert_epochs', type=int, default=3, help='就是M; how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=10, help='就是N; 训练时, 用浓缩数据集训练多少个epoch; how many steps to take on synthetic data')
    # parser.add_argument("--syn_sampling_num", type=int, default=5, help="用浓缩数据集训练模型时, 为anchor trajectory随机挑选多少个target trajectory")
    
    # 可变长度相关
    parser.add_argument("--syn_len", type=str, default="learnable_mask", choices=["fixed", "learnable"], help="合成数据集更新时, 长度是否可变")
    parser.add_argument("--ste", type=str, default="E_len_global", choices=["E_len_norm", "E_len_loc_glo", "cdf", "E_len_local", "E_len_global", "E_len_loc_dis_sen", "E_len_glo_dis_sen"], help="合成数据集更新时, 若长度可变, 生成mask的反向传播过程怎么传递梯度")
    parser.add_argument('--syn_len_max', type=int, default=200, help='合成数据集中轨迹的最大长度')    
    parser.add_argument('--lr_len', type=float, default=0.5, help='学习轨迹长度的lr')
    parser.add_argument('--syn_len_para_ini', type=str, default='{"method": "dis_laplace", "scale": 1.0}', help="len_logits的初始化参数字典")
    parser.add_argument('--len_norm_factor', type=float, default=10000, help='len_logits的范数系数, 0表示没有正则项')

    # augmentation和对比学习相关
    parser.add_argument('--track_aug', type=str, default='null', choices=["null", "straight", "simplify", "shift", "mask", "subset"], help="data augment methods, null表示原始数据") 
    parser.add_argument('--distill_aug', type=str, default='null', choices=["null", "straight", "simplify", "shift", "mask", "subset"], help="data augment methods, null表示原始数据") 
    parser.add_argument('--syn_eval_aug', type=str, default='null', choices=["null", "straight", "simplify", "shift", "mask", "subset"], help="data augment methods, null表示原始数据") 
    parser.add_argument("--traj_simp_dist", type=float, default=0.001, help="rdp的epsilon, 0.001的话阈值大概定在100米左右")
    parser.add_argument("--traj_mask_ratio", type=float, default=0.3, help="删除多少百分比的轨迹点")
    parser.add_argument("--traj_subset_ratio", type=float, default=0.5, help="截取多少百分比的轨迹点")
    parser.add_argument("--cl_K", type=int, default=2048, help="Moco负样本词典大小") 
    # parser.add_argument("--cl_m", type=float, default=0.999, help="Moco键编码器动量") 
    parser.add_argument("--cl_T", type=float, default=0.05, help="0.007Moco (InfoNCE)的温度系数; τ越小，分布越尖锐，会放大相似度的差异, 模型会越关注困难的负样本（即与正样本相似度高的负样本）；τ越大，分布越平缓，会缩小相似度的差异; 当batch_size较小或数据质量较差时, 使用较大的τ")
    parser.add_argument("--cl_K_syn_eval", type=int, default=300, help="Moco负样本词典大小")  #TODO: 改正比例
    # parser.add_argument("--cl_m_syn_eval", type=float, default=0.999, help="Moco键编码器动量") 
    parser.add_argument("--cl_T_syn_eval", type=float, default=1.0, help="Moco (InfoNCE)的温度系数; τ越小，分布越尖锐，会放大相似度的差异, 模型会越关注困难的负样本（即与正样本相似度高的负样本）；τ越大，分布越平缓，会缩小相似度的差异; 当batch_size较小或数据质量较差时, 使用较大的τ")
    # parser.add_argument("--cl_K_distill", type=int, default=300, help="Moco负样本词典大小") 
    # parser.add_argument("--cl_m_distill", type=float, default=0.999, help="Moco键编码器动量")    
    parser.add_argument("--cl_T_distill", type=float, default=0.01, help="Moco (InfoNCE)的温度系数; τ越小，分布越尖锐，会放大相似度的差异, 模型会越关注困难的负样本（即与正样本相似度高的负样本）；τ越大，分布越平缓，会缩小相似度的差异; 当batch_size较小或数据质量较差时, 使用较大的τ")
    parser.add_argument('--nce_factor', type=float, default=0.01, help='nce_loss前面乘的系数')
    return parser


def parse_args():
    """
    普通脚本传参
    """
    parser = build_parser()
    return parser.parse_args()


def parse_args_jupyter(args_list=None):
    """
    jupyter传参
    """
    if args_list is None:
        args_list = []

    parser = build_parser()
    return parser.parse_args(args=args_list)
