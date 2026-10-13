import torch 
from .dual_attention import TrajCL
from .shortcut import ShortCutCNN as ConvTraj
from .simformer import SIMformer
from .dualsimformer import DualSIMformer
from .other_models import my_GRU, my_LSTM, my_Linear


def get_model(args, config, traj_data):
    model_config = config["model"][args.model]
    if args.model == "ConvTraj":
        return ConvTraj(
            lon_input_size=traj_data.grid.grid_num_x,  # x轴的grid num 
            lat_input_size=traj_data.grid.grid_num_y,  # y轴的grid num
            target_size=args.hidden_dim,  # 轨迹embedding size
            batch_size=args.batch_size,  # 实际未用到
            sampling_num=args.sampling_num,  # 实际为用到
            max_seq_length=traj_data.max_traj_len.item(), 
            channel=model_config["channel"], 
            device=args.device, 
            head_num=model_config["head_num"]
            ).to(args.device)
    elif args.model == "TrajCL":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return TrajCL(
            grid_num=traj_data.grid.grid_num_x*traj_data.grid.grid_num_y,
            grid_embedding_size=args.grid_emb_size,  # grid embedding size
            ninput=model_config["ninput"], 
            nhidden=model_config["nhidden"],  
            output_embedding_size=args.hidden_dim,
            nhead=model_config["nhead"], 
            nlayer=model_config["hlayer"], 
            attn_dropout=model_config["attn_dropout"], 
            pos_dropout=model_config["pos_dropout"]
            ).to(args.device)
    
    elif args.model == "DualSIMformer":
        # 在模型初始化之前添加这些行
        torch.backends.cuda.enable_flash_sdp(False)  # 禁用Flash Attention
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # 禁用内存高效Attention
        torch.backends.cuda.enable_math_sdp(True)  # 启用数学实现（最稳定）
        return DualSIMformer(
            grid_num = traj_data.grid.grid_num_x*traj_data.grid.grid_num_y, 
            grid_embedding_size = args.grid_emb_size, 
            hidden_dim = args.hidden_dim,
        ).to(args.device)

    elif args.model == "SIMformer":
        # 在模型初始化之前添加这些行
        torch.backends.cuda.enable_flash_sdp(False)  # 禁用Flash Attention
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # 禁用内存高效Attention
        torch.backends.cuda.enable_math_sdp(True)  # 启用数学实现（最稳定）
        return SIMformer(
        feat_dim=2,  # input dim
        hidden_dim=args.hidden_dim,  # 输入2维轨迹输入SIMformer后被线性映射到args.hidden_dim维, 也是Transformer的input_dim, 亦是轨迹被编码为向量的维度
        num_layers=model_config["num_layers"],  # Transformer Encoder中block的数量
        n_heads=model_config["n_heads"], 
        dimfeedforward=2 * args.hidden_dim,  # Transformer中feedforward network的hidden_size
        pos_encoding=model_config["pos_encoding"],  # "learnable" or "fixed"
        ).to(args.device)
    
    elif args.model == "GRU":
        return my_GRU(
            feat_dim=2,
            hidden_dim=args.hidden_dim,
            num_layers=model_config["num_layers"],
            hidden_size=model_config["hidden_size"], 
            bidirectional=model_config["bidirectional"],
            pos_encoding=model_config["pos_encoding"],
        ).to(args.device)
    elif args.model == "LSTM":
        return my_LSTM(
            feat_dim=2,
            hidden_dim=args.hidden_dim,
            num_layers=model_config["num_layers"],
            hidden_size=model_config["hidden_size"], 
            bidirectional=model_config["bidirectional"],
            pos_encoding=model_config["pos_encoding"],
        ).to(args.device)
    else:
        return my_Linear(
            feat_dim=2, 
            hidden_dim=args.hidden_dim
        ).to(args.device)



