import os 
import torch 
import random 
import logging 
import numpy as np

def log_ini(save_dir):
    log_file = os.path.join(save_dir, "printed_info.txt")
    logging.basicConfig(
        filename=log_file, 
        level=logging.INFO, 
        format='%(asctime)s %(message)s', 
        filemode='w')   
    # 立即记录一条日志
    logging.info("Start logging")


def log_and_print(message):
    print(message)
    logging.info(message)


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def pad_traj_tensor_to_fix_length(traj_list, max_length, pad_value=0.0):
    device = traj_list[0].device
    dtype = traj_list[0].dtype
    padded_trajs = torch.full(
        size=[len(traj_list), max_length, 2], 
        fill_value=pad_value, 
        dtype=dtype, 
        device=device)
    for i, traj in enumerate(traj_list):
        padded_trajs[i, :traj.shape[0]] = traj 
    return padded_trajs


def get_padded_mask(padded_trajs):
    return padded_trajs[:, :, -1].ne(0)


def get_normalized_padded_traj( padded_trajs, padding_masks, mean_lon, mean_lat, std_lon, std_lat):  # 在原本tensor上修改
    # device, batch_size = padded_trajs.device, padded_trajs.shape[0]
    # return ((padded_trajs - self.mean_x.to(device)) / self.std_x.to(device)) * padding_masks.view(batch_size, -1, 1)
    padded_trajs[:, :, 0] = ((padded_trajs[:, :, 0] - mean_lon) / std_lon) * padding_masks  # lon z-score标准化
    padded_trajs[:, :, 1] = ((padded_trajs[:, :, 1] - mean_lat) / std_lat) * padding_masks  # lat z-score标准化
    return padded_trajs

def get_normalized_padded_traj_convtraj(padded_trajs, padding_masks, lon_min, lat_min):  # 在原本tensor上修改
    # device, batch_size = padded_trajs.device, padded_trajs.shape[0]
    #convtraj 原始连续轨迹处理
    padded_trajs[:, :, 0] = (100 * (padded_trajs[:, :, 0] - lon_min)) * padding_masks  # lon z-score标准化
    padded_trajs[:, :, 1] = (100 * (padded_trajs[:, :, 1] - lat_min)) * padding_masks  # lat z-score标准化
    return padded_trajs


# def getWeights(trajs, valid_mask, x_num, y_num, temperature, grid_size):
#     batch_size = trajs.shape[0]
#     device = trajs.device
#     dtype = trajs.dtype

#     x = trajs[..., 0] / grid_size
#     y = trajs[..., 1] / grid_size
#     center_x = torch.floor(x).to(torch.long)
#     center_y = torch.floor(y).to(torch.long)
#     valid_bool = valid_mask > 0
#     temp = max(float(temperature), 1e-12)

#     neighbor_offsets = torch.tensor(
#         [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
#         device=device,
#         dtype=torch.long,
#     )
#     offset_x = neighbor_offsets[:, 0].view(1, 1, -1)
#     offset_y = neighbor_offsets[:, 1].view(1, 1, -1)

#     ngx = center_x.unsqueeze(-1) + offset_x
#     ngy = center_y.unsqueeze(-1) + offset_y
#     inside = ( valid_bool.unsqueeze(-1) & (ngx >= 0) & (ngx < x_num) & (ngy >= 0) & (ngy < y_num) )

#     ndist2 = (x.unsqueeze(-1) - (ngx.to(dtype) + 0.5)) ** 2 + (y.unsqueeze(-1) - (ngy.to(dtype) + 0.5)) ** 2
#     # logits = -ndist2 / temp
#     # logits = torch.where(inside, logits, torch.full_like(logits, -1e9))
#     # weights = torch.softmax(logits, dim=-1)
#     weights=1/(1+ (ndist2 / temp)**2)
#     weights = torch.where(inside, weights, torch.zeros_like(weights))

#     return weights