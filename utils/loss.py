import torch
import torch.nn as nn
import torch.nn.functional as functional
from utils.similarity import pairwise_emd2sim, pairwise_emd2dist


def initialize_loss(args):
    if args.loss_type == "mse":
        my_loss = SimpleMSELoss(emb_sim_metric=args.emb_sim_metric)
    elif args.loss_type == "triplet":
        my_loss = TripletLoss(args.train_epochs)
    else:
        raise ValueError("Loss Type Error")
    return my_loss


class SimpleMSELoss(nn.Module):
    def __init__(self, emb_sim_metric="euc"):
        super(SimpleMSELoss, self).__init__()
        self.emb_sim_metric = emb_sim_metric  # 'cos', 'euc', 'chebyshev'
        self.mse = nn.MSELoss()
        print(f"[Loss Function] Using SimpleMSELoss")
        
    def get_pred_sim(self, traj_vecs1, traj_vecs2):
        # return pairwise_emd2sim(traj_vecs1, traj_vecs2, self.emb_sim_metric)
        return pairwise_emd2dist(traj_vecs1, traj_vecs2, self.emb_sim_metric)

    
    def forward(self, traj_vecs1, traj_vecs2, target_dist):
        pred_dist = pairwise_emd2dist(traj_vecs1, traj_vecs2, self.emb_sim_metric)
        return self.mse(pred_dist, target_dist)
    

class TripletLoss(nn.Module):
    def __init__(self, epochs):
        super(TripletLoss, self).__init__()
        self.mlp_r, self.m = 1, 1
        
        step = epochs // 5
        self.Ls = {
            step * 0: (0, 10),
            step * 1: (10, 10),
            step * 2: (10, 1),
            step * 3: (5, 0.1),
            step * 4: (1, 0.01),
        }

        self.init_loss(0)
        print(f"[Loss Function] Using TripletLoss")
    
    def init_loss(self, epoch):
        pass

    def forward(self,
                epoch,
                p_input,
                p_target,
                n_input,
                n_target,
                np_input,
                np_target,
                p_weights = None,
                n_weights = None):
        
        if epoch in self.Ls:
            self.mlp_r, self.m = self.Ls[epoch]

        #############################
        ######### MLP loss ##########
        #############################
        threshold = n_target - p_target
        rank_loss_mlp = functional.relu(p_input - n_input + threshold)

        #TODO: 有机会可以不用euc, 还可以尝试其他的距离, 例如cos, 切比雪夫等
        positive_mse_loss = (p_input - p_target) ** 2
        negative_mse_loss = (n_input - n_target) ** 2
        np_mse_loss       = (np_input - np_target) ** 2
        
        if p_weights is not None:
            positive_mse_loss *= p_weights
        if n_weights is not None:
            negative_mse_loss *= n_weights

        mse_loss = positive_mse_loss + negative_mse_loss + np_mse_loss

        return torch.mean(self.mlp_r * rank_loss_mlp),\
               torch.mean(self.m * torch.sqrt(mse_loss)),\
               torch.mean(self.mlp_r * rank_loss_mlp + self.m * torch.sqrt(mse_loss))


def gw_loss_chunked(C_real, C_syn, T, real_block_size=256):
    """
    C_real: [n_real, n_real]
    C_syn:  [n_syn, n_syn]
    T:      [n_real, n_syn]

    return:
        标量 gw loss
    """
    p = T.sum(dim=1)  # [n_real]
    q = T.sum(dim=0)  # [n_syn]

    term_real = (C_real.pow(2) * torch.outer(p, p)).sum()
    term_syn = (C_syn.pow(2) * torch.outer(q, q)).sum()

    cross = C_real.new_zeros(())
    Tt = T.transpose(0, 1)  # [n_syn, n_real]

    n_real = C_real.shape[0]
    for start in range(0, n_real, real_block_size):
        end = min(start + real_block_size, n_real)

        T_block = T[start:end]                  # [bs, n_syn]
        M_block = T_block @ C_syn @ Tt         # [bs, n_real]
        cross = cross + (C_real[start:end] * M_block).sum()

    loss = term_real + term_syn - 2.0 * cross
    return loss

def dm_loss(x_syn_vec, x_real_vec, T):
    # T: 合成轨迹，真实轨迹
    T_sum = T.sum(dim=1, keepdim=True)
    weighted_real = T @ x_real_vec
    weighted_real_squared = T @ (x_real_vec ** 2)

    weighted_diff = (
        (x_syn_vec ** 2) * T_sum
        - 2 * x_syn_vec * weighted_real
        + weighted_real_squared
    )

    return torch.mean(torch.sum(weighted_diff, dim=1))