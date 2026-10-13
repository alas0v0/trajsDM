import torch
import torch.nn as nn
import torch.nn.functional as functional
from utils.similarity import pairwise_emd2sim


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
        return pairwise_emd2sim(traj_vecs1, traj_vecs2, self.emb_sim_metric)
    
    def forward(self, traj_vecs1, traj_vecs2, target_dist):
        pred_dist = pairwise_emd2sim(traj_vecs1, traj_vecs2, self.emb_sim_metric)
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