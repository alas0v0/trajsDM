# 计算两个traj vector之间的相似度的函数, 来自于SIMformer
# Tool components for calculating similarity
# 1. pairwise_emd2sim: Given pairs of embedding vectors, calculate their pairwise similarity
# 2. emds2sim: Given embedding vectors, calculate the similarity matrix

from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import ot
import torch 
from torch import nn
import numpy as np

def chebyshev_distance(v1, v2):
    """
    Compute the Chebyshev distance between two batches of vectors.

    Parameters:
    - A (torch.Tensor): Tensor of shape (batch_size, hidden_dim)
    - B (torch.Tensor): Tensor of shape (batch_size, hidden_dim)

    Returns:
    - torch.Tensor: Tensor of shape (batch_size,) containing the Chebyshev distances
    """
    # calculate the absolute difference between v1 and v2
    diffs = torch.abs(v1 - v2)
    # following the direction of hidden_dim, calculate the maximum value of the absolute difference
    max_diffs = torch.max(diffs, dim=-1).values
    
    return max_diffs

def chebyshev_distance_matrix(X):
    """
    Compute the Chebyshev distance matrix for a batch of vectors.
    Parameters:
    - X (torch.Tensor): Tensor of shape (batch_size, hidden_dim)
    Returns:
    - torch.Tensor: Distance matrix of shape (batch_size, batch_size),
                    where element [i, j] is the Chebyshev distance between
                    vectors X[i] and X[j].
    """
    # expand the dimension of X to shape (batch_size, 1, hidden_dim) 
    X_expanded = X.unsqueeze(1)
    # expand the dimension of X to shape (1, batch_size, hidden_dim)
    X_tiled = X.unsqueeze(0)
    # calculate the absolute difference between X_expanded and X_tiled along the hidden_dim dimension
    diffs = torch.abs(X_expanded - X_tiled)
    # calculate the Chebyshev distance matrix on hidden_dim dimension
    distance_matrix = torch.max(diffs, dim=2).values
    return distance_matrix
        
def pairwise_emd2sim(v1, v2, metric="euc"):
    """
    Given two batches of vectors, calculate the pairwise similarity between them. Supports cos, euc, chebyshev.
    - v1: tensor, shape=(n, d), where n is the number of samples, and d is the feature dimension.
    - v2: tensor, shape=(n, d), where n is the number of samples, and d is the feature dimension.
   - metric: str, method of similarity calculation, named Representation Similarity Function in our paper 
        - cos: Cosine similarity
        - euc: Euclidean distance
        - chebyshev: Chebyshev distance
    """ 
    cos_sim = nn.CosineSimilarity(dim=-1) 
    if metric == "cos":
        sim_score = cos_sim(v1, v2)
    if metric == "euc":
        # predict_dis_matrix = torch.norm((v1 - v2), p=2, dim=-1)
        predict_dis_matrix = torch.sqrt(((v1 - v2)**2).sum(dim=-1) + 1e-5)
        sim_score = torch.exp(-predict_dis_matrix) # Map to a similarity space of 0-1
    if metric == "chebyshev":
        sim_score = torch.exp(-chebyshev_distance(v1, v2))
    return sim_score    

def pairwise_emd2dist(v1, v2, metric="euc"):
    """Calculate pairwise embedding distance for the selected metric."""
    if metric == "euc":
        return torch.linalg.vector_norm(v1 - v2, ord=2, dim=-1)
    if metric == "cos":
        return 1.0 - nn.CosineSimilarity(dim=-1)(v1, v2)
    if metric == "chebyshev":
        return chebyshev_distance(v1, v2)
    raise ValueError(f"Unsupported embedding metric: {metric}")

def emds2sim(emds, metric="euc"):
    """ DTW时, 计算向量间cos距离; Haus/费雷歇时, 计算向量间切比雪夫距离并用负对数处理
    Given embedding vectors, calculate the similarity matrix. Supports cos, euc, chebyshev.
    - emds: tensor, shape=(n, d), where n is the number of samples, and d is the feature dimension.
    - metric: str, method of similarity calculation, named Representation Similarity Function in our paper 
        - cos: Cosine similarity
        - euc: Euclidean distance
        - chebyshev: Chebyshev distance
    """
    if metric == "cos":
        sim_matrix = pairwise_cosine_similarity(emds, zero_diagonal=False)
    if metric == "euc":
        predict_dis_matrix = torch.cdist(emds, emds, p=2)  # a-b再求2范数
        sim_matrix = torch.exp(-predict_dis_matrix) # Map to a similarity space of 0-1
    if metric == "chebyshev":
        sim_matrix = torch.exp(-chebyshev_distance_matrix(emds))
    
    return sim_matrix

def emds2dist(emds, metric="euc"):
    if metric == "euc":
        dist_matrix = torch.cdist(emds, emds, p=2)
    elif metric == "cos":
        sim_matrix = pairwise_cosine_similarity( emds, zero_diagonal=False, )
        dist_matrix = 1.0 - sim_matrix
    elif metric == "chebyshev":
        dist_matrix = chebyshev_distance_matrix(emds)
    dist_matrix.fill_diagonal_(0.0)
    return dist_matrix

def pairwise_distance(X, metric="euclidean"):
    """
    X: [N, d] 特征矩阵
    return: [N, N] 距离矩阵
    """
    X = torch.as_tensor(X, dtype=torch.float32)

    if metric == "euclidean":
        D = torch.cdist(X, X, p=2)
    elif metric == "cosine":
        X = torch.nn.functional.normalize(X, dim=-1)
        D = 1.0 - X @ X.T
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return D.clamp_min(0)


def compute_T(
    C_real,
    C_syn,
    p=None,
    q=None,
    epsilon=0.05,
    tol=1e-4,
    max_iter=300,
    numItermax=None,
    verbose=False,
):
    if p is None:
        p = torch.full(
            (C_real.shape[0],),
            1.0 / C_real.shape[0],
            dtype=C_real.dtype,
            device=C_real.device,
        )
    else:
        p = torch.as_tensor(p, dtype=C_real.dtype, device=C_real.device)
        p = torch.clamp(p, min=1e-12)
        p = p / p.sum()

    if q is None:
        q = torch.full(
            (C_syn.shape[0],),
            1.0 / C_syn.shape[0],
            dtype=C_syn.dtype,
            device=C_syn.device,
        )
    else:
        q = torch.as_tensor(q, dtype=C_real.dtype, device=C_real.device)
        q = torch.clamp(q, min=1e-12)
        q = q / q.sum()

    gw_kwargs = {
        "C1": C_real,
        "C2": C_syn.detach(),
        "p": p,
        "q": q,
        "loss_fun": "square_loss",
        "epsilon": epsilon,
        "tol": tol,
        "max_iter": max_iter,
        "verbose": verbose,
        "log": verbose,
    }
    if numItermax is not None:
        gw_kwargs["numItermax"] = numItermax

    result = ot.gromov.entropic_gromov_wasserstein(**gw_kwargs)
    if verbose:
        T, ot_log = result
        errors = ot_log.get("err", [])
        final_error = errors[-1] if errors else None
        print(
            f"[compute_T] iterations={len(errors)}, "
            f"final_error={final_error}"
        )
    else:
        T = result
    T = torch.as_tensor(T, dtype=C_syn.dtype, device=C_syn.device)

    return T
