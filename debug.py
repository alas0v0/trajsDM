"""Run a short DM distillation loop and inspect the OT coupling matrix."""

import argparse
import datetime
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from evaluate import getEmbeddingsFromFull
from models import get_model
from utils.data_process import TrajData
from utils.loss import dm_loss
from utils.similarity import compute_T


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def coupling_stats(T):
    """Return statistics for row-normalized pi = T.T."""
    pi = T.T
    pi_cond = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-12)
    safe_pi = pi_cond.clamp_min(1e-12)
    entropy = -(pi_cond * safe_pi.log()).sum(dim=1)

    return {
        "pi_shape": list(pi.shape),
        "T_min": float(T.min().item()),
        "T_max": float(T.max().item()),
        "T_mass": float(T.sum().item()),
        "pi_row_sum_min": float(pi.sum(dim=1).min().item()),
        "pi_row_sum_max": float(pi.sum(dim=1).max().item()),
        "conditional_entropy_mean": float(entropy.mean().item()),
        "conditional_entropy_min": float(entropy.min().item()),
        "conditional_entropy_max": float(entropy.max().item()),
        "conditional_max_prob_mean": float(pi_cond.max(dim=1).values.mean().item()),
        "conditional_max_prob_min": float(pi_cond.max(dim=1).values.min().item()),
        "conditional_max_prob_max": float(pi_cond.max(dim=1).values.max().item()),
    }


def build_model_args(args, device):
    return SimpleNamespace(
        model=args.model,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        sampling_num=1,
        neighbor_num=args.neighbor_num,
        temperature=args.temperature,
        grid_emb_size=args.grid_emb_size,
        device=device,
        grid_mode=args.grid_mode,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="./config.yaml")
    parser.add_argument("--dataset", choices=["porto", "geolife", "ais", "tdrive"], default="porto")
    parser.add_argument("--model", choices=["ConvTraj", "TrajCL", "DualSIMformer", "SIMformer", "GRU", "LSTM", "Linear"], default="DualSIMformer")
    parser.add_argument("--target_measure", choices=["dtw", "haus", "fret"], default="dtw")
    parser.add_argument("--grid_mode", choices=["soft", "hard"], default="soft")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid_size", type=float, default=0.01)
    parser.add_argument("--grid_emb_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--neighbor_num", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--syn_num", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--lr_traj", type=float, default=0.1)
    parser.add_argument("--syn_ini", choices=["randomly_select"], default="randomly_select")
    parser.add_argument("--batch_embed", type=int, default=128)
    parser.add_argument("--ot_epsilon", type=float, default=0.05)
    parser.add_argument("--ot_tol", type=float, default=1e-4)
    parser.add_argument("--max_ot_iter", type=int, default=200)
    parser.add_argument("--output_dir", default="./debug_pi")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    with open(args.config_path) as file:
        config = yaml.safe_load(file)

    traj_data = TrajData(args.dataset, args.target_measure, config["datasets"], args.grid_size)
    # TrajSimLeaDataLoader normally adds these fields to the shared Grid.
    # debug.py calls getEmbeddingsFromFull directly, so initialize them here.
    traj_data.grid.neigh_num = args.neighbor_num
    traj_data.grid.temperature = args.temperature
    train_trajs, _ = traj_data.train_data[:2]
    if not 0 < args.syn_num <= len(train_trajs):
        raise ValueError(f"--syn_num must be between 1 and {len(train_trajs)}")

    indices = torch.randperm(len(train_trajs))[:args.syn_num]
    x_syn = [train_trajs[i].clone().detach().to(device) for i in indices]

    model_args = build_model_args(args, device)
    x_syn = torch.nn.ParameterList(x_syn)
    optimizer = torch.optim.SGD(x_syn, lr=args.lr_traj, momentum=0.5)
    output_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    # Keep the full real set in its original storage location; embeddings are
    # moved batch by batch inside getEmbeddingsFromFull.
    real_trajs = list(train_trajs)
    summary = []
    print(
        f"device={device}, model={args.model}, syn_num={args.syn_num}, "
        f"iterations={args.iterations}"
    )
    print(
        f"OT parameters: epsilon={args.ot_epsilon}, tol={args.ot_tol}, "
        f"max_iter={args.max_ot_iter}"
    )
    print(f"saving coupling matrices to {output_dir}")

    for iteration in range(args.iterations):
        # A fresh random model is used for every distillation iteration.
        net = get_model(model_args, config, traj_data)
        net.eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)

        real_vec = getEmbeddingsFromFull(
            net, args.model, real_trajs, model_args, traj_data,
            batch_size=args.batch_embed, required_grad=False,
        )
        syn_vec = getEmbeddingsFromFull(
            net, args.model, list(x_syn), model_args, traj_data,
            batch_size=args.batch_embed, required_grad=True,
        )
        C_real = torch.cdist(real_vec, real_vec, p=2)
        C_syn = torch.cdist(syn_vec, syn_vec, p=2)
        T = compute_T(
            C_real,
            C_syn,
            epsilon=args.ot_epsilon,
            tol=args.ot_tol,
            max_iter=args.max_ot_iter,
            verbose=True,
        )
        pi = T.T

        # T is fixed during this update, matching main_DM.py.
        loss = dm_loss(syn_vec, real_vec, pi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats = coupling_stats(T)
        stats.update({
            "iteration": iteration,
            "loss_dm": float(loss.item()),
            "C_real_mean": float(C_real.mean().item()),
            "C_real_max": float(C_real.max().item()),
            "C_syn_mean": float(C_syn.mean().item()),
            "C_syn_max": float(C_syn.max().item()),
            "T_abs_max": float(T.abs().max().item()),
            "T_exact_zero_count": int(torch.count_nonzero(T == 0).item()),
            "T_allclose_zero": bool(torch.allclose(T, torch.zeros_like(T))),
        })
        summary.append(stats)

        pi_cond = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-12)
        torch.save(
            {
                "T": T.detach().cpu(),
                "pi": pi.detach().cpu(),
                "pi_cond": pi_cond.detach().cpu(),
                "stats": stats,
            },
            os.path.join(output_dir, f"coupling_{iteration:02d}.pt"),
        )
        print(
            f"iter={iteration:02d} loss={loss.item():.6f} "
            f"shape={tuple(pi.shape)} "
            f"entropy={stats['conditional_entropy_mean']:.6f} "
            f"max_prob={stats['conditional_max_prob_mean']:.6f} "
            f"T_mass={stats['T_mass']:.6f} "
            f"T_max={stats['T_max']:.6e} "
            f"T_exact_zero={stats['T_exact_zero_count']} "
            f"T_allclose_zero={stats['T_allclose_zero']} "
            f"C_syn_mean={stats['C_syn_mean']:.6f}"
        )

    with open(os.path.join(output_dir, "summary.json"), "w") as file:
        json.dump(summary, file, indent=2)
    print(f"summary={os.path.join(output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
