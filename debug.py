"""One-step diagnostic for ordinary training and DM data distillation.

Examples:
    python debug.py --model DualSIMformer --loss_type mse
    python debug.py --model ConvTraj --loss_type triplet --use_syn_minibatch 4

The script deliberately runs one batch and one optimizer step only. It does
not import main_train_model.py or main_DM_test.py because those files execute
their full experiment at import time.
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import yaml
import torch
import torch.nn as nn

from parameter_setting import parse_args
from utils.tools import set_seed, get_padded_mask, get_normalized_padded_traj
from utils.data_process import TrajData
from utils.loss import initialize_loss
from dataloader import TrajSimLeaDataLoader
from models import get_model
from evaluate import getEmbeddings
from utils.syn_data_ini import k_medoids_sampling


def model_name(args):
    return args.model


def parameter_stats(model):
    grads = [p.grad.detach().norm().item() for p in model.parameters()
             if p.grad is not None]
    return {
        "grad_param_count": len(grads),
        "grad_norm_sum": sum(grads),
        "grad_norm_max": max(grads) if grads else 0.0,
    }


def make_optimizer(model, args, config):
    lr = config["model"][args.model]["lr_teacher"][args.opt_method]
    if args.opt_method == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=args.l2
        )
    return optimizer, lr


def original_loss(model, batch, loader, criterion, args):
    """Reproduce one batch from train.py without performing an optimizer step."""
    device = args.device
    if args.model == "ConvTraj":
        inputs_lon, inputs_lat, inputs_image, distance = batch
        anchor, positive, negative = model(inputs_lon, inputs_lat, inputs_image)
        pos_target = torch.as_tensor(distance[0], device=device)
        neg_target = torch.as_tensor(distance[1], device=device)
        cross_target = torch.as_tensor(distance[2], device=device)
        pos_dist = torch.linalg.vector_norm(anchor - positive, dim=1)
        neg_dist = torch.linalg.vector_norm(anchor - negative, dim=1)
        cross_dist = torch.linalg.vector_norm(positive - negative, dim=1)
        _, _, loss = criterion(
            0, pos_dist, pos_target, neg_dist, neg_target,
            cross_dist, cross_target
        )
        return loss

    similarity = loader.similarity.to(device)
    if args.model == "TrajCL":
        batch_idx, grid_trajs, spatial, lens, padding_mask = batch
        batch_idx = batch_idx.to(device)
        grid_trajs = (grid_trajs[0].to(device), grid_trajs[1].to(device))
        spatial = spatial.to(device)
        lens = lens.to(device)
        padding_mask = padding_mask.to(device)
        shuffled = torch.randperm(batch_idx.shape[0], device=device)
        shuffled_idx = batch_idx[shuffled]
        vecs = model(grid_trajs, spatial, lens, padding_mask)
    elif args.model == "DualSIMformer":
        batch_idx, trajs, weights, linear_idx, padding_mask = batch
        batch_idx = batch_idx.to(device)
        trajs = trajs.to(device)
        weights = weights.to(device)
        linear_idx = linear_idx.to(device)
        padding_mask = padding_mask.to(device)
        shuffled = torch.randperm(batch_idx.shape[0], device=device)
        shuffled_idx = batch_idx[shuffled]
        vecs = model(trajs, weights, linear_idx, padding_mask)
    else:
        padded, batch_idx = batch
        padded = padded.to(device)
        batch_idx = batch_idx.to(device)
        mask = get_padded_mask(padded)
        padded = get_normalized_padded_traj(
            padded,
            mask,
            loader.mean_lon,
            loader.mean_lat,
            loader.std_lon,
            loader.std_lat,
        )
        shuffled = torch.randperm(batch_idx.shape[0], device=device)
        shuffled_idx = batch_idx[shuffled]
        vecs = model(padded, mask)

    return criterion(vecs, vecs[shuffled], similarity[batch_idx, shuffled_idx])


def run_original_one_step(args, config, traj_data, loader, criterion):
    model = get_model(args, config, traj_data)
    optimizer, lr = make_optimizer(model, args, config)
    batch = next(iter(loader.get_dataloader()))

    optimizer.zero_grad(set_to_none=True)
    loss = original_loss(model, batch, loader, criterion, args)
    loss.backward()
    stats = parameter_stats(model)

    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    deltas = [((after.detach() - old).norm().item())
              for after, old in zip(model.parameters(), before)]
    stats.update({
        "loss": loss.item(),
        "lr": lr,
        "param_update_norm_sum": sum(deltas),
        "param_update_norm_max": max(deltas) if deltas else 0.0,
    })
    return stats


def initialize_synthetic(args, traj_data):
    train_trajs, train_dist = traj_data.train_data[:2]
    if args.syn_ini == "randomly_select":
        indices = torch.randperm(len(train_trajs))[:args.syn_num]
        values = [train_trajs[i].detach().clone() for i in indices]
        y_syn = train_dist[indices][:, indices].detach().clone().to(args.device)
    elif args.syn_ini == "k-medoids":
        indices = torch.tensor(
            k_medoids_sampling(
                train_dist.numpy(), args.syn_num, max_iters=1000
            )
        )
        values = [train_trajs[i].detach().clone() for i in indices]
        y_syn = train_dist[indices][:, indices].detach().clone().to(args.device)
    else:
        values = []
        mean = torch.tensor(
            [traj_data.mean_lon, traj_data.mean_lat],
            dtype=torch.float32, device=args.device
        )
        std = torch.tensor(
            [traj_data.std_lon, traj_data.std_lat],
            dtype=torch.float32, device=args.device
        )
        for _ in range(args.syn_num):
            length = int(torch.randint(20, args.syn_len_max, ()).item())
            values.append(torch.normal(mean, std.expand(length, -1)))
        y_syn = torch.zeros(args.syn_num, args.syn_num, device=args.device)
    return nn.ParameterList([nn.Parameter(value) for value in values]), y_syn


def run_dm_one_step(args, config, traj_data, train_loader):
    x_syn, _ = initialize_synthetic(args, traj_data)
    optimizer = torch.optim.SGD(x_syn, lr=args.lr_traj, momentum=0.5)
    model = get_model(args, config, traj_data)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    syn_count = len(x_syn)
    chosen = torch.randperm(syn_count)[:min(args.use_syn_minibatch, syn_count)]
    x_syn_batch = [x_syn[i] for i in chosen]

    real_indices = []
    for idx in chosen.tolist():
        start = idx * args.use_num_real_per_syn
        real_indices.extend(range(start, start + args.use_num_real_per_syn))
    real_indices = [i for i in real_indices if i < len(train_loader.dataset)]
    x_real_batch = [train_loader.dataset.traj_coor[i] for i in real_indices]

    real_vec = nn.functional.normalize(
        getEmbeddings(model, args.model, x_real_batch, args.device, traj_data), dim=1
    )
    syn_vec = nn.functional.normalize(
        getEmbeddings(model, args.model, x_syn_batch, args.device, traj_data), dim=1
    )
    loss = torch.sum((real_vec.mean(dim=0) - syn_vec.mean(dim=0)) ** 2)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norms = []
    missing = []
    for i, value in enumerate(x_syn):
        if value.grad is None:
            missing.append(i)
        else:
            grad_norms.append((i, value.grad.norm().item()))

    before = {i: x_syn[i].detach().clone() for i in chosen.tolist()}
    optimizer.step()
    updates = {
        i: (x_syn[i].detach() - old).norm().item()
        for i, old in before.items()
    }
    selected = set(chosen.tolist())
    selected_grad = [g for i, g in grad_norms if i in selected]
    unselected_grad = [g for i, g in grad_norms if i not in selected]
    missing_selected = [i for i in missing if i in selected]
    return {
        "loss": loss.item(),
        "selected_count": len(selected),
        "selected_indices": sorted(selected),
        "grad_norm_sum_selected": sum(selected_grad),
        "grad_norm_max_selected": max(selected_grad) if selected_grad else 0.0,
        "update_norm_sum_selected": sum(updates.values()),
        "update_norm_max_selected": max(updates.values()) if updates else 0.0,
        "nonzero_grad_unselected_count": len(unselected_grad),
        "missing_grad_count_total": len(missing),
        "missing_grad_count_selected": len(missing_selected),
        "loss_grad_fn": type(loss.grad_fn).__name__ if loss.grad_fn else None,
        "syn_vec_grad_fn": type(syn_vec.grad_fn).__name__ if syn_vec.grad_fn else None,
    }


def main():
    args = parse_args()

    with open(args.config_path) as handle:
        config = yaml.safe_load(handle)
    set_seed(args.seed)
    args.device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    print(f"device={args.device}, model={args.model}, loss_type={args.loss_type}")

    traj_data = TrajData(
        args.dataset, args.target_measure, config["datasets"], args.grid_size
    )
    train_loader = TrajSimLeaDataLoader(
        traj_data=traj_data, args=args, mode="train"
    )
    criterion = initialize_loss(args).to(args.device)

    print("\n[ordinary model: one batch]")
    print(run_original_one_step(args, config, traj_data, train_loader, criterion))

    print("\n[DM: one synthetic-data update]")
    print(run_dm_one_step(args, config, traj_data, train_loader))


if __name__ == "__main__":
    main()
