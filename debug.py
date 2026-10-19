"""Small training-only smoke test for the project's DualSIMformer."""

import argparse
import random

import numpy as np
import torch
import yaml

from dataloader import TrajSimLeaDataLoader
from evaluate import test
from models import get_model
from train import train_one_epoch_dualsim
from utils.data_process import TrajData
from utils.loss import initialize_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target_measure", choices=["dtw", "haus", "fret"], default="dtw")
    parser.add_argument("--emb_sim_metric", choices=["cos", "euc", "chebyshev"], default="cos")
    parser.add_argument("--grid_mode", choices=["soft", "hard"], default="soft")
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    traj_data = TrajData("porto", args.target_measure, config["datasets"], grid_size=0.01)
    train_data = traj_data.train_data
    if not 0 < args.num <= len(train_data[0]):
        raise ValueError(f"--num must be between 1 and {len(train_data[0])}")

    # Match main_DM_test.py's randomly_select initialization.
    index = torch.randperm(len(train_data[0]))[:args.num]
    selected = index.tolist()
    x_syn = [train_data[0][i].clone().detach() for i in selected]
    y_syn = train_data[1][index][:, index].detach().clone()

    model_args = argparse.Namespace(
        model="DualSIMformer", hidden_dim=128, batch_size=args.batch_size,
        sampling_num=1, neighbor_num=3, temperature=0.7, grid_emb_size=32,
        device=device, loss_type="mse", emb_sim_metric=args.emb_sim_metric,
        target_measure=args.target_measure, train_epochs=args.epochs,
        grid_mode=args.grid_mode,
    )
    eval_loader = TrajSimLeaDataLoader(traj_data, model_args, "eval", grid_mode=args.grid_mode)
    test_loader = TrajSimLeaDataLoader(traj_data, model_args, "test", grid_mode=args.grid_mode)
    syn_loader = TrajSimLeaDataLoader(
        traj_data,
        model_args,
        "syn",
        syn_data=(x_syn, y_syn),
        grid_mode=args.grid_mode,
    )

    model = get_model(model_args, config, traj_data)
    loss_fn = initialize_loss(model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"device={device}, model=DualSIMformer, train_samples={args.num}, epochs={args.epochs}")
    for epoch in range(args.epochs):
        loss, elapsed = train_one_epoch_dualsim(model, syn_loader, optimizer, loss_fn, device)
        print(f"epoch={epoch:03d}, loss={loss:.6f}, time={elapsed:.2f}s")
        print(f"  eval={test(model, eval_loader, model_args)[0]}")
        print(f"  test={test(model, test_loader, model_args)[0]}")


if __name__ == "__main__":
    main()
