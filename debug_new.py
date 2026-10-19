"""Small training-only smoke test for the project's SIMformer."""

import argparse
import random

import numpy as np
import torch
import yaml

from dataloader import TrajSimLeaDataLoader
from evaluate import test
from models import get_model
from train import train_one_epoch_other
from utils.data_process import TrajData
from utils.loss import initialize_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=300, help="random training trajectories")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Load the project's data split: train=3000, eval=1000, test=6000.
    traj_data = TrajData("porto", "dtw", config["datasets"], grid_size=0.01)
    train_trajs, train_dist, train_lens, x_grid, y_grid, grid_ids = traj_data.train_data
    if not 0 < args.num <= len(train_trajs):
        raise ValueError(f"--num must be between 1 and {len(train_trajs)}")

    index = torch.randperm(len(train_trajs))[:args.num]
    selected = index.tolist()
    traj_data.train_data = (
        [train_trajs[i] for i in selected],
        train_dist[index][:, index],
        train_lens[index],
        [x_grid[i] for i in selected],
        [y_grid[i] for i in selected],
        [grid_ids[i] for i in selected],
    )

    model_args = argparse.Namespace(
        model="SIMformer",
        hidden_dim=128,
        batch_size=args.batch_size,
        sampling_num=1,
        neighbor_num=3,
        temperature=0.7,
        grid_emb_size=32,
        device=device,
        loss_type="mse",
        emb_sim_metric="euc",
        target_measure="dtw",
        train_epochs=args.epochs,
    )

    train_loader = TrajSimLeaDataLoader(traj_data, model_args, mode="train", grid_mode="soft")
    eval_loader = TrajSimLeaDataLoader(traj_data, model_args, mode="eval", grid_mode="soft")
    test_loader = TrajSimLeaDataLoader(traj_data, model_args, mode="test", grid_mode="soft")
    model = get_model(model_args, config, traj_data)
    loss_fn = initialize_loss(model_args).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    print(f"device={device}, train_samples={args.num}, epochs={args.epochs}")
    for epoch in range(args.epochs):
        loss, elapsed = train_one_epoch_other(
            model, train_loader, optimizer, loss_fn, device
        )
        print(f"epoch={epoch:03d}, loss={loss:.6f}, time={elapsed:.2f}s")
        print(f"  eval={test(model, eval_loader, model_args)[0]}")
        print(f"  test={test(model, test_loader, model_args)[0]}")


if __name__ == "__main__":
    main()
