from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, n_features: int, n_targets: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    per_event = torch.mean((pred - target) ** 2, dim=1)
    return torch.sum(per_event * weight) / torch.sum(weight)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/ttbarh_angles.npz"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    weight = data["weight"].astype(np.float32)
    weight = np.clip(weight, 0.0, np.percentile(weight, 99.5)).astype(np.float32)
    weight = weight / np.mean(weight)

    indices = rng.permutation(len(X))
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    x_mean = X[train_idx].mean(axis=0)
    x_std = X[train_idx].std(axis=0)
    x_std[x_std == 0] = 1.0
    X = (X - x_mean) / x_std

    train_ds = TensorDataset(
        torch.from_numpy(X[train_idx]),
        torch.from_numpy(y[train_idx]),
        torch.from_numpy(weight[train_idx]),
    )
    val_x = torch.from_numpy(X[val_idx])
    val_y = torch.from_numpy(y[val_idx])
    val_w = torch.from_numpy(weight[val_idx])

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = MLP(X.shape[1], y.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad()
            loss = weighted_mse(model(batch_x), batch_y, batch_w)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = weighted_mse(val_pred, val_y, val_w)
            mae = torch.mean(torch.abs(val_pred - val_y), dim=0)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            mae_text = ", ".join(f"{name}={value:.4f}" for name, value in zip(data["target_names"], mae.tolist()))
            print(
                f"epoch {epoch:03d} train_loss={np.mean(losses):.5f} "
                f"val_loss={float(val_loss):.5f} val_mae=[{mae_text}]"
            )

    Path("outputs").mkdir(exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_names": data["feature_names"],
            "target_names": data["target_names"],
            "x_mean": x_mean,
            "x_std": x_std,
        },
        "outputs/mlp_angles.pt",
    )
    print("saved outputs/mlp_angles.pt")


if __name__ == "__main__":
    main()
