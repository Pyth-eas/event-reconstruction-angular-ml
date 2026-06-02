from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_mlp import MLP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/ttbarh_angles.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/mlp_angles.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction_diagnostics.png"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-points", type=int, default=4000)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    target_names = data["target_names"]

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(X))
    split = int(0.8 * len(indices))
    val_idx = indices[split:]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    x_mean = checkpoint.get("x_mean", X[indices[:split]].mean(axis=0))
    x_std = checkpoint.get("x_std", X[indices[:split]].std(axis=0))
    x_std = np.asarray(x_std, dtype=np.float32)
    x_std[x_std == 0] = 1.0

    X_scaled = (X - x_mean) / x_std

    model = MLP(X.shape[1], y.shape[1])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        pred = model(torch.from_numpy(X_scaled[val_idx].astype(np.float32))).numpy()

    truth = y[val_idx]
    residual = pred - truth

    if len(val_idx) > args.max_points:
        plot_idx = rng.choice(len(val_idx), size=args.max_points, replace=False)
    else:
        plot_idx = np.arange(len(val_idx))

    n_targets = y.shape[1]
    fig, axes = plt.subplots(2, n_targets, figsize=(4.2 * n_targets, 7.2))

    for i, name in enumerate(target_names):
        ax = axes[0, i]
        ax.scatter(truth[plot_idx, i], pred[plot_idx, i], s=8, alpha=0.3)
        low = min(float(truth[:, i].min()), float(pred[:, i].min()))
        high = max(float(truth[:, i].max()), float(pred[:, i].max()))
        ax.plot([low, high], [low, high], color="black", linewidth=1)
        ax.set_title(str(name))
        ax.set_xlabel("truth")
        ax.set_ylabel("prediction")

        ax = axes[1, i]
        ax.hist(residual[:, i], bins=60)
        ax.set_xlabel("prediction - truth")
        ax.set_ylabel("events")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
