from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _feature(features: np.ndarray, names: np.ndarray, name: str) -> np.ndarray:
    matches = np.flatnonzero(names == name)
    if len(matches) != 1:
        raise KeyError(f"feature {name!r} not found")
    return features[..., int(matches[0])]


def _selected_values(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    out = np.full(len(indices), np.nan, dtype=np.float32)
    valid = indices >= 0
    out[valid] = values[np.flatnonzero(valid), indices[valid]]
    return out


def _accuracy(name: str, selected: np.ndarray, labels: np.ndarray) -> str:
    labelled = labels >= 0
    if not np.any(labelled):
        return f"{name}: no truth labels available"
    return f"{name}: {np.mean(selected[labelled] == labels[labelled]):.4f} on {int(np.sum(labelled))} labelled events"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/bjet_candidates.npz"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/bjet_candidates"))
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    features = data["candidate_features"].astype(np.float32)
    mask = data["candidate_mask"].astype(bool)
    names = data["feature_names"]
    labels = data["truth_b_index"].astype(np.int64)
    highest = data["highest_btag_index"].astype(np.int64)
    heuristic = data["heuristic_index"].astype(np.int64)

    print(_accuracy("highest_btag", highest, labels))
    print(_accuracy("gn2_plus_topmass_heuristic", heuristic, labels))

    wj_mass = _feature(features, names, "wj_mass") / 1000.0
    gn2 = _feature(features, names, "gn2_quantile")
    dr_w_jet = _feature(features, names, "dr_w_jet")
    abs_m = _feature(features, names, "abs_m_wj_minus_mtop") / 1000.0

    highest_mass = _selected_values(wj_mass, highest)
    heuristic_mass = _selected_values(wj_mass, heuristic)
    highest_gn2 = _selected_values(gn2, highest)
    heuristic_gn2 = _selected_values(gn2, heuristic)

    args.outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    axes[0].hist(highest_mass[np.isfinite(highest_mass)], bins=60, alpha=0.65, label="highest GN2")
    axes[0].hist(heuristic_mass[np.isfinite(heuristic_mass)], bins=60, alpha=0.65, label="GN2 + top mass")
    axes[0].axvline(172.5, color="black", linewidth=1)
    axes[0].set_xlabel("m(W + selected jet) [GeV]")
    axes[0].set_ylabel("events")
    axes[0].legend()

    axes[1].hist(highest_gn2[np.isfinite(highest_gn2)], bins=np.arange(-0.5, 7.5, 1), alpha=0.65, label="highest GN2")
    axes[1].hist(heuristic_gn2[np.isfinite(heuristic_gn2)], bins=np.arange(-0.5, 7.5, 1), alpha=0.65, label="GN2 + top mass")
    axes[1].set_xlabel("selected jet GN2 continuous quantile")
    axes[1].set_ylabel("events")
    axes[1].legend()

    flat_mask = mask & np.isfinite(abs_m) & np.isfinite(gn2) & np.isfinite(dr_w_jet)
    axes[2].scatter(abs_m[flat_mask], gn2[flat_mask], s=3, alpha=0.12)
    axes[2].set_xlabel("|m(W + jet) - mtop| [GeV]")
    axes[2].set_ylabel("GN2 continuous quantile")

    axes[3].scatter(dr_w_jet[flat_mask], gn2[flat_mask], s=3, alpha=0.12)
    axes[3].set_xlabel("DeltaR(W, jet)")
    axes[3].set_ylabel("GN2 continuous quantile")

    fig.tight_layout()
    plot_path = args.outdir / "bjet_candidate_diagnostics.png"
    fig.savefig(plot_path, dpi=180)
    print(f"saved {plot_path}")

    if np.any(labels >= 0):
        labelled = np.flatnonzero(labels >= 0)
        correct_gn2 = gn2[labelled, labels[labelled]]
        correct_abs_m = abs_m[labelled, labels[labelled]]
        correct_dr = dr_w_jet[labelled, labels[labelled]]

        wrong_mask = mask[labelled].copy()
        wrong_mask[np.arange(len(labelled)), labels[labelled]] = False
        wrong_gn2 = gn2[labelled][wrong_mask]
        wrong_abs_m = abs_m[labelled][wrong_mask]
        wrong_dr = dr_w_jet[labelled][wrong_mask]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(wrong_gn2[np.isfinite(wrong_gn2)], bins=np.arange(-0.5, 7.5, 1), alpha=0.65, label="wrong candidates")
        axes[0].hist(correct_gn2[np.isfinite(correct_gn2)], bins=np.arange(-0.5, 7.5, 1), alpha=0.65, label="truth-matched b")
        axes[0].set_xlabel("GN2 continuous quantile")
        axes[0].legend()

        axes[1].hist(wrong_abs_m[np.isfinite(wrong_abs_m)], bins=60, alpha=0.65, label="wrong candidates")
        axes[1].hist(correct_abs_m[np.isfinite(correct_abs_m)], bins=60, alpha=0.65, label="truth-matched b")
        axes[1].set_xlabel("|m(W + jet) - mtop| [GeV]")
        axes[1].legend()

        axes[2].hist(wrong_dr[np.isfinite(wrong_dr)], bins=60, alpha=0.65, label="wrong candidates")
        axes[2].hist(correct_dr[np.isfinite(correct_dr)], bins=60, alpha=0.65, label="truth-matched b")
        axes[2].set_xlabel("DeltaR(W, jet)")
        axes[2].legend()

        fig.tight_layout()
        truth_plot_path = args.outdir / "truth_vs_wrong_candidate_features.png"
        fig.savefig(truth_plot_path, dpi=180)
        print(f"saved {truth_plot_path}")


if __name__ == "__main__":
    main()
