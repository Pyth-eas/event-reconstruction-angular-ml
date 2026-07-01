from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot


def _selected_values(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    out = np.full(len(indices), np.nan, dtype=np.float32)
    valid = indices >= 0
    out[valid] = values[np.flatnonzero(valid), indices[valid]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/Users/michaelquu/Desktop/Duke/courses/26 summer/HEP/Event_Reconstruction/"
            "paper & resources/input_4Angle_10x.root"
        ),
    )
    parser.add_argument("--candidates", type=Path, default=Path("data/bjet_candidates.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/bjet_candidates/root_reco_comparison.png"))
    parser.add_argument("--max-points", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    candidates = np.load(args.candidates, allow_pickle=True)
    features = candidates["candidate_features"].astype(np.float32)
    names = list(candidates["feature_names"])
    events = candidates["eventNumber"].astype(np.uint64)
    highest = candidates["highest_btag_index"].astype(np.int64)
    heuristic = candidates["heuristic_index"].astype(np.int64)

    wj_mass = features[:, :, names.index("wj_mass")]
    highest_mass = _selected_values(wj_mass, highest)
    heuristic_mass = _selected_values(wj_mass, heuristic)

    with uproot.open(args.root) as root_file:
        reco = root_file["reco"].arrays(["eventNumber", "tlep_m_NOSYS"], library="ak")

    root_events = ak.to_numpy(reco["eventNumber"]).astype(np.uint64)
    root_mass = ak.to_numpy(reco["tlep_m_NOSYS"]).astype(np.float32)
    root_lookup = {int(event): i for i, event in enumerate(root_events)}
    root_aligned = np.array([root_mass[root_lookup[int(event)]] for event in events], dtype=np.float32)

    root_gev = root_aligned / 1000.0
    highest_gev = highest_mass / 1000.0
    heuristic_gev = heuristic_mass / 1000.0
    residual_mev = highest_mass - root_aligned

    finite = np.isfinite(root_gev) & np.isfinite(highest_gev)
    rng = np.random.default_rng(args.seed)
    finite_idx = np.flatnonzero(finite)
    if len(finite_idx) > args.max_points:
        plot_idx = rng.choice(finite_idx, size=args.max_points, replace=False)
    else:
        plot_idx = finite_idx

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    ax = axes[0]
    ax.scatter(root_gev[plot_idx], highest_gev[plot_idx], s=7, alpha=0.25)
    low = float(np.nanpercentile(root_gev[finite], 0.5))
    high = float(np.nanpercentile(root_gev[finite], 99.5))
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_xlabel("ROOT tlep_m_NOSYS [GeV]")
    ax.set_ylabel("candidate highest-GN2 m(W+j) [GeV]")
    ax.set_title("Event-by-event mass closure")

    ax = axes[1]
    ax.hist(residual_mev[finite], bins=80)
    ax.set_xlabel("candidate highest-GN2 - ROOT tlep_m [MeV]")
    ax.set_ylabel("events")
    ax.set_title("Closure residual")

    ax = axes[2]
    bins = np.linspace(80, 500, 90)
    ax.hist(root_gev[np.isfinite(root_gev)], bins=bins, histtype="step", linewidth=2, label="ROOT tlep_m")
    ax.hist(highest_gev[np.isfinite(highest_gev)], bins=bins, alpha=0.45, label="highest GN2 candidate")
    ax.hist(heuristic_gev[np.isfinite(heuristic_gev)], bins=bins, alpha=0.45, label="GN2 + top-mass heuristic")
    ax.axvline(172.5, color="black", linewidth=1)
    ax.set_xlabel("m(W + selected jet) [GeV]")
    ax.set_ylabel("events")
    ax.set_title("Top-candidate mass distributions")
    ax.legend()

    ax = axes[3]
    diff_heuristic = heuristic_gev - root_gev
    ax.hist(diff_heuristic[np.isfinite(diff_heuristic)], bins=80)
    ax.set_xlabel("heuristic selected mass - ROOT tlep_m [GeV]")
    ax.set_ylabel("events")
    ax.set_title("Effect of replacing the ROOT/highest-GN2 choice")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"saved {args.output}")
    print(f"mean |highest - ROOT| = {np.nanmean(np.abs(residual_mev)):.4f} MeV")
    print(f"median |highest - ROOT| = {np.nanmedian(np.abs(residual_mev)):.4f} MeV")
    print(f"max |highest - ROOT| = {np.nanmax(np.abs(residual_mev)):.4f} MeV")


if __name__ == "__main__":
    main()
