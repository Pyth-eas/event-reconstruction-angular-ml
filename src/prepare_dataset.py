from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


RECO_BRANCHES = [
    "eventNumber",
    "actualInteractionsPerCrossing",
    "averageInteractionsPerCrossing",
    "pass_SUBcommon_NOSYS",
    "pass_ejets_NOSYS",
    "pass_mujets_NOSYS",
    "passNuReco_NOSYS",
    "el_e_NOSYS",
    "el_pt_NOSYS",
    "el_eta",
    "el_phi",
    "mu_e_NOSYS",
    "mu_pt_NOSYS",
    "mu_eta",
    "mu_phi",
    "jet_e_NOSYS",
    "jet_pt_NOSYS",
    "jet_eta",
    "jet_phi",
    "jet_select_GN2v01_FixedCutBEff_70",
    "nu_e_NOSYS",
    "nu_pt_NOSYS",
    "nu_eta_NOSYS",
    "nu_phi_NOSYS",
    "met_met_NOSYS",
    "met_phi_NOSYS",
    "weight_mc_NOSYS",
    "weight_pileup_NOSYS",
    "weight_leptonSF_tight_NOSYS",
    "weight_jvt_effSF_NOSYS",
    "weight_ftag_effSF_GN2v01_Continuous_NOSYS",
]

TRUTH_BRANCHES = [
    "PL_eventNumber",
    "PL_cos_theta_lep_NOSYS",
    "PL_cos_theta_star_lep_NOSYS",
    "PL_phi_lep_NOSYS",
    "PL_phi_star_lep_NOSYS",
]


def _pick_by_index(values: ak.Array, indices: ak.Array) -> np.ndarray:
    picked = ak.firsts(values[indices[:, None]], axis=1)
    return ak.to_numpy(ak.fill_none(picked, np.nan)).astype(np.float32)


def _leading_by_pt(data: ak.Array, prefix: str) -> dict[str, np.ndarray]:
    pt = data[f"{prefix}_pt_NOSYS"]
    idx = ak.argmax(pt, axis=1, mask_identity=True)
    return {
        "e": _pick_by_index(data[f"{prefix}_e_NOSYS"], idx),
        "pt": _pick_by_index(data[f"{prefix}_pt_NOSYS"], idx),
        "eta": _pick_by_index(data[f"{prefix}_eta"], idx),
        "phi": _pick_by_index(data[f"{prefix}_phi"], idx),
    }


def _jet_by_tag(data: ak.Array, *, highest: bool) -> dict[str, np.ndarray]:
    tag = data["jet_select_GN2v01_FixedCutBEff_70"]
    idx = ak.argmax(tag, axis=1, mask_identity=True) if highest else ak.argmin(tag, axis=1, mask_identity=True)
    return {
        "e": _pick_by_index(data["jet_e_NOSYS"], idx),
        "pt": _pick_by_index(data["jet_pt_NOSYS"], idx),
        "eta": _pick_by_index(data["jet_eta"], idx),
        "phi": _pick_by_index(data["jet_phi"], idx),
    }


def _target_lookup(truth: ak.Array) -> dict[int, tuple[float, float, float, float]]:
    events = ak.to_numpy(truth["PL_eventNumber"])
    targets = np.column_stack(
        [
            ak.to_numpy(truth["PL_cos_theta_lep_NOSYS"]),
            ak.to_numpy(truth["PL_cos_theta_star_lep_NOSYS"]),
            ak.to_numpy(truth["PL_phi_lep_NOSYS"]),
            ak.to_numpy(truth["PL_phi_star_lep_NOSYS"]),
        ]
    ).astype(np.float32)
    lookup: dict[int, tuple[float, float, float, float]] = {}
    for event, target in zip(events, targets):
        if int(event) not in lookup and np.all(np.isfinite(target)):
            lookup[int(event)] = tuple(float(x) for x in target)
    return lookup


def build_dataset(root_path: Path, output_path: Path) -> None:
    with uproot.open(root_path) as root_file:
        reco = root_file["reco"].arrays(RECO_BRANCHES, library="ak")
        truth = root_file["particleLevel"].arrays(TRUTH_BRANCHES, library="ak")

    truth_by_event = _target_lookup(truth)
    event_numbers = ak.to_numpy(reco["eventNumber"])
    target_rows = np.array(
        [truth_by_event.get(int(event), (np.nan, np.nan, np.nan, np.nan)) for event in event_numbers],
        dtype=np.float32,
    )

    el = _leading_by_pt(reco, "el")
    mu = _leading_by_pt(reco, "mu")
    is_ejets = ak.to_numpy(reco["pass_ejets_NOSYS"]).astype(bool)
    is_mujets = ak.to_numpy(reco["pass_mujets_NOSYS"]).astype(bool)
    lep = {
        key: np.where(is_ejets, el[key], np.where(is_mujets, mu[key], np.nan)).astype(np.float32)
        for key in ["e", "pt", "eta", "phi"]
    }
    bjet = _jet_by_tag(reco, highest=True)
    specjet = _jet_by_tag(reco, highest=False)

    feature_names = [
        "actual_mu",
        "average_mu",
        "is_ejets",
        "is_mujets",
        "lep_e",
        "lep_pt",
        "lep_eta",
        "lep_phi",
        "bjet_e",
        "bjet_pt",
        "bjet_eta",
        "bjet_phi",
        "specjet_e",
        "specjet_pt",
        "specjet_eta",
        "specjet_phi",
        "nu_e",
        "nu_pt",
        "nu_eta",
        "nu_phi",
        "met_met",
        "met_phi",
    ]
    X = np.column_stack(
        [
            ak.to_numpy(reco["actualInteractionsPerCrossing"]),
            ak.to_numpy(reco["averageInteractionsPerCrossing"]),
            is_ejets.astype(np.float32),
            is_mujets.astype(np.float32),
            lep["e"],
            lep["pt"],
            lep["eta"],
            lep["phi"],
            bjet["e"],
            bjet["pt"],
            bjet["eta"],
            bjet["phi"],
            specjet["e"],
            specjet["pt"],
            specjet["eta"],
            specjet["phi"],
            ak.to_numpy(reco["nu_e_NOSYS"]),
            ak.to_numpy(reco["nu_pt_NOSYS"]),
            ak.to_numpy(reco["nu_eta_NOSYS"]),
            ak.to_numpy(reco["nu_phi_NOSYS"]),
            ak.to_numpy(reco["met_met_NOSYS"]),
            ak.to_numpy(reco["met_phi_NOSYS"]),
        ]
    ).astype(np.float32)

    weights = (
        ak.to_numpy(reco["weight_mc_NOSYS"])
        * ak.to_numpy(reco["weight_pileup_NOSYS"])
        * ak.to_numpy(reco["weight_leptonSF_tight_NOSYS"])
        * ak.to_numpy(reco["weight_jvt_effSF_NOSYS"])
        * ak.to_numpy(reco["weight_ftag_effSF_GN2v01_Continuous_NOSYS"])
    ).astype(np.float32)

    selection = (
        ak.to_numpy(reco["pass_SUBcommon_NOSYS"]).astype(bool)
        & ak.to_numpy(reco["passNuReco_NOSYS"]).astype(bool)
        & (is_ejets | is_mujets)
        & np.all(np.isfinite(X), axis=1)
        & np.all(np.isfinite(target_rows), axis=1)
        & np.isfinite(weights)
    )

    X = X[selection]
    y = target_rows[selection]
    weights = np.abs(weights[selection])
    events = event_numbers[selection]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        weight=weights,
        eventNumber=events,
        feature_names=np.array(feature_names),
        target_names=np.array(
            [
                "PL_cos_theta_lep_NOSYS",
                "PL_cos_theta_star_lep_NOSYS",
                "PL_phi_lep_NOSYS",
                "PL_phi_star_lep_NOSYS",
            ]
        ),
    )
    print(f"wrote {output_path}")
    print(f"selected events: {len(X)}")
    print(f"features: {X.shape[1]}, targets: {y.shape[1]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/Users/michaelquu/Desktop/input_4Angle_10x.root"))
    parser.add_argument("--out", type=Path, default=Path("data/ttbarh_angles.npz"))
    args = parser.parse_args()
    build_dataset(args.root, args.out)


if __name__ == "__main__":
    main()
