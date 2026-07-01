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
    "jet_select_baselineJvt_NOSYS",
    "jet_GN2v01_Continuous_quantile",
    "jet_select_GN2v01_FixedCutBEff_70",
    "jet_select_GN2v01_FixedCutBEff_77",
    "jet_select_GN2v01_FixedCutBEff_85",
    "jet_select_GN2v01_FixedCutBEff_90",
    "nu_e_NOSYS",
    "nu_pt_NOSYS",
    "nu_eta_NOSYS",
    "nu_phi_NOSYS",
    "wlep_e_NOSYS",
    "wlep_pt_NOSYS",
    "wlep_eta_NOSYS",
    "wlep_phi_NOSYS",
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
    "PL_jet_pt",
    "PL_jet_eta",
    "PL_jet_phi",
    "PL_jet_nGhosts_bHadron",
]

MTOP_MEV = 172_500.0
DEFAULT_MAX_JETS = 8


def _to_numpy(values: ak.Array, dtype=np.float32) -> np.ndarray:
    return ak.to_numpy(values).astype(dtype)


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


def _pt_eta_phi_e_to_cartesian(
    pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, energy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return px, py, pz, energy


def _mass_from_pt_eta_phi_e(
    pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, energy: np.ndarray
) -> np.ndarray:
    px, py, pz, e = _pt_eta_phi_e_to_cartesian(pt, eta, phi, energy)
    mass2 = e * e - px * px - py * py - pz * pz
    return np.sqrt(np.maximum(mass2, 0.0)).astype(np.float32)


def _combine_pt_eta_phi_e(
    a_pt: np.ndarray,
    a_eta: np.ndarray,
    a_phi: np.ndarray,
    a_e: np.ndarray,
    b_pt: np.ndarray,
    b_eta: np.ndarray,
    b_phi: np.ndarray,
    b_e: np.ndarray,
) -> dict[str, np.ndarray]:
    ax, ay, az, ae = _pt_eta_phi_e_to_cartesian(a_pt, a_eta, a_phi, a_e)
    bx, by, bz, be = _pt_eta_phi_e_to_cartesian(b_pt, b_eta, b_phi, b_e)
    px = ax + bx
    py = ay + by
    pz = az + bz
    energy = ae + be
    pt = np.sqrt(px * px + py * py)
    phi = np.arctan2(py, px)
    eta = np.arcsinh(np.divide(pz, pt, out=np.zeros_like(pz), where=pt > 0))
    mass2 = energy * energy - px * px - py * py - pz * pz
    return {
        "e": energy.astype(np.float32),
        "pt": pt.astype(np.float32),
        "eta": eta.astype(np.float32),
        "phi": phi.astype(np.float32),
        "mass": np.sqrt(np.maximum(mass2, 0.0)).astype(np.float32),
    }


def _delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2)).astype(np.float32)


def _delta_r(
    eta1: np.ndarray, phi1: np.ndarray, eta2: np.ndarray, phi2: np.ndarray
) -> np.ndarray:
    deta = eta1 - eta2
    dphi = _delta_phi(phi1, phi2)
    return np.sqrt(deta * deta + dphi * dphi).astype(np.float32)


def _pad_jagged(values: ak.Array, max_jets: int, fill_value: float = np.nan) -> np.ndarray:
    padded = ak.pad_none(values, max_jets, axis=1, clip=True)
    return ak.to_numpy(ak.fill_none(padded, fill_value)).astype(np.float32)


def _truth_b_lookup(truth: ak.Array) -> dict[int, list[tuple[float, float, float]]]:
    lookup: dict[int, list[tuple[float, float, float]]] = {}
    events = ak.to_numpy(truth["PL_eventNumber"])
    pts = ak.to_list(truth["PL_jet_pt"])
    etas = ak.to_list(truth["PL_jet_eta"])
    phis = ak.to_list(truth["PL_jet_phi"])
    bghosts = ak.to_list(truth["PL_jet_nGhosts_bHadron"])
    for event, event_pt, event_eta, event_phi, event_bghost in zip(events, pts, etas, phis, bghosts):
        b_jets = [
            (float(pt), float(eta), float(phi))
            for pt, eta, phi, n_b in zip(event_pt, event_eta, event_phi, event_bghost)
            if n_b > 0
        ]
        if b_jets and int(event) not in lookup:
            b_jets.sort(reverse=True)
            lookup[int(event)] = b_jets
    return lookup


def _match_truth_b_indices(
    event_numbers: np.ndarray,
    jet_eta: np.ndarray,
    jet_phi: np.ndarray,
    candidate_mask: np.ndarray,
    truth_b_by_event: dict[int, list[tuple[float, float, float]]],
    max_delta_r: float,
) -> np.ndarray:
    labels = np.full(len(event_numbers), -1, dtype=np.int64)
    for i, event in enumerate(event_numbers):
        truth_b_jets = truth_b_by_event.get(int(event))
        if not truth_b_jets:
            continue
        best_index = -1
        best_dr = np.inf
        candidate_indices = np.flatnonzero(candidate_mask[i])
        for cand_idx in candidate_indices:
            for _, truth_eta, truth_phi in truth_b_jets:
                dr = float(_delta_r(jet_eta[i, cand_idx], jet_phi[i, cand_idx], truth_eta, truth_phi))
                if dr < best_dr:
                    best_dr = dr
                    best_index = int(cand_idx)
        if best_dr < max_delta_r:
            labels[i] = best_index
    return labels


def build_candidate_dataset(
    root_path: Path,
    output_path: Path,
    *,
    max_jets: int = DEFAULT_MAX_JETS,
    mass_scale_mev: float = 50_000.0,
    truth_match_delta_r: float = 0.4,
) -> None:
    with uproot.open(root_path) as root_file:
        reco = root_file["reco"].arrays(RECO_BRANCHES, library="ak")
        truth = root_file["particleLevel"].arrays(TRUTH_BRANCHES, library="ak")

    el = _leading_by_pt(reco, "el")
    mu = _leading_by_pt(reco, "mu")
    is_ejets = _to_numpy(reco["pass_ejets_NOSYS"], bool)
    is_mujets = _to_numpy(reco["pass_mujets_NOSYS"], bool)
    lep = {
        key: np.where(is_ejets, el[key], np.where(is_mujets, mu[key], np.nan)).astype(np.float32)
        for key in ["e", "pt", "eta", "phi"]
    }

    event_numbers = _to_numpy(reco["eventNumber"], np.uint64)
    jet_e = _pad_jagged(reco["jet_e_NOSYS"], max_jets)
    jet_pt = _pad_jagged(reco["jet_pt_NOSYS"], max_jets)
    jet_eta = _pad_jagged(reco["jet_eta"], max_jets)
    jet_phi = _pad_jagged(reco["jet_phi"], max_jets)
    jet_mass = _mass_from_pt_eta_phi_e(jet_pt, jet_eta, jet_phi, jet_e)

    gn2_quantile = _pad_jagged(reco["jet_GN2v01_Continuous_quantile"], max_jets, fill_value=-1.0)
    btag70 = _pad_jagged(reco["jet_select_GN2v01_FixedCutBEff_70"], max_jets, fill_value=0.0)
    btag77 = _pad_jagged(reco["jet_select_GN2v01_FixedCutBEff_77"], max_jets, fill_value=0.0)
    btag85 = _pad_jagged(reco["jet_select_GN2v01_FixedCutBEff_85"], max_jets, fill_value=0.0)
    btag90 = _pad_jagged(reco["jet_select_GN2v01_FixedCutBEff_90"], max_jets, fill_value=0.0)
    baseline_jvt = _pad_jagged(reco["jet_select_baselineJvt_NOSYS"], max_jets, fill_value=0.0)

    w_e = _to_numpy(reco["wlep_e_NOSYS"])
    w_pt = _to_numpy(reco["wlep_pt_NOSYS"])
    w_eta = _to_numpy(reco["wlep_eta_NOSYS"])
    w_phi = _to_numpy(reco["wlep_phi_NOSYS"])

    wj = _combine_pt_eta_phi_e(
        w_pt[:, None],
        w_eta[:, None],
        w_phi[:, None],
        w_e[:, None],
        jet_pt,
        jet_eta,
        jet_phi,
        jet_e,
    )
    abs_m_wj_minus_mtop = np.abs(wj["mass"] - MTOP_MEV).astype(np.float32)

    dphi_w_jet = _delta_phi(w_phi[:, None], jet_phi)
    deta_w_jet = (w_eta[:, None] - jet_eta).astype(np.float32)
    dr_w_jet = _delta_r(w_eta[:, None], w_phi[:, None], jet_eta, jet_phi)
    dphi_lep_jet = _delta_phi(lep["phi"][:, None], jet_phi)
    deta_lep_jet = (lep["eta"][:, None] - jet_eta).astype(np.float32)
    dr_lep_jet = _delta_r(lep["eta"][:, None], lep["phi"][:, None], jet_eta, jet_phi)

    candidate_mask = (
        np.isfinite(jet_pt)
        & np.isfinite(jet_eta)
        & np.isfinite(jet_phi)
        & np.isfinite(jet_e)
        & (jet_pt > 0)
        & (baseline_jvt > 0)
    )

    feature_names = np.array(
        [
            "jet_e",
            "jet_pt",
            "jet_eta",
            "jet_phi",
            "jet_mass",
            "gn2_quantile",
            "btag70",
            "btag77",
            "btag85",
            "btag90",
            "w_pt",
            "w_eta",
            "w_phi",
            "w_e",
            "wj_mass",
            "abs_m_wj_minus_mtop",
            "wj_pt",
            "wj_eta",
            "wj_phi",
            "deta_w_jet",
            "dphi_w_jet",
            "dr_w_jet",
            "deta_lep_jet",
            "dphi_lep_jet",
            "dr_lep_jet",
            "lep_pt",
            "lep_eta",
            "met_met",
            "met_phi",
            "actual_mu",
            "average_mu",
        ]
    )
    candidate_features = np.stack(
        [
            jet_e,
            jet_pt,
            jet_eta,
            jet_phi,
            jet_mass,
            gn2_quantile,
            btag70,
            btag77,
            btag85,
            btag90,
            np.broadcast_to(w_pt[:, None], jet_pt.shape),
            np.broadcast_to(w_eta[:, None], jet_pt.shape),
            np.broadcast_to(w_phi[:, None], jet_pt.shape),
            np.broadcast_to(w_e[:, None], jet_pt.shape),
            wj["mass"],
            abs_m_wj_minus_mtop,
            wj["pt"],
            wj["eta"],
            wj["phi"],
            deta_w_jet,
            dphi_w_jet,
            dr_w_jet,
            deta_lep_jet,
            dphi_lep_jet,
            dr_lep_jet,
            np.broadcast_to(lep["pt"][:, None], jet_pt.shape),
            np.broadcast_to(lep["eta"][:, None], jet_pt.shape),
            np.broadcast_to(_to_numpy(reco["met_met_NOSYS"])[:, None], jet_pt.shape),
            np.broadcast_to(_to_numpy(reco["met_phi_NOSYS"])[:, None], jet_pt.shape),
            np.broadcast_to(_to_numpy(reco["actualInteractionsPerCrossing"])[:, None], jet_pt.shape),
            np.broadcast_to(_to_numpy(reco["averageInteractionsPerCrossing"])[:, None], jet_pt.shape),
        ],
        axis=-1,
    ).astype(np.float32)
    candidate_features[~candidate_mask] = np.nan

    masked_gn2 = np.where(candidate_mask, gn2_quantile, -np.inf)
    highest_btag_index = np.argmax(masked_gn2, axis=1).astype(np.int64)
    highest_btag_index[~np.any(candidate_mask, axis=1)] = -1

    heuristic_score = gn2_quantile - abs_m_wj_minus_mtop / mass_scale_mev
    heuristic_score = np.where(candidate_mask, heuristic_score, -np.inf)
    heuristic_index = np.argmax(heuristic_score, axis=1).astype(np.int64)
    heuristic_index[~np.any(candidate_mask, axis=1)] = -1

    truth_b_by_event = _truth_b_lookup(truth)
    truth_b_index = _match_truth_b_indices(
        event_numbers, jet_eta, jet_phi, candidate_mask, truth_b_by_event, truth_match_delta_r
    )

    weights = (
        _to_numpy(reco["weight_mc_NOSYS"])
        * _to_numpy(reco["weight_pileup_NOSYS"])
        * _to_numpy(reco["weight_leptonSF_tight_NOSYS"])
        * _to_numpy(reco["weight_jvt_effSF_NOSYS"])
        * _to_numpy(reco["weight_ftag_effSF_GN2v01_Continuous_NOSYS"])
    ).astype(np.float32)

    event_selection = (
        _to_numpy(reco["pass_SUBcommon_NOSYS"], bool)
        & _to_numpy(reco["passNuReco_NOSYS"], bool)
        & (is_ejets | is_mujets)
        & np.any(candidate_mask, axis=1)
        & np.all(np.isfinite(np.column_stack([lep["pt"], lep["eta"], lep["phi"], w_pt, w_eta, w_phi])), axis=1)
        & np.isfinite(weights)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        candidate_features=candidate_features[event_selection],
        candidate_mask=candidate_mask[event_selection],
        eventNumber=event_numbers[event_selection],
        weight=np.abs(weights[event_selection]),
        truth_b_index=truth_b_index[event_selection],
        highest_btag_index=highest_btag_index[event_selection],
        heuristic_index=heuristic_index[event_selection],
        heuristic_score=heuristic_score[event_selection],
        feature_names=feature_names,
        max_jets=np.array(max_jets),
        mtop_mev=np.array(MTOP_MEV, dtype=np.float32),
        heuristic_mass_scale_mev=np.array(mass_scale_mev, dtype=np.float32),
        truth_match_delta_r=np.array(truth_match_delta_r, dtype=np.float32),
    )

    selected = int(np.sum(event_selection))
    labelled = int(np.sum(truth_b_index[event_selection] >= 0))
    print(f"wrote {output_path}")
    print(f"selected events: {selected}")
    print(f"candidate features: {candidate_features.shape[-1]}")
    print(f"events with truth b match label: {labelled}")
    if labelled:
        labels = truth_b_index[event_selection]
        print(f"highest-btag labelled accuracy: {np.mean(highest_btag_index[event_selection][labels >= 0] == labels[labels >= 0]):.4f}")
        print(f"heuristic labelled accuracy: {np.mean(heuristic_index[event_selection][labels >= 0] == labels[labels >= 0]):.4f}")


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
    parser.add_argument("--out", type=Path, default=Path("data/bjet_candidates.npz"))
    parser.add_argument("--max-jets", type=int, default=DEFAULT_MAX_JETS)
    parser.add_argument("--mass-scale-mev", type=float, default=50_000.0)
    parser.add_argument("--truth-match-delta-r", type=float, default=0.4)
    args = parser.parse_args()

    build_candidate_dataset(
        args.root,
        args.out,
        max_jets=args.max_jets,
        mass_scale_mev=args.mass_scale_mev,
        truth_match_delta_r=args.truth_match_delta_r,
    )


if __name__ == "__main__":
    main()
