# Event Reconstruction Angular Regression

This repository tracks the current ATLAS event-reconstruction ML workflow.

## Current baseline

The first baseline maps reconstructed-level event features to particle-level angular observables:

```text
22 reco-level event features
    -> MLP angular regression
    -> 4 particle-level angular targets
```

The four targets are:

- `PL_cos_theta_lep_NOSYS`
- `PL_cos_theta_star_lep_NOSYS`
- `PL_phi_lep_NOSYS`
- `PL_phi_star_lep_NOSYS`

## Main scripts

- `src/prepare_dataset.py` reads ROOT inputs, matches reco-level and particle-level information, applies event selection, and writes the NumPy training dataset.
- `src/train_mlp.py` trains the baseline MLP regression model.
- `src/plot_prediction_diagnostics.py` plots prediction-vs-truth and residual diagnostics for the saved model.

## Data and model artifacts

Large or generated artifacts are intentionally not committed:

- `data/ttbarh_angles.npz`
- `outputs/mlp_angles.pt`
- generated plots, reports, decks, and PDFs

Keep these files locally or share them through the group-approved storage location.

## Typical workflow

```bash
python3 src/prepare_dataset.py
python3 src/train_mlp.py
python3 src/plot_prediction_diagnostics.py
```

## Current physics direction

After the group meeting, the next planned decomposition is:

```text
reco objects + MET
    -> reconstructed neutrino
    -> reconstructed W/top system
    -> calculated angular observables
    -> signal/background classifier
```

The goal is to make the angular-observable pipeline less black-box by introducing physically interpretable intermediate objects.
