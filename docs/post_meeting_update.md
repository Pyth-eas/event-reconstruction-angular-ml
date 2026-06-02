# Post-Meeting Update: Angular Reconstruction Pipeline

## Current understanding

Our previous decomposition was already a step away from a direct black-box signal/background classifier. Instead of going directly from reconstructed event features to `P(signal)`, we first predict four physics-motivated angular observables.

```text
22 reco-level event features
    -> MLP angular regression
    -> 4 angular observables
    -> future signal/background classifier
```

This is more interpretable than direct classification, but it is still partly black-box because the model jumps directly from reconstructed objects to final angular variables.

## Suggested extra decomposition

Peter suggested inserting another physically meaningful reconstruction step before calculating the four angular observables:

```text
reco objects + MET
    -> reconstructed neutrino
    -> reconstructed W/top system
    -> calculate four angular observables
    -> classifier
```

The neutrino is not directly observed by the detector. We infer it from MET and the charged lepton, usually using the W-mass constraint:

```text
(lepton + neutrino)^2 = m_W^2
```

This lets us solve for the neutrino longitudinal momentum and build a neutrino four-vector.

## Why this helps

The direct baseline is:

```text
22 features -> MLP -> 4 angles
```

The more physical version is:

```text
lepton + MET -> neutrino -> W/top -> 4 angles
```

This gives us intermediate objects that can be checked separately:

- neutrino reconstruction quality
- reconstructed W mass
- reconstructed top mass
- angular-observable stability
- classifier behavior

## Decay topology and angular distributions

Two events may look similar in code, for example both containing lepton + MET + b-jets + light jets, but the reconstructed objects can come from different mother particles or decay chains.

Different decay histories can shift angular distributions:

- b-jets from `H -> b bbar` have different structure than b-jets from top decay or generic `ttbar + jets`.
- lepton directions can carry information from top decay, W polarization, and spin correlations.
- background-like processes may produce similar final reconstructed objects but different angular patterns.

This is why angular observables can help separate signal-like and background-like events.

## Practical next steps

- Clarify which neutrino object to use first: MC truth neutrino, chi-square reconstructed neutrino, or W-mass-constraint reconstructed neutrino.
- Implement or inspect neutrino reconstruction before the four-angle calculation.
- Compare truth neutrino vs reconstructed neutrino to quantify reconstruction quality.
- Recalculate angular observables from the reconstructed neutrino/W/top system.
- Compare angular distributions for signal-like vs background-like samples or decay categories.
- Feed reconstructed angular observables into the eventual signal/background classifier.
