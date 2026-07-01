# B-Jet Assignment Code Update

## Motivation

The original reconstruction in the ROOT file effectively uses the highest GN2/b-tagged jet as the leptonic-top b-jet candidate:

```text
selected b-jet = argmax(GN2 score)
top candidate  = reconstructed W + selected b-jet
```

This is a strong baseline, but it only answers:

```text
Which jet is most b-like?
```

Our event reconstruction task is slightly different:

```text
Which jet is most likely to be the b-jet paired with this reconstructed W?
```

To prepare for that task, we added a candidate-level feature layer. This keeps the GN2/b-tagging information, but also adds kinematic compatibility variables such as `m(W+jet)` and angular/topology variables such as `DeltaR(W, jet)`.

## New Files

### `src/prepare_bjet_candidates.py`

This script reads the ROOT file and builds a padded per-event, per-jet candidate dataset:

```text
candidate_features: [events, max_jets, features]
candidate_mask:     [events, max_jets]
```

Each event can contain a variable number of jets, so the output is padded to `max_jets = 8`. The `candidate_mask` marks which entries are real jets.

The output file is:

```text
data/bjet_candidates.npz
```

The candidate features include:

```text
jet_e, jet_pt, jet_eta, jet_phi, jet_mass
gn2_quantile
btag70, btag77, btag85, btag90
w_pt, w_eta, w_phi, w_e
wj_mass
abs_m_wj_minus_mtop
wj_pt, wj_eta, wj_phi
deta_w_jet, dphi_w_jet, dr_w_jet
deta_lep_jet, dphi_lep_jet, dr_lep_jet
lep_pt, lep_eta
met_met, met_phi
actual_mu, average_mu
```

The most important new variables are:

```text
gn2_quantile:
    how b-like the jet is

wj_mass:
    invariant mass of reconstructed W + candidate jet

abs_m_wj_minus_mtop:
    top-mass compatibility

dr_w_jet:
    angular separation between W and candidate jet
```

The script also saves two simple selection rules:

```text
highest_btag_index:
    current baseline, selecting the jet with highest GN2 quantile

heuristic_index:
    a simple test rule using GN2 plus top-mass compatibility
```

The heuristic score is:

```text
score = GN2_quantile - |m(W+jet) - mtop| / mass_scale
```

This heuristic is only a diagnostic baseline, not the final method.

### `src/evaluate_bjet_candidates.py`

This script compares:

```text
highest GN2 baseline
vs
GN2 + top-mass heuristic
```

It produces:

```text
outputs/bjet_candidates/bjet_candidate_diagnostics.png
outputs/bjet_candidates/truth_vs_wrong_candidate_features.png
```

These plots show how GN2, `m(W+jet)`, and `DeltaR(W, jet)` separate truth-matched b candidates from wrong jet candidates.

### `src/plot_root_reco_comparison.py`

This script checks whether our new highest-GN2 candidate reconstruction matches the original ROOT reconstruction.

It compares:

```text
ROOT tlep_m_NOSYS
vs
our highest-GN2 m(W+jet)
```

The output plot is:

```text
outputs/bjet_candidates/root_reco_comparison.png
```

This is mainly a closure test: if these two agree, then our candidate-feature code is using the same jet choice, units, and four-vector combination as the ROOT reconstruction.

## Current Results

Running:

```bash
python3 src/prepare_bjet_candidates.py
python3 src/evaluate_bjet_candidates.py
python3 src/plot_root_reco_comparison.py
```

gives:

```text
selected events: 26435
candidate features: 31
events with truth b match label: 16763
highest-btag labelled accuracy: 0.9417
heuristic labelled accuracy: 0.9312
```

The ROOT closure test gives:

```text
mean |highest - ROOT|   = 0.0228 MeV
median |highest - ROOT| = 0.0156 MeV
max |highest - ROOT|    = 1.25 MeV
```

This confirms that:

```text
our highest-GN2 m(W+jet) reproduces ROOT tlep_m_NOSYS event by event.
```

Therefore, the new candidate layer is consistent with the existing ROOT reconstruction.

## Interpretation

The diagnostic plots show that GN2 is already a very strong feature:

```text
truth-matched b jets mostly have high GN2 quantile
wrong candidates mostly have low GN2 quantile
```

The additional top-compatibility features are still useful:

```text
m(W+jet) is generally closer to mtop for the correct b candidate
DeltaR(W, jet) also carries topology information
```

However, a simple hand-written heuristic does not improve the truth-matching accuracy. It slightly narrows the reconstructed top-mass distribution, but it can also select wrong jets that accidentally give `m(W+jet)` close to the top mass.

The current conclusion is:

```text
highest GN2 is a strong baseline.
top-mass and topology variables are useful auxiliary features.
manual heuristic rules are too crude.
the next step should be an NN/MLP assignment model.
```

## Next Step

The next recommended file is:

```text
src/train_bjet_assignment.py
```

This model should read:

```text
data/bjet_candidates.npz
```

and learn an event-level assignment score:

```text
score_i = f(candidate_features_i)
```

for each jet candidate. The selected jet would then be:

```text
selected jet = argmax_i score_i
```

The model should replace the final selection rule:

```text
argmax(GN2)
```

but still use GN2 as one of the input features.

In other words:

```text
replace the highest-btag rule,
but build on the GN2 information.
```
