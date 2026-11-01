# Conserved source-width and signal-amount controls — 2026-10-28

Implemented and run on `codex/source-prior-fixed-evidence`, starting HEAD
`5cac633f47001d3aa455f8b6f3cc00b11c550ed3`, preserving previous uncommitted studies.
Production inference/policies and original evidence are unchanged. See the [protocol](../specs/conserved-source-width-v1.md).

## Answer to the experiment question

There is no single width-versus-strength answer across both sides of the baseline.
For narrower sources, restoring the original total injected signal removes most
of the excess localization error in this fixture; reducing amount alone also
creates large errors. This supports amount reduction as an important driver of
the earlier narrow-source result. For wider sources, shape mismatch remains large
at fixed amount; increased amount in the old fixed-peak experiment partially
counteracts that error. Shape and amount effects do not add linearly.

The amplitude-only .5-reference-width condition also exposes a new failure:
its total is .5712 times baseline, and its broad-prior mean error is .05968 FOV.
Three of ten posterior means move near x=.012–.015, versus true x=.15047; their
errors are about .136–.138. Other seed errors are .00956–.03623. The mean displacement
is approximately (y=+.00381,x=-.04417), opposite the original demo's large +x
shift. Similar error magnitude therefore does NOT identify the original demo's
cause. Boundary/input-position confounding is a plausible mechanism to investigate,
not a mechanism isolated by this experiment.

## Design and meaning of conservation

13 conditions, each with a zero-perturbation mean control and the same ten seeds
20261028–20261037: 143 histories. The matched plus four fixed-peak conditions
(55 histories) exactly reproduce the preceding saved observations and latent truth.
The other 88 histories test fixed-total width changes and amount-only changes.
Every prior receives the same evidence within a history, on the unchanged 40 s,
708-observation path, with acquisition/delivery times and sensor footprint intact.

Let I0 be the actual baseline injection vector's mean on the 12x16 grid. For a
width multiplier k, r(k) is the changed-width injection mean at the old peak rate,
divided by I0. The three arms are:

- joint/fixed peak: width k*sigma0, peak q0, amount r*I0 (old experiment);
- shape/fixed total: width k*sigma0, peak q0/r, amount I0;
- amount only: width sigma0, peak q0*r, amount r*I0.

All inference hypotheses keep baseline width/peak conventions. Normalization
belongs to evaluator truth only. This conserves the actual discrete source term,
including grid alignment and boundary truncation, not the continuous Gaussian
integral or the evolving plume's mass. It is synthetic signal input, not calibrated
physical methane emission. Observed SNR may still change as shape redistributes
signal across the fixed measurement path.

## Broad-prior numerical results

Distances are in normalized FOV coordinates. The matched 513-grid baseline has
mean stochastic error **.00678015** and deterministic error **.00254990**.
The reference width labels the amount-only arm's corresponding old experiment;
its actual width always remains at baseline.

| Reference width | Old input / baseline | Old fixed-peak mean error | Fixed-total mean error | Amount-only mean error |
|---|---:|---:|---:|---:|
| 0.5 | 0.5712 | 0.02601 | 0.00785 | 0.05968 |
| 0.75 | 0.6994 | 0.01832 | 0.00715 | 0.02237 |
| 1.25 | 1.4878 | 0.00893 | 0.01507 | 0.00770 |
| 1.5 | 2.1249 | 0.01132 | 0.02419 | 0.00803 |

No-perturbation controls separate deterministic displacement from random scatter:

| Reference width | Old fixed-peak error | Fixed-total error | Amount-only error |
|---|---:|---:|---:|
| 0.5 | 0.00711 | 0.00651 | 0.08344 |
| 0.75 | 0.00615 | 0.00581 | 0.02037 |
| 1.25 | 0.00672 | 0.01406 | 0.00660 |
| 1.5 | 0.00958 | 0.02217 | 0.00740 |

Accurate-region and biased-region priors show the same qualitative contrast.
For width .5, fixed-total mean errors are .00782 and .00849, compared with old
fixed-peak errors .02582 and .02744. For width 1.5, fixed-total errors are .02408
and .02565, compared with .01128 and .01195. All five priors are saved in JSON;
point-prior locations remain prescribed controls.

## Paired evidence and interactions

For each seed, the full 2x2 design yields shape-only, amount-only, joint and
interaction contrasts in both distance and squared distance. `summary.json`
stores every contrast, mean, standard error and sign count. Do not convert these
nonlinear outcomes to percentages purportedly partitioning a universal cause.

For broad priors:

- width .5: old joint minus fixed-total error is +.018162 (paired SE .001928),
  positive in 10/10 seeds. Shape-only minus baseline is +.001068 (SE .000518);
  amount-only minus baseline is +.052904 (SE .017136). The interaction is -.034742
  (SE .016657): the joint error is much smaller than adding both separate effects.
- width .75: restoring total reduces error by .011163 on average (SE .002178),
  with improvement in 9/10 seeds. Shape-only excess is only .000375 (SE .000495),
  so a small residual difference is not a robust superiority claim.
- width 1.5: joint minus fixed-total error is -.012870 (SE .001539), negative in
  all ten seeds. Shape-only excess is +.017409 while amount-only excess is
  +.001246; interaction is -.014115. Added amount can mask shape mismatch here.

These are descriptive conditional effects on a fixed path and source, not a
population-level causal or calibration claim. Source amount is fixed by the
estimator, so it may explain a wrong amplitude by moving its source hypothesis.

## Verification and remaining uncertainty

40 targeted tests pass, including conservation and factorial input-vector
identities at both interior and boundary source positions, unchanged nuisance
parameters, unit multiplier and invalid-input tests, plus earlier generator,
likelihood, source-bank and calibration checks. Initial tail-identity test
assertions were adjusted to allow two subnormal floating-point ULPs; no physical
or statistical parameter was changed, and the integrated conservation tolerance
remained 5e-14.

- Maximum relative error of intended discrete input: 2.22e-16.
- Matched/fixed-peak old evidence and truth reproduce exactly for all 55 histories.
  Source means reproduce within 1e-10 at shared saved grids.
- Inference likelihood precision is exactly equal across all 143 histories,
  validating reuse of the exact Gaussian quadratic term.
- RNG state matches after every event. Independently subtracting each arm's
  deterministic truth confirms identical realized random components: max field
  noise difference 2.55e-15, max reading noise difference 1.11e-15. Source-rate
  normalization has not inadvertently scaled the noise.
- All histories run the same 257 and 513 source grids. Final source-mean and
  .05-radius mass checks all pass: maximum changes .00064143 FOV and .00167182.
- **Complete refinement is not globally passed.** Four histories have six
  prior-specific HPD membership changes between the grids: one inherited
  fixed-peak 1.25 case, two fixed-total .5 cases, one fixed-total .75 case. Their
  source-mean shifts are <=3.09e-6 FOV. These are unresolved boundary classifications,
  retained as HPD count ranges in the summary. Full density/HPD-rank convergence
  or calibration is not claimed.

Conserving input improves location without guaranteeing well-calibrated source
uncertainty. For broad priors at fixed-total width .5, truth is in the approximate
95% HPD set in only 4–5/10 seeds; at width .75 the count is 8–9/10. At widths 1.25
and 1.5, it is 0/10. Counts are descriptive for ten fixed-geometry samples.

## Limits and next research question

The narrow source is under-resolved relative to the 12x16 latent grid; normalization
is exact for this model but not a grid-independent physical-source construction.
Results may depend on source alignment, boundary, path, prior and assumed strength.
The path was originally chosen with source knowledge. Different source positions,
latent resolutions and real observations are untested. The original source-location
hypotheses also retain their position-dependent discrete input totals.

A useful next test is joint inference of source location and injection amplitude,
with numerical grid/position checks: does allowing amplitude to change prevent
these wrong-location explanations? It must preserve explicit source uncertainty
and be evaluated against the same frozen evidence. That followup is not implemented
in this round.

## Artifacts and reproduction

Everything is local under `outputs/2026-10-28-conserved-width-01/`: frozen config,
original history, generated evidence/truth per arm/seed, normalization parameters,
likelihood statistics, 257/513 posterior weights and field means, refinement
history, paired contrasts, common-noise/observable-signal diagnostics, plot,
code/environment, logs and SHA-256 manifests.

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/plume-study-mpl .venv/bin/python -u scripts/conserved-width-source-control.py --config configs/source-prior-study/conserved-width-v1.json --output outputs/<fresh-directory>
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check-conserved-width-noise.py --input outputs/<fresh-directory>
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_conserved_source_control.py tests/test_mismatch_control.py tests/test_model_consistent_control.py tests/test_source_likelihood_quadrature.py tests/test_source_prior_belief.py tests/test_scanner_calibration.py
```

Preserve the preceding one-factor output referenced by the config to reproduce
exact-history checks. No computational-performance claim is made.
