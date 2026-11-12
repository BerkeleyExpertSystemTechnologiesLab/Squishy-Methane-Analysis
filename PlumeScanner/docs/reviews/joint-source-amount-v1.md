# Matched source location and amount comparison v1

Executed locally on 2026-11-01. The question is whether an incorrect fixed source
amount can be explained by an incorrect location, and whether allowing both to
vary improves reconstruction. The answer is supported for the declared low-input
case, with an important asymmetry: high input worsens maps without a large source
displacement. Joint inference improves the low/high reconstruction scores but
slightly worsens the nominal-input score.

## Reproduction and evidence

Starting local revision: `96ccb71c182658e10352ed9ff61e7983f959337f`.
Historical software baseline: `plume_scanner_new` revision
`5cac633f47001d3aa455f8b6f3cc00b11c550ed3`. This is a separate offline inference
experiment; it does not modify the baseline estimator or scanner policy.

Frozen [configuration](../../configs/source-prior-study/joint-source-amount-v1.json)
and [specification](../specs/joint-source-amount-v1.md) declare the controls before
execution. Run from `PlumeScanner/` with a new output destination:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compare-source-amount.py \
  --config configs/source-prior-study/joint-source-amount-v1.json \
  --output outputs/2026-11-01-joint-source-amount-01
MPLCONFIGDIR=/tmp/plume-amount-mpl PYTHONPATH=src .venv/bin/python scripts/plot-source-amount.py \
  --output outputs/2026-11-01-joint-source-amount-01
```

The local output contains 33 truth/evidence histories, configuration, original
schedule, implementation snapshots, affine likelihood checkpoints, three numerical
quadrature levels, posterior/field arrays, trace and summary JSON, verification,
refinement, figures and SHA-256 manifests. The archived schedule is not bundled
with a fresh clone; its required SHA-256 is
`9545e6c6f9495f082ca845eafecf2d5cdef82ff19160d6eb2f8481aff9cedf10`.
All new work remains local; no remote upload accompanies this checkpoint.

## Controlled setup

- Total in-FOV source input is 0.5/1/1.5 times
  `Q0 = 0.002305094376073324` synthetic input units per second, using
  `cell_integrated_fov_normalized_v1`. This is an area-integrated source-input
  rate on a unit FOV, not a measured concentration or methane emission rate.
  The constant source remains on throughout the 40 s monitoring window and
  subsequent forecasts.
- Source position `(y,x)=(0.5479452055,0.1504702194)`, width .035 FOV, half-life
  6 s, supplied wind, 12x16 field grid and Gaussian P0/Q/R stay fixed.
- Reuse the original 40 s, 708-reading path (one broad and two focused scans),
  positions, footprints, travel/acquisition times and .0125 s delivery delay.
  Generate new readings with the same model used in inference. The statistical
  generator advances at both acquisition and delivery; this explicitly matched
  time partition differs from older control histories.
- Each amount has a zero-realized-noise control and ten seeds 20261101–20261110.
  Random draws are shared across amounts, never scaled with signal. All three
  estimators see exactly the same readings within each history.
- All methods integrate an unknown position with a uniform density over the
  unit FOV. The true position is evaluator-only, never inserted as a candidate.
  The first estimator fixes amount at Q0; the joint estimator integrates an
  independent uniform amount multiplier on [.25,1.75]; the oracle is supplied
  the true amount, but still must infer position. Posterior field means are
  mixtures of conditional maps, not maps evaluated at mean source parameters.
- Score after reading delivery, using only available evidence. The primary score
  averages field MSE at every 20th reading plus reading 708 (36 checkpoints).
  It is not physical-time-integrated J/T. Separate final maps refer to t=40 s.

## Results across the ten noisy histories

Numbers below are means across the ten paired seeds. Smaller MSE is better.
Percent change compares joint inference with fixed nominal amount.

| True amount/Q0 | Fixed-amount checkpoint MSE | Joint checkpoint MSE | True-amount checkpoint MSE | Joint change | Seeds improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | .00258382 | .00175833 | .00174679 | -31.95% | 10/10 |
| 1.0 | .00178416 | .00179999 | .00178416 | +0.89% | 1/10 |
| 1.5 | .00239813 | .00185166 | .00182286 | -22.79% | 10/10 |

Paired joint-minus-fixed changes (standard error across seeds in parentheses):
low `-.00082549 (.00006697)`, nominal `+.00001584 (.00000743)`, high
`-.00054647 (.00002591)`. These are descriptive paired summaries at one geometry,
not a generalization or significance claim.

| True amount/Q0 | Fixed source error | Joint source error | Oracle source error | Joint estimated amount/Q0 |
| --- | ---: | ---: | ---: | ---: |
| 0.5 | .140881 | .013374 | .011909 | .544649 |
| 1.0 | .006757 | .007318 | .006757 | 1.040264 |
| 1.5 | .004714 | .005050 | .004738 | 1.538107 |

Source error is distance between posterior mean position and truth in normalized
FOV coordinates at 40 s, not meters or posterior spread. The joint amount means
remain positively biased across these ten seeds; recovery is not exact.

In the zero-realized-noise low-input control, fixing amount at Q0 moves the mean
source to `(y,x)=(.576846,.010290)` with .143128 FOV error. Joint inference gives
`(.547052,.148861)`, error .001841 and amount .503561 Q0. Thus the low-input
boundary displacement exists even without a realized random perturbation. The
inference covariance/noise assumptions remain present in this control.

High input is a useful counterexample: fixed-amount source error stays small,
yet reconstruction is worse. Joint inference slightly increases its mean source
error while improving maps. Do not summarize this as universally better source
localization, or assume source-location accuracy validates the full signal model.

Final t=40 s mean map MSE changes from .00153147 to .00022459 at low input,
.00021577 to .00022529 nominal, and .00056142 to .00022569 high. These are final
snapshots, not the primary during-scan scores.

## Prediction after monitoring stops

No readings are delivered after 40 s. Truth continues with the same source and
matched dynamics; forecasts extend to 46/52 s.

| True amount/Q0 | Fixed MSE at +6/+12 s | Joint MSE at +6/+12 s |
| --- | ---: | ---: |
| 0.5 | .00172655 / .00166553 | .00026414 / .00023026 |
| 1.0 | .00026065 / .00022164 | .00026512 / .00023148 |
| 1.5 | .00089566 / .00100245 | .00026572 / .00023213 |

At the declared validation footprint `(y,x)=(.55,.25)`, the exact mixture central
95% predictive interval contains the realized latent footprint signal in 8/10,
9/10, 0/10 seeds for the fixed-amount low/nominal/high cases at either horizon.
The joint and true-amount controls have 9/10 at both horizons in every scenario.
These ten same-geometry observations do not establish calibrated intervals.
The evaluated target is a latent footprint signal, not a noisy future voltage.

## Verification and numerical refinement

- 49 targeted tests pass, covering source-input conservation, direct bank versus
  affine likelihood, amount integration against independent quadrature, mixture
  field moments, predictive CDF, screening bounds/fallback and generator controls.
- Exact true-source/true-amount conditional mean matches deterministic truth
  within `4.67e-15`. Two full-history direct reference banks agree with the affine
  calculation: component means `2.56e-15`, weights `7.00e-15`, covariance exactly.
- Shared realized field and reading perturbations agree across amounts within
  `3.53e-15` and `1.48e-15`. RNG signatures are identical. Original record unchanged.
- Source/amount quadrature 65/65 → 129/129 passes only 46/99 frozen checks;
  retain this failure. 129/129 → 257/257 passes 99/99: maximum source-mean change
  .001324 FOV, amount-mean change .006529 Q0; all primary-MSE differences satisfy
  the frozen `max(1%,1e-6)` tolerance. Report the finest 257/257 results.
- Maximum numerical screening omitted-posterior-mass bound is `1.17e-18`, below
  the declared `1e-10`. No truth-dependent search region is used.
- These refinement checks establish stability of declared means and primary MSE,
  not convergence of entire densities or every interval endpoint. Computational
  wall time for the inference experiment was about 145 s on this local run;
  offline runtime is separate from the frozen 40 s scanner schedule.

## Interpretation and remaining work

This provides a concrete mechanism: on this path/model, a too-large assumed
input can be partly compensated by moving a source toward an edge where more
transported signal leaves the FOV. The actual posterior displacement is measured;
the boundary-transport explanation is an interpretation, not a separate causal
ablation. Joint inference avoids most of this low-input failure and approaches
the true-amount control in map error. It adds a small cost when amount was already
correct and does not recover source parameters perfectly with noisy data.

The geometry, source-informed historical path, broad location density and bounded
amount range are fixed. This is not a blind source-search test, independent-source
validation, real-sensor calibration, source-activity inference or adaptive-policy
comparison. The Gaussian field can be negative. Cell integration normalizes source
input inside the FOV and does not fix coarse field representation or boundary
physics. Later work can test new source positions/paths and amount bounds, then
evaluate whether better beliefs improve the next physical scan. None is claimed
by this checkpoint.
