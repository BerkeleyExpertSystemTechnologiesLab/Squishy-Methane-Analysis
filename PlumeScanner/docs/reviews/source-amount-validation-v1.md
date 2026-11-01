# Source-amount reconstruction and independent-scene validation v1

Completed locally on 2026-11-01. Reconstruction and uncertainty benefits are
supported for the declared strength-mismatch simulations; improved acquisition
is `not_demonstrated`. Production inference and scanner policy remain unchanged.

## Question and controls

Can uncertainty about source input amount prevent misleading location explanations
and improve reconstruction of a changing field? Separate three outcomes:
monitoring reconstruction, predictions without new readings, and uncertainty when
the reconstruction is wrong. Source location and amount are auxiliary diagnostics.
Then change independent source geometry/boundary distances and matched field grids,
before reconnecting the same scanner policy to two different estimators.

The [protocol](../specs/source-amount-validation-v1.md) and
[configuration](../../configs/source-prior-study/source-amount-validation-v1.json)
freeze the setup. Historical baseline is `plume_scanner_new` revision
`5cac633f47001d3aa455f8b6f3cc00b11c550ed3`; starting local revision is `bf0500b`.
All work is local. The fixed-default uncertain-location control retains the old
fixed-amount idea, but is not the unchanged historical production estimator.
The candidate additionally infers amount; the information control knows true
amount but not location. Every method integrates a uniform location prior over
the FOV, without truth-dependent source candidates or search windows.

## Evidence and physical clock

Use a declared .0125 s model clock, independent of scanner choices. Precompute
zero-source Gaussian perturbations and deterministic source responses separately.
Scale only the source response by .5/1/1.5 Q0. The measurement-noise key depends
on seed, physical acquisition tick and position, not previous sample count.
Latent environments are immutable and shared between policies/estimators.

Cell-integrated in-FOV-normalized input, Q0=.002305094376073324, .035 source width,
6 s decay, supplied wind, optical footprint, 20 Hz cadence, integration, motion
speeds and .0125 s acquisition-to-availability separation stay fixed. Q0 has
synthetic source-input units; it is not a methane emission measurement. Source
input stays on during monitoring and forecasts. Signed Gaussian signal states
are intentional.

This creates new histories: the earlier event-dependent matched generator is
not the same fixed-clock process. Earlier results remain separate and unchanged.
J/T is the exact spatial-error time average for the declared piecewise-constant
clock model on [0,T), using current knowledge at each tick. A delivered reading
updates the acquisition-time state only when available; earlier scores never
change. Detached forecasts between delivery events cannot modify live filters.
Subsampled time scores are sensitivity diagnostics on the same fine trajectory.

Development uses the original source position, four old seeds and a mean control.
Holdout uses four new source positions and seeds 20261111–20261114:

| Scene | y | x | Nearest FOV boundary distance |
| --- | ---: | ---: | ---: |
| Interior left | .30 | .35 | .30 |
| Interior right | .75 | .65 | .25 |
| Near inflow | .15 | .045 | .045 |
| Near outflow | .55 | .94 | .06 |

There are 63 histories per matched field grid, 126 total including six mean
controls, and three method evaluations per history. Each amount has 16 heldout
noise histories per grid. The 12x16 and 18x24 models have their own grid-sampled
Gaussian fields; contrasts within a grid share evidence. This is grid robustness
of matched models, not convergence against one common physical continuous truth.
The same seed trajectory is reused across amounts and source locations. Sixteen
location-seed combinations are not sixteen independent random environments.
The legacy `model.source_rate` field is not used for input in these controls;
the separately recorded Q0 and source-input formulation define actual injection.

## Prespecified outcomes and limits

Report heldout J/T, final and +6/+12 s MSE, paired signs, nominal regressions,
source/amount error, and actual finite-mixture predictive CDF at the five global
validation footprints. The separately declared
[uncertainty audit](../../configs/source-prior-study/source-amount-uncertainty-audit-v1.json)
also checks source-relative evaluator-only footprints and whole-field variance,
because fixed sparse validation points may miss a narrow plume. Source-relative
targets never enter inference or scan selection. Squared-error/variance ratios
are moment diagnostics, not Gaussian 95% confidence claims. Pointwise coverage
counts are descriptive, correlated, and not whole-field calibration guarantees.

Scientific checks require lower heldout mean J/T and both forecast errors at low
and high amounts on each grid, improved amount absolute error in those cases,
and no more than 2% nominal J/T regression. Failures must be retained without
changing scenes, priors or criteria. Numerical refinement is a separate gate.
Amount prior [.25,1.75] contains the three true amounts; performance outside that
range, multiple sources, source shutoff, wind error and real sensors are untested.

The closed-loop stage uses the existing evidence-gated policy and physical
compiler, the same 60 s budget and two belief adapters. Actual finite-mixture
quantiles feed policy; no aggregate-Gaussian substitute is used. Mean-only event
views explicitly leave standard deviation unavailable; predictive moments and
CDF diagnostics calculate actual mixture uncertainty. The field truth and
noise are shared, while different selected paths may produce different readings.
Offline computational time is recorded but not charged as control latency;
hardware real-time feasibility remains unproven.
The first full scan costs 33.575 s: 32 s of column sweeps plus 1.575 s of
repositioning. The remaining 26.425 s cannot fit another complete full refresh,
even before return travel. Unsupported/infeasible focus may therefore end
acquisition early; subsequent idle prediction error remains in the 60 s score.
Keep this budget limitation rather than changing duration after inspecting paths.

The separately declared
[path audit](../../configs/source-prior-study/source-amount-path-audit-v1.json)
cross-replays both acquired histories through each fixed estimator. Comparing
paths with the same replay estimator separates the value of new measurements
from improved interpretation of unchanged measurements. Identical physical
readings are explicitly marked and cannot support an acquisition-improvement
claim. The path audit does not influence the policy or any acquired reading.

## Fixed-path results at the common 257/128 level

Each row averages 16 heldout location-seed combinations. Values are joint-minus-
fixed percentage changes; negative means lower error. Supplemental numerical
checks are reported separately rather than deleting the failed first comparison.

| Field grid | True amount | Monitoring J/T | Forecast +6 s | Forecast +12 s | Histories with lower J/T |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12x16 | .5 Q0 | -14.35% | -78.18% | -75.46% | 16/16 |
| 12x16 | 1 Q0 | +.97% | +11.69% | +10.03% | 4/16 |
| 12x16 | 1.5 Q0 | -9.32% | -69.07% | -67.02% | 16/16 |
| 18x24 | .5 Q0 | -12.37% | -67.54% | -69.56% | 16/16 |
| 18x24 | 1 Q0 | +.57% | +4.89% | +7.60% | 3/16 |
| 18x24 | 1.5 Q0 | -8.83% | -68.97% | -69.96% | 16/16 |

All 18 prespecified scientific checks pass at these saved scores, conditional on
numerical closeout. This does not mean every outcome improves: nominal forecast
regression is material and remains explicit. Monitoring errors include startup,
unobserved locations and all idle physical time, unlike the earlier checkpoint
average. These are new fixed-clock histories; do not interpret percentage changes
relative to earlier studies as an additional algorithm gain.

Joint amount absolute error at 40 s is .055–.061 Q0 across the four mismatched
grid/amount combinations, versus .5 Q0 for fixed default. Nominally the fixed
value is exactly right and joint absolute error is .061–.063 Q0. Amount means
are still systematically higher than true amounts in this finite set; unbiased
physical strength estimation is not established.

Source-location error is substantially grid dependent. At low amount, fixed
mean-position error is .08662 FOV on 12x16 and .02116 on 18x24; joint errors are
.01616 and .01326. At high amount, fixed/joint errors are .01345/.00764 and
.01008/.00604, respectively. Nominal location errors slightly worsen under joint
inference. Thus the strong displacement observed in the original fixture is not
a universal error magnitude or a grid-independent physical mechanism. Map and
forecast gains are the stronger reconstruction evidence here.

## Uncertainty when the assumed amount is wrong

The independent uncertainty replay reproduces saved maps within `1.15e-14`.
At +12 s without new readings, exact finite-mixture central-95% membership for
the evaluator-only source footprint is:

| Grid | True amount | Fixed default | Joint | True-amount control |
| --- | ---: | ---: | ---: | ---: |
| 12x16 | .5 Q0 | 2/16 | 15/16 | 15/16 |
| 12x16 | 1 Q0 | 15/16 | 14/16 | 15/16 |
| 12x16 | 1.5 Q0 | 0/16 | 15/16 | 15/16 |
| 18x24 | .5 Q0 | 3/16 | 15/16 | 16/16 |
| 18x24 | 1 Q0 | 16/16 | 15/16 | 16/16 |
| 18x24 | 1.5 Q0 | 0/16 | 15/16 | 16/16 |

At low/high amount the whole-field mean squared standardized error is
1.93/2.33 on 12x16 and 2.16/2.81 on 18x24 with fixed amount, versus approximately
1.04–1.06 with joint amount and 1.03–1.06 with true amount. These moment ratios
and missed pointwise intervals indicate underrepresented error under fixed
wrong amount on these simulations. They do not prove general probability
calibration; the sixteen combinations share four noise trajectories, points and
times are correlated, and interval-endpoint convergence is not the frozen
source-summary/MSE refinement criterion. Nominal interval performance also does
not uniformly improve. The original five global validation footprints remain
available alongside this supplementary signal-relative evaluation.

## Numerical closure and reproduction

The original 129/64 → 257/128 checks pass 176/189 on 12x16 and 148/189 on 18x24.
Retain all 54 failures. Repeating all three methods for the affected 8 and 19
histories on 513/256 passes 24/24 and 56/57 checks. The remaining check is the
development source, 1.5 Q0, seed 20261104, true-amount control on 18x24: its
forecast-MSE change is about 1.04%, just above the frozen 1% rule. A full-history
1025x1025 global-source pass of that method reproduces the 513-level J/T within
3.35e-11 and forecast MSE within 1.35e-11; all unchanged criteria pass. A temporary
source-input cache bounds memory and is removed afterward. The global numerical
resolution record preserves each intermediate failure and closes every case.
No source candidates are inserted at truth and no tolerance is relaxed.

The figures use a common 257/128 level for heldout comparisons, with affected
histories checked against 513. The exceptional 1025 pass concerns a development
oracle, not a heldout candidate result. This verifies the declared source means,
amount means and MSE summaries, not complete densities, every interval endpoint
or allocation convergence. Holding the same fine-clock trajectory fixed,
subsampling score times at .025/.05 s changes heldout J/T by at most .36%/1.07%
on 12x16 and .32%/.95% on 18x24. The reported primary score uses every .0125 s
tick; it is exact for the declared clock model rather than a continuum guarantee.

Core local reproduction commands (use new output destinations):

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/validate-source-amount.py \
  --config configs/source-prior-study/source-amount-validation-v1.json \
  --output outputs/source-amount-validation-new
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit-source-amount-uncertainty.py \
  --study outputs/2026-11-01-source-amount-validation-04 \
  --output outputs/source-amount-uncertainty-new
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run-source-amount-loop.py \
  --study outputs/2026-11-01-source-amount-validation-04 \
  --resolution outputs/2026-11-01-source-amount-validation-04/numerical-resolution.json \
  --output outputs/source-amount-loop-new
```

The local record path and required SHA-256 are frozen in configuration; a fresh
clone alone does not contain this ignored archived evidence. The experiment
archive includes copied schedule/configuration, immutable truth/noise, readings,
full-time errors, posterior diagnostics, producer snapshots and manifests.
The numerical-refinement and oracle-refinement CLIs retain the original failed
checks and use the same saved readings. `resolve-source-amount-checks.py` records
supplemental closure without overwriting those failures.

## Same-policy closed loop and crossed measurement replay

Run 12 paired cases (interior-left and near-inflow sources, three amounts, two
seeds), with fixed and joint beliefs: 24 executions on the same 60 s budget.
Each pair shares the identical latent trajectory, keyed sensor noise and initial
full-scan events. The same policy code, thresholds, footprint, cadence and
mechanical compiler govern both. Each forward execution has an identical causal
replay. Independently reloading all 24 saved JSON event histories reproduces the
plans, receipts, complete error arrays and rendered field frames bit-for-bit.

| True amount | Fixed J/T | Joint J/T | Joint change | Pairs improved | Mean readings fixed/joint |
| --- | ---: | ---: | ---: | ---: | ---: |
| .5 Q0 | .00196550 | .00141772 | -27.87% | 4/4 | 640 / 640 |
| 1 Q0 | .00148541 | .00149087 | +.37% | 0/4 | 640 / 640 |
| 1.5 Q0 | .00202397 | .00162461 | -19.73% | 4/4 | 972.25 / 1004.75 |

All eight low/nominal pairs acquire identical readings: one full scan, then no
legal refresh fits the remaining budget. Those gains/regressions are entirely
belief interpretation, not a changed acquisition schedule. All four high-input
pairs have different measurement histories. Average charged motion is 11.15 s
fixed versus 9.7375 s joint; total scan time is 59.7625 versus 59.975 s. Extra
readings and fewer/larger focused passes do not alone establish useful acquisition.

The 2x2 crossed replay holds the estimator constant while changing acquired data:

| True amount | Changed-history pairs | Change under common joint estimator | Change under common fixed estimator |
| --- | ---: | ---: | ---: |
| .5 Q0 | 0/4 | 0% | 0% |
| 1 Q0 | 0/4 | 0% | 0% |
| 1.5 Q0 | 4/4 | -.0308% | +.2709% |

For high input, individual common-joint path changes range from -.235% to +.267%
and only three of four improve. Signs also depend on the replay estimator. These
effects are small relative to the fixed-path 1% numerical verification tolerance;
allocation convergence has not been demonstrated. Do not attribute the ~20%
end-to-end gain to smarter scanning. The result supports better belief updates
under incorrect amount, while independent acquisition benefit remains
`not_demonstrated`. A longer budget, additional independent scenes and a separate
allocation-sensitive numerical audit are future work, not claimed here.

## Verification, media and practical limits

116 targeted tests pass, including the unchanged full-FOV, evidence-gate and
calibration tests, direct affine filtering, independent amount quadrature,
analytic truncated-normal moments, streaming versus dense actual-mixture CDF and
quantiles, causal acquisition, cache invalidation and parallel equivalence.
Cheap event snapshots are explicitly mean-only; uncertainty evaluation uses
actual mixture moments/CDFs. The online adapter uses analytic amount moments
with quadrature fallback, verified against 256-point integration; policy
quantiles still use the finite component distribution, in bounded-memory chunks.

The persisted replay check also verifies the final negligible-precision guard
against recorded output. JSON comparison normalizes tuples/lists through their
canonical serialized representation; scalar values, plans and arrays are not
relaxed. A harmless unused reciprocal overflow for subnormal precision in the
initial producer is guarded in final code; all persisted arrays replay exactly.

The observer-only demo selects the first predeclared high-input pair, not the
best result. MP4 decodes as H.264, 4 fps, 121 distinct frames, 30.25 s playback
for 60 s physical time; GIF also has 121 frames. A shared color scale shows truth,
fixed and joint maps; orange points are recently delivered samples and the white
cross is mechanical pointing. Playback does not alter simulation timing.

All configs, truth/noise components, readings, full-clock losses, source/amount
summaries, uncertainty audits, crossed replays, code snapshots and SHA-256
manifests are local. Performance-only partial attempts are retained, and their
completed 12x16 coarse level is reused byte-for-byte. No GitHub upload or slides.
Claims remain limited to a constant single source, matched synthetic physics,
bounded correct-containing amount prior and the stated Gaussian measurement
model. Real gas calibration, physical boundary/representation validity, source
activity, computational control latency and real-time hardware readiness remain
unvalidated. Joint inference is not uniformly better when default amount is true.
