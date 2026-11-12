# Source amount under wind and decay mismatch

Completed locally on 2026-11-12. The joint estimator can compensate for incorrect
wind or decay by changing inferred source amount even though true amount stays
fixed. Better maps do not establish correct strength recovery or general
predictive calibration. This is a bounded synthetic mechanism diagnostic.

## Question, protocol and provenance

The B3 question is whether location/amount inference misattributes wind or decay
error when true input remains correct. Starting code was
`f93b2a430f21f30179c58e0afb058d2a54c07499`;
implementation/execution producer is `0828b9e`. The
[protocol](../specs/source-amount-nuisance-v1.md) and
[configuration](../../configs/source-prior-study/source-amount-nuisance-v1.json)
were frozen before scientific execution. No parameters or thresholds were tuned
on evaluation outcomes. The older location-only nuisance study is different:
this study evaluates the joint amount candidate and holds per-tick Q fixed.

Three methods receive the same readings, locations, footprints and acquisition/
availability times within each condition:

1. **Fixed:** nominal wind/decay, unknown location, amount fixed to 1 Q0.
2. **Joint:** nominal wind/decay, unknown location and amount.
3. **Correct dynamics:** correct perturbed wind/decay, unknown location, amount
   fixed to 1 Q0. It is explicitly an information control, not deployable inference
   supplied with an unknown true parameter. Knowing amount adds no new comparison
   here because true amount equals the default. The control is not a guaranteed
   per-realization error lower bound.

All location priors are uniform over the full FOV. Joint amount prior is uniform
on [.25,1.75] Q0. No source candidates are inserted at truth. True source remains
active with constant total input Q0 = .002305094376073324 synthetic input units
per second and .035 FOV width. These are neither ppm nor a physical emission rate.

The 12x16 field, .0125 s model clock, zero-inflow transport, finite optical
footprint and inherited 708-reading/40 s schedule are fixed. The path was
originally selected with source information; this is not unknown-source search
or a scanning-policy experiment. Score causal map MSE over every tick on [0,40),
including startup/idle time; J/T is its time average for this finite clock model.
Pause only new readings at 40 s, leaving source input active for +6/+12 s forecasts.

Nominal half-life is 6 s. The eight mismatches multiply true wind speed or true
half-life by .5/.75/1.25/1.5, one at a time, with a ninth matched condition.
These exploratory multipliers are not measured hardware error distributions.
Initial covariance and discrete per-tick process Q stay nominal for all truth and
inference models. Half-life changes A and within-step source retention, without
also rescaling process innovation amplitude. Standard-normal initial/process
draws and keyed sensor noise are paired within each seed; realized field noise
differs after propagation by different A. This is an explicit diagnostic noise
convention, not an independently validated atmospheric model.

There are 13 histories per condition, 117 total (351 method/history evaluations):
the original development source with mean control and two old seeds; new
evaluation positions (.32,.46) and (.68,.07) in normalized (y,x), each with a mean
control and seeds 2026111201-2026111204. Tables below average only the eight noisy
evaluation combinations per condition. Mean controls are retained separately.
The four evaluation noise trajectories are reused across positions and
conditions, so the eight combinations are not eight independent environments.

## Reconstruction and inferred amount

Negative error changes mean joint inference has lower error than fixed inference.
Amount means and central-95% amount intervals are evaluated at 40 s. An interval
excluding 1 is misleading about the known constant amount in that history.

| True dynamics / nominal | Monitoring J/T change | +12 s MSE change | Mean inferred amount / Q0 | Amount intervals excluding 1 |
| --- | ---: | ---: | ---: | ---: |
| Matched | +0.673% | +4.742% | 1.0323 | 0/8 |
| Wind .5 | -1.904% | -13.198% | 1.1469 | 5/8 |
| Wind .75 | -0.682% | -13.879% | 1.1090 | 3/8 |
| Wind 1.25 | +0.225% | +3.923% | .9469 | 1/8 |
| Wind 1.5 | -2.184% | -16.734% | .8670 | 4/8 |
| Half-life .5 (3 s) | -8.347% | -58.248% | .6654 | 8/8 |
| Half-life .75 (4.5 s) | -0.496% | -16.483% | .8767 | 1/8 |
| Half-life 1.25 (7.5 s) | -0.094% | -9.659% | 1.1518 | 3/8 |
| Half-life 1.5 (9 s) | -1.236% | -23.322% | 1.2458 | 7/8 |

These are descriptive paired simulation summaries, not confidence estimates.
Sub-percent differences do not support strong method rankings under the 1%
per-method refinement criterion. Nominally joint inference also raises +6 s MSE
9.385%; wind .75 and half-life 1.25 have slightly worse +6 s forecasts even when
their +12 s forecasts improve. No uniform advantage is claimed.

The clearest discriminator is half-life .5: map and forecast errors decrease
while amount is underestimated by about 33.5% on average, and every evaluation
amount interval excludes truth. For half-life 1.5 the mean estimate is about
24.6% too high, with 7/8 exclusions. Wind errors also shift amount, in a way
that depends on source geometry; not every individual history follows the
ensemble trend. The zero-perturbation development example at wind 1.5 estimates
.7995 Q0 although true input stays 1 Q0. This mechanism is not solely a favorable
measurement-noise realization.

Maximum posterior amount mass within .05 Q0 of either prior edge is 2.055% across
noisy evaluation histories (half-life 1.5). It is not a dominant boundary pileup
in these cases; a wider-prior sensitivity experiment remains untested.

The correct-dynamics control has lower aggregate monitoring and forecast MSE in
all eight mismatch conditions than the joint wrong-dynamics candidate. Amount
flexibility is an incomplete substitute for correcting the actual dynamics.
This aggregate result does not imply an oracle lower bound in every individual
history or establish which physical transport model is correct.

## Predictive uncertainty is a separate outcome

The actual finite-mixture CDF is evaluated at five fixed global footprints and
two truth-relative footprints. Truth-relative positions never enter inference.
At the source footprint after 12 s without new readings, central-95% membership is:

| True dynamics / nominal | Fixed | Joint | Correct dynamics |
| --- | ---: | ---: | ---: |
| Matched | 7/8 | 8/8 | 7/8 |
| Wind .5 | 0/8 | 0/8 | 8/8 |
| Wind .75 | 1/8 | 6/8 | 8/8 |
| Wind 1.25 | 7/8 | 8/8 | 8/8 |
| Wind 1.5 | 5/8 | 8/8 | 8/8 |
| Half-life .5 | 4/8 | 6/8 | 7/8 |
| Half-life .75 | 6/8 | 7/8 | 7/8 |
| Half-life 1.25 | 4/8 | 8/8 | 8/8 |
| Half-life 1.5 | 1/8 | 8/8 | 8/8 |

Wind .5 is a strong negative case: lower joint map MSE coexists with a source
prediction interval that misses truth in every evaluation combination. The
correct-dynamics control includes truth in all eight. Global membership is
37/40 fixed and 38/40 joint in that same condition: sparse background-dominated
global diagnostics can miss a local failure. Neither 8/8 at one footprint nor
38/40 across correlated footprints establishes general calibration.

Amount intervals concern the source parameter; predictive intervals concern a
footprint signal. Their counts answer different questions and are never pooled.

## Numerical closure and reproducibility

All 117 histories run source/amount orders 129/64 and 257/128. The original
comparison has 37 failed method/history gates across 24 condition/history pairs.
All failures are retained in each condition's `refinement.json`. Rerun all three
methods for those 24 histories at 513/256; all unchanged criteria pass. No
1025 fallback was needed. The gates include source/amount means, amount interval
endpoints, J/T and forecasts, predictive PIT, and stable predictive/amount
membership at 20/40/46/52 s. Final tables use each history's finest completed level,
with exact level provenance. This verifies the declared finite-model diagnostics,
not continuum physics or arbitrary uncertainty tails.

104 targeted tests pass in 3.86 s, covering direct candidate filtering,
independent amount integration, actual-mixture CDFs, nuisance/Q isolation, input
conservation, paired noise and causal delivery. A changed reading cannot alter
earlier scores. The smoke run is engineering-only and retains unresolved small-
grid quadrature; it is never used for scientific tables.

Separate disk audits verify all condition manifests and regenerate scalar
evidence from saved truth and physical-slot keyed noise (maximum difference
1.11e-16). Replaying saved scalar evidence reproduces final m/P/B/L/K arrays
exactly. Reconstructed saved maps differ by at most 2.21e-14; independently
checked 20 s scores agree exactly. Matched fixed/correct-dynamics controls are
identical. All nine audits pass. The archived 708-reading source record retains
SHA-256 `9545e6c6f9495f082ca845eafecf2d5cdef82ff19160d6eb2f8481aff9cedf10`.

## Artifacts and commands

Local frozen execution root:
`outputs/2026-11-12-source-amount-nuisance-01/`.
Each condition retains deterministic/noise trajectories, common scalar readings,
schedule, sufficient statistics, full-time errors, map snapshots, diagnostics,
refinement failures and artifact hashes. Producer source/configuration snapshots
and Git provenance are saved at the root. Generated data/media are not committed.

The reviewed `figures/` directory contains:

- `reconstruction-and-forecast.png`: three-method monitoring and +6/+12 s errors.
- `inferred-amount.png`: individual posterior means/95% amount intervals and
  aggregate means; true source input remains 1 Q0 at every point.
- `predictive-coverage.png`: source-footprint membership at 40 and 52 s.
- `worked-example.png`: prespecified development mean control, true wind 1.5x,
  with shared map color scale and evaluator-only true-source marker.
- `summary.json` and `summary.csv`: all conditions, global/source/downwind
  uncertainty and paired outcome summaries.

An earlier reviewed figure directory is retained separately; final scientific
numbers did not change when shortening overlapping coverage-axis labels.
Audit directories are `2026-11-12-source-amount-nuisance-wind-audit-01`,
`2026-11-12-source-amount-nuisance-decay-audit-01` and
`2026-11-12-source-amount-nuisance-final-decay-audit-01`, under `outputs/`.

Run from `PlumeScanner/`, choosing fresh output destinations:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/diagnose-source-amount-nuisance.py \
  --config configs/source-prior-study/source-amount-nuisance-v1.json \
  --output outputs/nuisance-new --jobs 3
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/plume-nuisance-mpl \
  .venv/bin/python scripts/summarize-source-amount-nuisance.py \
  --study outputs/nuisance-new --output outputs/nuisance-new/figures
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/audit-source-amount-nuisance.py \
  --study outputs/nuisance-new --output outputs/nuisance-new-audit
```

A fresh clone lacks the ignored historical schedule/evidence. The publication
entry `--fresh-schedule` regenerates the baseline record and checks that its
physical acquisition schedule exactly matches the frozen 708-reading contract.
For the fresh-checkout command, see the [README](../../README.md#source-amount-and-nuisance-diagnostics).
The generated record has a new file hash; the archived experiment and its hashes
remain unchanged. Unit tests use constructed fixtures, without archive inputs.

## Scientific scope

Supported: under the declared fixed-path Gaussian model, wrong wind/decay can
shift joint source-amount inference, sometimes excluding true Q0, while map error
falls. The candidate's earlier strength-mismatch reconstruction gains therefore
do not establish physically correct strength estimation under other model errors.
Prediction-interval reliability is not restored in every condition.

Unvalidated: actual methane calibration, a second field discretization or common
continuous truth, wider amount priors, rotating/spatially varying wind, source
activity/multiple sources, external simulations, independent field recordings,
new scanning rules and real-time feasibility. This step is complete; combined
wind/amount inference or residual diagnostics would be separately scoped work.
