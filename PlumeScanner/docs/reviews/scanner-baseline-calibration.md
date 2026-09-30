# Scanner baseline calibration

Started from `b5b48e6` on `codex/scanner-baseline-calibration`. Robotics integration
and simulator choice are deferred. The [protocol](../specs/scanner-baseline-calibration.md)
was declared before development selection and held-out execution. This review
records the completed experiments and verified closeout.
Implementation conformance is separate from scientific performance.

## Implementation

`07f844d` adds separately named margin/decay bindings and the declared pipeline;
`08ed40d` verifies an exact faster evaluator-only truth calculation. Historical
constructors/messages are preserved. The new `CalibrationSession` uses the same
full startup, focus gate and charged home/full-rescan loop. Padding precedes path
compilation and remains charged. The three public acquisition operations remain
unchanged. A full-scan-only comparator keeps the original estimator.

Decay integrates source and process input using the declared exponential factors.
Fixed-evidence source-rate fitting shares covariance and propagates two affine
endpoint means. Independent replays at intermediate rates reproduce the resulting
quadratic causal loss. No signal-activity schedule is passed to the controller.
Held-out phase seeds determine source phases before calibration; they are paired
with their declared noise levels, so those two effects are not separately inferred
from held-out comparisons.

## Focus-margin development

All 36 declared executions completed. Neither nonzero margin meets the geometric
core criterion; **zero margin is retained**. This is unsuccessful margin tuning.

| Margin per vertical side | Coarse core coverage, noise 0 / .05 | Doubled core coverage, noise 0 / .05 | Aggregate continuous J/T |
|---|---:|---:|---:|
| 0 | 99.96% / 99.89% | 86.73% / 86.23% | .06419895 |
| .05 | 99.99% / 99.56% | 52.57% / 52.38% | .06324336 |
| .10 | 100.00% / 99.30% | 51.04% / 50.95% | .06382112 |

Coverage is time-averaged geometric overlap during focus, using the predeclared
bright-core definition. It is not actual measurement coverage or physical
resolution. The aggregate column is a sum across four continuous cases, not a
single run's loss. Lower aggregate loss does not satisfy the separate core criterion.

For the first doubled zero-noise focus decision at 129.5875 s, zero margin selects
189 cells and spans 56 columns in 11.925 s. Margin .05 selects 45 cells, adds 148
padding cells and spans 37 columns in 11.95 s. Margin .10 selects 21 cells, adds
168 padding cells and spans 21 columns in 11.625 s. Padding competes with useful
horizontal extent under the unchanged time ceiling. It cannot simply be added
without charging that tradeoff.

Source-anchored blank controls falsely focus even without noise. For example,
coarse zero-margin/noise-zero makes nine false-focus decisions, each after a
background full scan; .05 padding makes 121 false-focus decisions. A larger
window does not repair the source-persistence assumption. The diagnostic video
retains this model's visible excessive mean tail; layout and scales are unchanged.

## Fixed-evidence calibration and test freeze

All eight lifetime/width combinations were fitted on the same 12 selected-margin
development histories. The frozen selection is half-life **6 s**, source sigma
**.035 FOV**, source rate **.2829738193616142 synthetic units/s**, and margin **0**.
Its summed causal development loss is 34.18635839 versus 39.47575270 for the
unchanged model, a 13.40% fixed-evidence reduction. This does not establish a scan
improvement; the paths and readings were held fixed during fitting.

| Half-life | Width .0175: fitted rate / loss | Width .035: fitted rate / loss |
|---|---:|---:|
| Infinite | .225492 / 38.980412 | .138471 / 39.433174 |
| 3 s | .300000 / 36.584377 | .300000 / 34.895592 |
| 6 s | .300000 / 34.770152 | .282974 / 34.186358 |
| 12 s | .300000 / 35.143556 | .214001 / 35.496720 |

Several fits reach the declared rate ceiling. No bound was expanded. The selected
fit does not reach it, and no alternative falls within the declared numerical tie
tolerance. The operational inference grid remains 12 x 16.

Freeze SHA256: `76ebf8e5f330eb74bda04dcbe754c6df6fb96f65eebc0ad54fef51e28215c4e4`.
The 72 fully bound test configurations were hashed before starting any test run:
`f2aa73ce700c02e151466356a6a3492a9d192967323427ab50044d03d0aa7f03`.
Session hashes are identical across scenario activity/noise realizations within
each profile/variant: source cessation, intermittency and blank status do not leak
into inference configuration. All 72 held-out executions are complete without retuning.

## Held-out outcomes

All 24 paired calibrated/baseline comparisons reduce causal J/T by more than 1%.
Individual reductions range from 11.60% to 40.76%. Across the 24 configurations,
mean J/T is .01053931 for the unchanged baseline, .00869057 for calibration and
.01162844 for repeated full scanning. These are descriptive two-realization results.

| Profile | Scenario | Baseline J/T | Calibrated J/T | Full-scan J/T | Calibrated change |
|---|---|---:|---:|---:|---:|
| Coarse | Continuous | .01719252 | .01477380 | .01906737 | -14.07% |
| Coarse | Wind error | .01657379 | .01464977 | .01856902 | -11.61% |
| Coarse | Weak | .00149265 | .00114382 | .00235388 | -23.37% |
| Coarse | Blank | .00179264 | .00107233 | .00259507 | -40.18% |
| Coarse | Cessation | .00977043 | .00802439 | .01101299 | -17.87% |
| Coarse | Intermittent | .01227202 | .01028211 | .01371964 | -16.21% |
| Doubled | Continuous | .01935130 | .01604091 | .02031271 | -17.11% |
| Doubled | Wind error | .01904585 | .01574688 | .01988910 | -17.32% |
| Doubled | Weak | .00183152 | .00131028 | .00254192 | -28.46% |
| Doubled | Blank | .00232148 | .00145862 | .00271884 | -37.17% |
| Doubled | Cessation | .01107701 | .00868877 | .01268943 | -21.56% |
| Doubled | Intermittent | .01375049 | .01109511 | .01407128 | -19.31% |

Rows average the two seeds. Common 360 s horizons are not exactly matched
expenditure: in continuous cases the adaptive variants spend approximately
359.4–360 s; full scanning spends 349.925 s coarse and 260.763 s doubled, then
cannot complete another full scan. Idle time remains in the loss. Computation is
recorded separately and includes online evaluation; concurrent runs do not
establish real-time performance.

Continuous bright-core geometric coverage rises from 82.08%/84.57% to
99.997%/99.998% on the doubled profile. Coarse coverage is already about 99.9% and
becomes 100%. Actual core readings rise from 251/268 to 434/418 coarse and from
347/353 to 537/529 doubled. This improvement uses zero padding and unchanged
physical resolution. It also leaves the wider scene older: continuous coarse
mean observed sample age rises from about 138 s to 152 s. Missing/unvisited age is
reported separately, and geometric coverage is not a count of measurements.

Enhancement-index MSE improves in 20 pairs but worsens in four: coarse continuous
by 17–18%, and coarse wind-error by 52–54%. Signed mean-index bias in those cases
becomes more positive. Better spatial reconstruction does not establish better
strength estimation, and the index is not a calibrated emission rate.

### Required operating behavior fails

- All four calibrated blank cases falsely focus after startup. Coarse produces
  228/238 focus sweeps and doubled 48/48, with only one full scan in each run.
  The unchanged coarse baseline makes 9/10 false-focus decisions but returns to
  full scanning; calibration can sustain the false focus instead.
- None of the four calibrated cessation cases starts another full scan after
  cessation. Coarse makes 127/135 false-focus decisions after signal loss and
  doubled 37/37. An all-background rescan is never reached, so a zero
  "refocus-after-background-rescan" count is not a passing recovery result.
- Intermittent cases also falsely focus during the inactive interval. No source
  activity schedule, reset or threshold adjustment was introduced to repair this.

**Classification: useful reconstruction improvement in these fixtures, unsuccessful
calibration of the complete operating loop. Do not promote the candidate to the
default.** Constant source input plus the selected decay does not adequately
represent both continued emission and disappearance. Lower integrated loss can
coexist with a small persistent positive peak that keeps the focus gate enabled.

The smallest justified next stage is an evidence-driven source-activity/emission
model, first tested on fixed paths with continuous, blank, cessation and intermittent
controls. It must allow absent-source evidence to reduce subsequent source input
while retaining continuing signal. Simply removing source input previously lost
real continuing signal. Do not tune away these failures with a new cutoff, reset
or another allocator during this round. Representation error, changing local drift,
uncertainty reliability and physical sensor calibration remain open.

## Numerical and visual checks

All four representative evaluation refinements pass at 240 x 320 / .025 s:
relative loss changes are .363% coarse continuous, .114% coarse blank,
.330% doubled continuous and .213% doubled blank. No further refinement is
triggered. Reported comparisons use the predeclared common 120 x 160 / .05 s mesh.
The identical clean-field quadrature is reused across profiles only after source,
time and inference-grid identity checks; mechanical coverage is not reused.

Fixed-evidence 24 x 32 inference probes change forecast MSE by 1.1–3.9% at the
declared 0/1/3/6/12 s horizons. Representation error remains unresolved; this is
a sensitivity probe, not a convergence proof. Operational inference remains 12 x 16.

Five primary physical/summary video pairs and one alternate-playback pair decode
completely. Selected startup, acquisition, completion and failure keyframes were
visually inspected. Continuous core readings and scan progression are visible;
blank/ceased truth and near-zero readings coexist with a persistent inferred
positive source region. The existing six-panel layout and scales are preserved.
The 4 FPS/4x and 2 FPS/16x blank views have identical execution/configuration hashes.
Visual review covers the recorded keyframes, not a claim of watching every frame.

Local videos: `outputs/scanner-calibration/inspection/`, with per-video hashes,
frame counts and specific reviewed frames in each `manifest.json`. Separate plots
in `plots/` show online error and the executed mode timeline. The failed padded
doubled run remains visible alongside the frozen candidate.

## Verification and local evidence

Four complete historical continuous runs (both profiles, noise 0/.05) reproduce
identical acquisition events, paths and causal losses after removing only the new
plan diagnostic fields. Older v1–v5/legacy compatibility checks also pass. The
truth optimization reproduces both checked complete executions and losses exactly.
Independent arithmetic, serialization, lifecycle and affine-loss tests pass.
The full suite reports **520 passed, five existing Windows symlink skips**.
All **108 public-call/causal-loss replays** pass. Import-boundary checks, compilation
and `git diff --check` pass. Legacy compatibility was rechecked at `19a78d4`.

Local evidence is in `outputs/scanner-calibration/`: immutable declaration/configs,
margin selection with all 36 hashes, execution records, checks and inspection
media. Frozen candidate reference configurations are committed under
`configs/scanner-calibration/`; the complete test manifests and generated evidence
remain local. Historical `not_demonstrated` findings remain unchanged.

The no-decay evaluator optimization in `19a78d4` uses the already tested vector
forecast rather than constructing unused covariance at every long-horizon mean
query. Operational inference and planning are unchanged. The 91 pre-optimization
artifacts (83 executions and eight fits) retain their original bytes/revisions.
`query-compatibility.json` binds their hashes to independently verified query
equivalence (maximum absolute mean difference 1.9984e-15 through 100 s). Saved
executions are additionally replayed against their original J with 1e-12 tolerance.

## Reproduce

Use the repository virtual environment and keep generated outputs local. The
initial lean declaration supplies the unchanged parent configurations; it does
not run or retune that historical experiment.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:MPLCONFIGDIR="$PWD/outputs/.matplotlib"
.\.venv\Scripts\python.exe scripts/run-full-fov-demo.py --phase declare
.\.venv\Scripts\python.exe scripts/run-scanner-calibration.py --phase declare
.\.venv\Scripts\python.exe scripts/run-scanner-calibration.py --phase development --workers 6
.\.venv\Scripts\python.exe scripts/run-scanner-calibration.py --phase select-margin
.\.venv\Scripts\python.exe scripts/run-scanner-calibration.py --phase fit --workers 8
# Freeze is a one-time immutable artifact; retain it on resume.
if (-not (Test-Path outputs/scanner-calibration/freeze.json)) {
  .\.venv\Scripts\python.exe scripts/run-scanner-calibration.py --phase freeze
}
.\.venv\Scripts\python.exe scripts/report-scanner-calibration.py --phase bind
.\.venv\Scripts\python.exe scripts/run-scanner-calibration.py --phase heldout --workers 6
.\.venv\Scripts\python.exe scripts/check-scanner-calibration.py --phase replay --workers 8
.\.venv\Scripts\python.exe scripts/check-scanner-calibration.py --phase refinement --workers 2
.\.venv\Scripts\python.exe scripts/check-scanner-calibration.py --phase representation --workers 2
.\.venv\Scripts\python.exe scripts/report-scanner-calibration.py --phase summarize
.\.venv\Scripts\python.exe scripts/plot-scanner-calibration.py
.\.venv\Scripts\python.exe scripts/render-full-fov-demo.py --output outputs/scanner-calibration --name heldout-double-continuous-seed101-noise0.01-calibrated --fps 4 --speed 4
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts
git diff --check
```

For the preserved historical artifacts, `check-scanner-calibration.py --phase
historical-zero` verifies four complete lean runs; `check-full-fov-compatibility.py
--output outputs/scanner-calibration/compatibility --revision b5b48e6` checks older
interfaces. The query-equivalence script applies only when resuming the archived
pre-optimization evidence with its `pre-vector-query.json`; fresh studies do not
need that compatibility exception. `report-scanner-calibration.py --phase inventory`
produces the final artifact/hash index.

## Resolved and unresolved

Resolved: charged/clipped margin implementation, zero-margin historical parity,
versioned decay and affine fitting, frozen development/test separation, complete
108-run evidence, numerical evaluation checks, public replay and faithful media.

Unresolved: successful margin tuning, absent-source/cessation recovery, four
strength-index regressions, inference representation sensitivity, calibrated
uncertainty, changing local drift and real instrument calibration. No policy is
newly frozen or promoted. Robotics integration remains deferred.
