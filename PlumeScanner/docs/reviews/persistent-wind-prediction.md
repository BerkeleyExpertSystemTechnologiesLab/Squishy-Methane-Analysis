# Persistent wind prediction experiment

Implementation, declared evaluation and video inspection complete, 2026-09-15.
Start: `66ea099299599f0d253b720424da81dba8bc8068`;
branch: `codex/persistent-wind-prediction`. The
[candidate specification](../specs/persistent-wind-motion-v3.md) is experimental;
frozen v1 and historical `not_demonstrated` findings are preserved.

**Hypothesis classification: `numerically_unresolved`.** All 84 records are
complete. The finest tested approximation gives mixed map-reconstruction effects
and does not establish joint reconstruction/strength-index improvement or a new freeze.

## Implemented hypothesis

`beam_reconstruction_motion_v3_candidate` represents session-constant two-axis
wind bias using persistent quadrature hypotheses. Each hypothesis retains the
existing Gaussian field dynamics, spatial covariance, source input, diffusion,
process noise and scalar observation model. Predictive likelihoods update log
weights before conditioning. No hypothesis is pruned, tempered, reset or resampled.

The public session selects this estimator through `SensingController.from_session`.
The three acquisition operations and fixed active paths are unchanged. Detached
component distributions supply exact mixture expected excess, aggregate signal
and within/between variance. Coverage diagnostics use component envelopes,
including hypotheses with underflowed floating-point weights. They are conditional
on the finite model, not calibrated physical guarantees or Gaussian bands around
aggregate moments.

Allocation remains v2: threshold 0.15, aggregate-mean readiness, descending 90%
prefix, preserved column closure, own-completion checks, charged surveillance and
the existing `error_speed` trigger. Bias covariance is separately declared in
FOV²/s². The prior is not inferred from that trigger's bound.

Both synthetic instruments retain 20 Hz acquisition and 0.025 s integration.
The 20 x 32 profile uses full-scene startup; tentative 40 x 64 uses prior startup.
Both use the 12 x 16 inference grid, 75 s horizon and 36/12 s ceilings. Retrieved
signal and noise use identity synthetic calibration; assumed and simulated scalar
noise remain 0.05. All remaining coefficients and source assumptions are serialized
unchanged from v2 in the run configurations. These profiles are not hardware calibration.

**Scope decision:** persistent error is this round. Changing local drift is also
required and remains the next motion-model stage. A learned constant correction
cannot describe unpredictable future deformation. No plume lifetime, threshold,
noise or mechanical settings were tuned in response to these results.

## Fixed evidence and controlled prediction

The initial nine-hypothesis diagnostic replays six saved v2 decision snapshots
across coarse/doubled biased-wind runs, including acquisition, delivery, decision
and receipt propagation boundaries. It forecasts +0/1/3/6/12 seconds and evaluates
hypothetical plans. These are not alternative executions of the saved path.

Controlled compact-field fixtures show local spreading across plausible transport
offsets, unchanged weights in the absence of observations and loss outside the
FOV without renormalization. The inherited interpolation also spreads signal
numerically; a strict compact-support or exact conservation claim is inappropriate.

The coarse first post-startup snapshot illustrates a failure. At +6 seconds the
baseline map MSE is 0.017349 and mixture MSE is 0.021009. Expected excess rises
from 0.00229 to 0.02217 and its background fraction from 0.427 to 0.649. The
mixture has concentrated on bias (-0.01732, -0.01732) FOV/s. The generator's nominal
wind correction is (-0.012, +0.015): its horizontal direction differs. This is
consistent with limited identifiability or compensation for model mismatch;
it does not identify one model defect as the sole cause. Some doubled snapshots
improve and others worsen. The initial diagnostic does not establish a fix.
In that same +6-second example, expected-excess centroid displacement is
(+0.00391, +0.00745) FOV for the baseline and (-0.01318, +0.00693) for the mixture.
The saved forecasts retain centroid and spatial covariance at every requested
horizon. These centroid changes include source input, thresholding and FOV loss;
they are not the displacement of one conserved material parcel.

## Numerical refinement

Both representative biased-wind adaptive seed-0 runs failed the predeclared 1%
test for 9 versus 25 hypotheses and again for 25 versus 49. The finest tested
49-hypothesis approximation is provisional; hypothesis convergence is incomplete.

| Profile | Hypotheses | Causal J/T | Integrated expected excess |
|---|---:|---:|---:|
| Coarse | 9 | 0.01999456 | 1.214979 |
| Coarse | 25 | 0.02137903 | 1.432295 |
| Coarse | 49 | 0.01985931 | 1.627695 |
| Doubled | 9 | 0.03822680 | 1.016871 |
| Doubled | 25 | 0.04105821 | 0.680503 |
| Doubled | 49 | 0.02596912 | 0.663239 |

The nonmonotone values matter: more hypotheses change learned weights and
executed paths. The declared limit does not support choosing whichever order
looks best. Controlled expected-excess refinements also have failures.
Spatial/time refinement passed at 120 x 160 / 0.05 s against 240 x 320 / 0.025 s.
Coarse J changed from 1.48944837 to 1.48922943 (0.0147%); doubled J changed from
1.94768409 to 1.94595559 (0.0888%). The common accepted evaluation mesh is therefore
120 x 160 / 0.05 s. This does not resolve the separate hypothesis-count failure.
All 84 common-configuration records completed on that frozen mesh. Profiles
remain 20 x 32 full-scene startup and tentative 40 x 64 prior startup.

## Main comparison

All 72 main records are complete. The following are two-seed means for the
biased-wind case, using the provisional 49-hypothesis model. Lower J/T is better.

| Profile | Allocation | Gaussian J/T | Mixture J/T | Change |
|---|---|---:|---:|---:|
| Coarse | Adaptive threshold90 | 0.01982008 | 0.02001911 | +1.00% |
| Coarse | Fixed corridor | 0.01786152 | 0.01898772 | +6.31% |
| Coarse | Age rescan | 0.01886197 | 0.01852806 | -1.77% |
| Doubled | Adaptive threshold90 | 0.02488442 | 0.02549843 | +2.47% |
| Doubled | Fixed corridor | 0.01980707 | 0.02008981 | +1.43% |
| Doubled | Age rescan | 0.01980707 | 0.02008981 | +1.43% |

The corridor and age-rescan estimator pairs have identical delivered readings,
delivery times and actual mechanical segments. Their differences isolate the
estimator on those evidence paths. Adaptive scans obtain different readings:
mean tracking count rises from 6.5 to 16, while mean charged time is 74.95 s for
both estimators. More tracking does not establish better reconstruction. The
adaptive mean signed enhancement-index error changes from +0.01253 to -0.01865.
Its mixture still loses to the corridor with that same estimator.

For doubled adaptive scanning, mean tracking count increases from 1 to 2 and
charged time from 74.6125 to 74.8875 s. Mean signed enhancement-index error changes
from +0.01311 to -0.00110, even though squared map error increases. Its seed-0 loss
improves by 4.45%, while seed 1 worsens by 10.78%. Neither one selected frame nor
one selected seed summarizes the result. Doubled biased-wind corridor and age
policies execute identical evidence paths: all five actions are surveillance.
Their matching outcomes therefore do not demonstrate two distinct allocations.

Known-wind and source-free known-wind zero-covariance controls reproduce v2.
All 36 main configuration pairs differ only in the declared motion configuration
and version identifiers. Thirty-two have identical delivered evidence and actual
motion; only the four biased adaptive pairs change paths. All 36 fresh Gaussian
baselines reproduce historical execution hashes and J exactly.
Observed biased-wind posteriors concentrate on (-0.01154, -0.01154) FOV/s,
opposite the generator's nominal horizontal correction. This is a model-conditional
likelihood result, not validated wind estimation. The overall classification
remains numerically unresolved because the hypothesis-count check failed.

## Signal-scale diagnostics and enhancement-index accuracy

All 12 supplementary records are complete. These use coarse prior startup,
source-free inference, biased wind and the same fixed noise at every signal scale.

| Signal scale | Gaussian J/T | Mixture J/T | Mean tracking scans, Gaussian → mixture |
|---|---:|---:|---:|
| 1 | 0.02195595 | 0.01374666 | 3 → 20.5 |
| 0.1 | 0.00031725 | 0.00030705 | 0 → 0 |
| 0 | 0.00009610 | 0.00007287 | 0 → 0 |

The completed full-strength source-free diagnostics give a partial positive
result: mean J/T decreases from 0.02195595 to 0.01374666 (37.4%), with mean charged
time 74.95 s for both estimators. Individual expenditures differ by 0.05 s.
Mean tracking count increases from 3 to 20.5. Both mixtures infer bias
(-0.01154, -0.02367), almost canceling nominal horizontal transport. This may
compensate for model mismatch; it does not validate recovery of physical wind.

Mean signed index error can cancel across time. The existing time-varying index
MSE, included in every reconstruction report, makes this distinction visible:

| Adaptive case | Gaussian index MSE | Mixture index MSE |
|---|---:|---:|
| Coarse biased wind | 0.00140407 | 0.00202817 |
| Doubled biased wind | 0.00139465 | 0.00249172 |
| Source-free biased, signal scale 1 | 0.00070150 | 0.00128915 |
| Source-free biased, signal scale 0.1 | 0.00002912 | 0.00003445 |
| Source-free biased, blank | 0.00002117 | 0.00002636 |

These are enhancement-field indices, not calibrated emissions. The source-free
full-strength map improvement therefore does not establish better strength-proxy
accuracy. Weak-signal mean map loss decreases by 3.2%, but both estimators execute
only surveillance. This is an estimator effect on that evidence path, not weak-
plume tracking success. Blank mean map error also decreases, while index MSE
increases. Both blank estimators execute five surveillance scans and no tracking.
All nonzero predicted excess is background; signal recall is unavailable because
the truth signal is zero. Blank observations cannot validate gas-transport velocity.

Fourteen of the 18 nonzero-prior mixture runs end with maximum weight above 99%:
the 12 main biased-wind runs and two full-strength source-free diagnostics.
The weak/blank runs instead retain effective counts 2.94–3.64 and maximum weights
0.405–0.536. The 24 zero-covariance controls start with one hypothesis and are not
counted as learned collapse. No component state is pruned even when its weight
underflows; component-envelope bands remain explicitly conservative.

## Coverage, age and actual spending

The following are supplemental geometric diagnostics, not substitutes for J/T.
The miss proxy is one minus full-executed-sweep-mask recall at five equally spaced
truth times per scan, averaged equally across scans. It is not contemporaneous
acquisition coverage or a physical-time-weighted capture fraction. Background
weight is the background fraction of predicted expected excess, averaged per
decision. The two-seed means below are Gaussian → mixture.

| Adaptive case | Geometric miss proxy | Background weight | Last-receipt stale/unvisited cells |
|---|---:|---:|---:|
| Coarse biased wind | 12.43% → 6.04% | 53.97% → 18.04% | 4.92% → 12.97% |
| Doubled biased wind | 68.72% → 64.16% | 60.97% → 51.79% | 60.72% → 62.52% |
| Source-free biased, scale 1 | 56.97% → 6.37% | 83.20% → 13.08% | 0% → 11.72% |
| Source-free biased, scale 0.1 | 48.41% → 48.41% | 99.28% → 98.65% | 0% → 0% |

The doubled adaptive runs visit only 50.18% → 48.75% of physical cells. All coarse
cases above eventually visit every cell, but focused tracking leaves older cells
stale. More favorable geometric coverage can coexist with worse causal map loss
and incomplete refresh. The full reports retain observed-age quantiles separately
from unvisited cells, plus every shortfall, fallback and no-action event.

Coarse biased adaptive travel/acquiring time changes from 11.90/63.05 s to
14.325/60.625 s; doubled changes from 7.5625/67.05 to 8.0625/66.825 s. Full-strength
source-free travel/acquiring time changes from 8.675/66.275 to 13.175/61.775 s.
Settling, detector waiting and extra receipt latency are zero in these frozen
fixtures. The remainder of the 75 s horizon is still included in causal loss.
Other actual-time components and per-seed spending are retained in the reports;
common ceilings and equal rounded means are not claims of exactly paired costs.

## Completed video inspection

The coarse and doubled pairs plus the blank diagnostic are rendered in both modes:
10 videos, 4,550 fully decoded frames. Selected frames were visually reviewed;
this is not a claim that every decoded frame received manual inspection. These frames
verify startup, scanner progression, accumulating scalar deliveries, acquisition
versus availability clocks, focused tracking, completion and final fallback/no
action. The six-panel layout is preserved; the mixture uncertainty subtitle is
the only panel change. Signal and uncertainty are current causal estimates; the
mass panel retains the start-of-sweep forecast for that scan's planned completion.
The white circle is the scanner marker; other white contours show evaluator
support. Measurement and age panels accumulate the current sweep's readings.
Full-session visit ages and unvisited fractions are reported numerically.

The existing signal display clips at 1, while inspected synthetic truth peaks
are approximately 1.98–2.09. Measurement display also clips outside -0.1..1.
The fixed scales are preserved, and raw values remain in inference and scoring.
Saturated colors therefore cannot establish peak-strength accuracy. Some snapshots
look closer even when the full-horizon loss does not improve. Post-hoc 25-second
loss bins aid inspection; they do not replace the predeclared 75-second outcome.

The doubled mixture's second action explicitly fails the tracking duration budget
and scans mostly background. Three later sweeps meet the predicted mass target;
the final action again reports a budget shortfall and charged surveillance. Its
more localized final image does not establish lower full-horizon loss across seeds.
Main videos stream readings at exposure completion; delayed batches and actual
overruns are covered separately by regression tests. Blank physical and summary
frames show available zero truth, actual noisy measurements, decreasing standard
deviation and charged surveillance. The readiness condition leaves mass ratio
unavailable; no false tracking-success label appears. Summary frames explicitly
mark completed sweeps and planned versus actual completion.

Local reviewed views (summary playback is compressed to one frame per sweep):

| Run, seed 0 | Physical time | Compressed summary |
|---|---|---|
| Coarse Gaussian | [Video](../../outputs/persistent-wind-v3/development/coarse-full-startup-wind-error-threshold90_segment_v2-seed-0-baseline/physical.mp4) | [Video](../../outputs/persistent-wind-v3/development/coarse-full-startup-wind-error-threshold90_segment_v2-seed-0-baseline/summary.mp4) |
| Coarse mixture | [Video](../../outputs/persistent-wind-v3/development/coarse-full-startup-wind-error-threshold90_segment_v2-seed-0-motion/physical.mp4) | [Video](../../outputs/persistent-wind-v3/development/coarse-full-startup-wind-error-threshold90_segment_v2-seed-0-motion/summary.mp4) |
| Doubled Gaussian | [Video](../../outputs/persistent-wind-v3/development/double-prior-startup-wind-error-threshold90_segment_v2-seed-0-baseline/physical.mp4) | [Video](../../outputs/persistent-wind-v3/development/double-prior-startup-wind-error-threshold90_segment_v2-seed-0-baseline/summary.mp4) |
| Doubled mixture | [Video](../../outputs/persistent-wind-v3/development/double-prior-startup-wind-error-threshold90_segment_v2-seed-0-motion/physical.mp4) | [Video](../../outputs/persistent-wind-v3/development/double-prior-startup-wind-error-threshold90_segment_v2-seed-0-motion/summary.mp4) |
| Blank mixture | [Video](../../outputs/persistent-wind-v3/development/diagnostic-source-free-biased-scale-0-seed-0-motion/physical.mp4) | [Video](../../outputs/persistent-wind-v3/development/diagnostic-source-free-biased-scale-0-seed-0-motion/summary.mp4) |

The [inspection manifest](../../outputs/persistent-wind-v3/inspection-manifest.json)
records media/input hashes, producing revisions and selected frame paths.
The [visual review](../../outputs/persistent-wind-v3/visual-review.json) records
findings and display limitations. Both playback modes reproduce the same saved
operational hash for each run.

## Additional inherited boundary defect

A post-hoc engineering probe isolates incoming-boundary clipping. Initialize the
first inference column to one and the rest to zero; disable source input,
diffusion and process noise; apply horizontal wind +0.04 FOV/s. With no readings,
the first column remains one. The field mean integral rises from 0.0625 initially
to 0.3025 at 6 s, 0.54246 at 12 s and 0.98755 at 30 s. A zero-external-inflow
translation would not maintain that incoming signal.

The inherited interpolation clips locations between the FOV boundary and the
first cell center to the first cell. Small propagation steps can therefore pin
incoming-edge values and supply artificial persistent inflow. This is distinct
from renormalizing lost signal and from the missing decay term. It affects both
estimators and is a plausible additional source of misleading wind likelihoods;
the probe does not establish its contribution to the study's loss differences.

This diagnostic was added after observing persistent signal and concentrated
weights. It is separate from the frozen 84 comparisons. Their transport and all
main-study parameters remain unchanged. Boundary treatment requires correction
and validation in subsequent model work, alongside the already required changing
local drift stage. Compact interior FOV-exit checks alone do not cover this defect.

```powershell
.\.venv\Scripts\python.exe scripts/diagnose-persistent-wind.py --output outputs/persistent-wind-v3/diagnosis --boundary-probe
```

## Verification and provenance

- Full suite at `bfdaecc`: 403 passed, five existing Windows file/directory
  symlink capability skips.
- Independent frozen-v1 specification arithmetic: passed.
- Mixture tests cover independent batch likelihood/posterior calculations,
  aggregate moments, nonlinear excess, correlated/zero covariance, sparse
  footprints, repeated/zero/positive evidence, duplicate/conflict rejection,
  delayed replay, actual overruns, immutable paths and causal read-only queries.
- Zero covariance reproduces v2 evidence/paths and completed inference. Fake
  scanner/public replay, configuration round trips, import boundaries, unavailable
  truth and playback-independent execution have regression coverage.
- Offline SciPy CDF and separable bilinear variance evaluation agree with reference
  arithmetic. Planning retains the original summation. Saved order-3 motion and
  historical v1 execution hashes reproduce after this optimization.
- Blank available truth now reports all nonzero allocation as background; recall
  without signal remains unavailable. This is reporting only.
- All 42 estimator pairs pass the configuration audit. Thirty-six pairs have
  identical actual evidence/motion; six biased adaptive pairs change paths
  (four main and two full-strength diagnostics). All 36 historical baseline
  executions and J values reproduce exactly.
- Summary annotation preserves every existing numerical result. All mean index
  MSE values satisfy the area-mean Jensen bound against map MSE. Observer checks
  supplement the full core regression; no allocation or model tuning follows them.
- Final compilation and `git diff 66ea099 --check` pass. The original 301-file
  evidence index is preserved and verified separately from the final collection.

Implementation: `81e3192`; frozen declaration: `c9d71e0`; fixed-evidence clock
correction: `ba4b1eb`; offline evaluation optimization: `0201240`; inspection and
blank-truth reporting: `bfdaecc`. Orders 3/5 were produced before the observer
optimization; order 7 records `0201240` in its reports. Generated evidence remains
local under `outputs/persistent-wind-v3/`; untracked historical `output/` is preserved.

```powershell
.\.venv\Scripts\python.exe scripts/run-persistent-wind-study.py --output outputs/persistent-wind-v3 --phase all --workers 2
.\.venv\Scripts\python.exe scripts/summarize-persistent-wind-study.py outputs/persistent-wind-v3
.\.venv\Scripts\python.exe scripts/plot-persistent-wind-diagnostics.py outputs/persistent-wind-v3
.\.venv\Scripts\python.exe scripts/render-persistent-wind-inspection.py outputs/persistent-wind-v3
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check-policy-v1-spec.py
.\.venv\Scripts\python.exe -m compileall -q src scripts examples
git diff --check
```

Use a fresh output directory for a new study; preserve saved media. Resuming the
study checks declared configurations. Two noise seeds are descriptive, not
independent plume realizations. Common time ceilings do not imply equal spending.
Compute cost is reported separately from mechanical time; this experiment does
not establish real-time or embedded feasibility. These are elapsed compute times
under concurrent run/rendering load, not controlled latency benchmarks.
For the 18 nonzero-prior runs, simulation takes 232–393 s and causal replay/evaluation
591–752 s per 75 s fixture. Gaussian baselines take 10–16 s and 58–85 s respectively.
Zero-covariance mixture controls use one hypothesis. These costs preclude claiming
that the 49-hypothesis implementation is a demonstrated real-time controller.

## Local evidence and hashes

The [complete paired summary](../../outputs/persistent-wind-v3/descriptive-summary.json)
contains all 42 pair checks, group means, per-run timing/age/background diagnostics,
posterior concentration and configuration/evidence checks. Each run directory
contains its exact configuration, public-call trace and full reconstruction report.
The [original data index](../../outputs/persistent-wind-v3/original-data-artifact-manifest.json)
was verified and archived before final rendering. Its listed files remain unchanged.
Final inspection/closeout metadata is separate from that original data snapshot.
The [closeout provenance](../../outputs/persistent-wind-v3/closeout-provenance.json)
records the exact final revision, verification and review hashes. The
[final artifact index](../../outputs/persistent-wind-v3/artifact-manifest.json)
records hashes for the collected evidence and that closeout record.

SHA-256 values below hash file bytes. Canonical configuration hashes, including
separate instrument/model/policy/motion identities, are also retained in reports.

| Artifact | SHA-256 |
|---|---|
| `declared-study.json` | `2d79a5ab24865b5a21f361aeecd5817c1d92f2393af049c98017b6fa2ff4fc80` |
| `execution-configurations.json` | `d7c2ae7cbf9a3bb5d6640760d651970a8a507354b8bb114495117a9d7a8cbcc0` |
| `numerical-freeze.json` | `51e10c6a1753ed0f425acbabcec41ad8309a8339969c01c46cd78dba6f21857c` |
| `study-summary.json` | `4af321684a612e66e545d3bdab46fc189b7ecac003e5c7afc841dfa99294e122` |
| `descriptive-summary.json` | `53df57fa84eee3d592b686e5a62954d3b4e5485c5aa29d7af593e4cd8443127e` |

The fixed declaration's source manifest hash is
`d5079b51453b24a10b99712774c062cb1fd01656fcf9e9753850293506a9efd1`.
Run reports record job-start revisions; `phase-provenance.json` records earlier
loaded-code phases and runtime versions. Observer changes after `bfdaecc` preserve
the estimator, planner and physical settings. The final artifact index records
collection provenance; it is not a claim that every earlier artifact was produced
by the final documentation revision.

## Resolved and unresolved

Completed: persistent conditional inference and API integration; fixed-evidence
and controlled forecasts; 72 main plus 12 diagnostic records; declared bounded
numerical checks; preserved-layout media inspection; regression and evidence
verification; synchronized changelog, specification, implementation and status.
No remaining engineering work is required to close this declared experiment.

Unresolved: hypothesis-count convergence, calibrated posterior/transport and
strength estimation, persistent tails, weak-signal behavior, the hard-budget gate
and incomplete refresh. The positive source-free map result is partial and
provisional; it does not establish joint map/index improvement or adaptive
superiority. Claim status stays `not_demonstrated`.

Next model work must address changing local drift and correct/validate the
isolated incoming-boundary defect. Constant-bias learning does not solve either.
Their effects must be evaluated separately rather than attributed retrospectively
as the sole explanation of this study. Hardware calibration, networking,
real-data expansion, new allocation and within-sweep replanning remain outside
this completed goal; recorded-data expansion stays paused.
