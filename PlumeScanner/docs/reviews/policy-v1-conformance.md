# Frozen v1 implementation review

Status: **bounded implementation complete; adaptive advantage `not_demonstrated`**.
Date: 2026-09-13. The frozen policy is implemented and its behavior is inspectable.
The development comparison exposes resource and model limitations; it does not
establish calibrated reconstruction, physical strength estimation, or superiority.

Starting revision: `4a598acc38fb2ad8be0cfaf370c5955fa8ba275b`.
Final implementation revision: `8de3a3e81aae1975ff366fe216478c3f1e3a01a2`.
Branch: `codex/policy-v1-conformance`. Identify the closing documentation commit
with `git log -1 --format=%H -- docs/reviews/policy-v1-conformance.md`; it is also
recorded in local `outputs/policy-v1-conformance/closeout.json`.
Untracked user PDFs under `output/` are preserved. No remote push or history
rewrite occurred. Generated evidence remains local under `outputs/`.

## Implemented contract and repository boundaries

The [frozen algorithm](../specs/reconstruction-policy-v1.md) and
[instrument schema](../specs/instrument-profile-v1.schema.json) are unchanged.
The explicit entry point is `SensingController.from_session(...)`; legacy
construction, CLI invocations, packing and timing remain available separately.

| Responsibility | Implementation | Result |
|---|---|---|
| Immutable experiment settings | `sensing_session`, `sensing_execution_v1` | Explicit instrument, calibration, model, policy, startup, scenario and evaluation settings; canonical hashes; no physical overrides during a session |
| Gaussian inference | `signal_belief` | Existing dynamics and scalar conditioning; signed mean/covariance distinct from total positive signal and optional normalized scan mass |
| Scan decisions | `sensing_policy_v1` | Detached forecasts; descending mass prefix; cell-preserving column closure; bounded own-completion tests; shortest valid candidate; visible cycles and shortfalls |
| Physical compilation | `physical_scan` | Separate profile compiler fixes speed/cadence and charges transit, acquisition, settling and recovery waits; no implicit slowing or tracking subsets |
| Lifecycle and evidence | `controller`, `sensing_messages_v2` | Fixed active paths, actual visits/startup, duplicate/conflict handling, acquisition-order open-scan replay, actual receipts and causal availability guards |
| Adapter and trace | `sensing_execution_v1`, `sensing_cli_v1` | Actual scalar readings through the public API; exact delivery calls and completion events; controller inputs exclude evaluator truth |
| Reconstruction and inspection | `sensing_reconstruction`, existing `sensing_rendering` | Public-call causal replay, common-time loss, null/incomplete truth, model-conditional band diagnostic, existing six-panel layout with actual motion |

Public operations remain `plan_next`, `ingest_readings`, and `finish_scan`.
Version-two plans represent unavailable mass, candidate failures and
`no_legal_action`; reading/wind/receipt transport v1 decoding is preserved.
New evidence for closed scans is rejected. Earlier online scores use only
deliveries already available at that reference time. Querying a view does not
advance the live execution. Wide footprints remain one observation and do not
fabricate extra physical-cell visits.

Focused commits: A `c1ced3b` (profiles/mechanics), B `37bb8c7` (policy/controller),
C `d4d3d44` (causal evaluation), D `4dbfa8e` (adapters/configs/inspection),
`043549d` (baseline audit correction), `698c793` (profile/units/idle labels and
strict validation), `8de3a3e` (controller immutability and inspection provenance).
Changelog and status updates accompanied the checkpoints.

Before accepting results, an audit found the fixed corridor was incorrectly
gated by adaptive 90% retention. Corrected: feasible fixed geometry executes as
`static_corridor` with its actual retention, sharing startup, age and resource
rules. The frozen mass90 rule did not change. Interrupted preliminary quadrature
files remain in `development-baseline-audit` and are excluded from final results.
No held-out tuning or estimator-dynamics change was introduced.

## Declared fixtures and numerical results

Both profiles use synthetic retrieved-scalar units and identity calibration,
the existing 12 x 16 Gaussian model coefficients, 20 Hz cadence/capacity,
0.025 s integration and zero settling. They are numerical fixtures, not validated
hardware or recovered missing May instrument parameters.

| Physical setting | Coarse | Narrow |
|---|---:|---:|
| Sampling lattice | 20 x 32 | 220 x 320 |
| Beam height x width, FOV | 1/20 x 1/32 | 1/220 x 1/320 |
| Vertical sweep/transit, FOV/s | 1 | 20/220 |
| Horizontal transit, FOV/s | 0.625 | 0.625 |
| Full-height acquiring pass | 1 s | 11 s |
| Full scene from `(0, 0)` including initial move | 33.575 s | 3521.5975 s |

Common horizon is 75 s, with a 36 s startup ceiling and subsequent 12 s ceilings
capped by the remaining horizon. There is no arbitrary scan-count limit.
Gamma is 0.90, minimum readings four, maximum forecast iterations five; frozen
tolerances and the declared mass floor, stale/wind thresholds and delta remain
unchanged. Configurations serialize corridor, source, noise and wind coupling.
Source-free corridor geometry does not use hidden source coordinates.

All **36 descriptive runs** completed: two profiles x three cases x three
policies x seeds 0 and 1. Common source/noise realizations and instrument settings
are used within comparisons; different paths receive their own actual readings.
Six declared convergence fixtures accepted the common **120 x 160 / 0.05 s**
quadrature after comparison with 240 x 320 / 0.025 s. Maximum relative change in
`J` was **0.3772%**, below `max(1e-10, 1%)`; no further refinement was required.
This establishes the declared development check, not universal discretization
accuracy. Every final run has a complete 75 s truth horizon, including startup
and idle periods. Narrow startup is incomplete even though evaluation is complete.

Means over the two seeds follow. `J/T` is squared error of the signed enhancement
map; lower is better. `J = 75 * J/T`. Recall is the reported mean actual-sweep
positive-signal recall, a separate coverage diagnostic.

| Profile / case | Policy | J/T | Charged scan time, s | Mean sweep recall |
|---|---|---:|---:|---:|
| Coarse / known wind | mass90 = age rescan | 0.0195730 | 74.975 | 0.5263 |
| Coarse / known wind | fixed corridor | 0.0176410 | 74.425 | 0.9692 |
| Coarse / wind error | mass90 = age rescan | 0.0188490 | 74.975 | 0.5263 |
| Coarse / wind error | fixed corridor | 0.0179584 | 74.425 | 0.9692 |
| Coarse / source-free | mass90 = age rescan | 0.0236618 | 74.975 | 0.5263 |
| Coarse / source-free | fixed corridor | 0.0219300 | 74.325 | 0.9189 |
| Narrow / known wind | all three | 0.0168653 | 66.0275 | approximately 0 |
| Narrow / wind error | all three | 0.0181311 | 66.0275 | approximately 0 |
| Narrow / source-free | all three | 0.0222413 | 66.0275 | approximately 0 |

The coarse adaptive run completes startup, then uses **four reacquisitions**:
each tracking candidate exceeds the cycle budget. Age rescan takes the same
paths and therefore has the same evidence and loss. The fixed corridor executes
three corridor scans and one fallback after startup, obtaining roughly 4.7-9.9%
lower loss with slightly less scan time. These are **common ceilings, not exactly
matched expenditure**. There are no successful adaptive tracking actions in this
suite; localized-prior regression examples exercise that branch.

Narrow runs visit only **1.875% of the scene (six of 320 columns)** through four
partial-startup actions. The last 8.9725 s cannot accommodate another whole
column, producing explicit no action. All three policies share the startup
schedule; their equality is not evidence of equivalent tracking quality.
Field-wide loss is dominated by background and the prior when virtually no
plume is measured. Its sometimes lower value does not establish a better narrow
reconstruction or usable plume detection.

Reports include squared error of the **signed spatial-mean enhancement index**,
using the same quadrature; it is not an emission-rate error. For example, coarse
known-wind adaptive seed 0 has `J=1.4787350137`, `J/T=0.01971646685` and index
squared-error mean `0.000617348640`. Its 74.975 s comprises 69 s acquisition and
5.975 s travel. Narrow seed 0 spends 66 s acquiring and 0.0275 s travelling.
Settling/detector waits are explicitly zero for these compatible fixtures;
nonzero waits and partial-execution recovery are tested separately.

Nominal 95% pointwise posterior coverage is not calibrated: coarse adaptive case
means range from 74.36% to 84.75%, and fixed-corridor means from 76.35% to 83.08%.
Every inspected finite-grid simultaneous-band retention diagnostic is
`not_certified`. This diagnostic never changes allocation and is conditional on
the model. The unchanged no-decay dynamics produce an overly persistent downstream
tail relative to the compact finite-lifetime synthetic plume.

Compute time is separate from physical time. For coarse known-wind seed 0 the
recorded simulation took 6.41 s and replay/evaluation 26.59 s; narrow took 10.07 s
and 40.94 s. These are descriptive wall times from the local Windows NumPy/OpenCV
run with two suite workers, not hardware latency claims.

## Inspection and reproducible evidence

Numerical data were produced from clean revision
`043549daefa5b23f8f995c7c763a903b25bf1303`. Labelled videos were produced from
`698c793450c46254ccacaff011ed937b3f978eed`. The final implementation reproduces
the original coarse, narrow and delayed operational hashes. Label/palette changes
also reproduced those hashes; all 256 cached age colors match the original
OpenCV palette bit for bit.

The existing six panels and layout were preserved. Reviewed decoded frames
cover travel/acquisition, completion, delayed delivery, summary and idle states.
Measurements remain separate from outlines, uncertainty is explicitly standard
deviation, mass is explicitly a normalized scan quantity, and truth is labelled
evaluator-only. All **12 videos / 5,448 frames** decoded, including six original
and six labelled variants; expected frame counts/FPS and all 137 primary artifact
hashes validated. `inspection-manifest.json` records frame hashes;
`visual-review.json` records reviewed frames and findings.

Representative labelled local videos (relative to this review):

- Coarse: [physical time](../../outputs/policy-v1-conformance/development/coarse-known-wind-mass90_column_v1-seed-0/physical-profile.mp4), [compressed summary](../../outputs/policy-v1-conformance/development/coarse-known-wind-mass90_column_v1-seed-0/summary-profile.mp4).
- Narrow: [physical time](../../outputs/policy-v1-conformance/development/narrow-known-wind-mass90_column_v1-seed-0/physical-profile.mp4), [compressed summary](../../outputs/policy-v1-conformance/development/narrow-known-wind-mass90_column_v1-seed-0/summary-profile.mp4).
- Delayed batches and overruns: [physical time](../../outputs/policy-v1-conformance/development/delayed-inspection/physical-profile.mp4), [compressed summary](../../outputs/policy-v1-conformance/development/delayed-inspection/summary-profile.mp4).

At coarse t=37.5 s, 54/200 readings have arrived and the scanner is on its current
fixed fallback path. The completed summary at 45.125 s shows 200/200, 28% predicted
swept mass and `reacquisition`, without claiming target success. Narrow t=75 s
shows `NO LEGAL ACTION` and its fixed profile. In the separate delayed fixture,
the first sweep is planned to finish at 33.575 s, actually finishes at 33.825 s,
and is delivered/completed at receipt time 34.325 s. The 34.25 s frame correctly
has no delivered measurements; the 34.325 s summary has 640. Its physical video
reaches 75.575 s because of actual overruns; reconstruction still uses the declared
75 s horizon. That fixture changes delivery/execution and is outside the 36 comparisons.

Run these from the repository; use a fresh local output directory when
regenerating evidence you wish to keep separately:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check-policy-v1-spec.py --output outputs/policy-v1-conformance/worked-checks.json
.\.venv\Scripts\python.exe -m plume_focus.sensing_cli --suite-config configs/sensing-v1/development.json --output-dir outputs/policy-v1-conformance/development --workers 2 --render
.\.venv\Scripts\python.exe scripts/render-sensing-v1-inspection.py outputs/policy-v1-conformance/development
.\.venv\Scripts\python.exe scripts/inspect-sensing-v1-artifacts.py outputs/policy-v1-conformance/development
.\.venv\Scripts\python.exe examples/sensing_controller_v1.py --run-config configs/sensing-v1/coarse.json
.\.venv\Scripts\python.exe -m compileall -q src tests scripts examples
git diff --check
```

Verification: **366 passed, five Windows symlink-capability skips**. Import
boundary checks run in the suite. Nine independent specification arithmetic
groups pass; dynamic sequential/joint-batch disagreement is at most
`2.220446049250313e-16`. A production-dynamics test separately compares an explicit
augmented joint Gaussian calculation to the live filter at `1e-12`.
Profile schema validation, message round trips, fake-scanner/simulator equality,
causal batch/stream losses, duplicates/conflicts, partial startup, exact timing,
cycles/no action, finite truth and playback/display invariance pass. Compilation,
documentation links and Git diff checks pass. Historical recovery operational
SHA-256 remains `8a2692a5b741801911ca5e1d04f5614e8943ca5cfca8ff85c7c3442201d0d449`.

SHA-256 file anchors (canonical component hashes also appear in every report):

| Artifact | SHA-256 |
|---|---|
| `configs/sensing-v1/coarse.json` | `19c4d8005c7aa4711d8d7a3dcc02b1927200133f66da04045f96bce6777537ce` |
| `configs/sensing-v1/narrow.json` | `a732c6cf47451ca9929d12b4334230b9da5f07f25bcbe6fa149888aca124a790` |
| `configs/sensing-v1/development.json` | `9e3080a43867a9a790c5b32906260c38d8a15280c1cd9176a89a4b5b64704255` |
| `development/artifact-manifest.json` | `df26ee3bc802394b430689ae8a683cb91f1e649de25c2c8fa90f8dac2159c8b4` |
| `development/evaluation-freeze.json` | `e6cf77596db080b32f910a2ba3499cfd235cb3a8b7391514ff91c6125e0be141` |
| `development/development-summary.json` | `36f146438eac222aa64b9ffb45292333ae142a6e89fe2638aba09886577e1539` |
| `development/inspection-manifest.json` | `6612d9a155b6bb263d6cd652bc9c39eef849d63232e571544f130bfc03daa3b0` |

`development/` above is under local `outputs/policy-v1-conformance/`.
The manifest retains original and labelled videos and separate data/renderer
revisions. Derived inspection files are hashed separately to avoid circular
manifests. Runtime: Python 3.12.14, NumPy 2.4.4, OpenCV 4.13.0 on Windows.

## Resolved and unresolved

Resolved within this implementation: implicit physical defaults in the v1 entry
point; silent sweep slowing and tracking-subset packing; incomplete startup
mistaken for full startup; past causal views containing late evidence; missing
actual-time reconstruction evaluation; baseline coverage gating; and video
timestamp/profile/idle ambiguity. Settings remain configurable between explicit
experiments and fixed within each one.

Unresolved scientific/instrument issues: adaptive benefit under these ceilings,
narrow-profile startup feasibility, excessive signal persistence, posterior
calibration, real beam retrieval/noise/response calibration, and the physical
meaning of strength. Those need separately planned evidence or a newly named
model/policy version where algorithmic rules change. Hardware drivers, networking,
UCB allocation, emissions quantification, real-data expansion and replanning
during a sweep remain outside this goal. Historical held-out and recovery
`not_demonstrated` results remain intact; recorded-data expansion stays paused.
