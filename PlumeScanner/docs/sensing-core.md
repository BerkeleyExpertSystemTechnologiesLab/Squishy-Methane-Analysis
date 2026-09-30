# Callable sensing core

## Evidence-gated focus candidate (study executed, engineering partially complete)

As of 2026-09-23 this is an accepted partial research baseline, still opt-in. The
[open review findings](reviews/evidence-gated-focus.md#open-review-findings-and-partial-checkpoint)
cover actual-timing support, reacquisition scoring across ongoing scans, diagnostic
alpha provenance and the normal report producer's verdict. Those fixes remain pending;
the existing study outcomes do not establish a completed general integration boundary.

The [assignment](handoffs/evidence-gated-focus.md), its
[context package](handoffs/evidence-gated-focus-context.json) and the
[candidate specification](specs/evidence-gated-focus-v3.md) define
`evidence_gated_focus_loop_v3_candidate`. `evidence_gated_loop.EvidenceGatedSession`
binds `EvidenceGateSettings` (alpha, entry level, retain level) to the same calibrated
estimator, transport and physical compiler. Use the same three acquisition calls and
the existing dispatch:

```python
controller = SensingController.from_session(candidate.session)
helper = controller.plan_next(now_seconds, pose, ScanRequestV1(remaining_seconds), wind=wind)
```

The gate reads delivered scalar readings only. `evidence_gate` builds Bonferroni
measurement-error bounds at the frozen schedule length, classifies the diagnostic state
(`supported`, `below_level_on_observed_footprints`, `inconclusive`, `unavailable`)
and records decisions, transitions and evidence as versioned messages. Entry needs fresh
support from the current full epoch and retention needs fresh support from the sweep just
executed; otherwise the existing charged return-home/complete full scan runs. The
predicted upper-signal field still chooses where to focus. `EvidenceGatedPlan` carries
the old mean-gate value, evidence state, levels, alpha allocation, slot counts, bounds,
support identities with ages, fallback reason and coverage status. Gate support is not a
source-active probability and the bounds are not field posterior confidence. The
[handoff](handoffs/evidence-gated-focus.md) study driver is
`scripts/run-evidence-gated-focus.py` with resumable declare/verify/shadow/development/
freeze/heldout/report/refine/media phases. All 104 declared executions completed and
replay-verified: false focus falls from 1425 to 0 decisions in the blank pairs, 849 to 9
in cessation (each with a completed post-change full scan) and 533 to 10 in
intermittency (26 s coarse reacquisition), while continuous, wind-error and weak cases
are bit-identical to the calibrated control. Reconstruction is `mixed` (blank J/T 91%
higher); the [review](reviews/evidence-gated-focus.md) holds the tables and limitations.
Historical `FullFovSession`, calibration and v1-v5 bindings are unchanged.
Source-activity inference and robotics remain outside this assignment.

## Versioned scanner calibration experiment

`calibrated_scanner.CalibrationSession` binds `CalibrationSettings` and an optional
signal-decay half-life to the same lean loop. Use
`SensingController.from_session(session)` and the same three acquisition calls.
The historical `FullFovSession` is unchanged. These experimental identifiers are
`full_fov_margin_loop_v2_candidate` and `beam_reconstruction_decay_v1_candidate`;
they do not replace a frozen policy.

`vertical_margin_fov` expands each ranked prefix before mechanical compilation.
Every padded physical cell is charged; padding is rounded outward and clipped at
the field boundary. `CalibrationPlan` separates selected, padded, boundary-clipped
and mechanically swept cells. Zero margin retains historical paths. The
`full_scan_only` setting supplies the repeated-full-scan comparator; interpret its
FULL actions using that explicit setting, rather than as failures of signal support.

`DecayGaussianBelief` retains correlated conditioning. Its versioned propagation
uses exponential mean/covariance attenuation and integrated source/process input;
`half_life_seconds=None` preserves historical arithmetic. The evaluator's
`calibration_evaluation` uses the same transport/step partition and shared
covariance with two affine source-rate endpoint means. It never changes allocation
while fitting fixed evidence. Signal windows and clean truth belong to
`calibration_execution` and evaluator adapters, never controller inputs.

The [protocol](specs/scanner-baseline-calibration.md) declares development tuning,
offline fitting, then frozen held-out execution. Commands are in
`scripts/run-scanner-calibration.py`; `check-scanner-calibration.py` verifies public
replay, historical zero-mode agreement, refinement and representation probes.
`report-scanner-calibration.py` locks the final test configurations and summarizes
results. The completed [review](reviews/scanner-baseline-calibration.md) records
lower J/T in all 24 held-out pairs but persistent false focus in blank/cessation
cases. Keep the candidate opt-in; reference configs are in
`configs/scanner-calibration/`. Generated records/videos remain local. Evidence-
driven source activity is proposed next; robotics integration remains deferred.

## Fixed-evidence focus diagnosis

The [focus-extent review](reviews/focus-extent-diagnosis.md) records 42 saved
decision snapshots, six isolated forecast variants and a redundancy audit.
`focus_extent_diagnostics` is evaluator-only: it follows saved paths through the
shared acquisition engine and verifies unchanged-model parity against public
controller replay. It is not a new controller or policy entry point. Truth is
used only to score detached forecasts; diagnostic source-off replay does not
change operational settings or require historical policy agreement.

Use `scripts/diagnose-focus-extent.py` for declaration, controlled probes and
resumable replay, `scripts/report-focus-extent.py` for figures, and
`scripts/verify-focus-extent.py` for input/artifact and source-code integrity.
Versioned local records distinguish current/forecast mean, upper-bound eligibility,
requested and swept geometry, observed values and ages. Keep historical runtime
and media unchanged. Its proposed source/persistence calibration is now being
tested through the separate binding above.

## Lean full-FoV loop (current mechanism demonstration)

`full_fov_loop.FullFovSession` is an independent immutable binding. Pass it to
`SensingController.from_session(session)`; the three public operations remain
`plan_next`, `ingest_readings`, and `finish_scan`. Its Gaussian estimator and
`zero_inflow_v1` transport are unchanged. Historical v1–v5 remain available.

The state sequence is complete full scan → mean-supported focus, or charged
return home → another complete full scan. Every full scan requires fresh actual
readings at every physical lattice cell. An interrupted scan keeps its missing
cells; it cannot activate focus. Focus uses the existing .15 threshold and .95
upper signal quantile across the entire field, with a 12 s ceiling. Full scans
use their actual compiled duration, constrained by remaining session time.
Insufficient time yields an explicit incomplete full-scan obligation, not an
age-block substitute. No prior bypass, surveillance detour, belief reset, risk
scoring or percentage-mass gate exists in this binding.

The runtime responsibilities are deliberately small:

| Module | Responsibility |
| --- | --- |
| `full_fov_loop` | Settings, full/focus decision, per-sweep completion |
| `acquisition` | Evidence identities, acquisition-order replay, receipts |
| `physical_scan` | Neutral ordered paths, fixed mechanics, charged home return |
| `signal_belief` | Existing Gaussian inference and transport |
| `sensing_compat` | Historical policy dispatch, outside the lean decision loop |
| `full_fov_execution` | Fake-scanner execution and causal evaluation adapter |

`examples/full_fov_scanner.py` demonstrates two cycles with a declared JSON run
configuration. The mechanism suite is reproducible with:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
python scripts/run-full-fov-demo.py --phase declare
python scripts/run-full-fov-demo.py --phase run --noise 0 --workers 2
python scripts/run-full-fov-demo.py --phase verify --zero-only
python scripts/run-full-fov-demo.py --phase run --workers 2
python scripts/run-full-fov-demo.py --phase verify
python scripts/render-full-fov-demo.py --name coarse-continuous-noise-0
python examples/full_fov_scanner.py outputs/lean-full-fov/configs/coarse-blank-noise-0.json
python scripts/check-full-fov-compatibility.py
```

Outputs stay local. The 360 s fixtures retain source and instrument settings;
simulated noise is separately 0/.01/.05 while assumed likelihood noise remains
.05. Blank truth is explicitly source-free; the cessation control abruptly
removes truth at 180 s without changing the source-anchored estimator. That
control is not a physical source-shutdown model. Neither conformance nor these
descriptive runs establishes calibrated uncertainty or reconstruction advantage.

## Signal-gated threshold scanning (experimental v5)

`sensing_session_v5` provides immutable Gaussian and persistent-motion session
bindings. `with_threshold_policy(config, UPPER_POLICY)` or `MEAN_POLICY` converts
a compatible run configuration while preserving its estimator and instrument.
Both require explicit `zero_inflow_v1` transport. Instantiate through
`SensingController.from_session(config.session)` and use the unchanged
`plan_next`, `ingest_readings`, `finish_scan` lifecycle.

The current physical-lattice mean must exceed 0.15 somewhere to enable focus.
At the next decision, forecast the full field to half the available duration;
rank cells exceeding 0.15 by their latent 95th percentile (or mean in the
ablation), then actual age and deterministic rotation. Execute the longest legal
prefix plus due service. An absent signal or infeasible focused prefix produces
costed age scanning. Startup, delivery, detector recovery and actual service
credit remain causal. The active path never changes during a sweep.

`ScanPlanV4` is transport version 4, separate from policy version 5. Its gate,
provenance, eligible/requested/swept/omitted cells, partial-selection and fallback
fields explain the decision. Its inherited `target_mass` is inactive compatibility
metadata; no v5 status means `target_met`. `PredictiveDistribution.cdf/quantile`
uses exact finite-mixture marginal distributions, including zero-variance atoms.
It adds no detector-noise exploration bonus. These pointwise bounds are conditional
on the model; this experiment does not establish calibrated containment.

`substitute_snapshot_policy` changes only a compatible session binding on a
detached READY snapshot. The study declaration records parent snapshot identity.
The evaluator records autonomous public calls; saved delivery replay is distinct
from replanning, and later evidence never improves past online losses.

Resumable study commands, from the repository root:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe scripts/run-threshold-bound-study.py --phase declare
.\.venv\Scripts\python.exe scripts/run-threshold-bound-study.py --phase run --workers 2
.\.venv\Scripts\python.exe scripts/threshold_bound_observers.py outputs/threshold-bound-v5 --phase bounds
.\.venv\Scripts\python.exe scripts/threshold_bound_observers.py outputs/threshold-bound-v5 --phase refine
.\.venv\Scripts\python.exe scripts/threshold_bound_observers.py outputs/threshold-bound-v5 --phase contrasts
.\.venv\Scripts\python.exe scripts/threshold_bound_observers.py outputs/threshold-bound-v5 --phase contrasts --mode full
.\.venv\Scripts\python.exe scripts/threshold_bound_observers.py outputs/threshold-bound-v5 --phase readings
.\.venv\Scripts\python.exe scripts/threshold_bound_observers.py outputs/threshold-bound-v5 --phase coverage
.\.venv\Scripts\python.exe scripts/summarize-threshold-bound-study.py outputs/threshold-bound-v5
.\.venv\Scripts\python.exe scripts/plot-threshold-bound-study.py outputs/threshold-bound-v5
.\.venv\Scripts\python.exe scripts/audit-threshold-bound-study.py outputs/threshold-bound-v5 --replay --final
```

The runner includes both estimators by default; `--gaussian` and `--mixture`
partition jobs. `--fixture`, `--policy`, and `--mode` select frozen jobs without
altering settings. The job's seed is authoritative over the shared seed-0
configuration template. Resume checks reject changed inputs. Reading diagnostics
record decision-only bounds at every planned acquisition and completion, plus
separate noisy reading residuals. The final audit requires all 348 executions,
coverage records, 152 threshold reading reports and reviewed media.

Render a saved run with `scripts/render-threshold-bound-study.py
outputs/threshold-bound-v5 --fixture <name> --policy upper`. Physical and summary
views use the same trace and existing panels; `--fps`, `--speed`, and `--variant`
change observation only. Full decoding and operational SHA checks accompany
keyframe review. See the [protocol](specs/threshold-bound-scanning-v5.md) and
[study review](reviews/threshold-bound-scanning-v5.md).

## Detached scan-value diagnosis

The [diagnostic protocol](specs/scan-value-diagnosis.md) adds opt-in
`TransportDiagnosticSession` and `MotionTransportDiagnosticSession` bindings.
Use `with_transport_diagnostic(run_config, "zero_inflow_v1")` for exterior-zero
advection or `"legacy_clipped_v1"` for the historical operator. This identifier
is serialized and included in effective-model hashes, snapshots and scorer
cache keys. Measurement interpolation, physical profiles and public acquisition
operations are unchanged. Old configuration decoding and hashes stay unchanged.

At a READY boundary, `controller.decision_snapshot()` returns an immutable
`DecisionSnapshot`; `SensingController.from_snapshot(snapshot)` restores a
detached controller. Snapshots include component moments/weights, evidence and
availability clocks, causal wind, pose, detector recovery timing, actual visits,
startup and service state. An active scan cannot be replaced by a snapshot.

`sensing_counterfactual` is an evaluator adapter. Its forced-path entry point
recompiles the supplied geometry from actual pose/timing before execution, then
uses ordinary reading and completion operations. Its events explicitly label
forced actions; this is not an assertion that the production policy selected
them. `evaluate_saved_branch` replays recorded deliveries without re-running
planning and without allowing later evidence to improve earlier online scores.
Truth, file access and experiment orchestration remain outside the controller.

Resumable commands (PowerShell, repository root):

```powershell
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe scripts/run-scan-value-diagnosis.py --phase declare
.\.venv\Scripts\python.exe scripts/run-scan-value-diagnosis.py --phase run --workers 6
.\.venv\Scripts\python.exe scripts/scan_value_diagnosis_checks.py outputs/scan-value-diagnosis --phase verify
.\.venv\Scripts\python.exe scripts/scan_value_diagnosis_checks.py outputs/scan-value-diagnosis --phase refine
.\.venv\Scripts\python.exe scripts/scan_value_diagnosis_observers.py outputs/scan-value-diagnosis --phase evaluation
.\.venv\Scripts\python.exe scripts/scan_value_diagnosis_observers.py outputs/scan-value-diagnosis --phase exposure
.\.venv\Scripts\python.exe scripts/summarize-scan-value-diagnosis.py outputs/scan-value-diagnosis
```

The default run excludes mixture follow-ups. Select one frozen fixture with
`--fixture <name> --include-mixture`; use `--transport <identifier> --seeds 16`
or `32` only for a summary-requested unresolved comparison. Declaration verifies
historical trace hashes; resumes verify snapshot/declaration/candidate identity.
The checks command also provides `mixture-refine` and `compact-exposure`; the
observer provides `same-initial`, `projection` and `render`. Render requires
`--fixture`, `--candidate` and optionally `--transport`; `--variant` keeps a
different playback setting in a separate output directory. All generated
diagnostic evidence stays local under `outputs/scan-value-diagnosis/`.
For finer resume scheduling, `--phase branches` executes missing independent
continuations after scores exist. Stop fixture-level `--phase run` workers first;
never run both scheduling modes on the same fixture concurrently. The `contrasts`
and `innovations` observers check numerical attribution margins and pre-delivery
mixture residuals. `audit-scan-value-diagnosis.py <output> --final` verifies lineage,
causality, identical paired transport evidence and completion before hashing the
artifact set. `plot-scan-value-diagnosis.py <output>` writes separate scientific
diagnostic figures without changing the video panels.

Exact Gaussian risk memoization excludes the mean because fixed linear-Gaussian
posterior covariance is value-independent. It includes covariance bytes, model,
wind, transport, physical observation/delivery schedule and scoring settings;
returned diagnostics are detached copies. Mixture scores never use this cache.
The completed round has 9,536 paired continuation records plus 16 idle references,
447 passing regressions and five Windows symlink skips. The `coverage` observer
reports requested/swept area and predicted/actual excess separately; unavailable
blank truth coverage stays null. Final attribution supports Gaussian value
misranking in multiple fixtures, with limited candidate and continuation effects.
Mixture attribution and a strict blank-signal contrast remain unresolved. See the
[review](reviews/scan-value-diagnosis.md) for evidence, hashes and the proposed
source/lifetime calibration handoff. This does not activate a new default model.

## Reconstruction-value acquisition candidate

The [v4 acquisition specification](specs/reconstruction-value-policy-v4.md)
implements `reconstruction_risk_segment_v4_candidate` with either unchanged
estimator. `with_risk_policy(base)` binds a v2 or motion-v3 run to
`SensingSessionV4` / `SensingMotionSessionV4`; `SensingController.from_session`
keeps the same three acquisition operations. The existing `--run-config` CLI
and fake-scanner example decode both new run messages.

The detached scorer compares at most 14 physically compiled repeated patterns
over the request ceiling. Gaussian risk uses the evaluator's bilinear projection
of covariance. Mixture risk keeps its components: expected within-component
variance is analytic, and possible reading histories estimate remaining component
disagreement. No 90% mass or mean-readiness gate applies. Small oldest-block
service is included in candidate geometry and credited only after actual visits.

Transport-v3 plans use `value_selected`, include candidate risks, achieved mass
coverage, delivery/receipt and fixed causal-wind assumptions, numerical settings
and service requirements. They do not claim `target_met`. Frozen v1-v3 message
and allocator semantics are unchanged. Risk-model assumptions remain uncalibrated.
Use `value_selected_count` and per-decision risk/coverage fields for v4. Legacy
target-met/tracking counters retain their old meanings. Transport required-cell
fields also represent actual-visit startup; the study summary counts scheduled
refresh blocks specifically where `service_status == "required"`.

`expected_integrated_map_risk` describes hypothetical repetitions of the selected
core region. `actual_followon_integrated_map_error` includes later decisions made
from actual evidence, so the two values have different continuations. Their
comparison is descriptive, not an unbiased calibration test. The summary's clean
sample diagnostics are evaluator-only: they distinguish actual synthetic signal
from added noise without changing any reading supplied to the controller.

The experiment clock charges scanner motion, exposures, waiting and declared
delivery. It does not charge planner computation; decision latency is explicitly
zero in these synthetic comparisons. Per-run computation is reported separately.
The finite-mixture implementation does not establish real-time feasibility.

```python
from pathlib import Path
from plume_focus.sensing_cli_v1 import load_config
from plume_focus.sensing_execution_v4 import with_risk_policy
from plume_focus.controller import SensingController

base = load_config(Path("configs/sensing-v2/coarse-full-startup.json"))
experiment = with_risk_policy(base)
controller = SensingController.from_session(experiment.session)
```

The comparison declaration is `configs/sensing-risk-v4/study.json`. Run its
`declare`, `refine`, then `run` phases with
`scripts/run-reconstruction-risk-study.py --output outputs/reconstruction-risk-v4`.
The run phase requires a completed numerical freeze. Its 60 historical controls
are verified by configuration, trace and file hashes; 36 new runs are required.
`scripts/summarize-reconstruction-risk-study.py outputs/reconstruction-risk-v4`
checks paired configuration differences and separates map, index and cost outcomes.
Representative completed runs receive evaluation refinement and the existing
physical-time / decision-summary videos. Generated evidence remains local.

After generating the study, the same CLI and serialized fake-scanner example
accept an individual saved v4 configuration:

```powershell
.\.venv\Scripts\python.exe -m plume_focus.sensing_cli --run-config outputs/reconstruction-risk-v4/development/coarse-full-startup-wind-error-threshold90_segment_v2-seed-0-baseline-risk/configuration.json --output-dir outputs/risk-demo --render
.\.venv\Scripts\python.exe examples/sensing_controller_v1.py --run-config outputs/reconstruction-risk-v4/development/coarse-full-startup-wind-error-threshold90_segment_v2-seed-0-baseline-risk/configuration.json
```

Study directory names retain the reference policy name to identify their paired
historical run. The saved configuration and each plan identify the actual v4
policy; the directory name does not select the algorithm.

Implementation and joint Gaussian/mixture allocation refinement are verified:
0.1 s integration and 32 histories were frozen before the completed 96-record
study. All five evaluation refinements pass on the retained 120 x 160 / 0.05 s
mesh. The [completed review](reviews/reconstruction-value-policy-v4.md) records
mixed Gaussian outcomes, unresolved mixture quadrature, reviewed media and hashes.
The inherited boundary defect, changing
local drift and model calibration remain subsequent work. No performance claim
follows from a lower model-predicted risk.

## Persistent wind candidate

The [motion candidate](specs/persistent-wind-motion-v3.md) binds immutable
PersistentMotionConfig through SensingSessionV3 / SensingRunConfigV3. It retains
the v2 scan policies while carrying conditional Gaussian fields for a persistent
wind-bias prior. The covariance is explicit; error_speed keeps its old scheduler
meaning. Repeated wind messages and scan completion never restart the bias prior.

The three acquisition operations remain unchanged. The detached read-only
predictive_distribution(time, shape) exposes component weights/means/variances;
motion_diagnostics() exposes learned bias weights. Policy, trace, reporting and
rendering use mixture expected excess, rather than a Gaussian approximation of
its moments. Band diagnostics use a conservative component envelope. No truth is
accepted by the controller. Existing constructor/v1/v2 decoding remains available.

For a synthetic integration example, bind the motion prior explicitly:

```python
from pathlib import Path
from plume_focus.controller import SensingController
from plume_focus.sensing_cli_v1 import load_config
from plume_focus.sensing_execution_v3 import with_persistent_motion

base = load_config(Path("configs/sensing-v2/coarse-full-startup.json"))
candidate = with_persistent_motion(base, ((0.0001, 0.0), (0.0, 0.0001)), order=7)
controller = SensingController.from_session(candidate.session)
# Use plan_next, ingest_readings and finish_scan as in the existing example.
```

Order 7 is the study's finest tested provisional approximation, not a converged
hardware setting. The covariance above is the declared synthetic assumption;
it is independent of the scheduler's error_speed. The existing
`examples/sensing_controller_v1.py --run-config <saved-v3-configuration.json>`
command demonstrates two serialized plan/readings/completion cycles.

Run the declared study with:

```powershell
.\.venv\Scripts\python.exe scripts/run-persistent-wind-study.py --output outputs/persistent-wind-v3 --phase all --workers 2
```

The existing --run-config CLI and fake-scanner example accept saved v3 run
configurations. Changing local drift is required next-stage work, not represented
by learning this session-constant bias. The candidate is not a policy freeze or
an established reconstruction advantage; the 84-record evaluation and ten-video
inspection are complete. The [study review](reviews/persistent-wind-prediction.md) records failed
9/25/49-hypothesis convergence and accepted spatial/time loss quadrature. The
84-record comparison uses the finest tested, provisional 49-hypothesis model
(zero-covariance controls collapse to one). Its compute cost is separate from
physical scan duration; real-time feasibility is not established.
Main biased-wind adaptive mean loss increases by 1.00% coarse and 2.47% doubled.
Full-strength source-free map loss improves by 37.4%, but enhancement-index MSE
worsens. Numerical classification remains unresolved, and an inherited incoming-
boundary clipping defect is recorded for subsequent correction without changing
the completed study's transport model.
Offline evaluation uses SciPy's vectorized Gaussian CDF and a separable evaluation
of the same bilinear covariance polynomial. Reference-arithmetic tests bound their
rounding differences; planning retains its original calculation. Loss-only mesh
refinement queries aggregate moments and marks omitted band/excess diagnostics
unavailable. These observer optimizations do not change acquisition or evidence.

## Experimental mechanism revision

The [v2 candidate](specs/mechanism-grounded-policy-v2.md) reuses the controller,
Gaussian estimator, fixed-speed compiler and causal replay. `SensingSessionV2`
selects separate `threshold90_segment_v2`, `mass90_segment_v2` (zero-threshold
ablation), `static_corridor_segment_v2` or `age_segment_v2` behavior. V1 decoding
does not silently accept these settings. The existing CLI detects the explicit
`SensingRunConfigV2` message; the three public controller operations are unchanged.

```powershell
.\.venv\Scripts\python.exe -m plume_focus.sensing_cli --run-config configs/sensing-v2/coarse-full-startup.json --output-dir outputs/sensing-v2-coarse --render
.\.venv\Scripts\python.exe examples/sensing_controller_v1.py --run-config configs/sensing-v2/coarse-full-startup.json
.\.venv\Scripts\python.exe scripts/run-sensing-v2-development.py --output outputs/mechanism-revision/development --workers 2
```

The doubled physical profile uses 40 x 64 samples, 20 Hz and 0.5 FOV/s vertical
motion. Its full scene takes 129.5875 s; the 75 s tracking fixture therefore uses
explicit prior startup. A separate full-startup fixture preserves that resource
failure. These are tentative numerical settings, not calibrated hardware.
Reports separate injected/assumed noise, reportable/full signal, size sequences,
actual visits and outside-region surveillance. The [review](reviews/mechanism-revision.md)
records results without changing the historical claim status.
The completed study contains 36 main runs and seven diagnostics; six inspection
videos reproduce their saved executions. Coarse focusing expands and contracts,
but the fixed corridor has lower reconstruction loss in its three cases. Doubled
results are mixed and startup/coverage limits remain visible. The candidate is
implemented, not frozen or validated as superior.

## Frozen v1 implementation

The callable implementation now includes `beam_reconstruction_v1` and
`mass90_column_v1`, following the [frozen contract](specs/reconstruction-policy-v1.md).
The [implementation review](reviews/policy-v1-conformance.md) records verification
and the completed 36-run development comparison, reviewed videos and artifact hashes.
No numerical reference hardware or superiority claim is established.

```powershell
.\.venv\Scripts\python.exe -m plume_focus.sensing_cli --run-config configs/sensing-v1/coarse.json --output-dir outputs/sensing-v1-coarse --render
.\.venv\Scripts\python.exe -m plume_focus.sensing_cli --suite-config configs/sensing-v1/development.json --output-dir outputs/policy-v1-conformance/development --workers 2 --render
.\.venv\Scripts\python.exe examples/sensing_controller_v1.py --run-config configs/sensing-v1/coarse.json
```

`coarse.json` and `narrow.json` are complete synthetic configurations: a shared
12 x 16 estimator, 20 Hz cadence/capacity and 0.025 s integration, with physical
lattices 20 x 32 and 220 x 320. The narrow profile uses an 11 s full-height pass;
its roughly 3,522 s full scene cannot fit the 75 s monitoring horizon. Incomplete
startup is a resource-limit result. Neither profile supplies missing May hardware
parameters. Regenerate fixtures with `scripts/build-sensing-v1-configs.py`.

The suite checks quadrature on six declared development fixtures before choosing
one common evaluation mesh and running all 36 comparisons. Agreement with a
finer mesh accepts the tested coarser mesh at the declared tolerance. Failed
convergence remains explicit. Comparisons share ceilings and physical settings
within each profile/case; reports retain actual expenditure.
The static-corridor comparator executes its full declared geometry whenever
physically feasible. It shares startup, age and resource checks, but does not
have to satisfy the adaptive 90% coverage test. Its measured/predicted coverage
is reported, including poor coverage. Reacquisition is explicit when its full
geometry cannot fit; it is never packed into a smaller corridor.

Bind a validated `SensingSessionV1`, then use:

```python
controller = SensingController.from_session(session)
plan = controller.plan_next(now_seconds, pose, ScanRequestV1(remaining_cycle_seconds), wind=wind)
if plan.samples:
    controller.ingest_readings(actual_readings, available_at_seconds=delivery_time)
    view = controller.finish_scan(actual_receipt)
```

The controller returns transport-v2 plans with fixed paths, null-capable mass,
candidate/shortfall records, completion-forecast convergence and a model-conditional
coverage diagnostic. Empty `no_legal_action` responses leave it ready. Reading,
wind and receipt messages remain transport v1. Policy/formulation version v1 is
independent of message version v2. Explicit session policy parameters cannot be
overridden by legacy `ScanConstraints` or CLI physical flags.
Bound controller attributes also reject replacing session settings; construct a
new explicit session for a different instrument, model or policy.

`predictive_view` and `predictive_moments` reject a query before incorporated
evidence availability. For past online estimates, `CausalReplayV1` replays recorded
public calls in delivery order and verifies plans. It is not retrospective smoothing.
Reports compute common-time squared reconstruction loss `J`/`J/T` on signed
enhancement, squared enhancement-index error, and descriptive pointwise posterior
coverage. Missing truth produces null
errors; a short truth horizon produces an incomplete result. The posterior band
on each plan is a distinct finite-grid/time diagnostic and never changes allocation.

The existing six-panel renderer uses causal time-matched estimates and actual
executed motion for v1. Physical videos include idle time; summary videos contain
one completed sweep per frame. No redesign accompanies the policy change.
The suite also emits a separately configured delayed-batch/overrun inspection run.
The six development quadrature checks accepted 120 x 160 / 0.05 s, with maximum
refinement difference 0.3772%. The 36 comparisons remain `not_demonstrated`:
coarse adaptive tracking falls back to age rescan, while the fixed corridor has
lower loss; narrow full-scene startup visits only 1.875% of the scene. The model's
overly persistent signal and uncalibrated uncertainty remain visible. Reviewed
physical/summary video links and reproduction commands are in the implementation
review. The final regression suite passes 366 tests with five platform skips.

## Historical recovery interface and fixture

The remainder describes the preserved `recovery_scalar_v1` implementation and
its reproducible historical fixture. The authoritative v1 specification
is [reconstruction and scan policy v1](specs/reconstruction-policy-v1.md), with
an explicit [instrument profile](specs/instrument-profile-v1.md). This legacy path
uses its historical packing/timing rules; see the [original gap review](reviews/policy-v1-freeze.md).
The commands below continue to run the recovery fixture, not a newly selected
hardware reference or the coverage-preserving v1 allocator.

The recovery demonstrator uses a 12 x 16 correlated Gaussian signal grid,
timestamped scalar observations, expected positive-signal mass normalized for
a 90% scan target, and decisions after completed sweeps. Simulator and external
scanner adapters consume the same ordered physical plans. Hardware drivers,
networking, and replanning during a sweep are extension work.

## Run and inspect

From the repository, using the local environment:

```powershell
.\.venv\Scripts\python.exe -m plume_focus.sensing_cli --output-dir outputs/sensing-recovery --max-scans 5 --duration 75 --render
.\.venv\Scripts\python.exe -m plume_focus.sensing_cli --output-dir outputs/sensing-recovery-development --max-scans 5 --duration 75 --suite
.\.venv\Scripts\python.exe examples/sensing_controller.py
```

Optional static coverage/time plots use the `plots` extra:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[plots]"
.\.venv\Scripts\python.exe scripts/plot-sensing-results.py outputs/sensing-recovery-development
```

Installing the package also exposes `plume-focus-sensing`. The CLI produces a
report, complete serializable operational trace, measurement replay, and artifact
hashes. `--render` adds physical-time and decision-summary videos plus matched
rendering-only and measurement-only diagnostics. `--fps` and `--playback-speed`
change encoding only; the videos carry the same operational hash. At speed 1,
the physical-time view plays approximately one physical second per video second.
The summary deliberately emits one completed sweep per frame.

The legacy `plume-focus-video`, benchmark, and recorded-data commands remain
available with their original named model profiles. Their historical schema-v3
reports and frozen evidence are not silently migrated. New recovery reports
use `report_type=sensing_recovery`, schema version 1, with null spatial metrics
when evaluator truth is unavailable.

## Public Python API and messages

Import `SensingController`, `SignalModelConfig`, `ScannerCapabilities`,
`ScanConstraints`, `SensorReading`, `CausalWind`, `ScanPlan`, and
`ExecutionReceipt` from `plume_focus.controller`.

```python
controller = SensingController(capabilities, model, source_position=None)
plan = controller.plan_next(now_seconds, current_position, constraints, wind=wind)
# A simulator or hardware adapter executes plan.segments and acquires readings.
controller.ingest_readings(readings, available_at_seconds=delivery_time)
view = controller.finish_scan(receipt)
# The next plan starts no earlier than receipt/evidence availability.
```

`plan_next` locks an executable plan until `finish_scan`. An infeasible result
contains no samples and leaves the controller ready for another request.
`ingest_readings` accepts a chronological stream or a completed-sweep batch.
Out-of-order deliveries within an open scan replay from its saved starting
state. Earlier availability snapshots remain in the diagnostic event history;
replayed states become available at delivery time, not retroactively.
Batch and stream modes yield the same completed posterior for identical
acquisitions, values, calibration, and causal wind.

Duplicate reading identities with identical payloads are idempotent, including
after closure. Conflicting duplicates, future/unavailable readings, incompatible
integration, excessive readout cadence, and evidence for another scan are
rejected before observation mutation. A completion receipt cannot precede
acquired exposures or already delivered evidence. Receipts may report partial
execution, delays, and a different actual final position. Future wind is rejected;
a supplied wind estimate is held through the open sweep. The next decision may
provide a newer estimate. Wind, frame arrays, and truth are never inferred by
the controller from a simulator or video.

`belief_view()` returns copies of mean, standard deviation, and normalized
mass at the latest availability time. `predictive_view(at_seconds)` makes a
read-only forecast from available evidence. `assimilation_events` records
acquisition and delivery times and small grid snapshots, without storing a
full covariance per reading.

All transport messages support `to_dict()` and `Type.from_dict(payload)` and
round-trip through JSON. Top-level `schema_version=1` and `message_type` are
mandatory; unsupported versions/types and unknown fields are rejected. Nested
objects use the containing version. Required fields retain explicit names:

| Message | Contents |
| --- | --- |
| `ScannerCapabilities` | Physical lattice shape, axis speeds, readout limit, integration, footprint, point/column mode, coordinate frame |
| `SensorReading` | Scan/reading identities, actual position/footprint, floating-point value, acquisition midpoint, availability, integration duration |
| `CausalWind` | Measurement/availability times, velocity vector, declared error speed |
| `ScanConstraints` | Maximum scan seconds, target mass, minimum samples, bounded planning iterations |
| `ScanPlan` | Ordered timed movement/sweep/dwell segments, expected exposures, start/end pose and time, masks as lattice indices, retained mass, convergence/status/reason |
| `ExecutionReceipt` | Scan identity, actual completion and final position, receipt availability |

Coordinates are `(y, x)` in `[0,1]` across the field of view, with y increasing
downwards and x rightwards. Speeds are FOV fractions per second; footprints are
FOV fractions. The hardware adapter must map these to calibrated motor angles
and verify its capabilities. No physical camera/scanner calibration is implied.
All timestamps use a monotonic session origin in seconds. Detector values are
in the declared `signal_units`, with an explicit fixed offset and gain.

## Mathematical contract

The equations below describe the synthetic recovered estimator. For inferred
beam-integrated concentration, units, retrieval conventions, causal reconstruction
loss, and approximation limits, the frozen formulation linked above takes precedence.

The latent signal is Gaussian, with mean `mu` and covariance `P`. The initial
covariance is a squared-exponential spatial kernel in FOV coordinates, plus a
small numerical diagonal. For a footprint, four spatial Gauss-quadrature points
produce bilinear weights `h` over the coarse state. A detector reading is one
linear observation, even when its display footprint covers many pixels:

```text
y = (supplied_scalar - offset) / gain
r = (sensor_noise_std / gain)^2
K = P h / (h^T P h + r)
mu_new = mu + K (y - h^T mu)
P_new = P - K h^T P
```

A positive innovation raises positively correlated neighbors; an unexpectedly
low reading lowers them. Cells with zero cross-covariance do not change.
Conditioning reduces the relevant uncertainty. Unsampled cells have no fabricated
observation, and measured zero remains evidence. Signal means are not normalized
or clamped: this Gaussian model permits negative latent contrast.

Between readings, explicit substeps apply a linear bilinear advection operator,
an absorbing-boundary diffusion stencil, and correlated process noise:

```text
mu_next = A mu + source_input * dt
P_next = A P A^T + process_std^2 * dt * spatial_kernel
```

Substeps limit travel to half an inference cell and ensure diffusion stability.
A source Gaussian contributes continuously only when a source position is
explicitly supplied. It is a declared prior/model input. Source-free mode has
no reinjection. Finite-grid interpolation, diffusion stepping, and midpoint
exposure likelihood are numerical/model approximations, not calibrated physics.
The simulator independently averages signal across the moving beam exposure.

For each cell, the scan surface is its area times expected positive signal
above the threshold. For `d = mu - threshold`, `s = sqrt(variance)`,
`E[(X-threshold)+] = d*Phi(d/s) + s*phi(d/s)`; the zero-variance limit is
`max(d,0)`. Normalize these masses by their sum. A zero total triggers
reacquisition; it is never divided by zero. A 90% region retains predicted
signal mass. It is neither a map of independent occupancy probabilities nor
a 90% guarantee that the whole plume is contained.

The implementation keeps finite covariance on the coarse grid (default 192
states; at most 1024). It does not allocate a covariance over display pixels.
The conditioning identity is standard; see Rasmussen and Williams,
[Chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf).

## Declared demonstration assumptions

The `recovery_scalar_v1` configuration is a development fixture, not hardware
or uncertainty calibration. All values are included in generated reports.

| Parameter | Default |
| --- | --- |
| Inference grid / physical scan lattice | 12 x 16 / 20 x 32 |
| Initial signal mean / standard deviation | 0 / 0.2 |
| Sensor noise standard deviation / gain / offset | 0.05 / 1 / 0 |
| Correlation length / background threshold | 0.08 FOV / 0 |
| Process standard deviation | 0.01 signal units per sqrt(second) |
| Diffusivity | 0.00005 FOV squared per second |
| Known-source emission rate / sigma | 0.15 signal units per second / 0.035 FOV |
| Vertical / horizontal speed | 1 / 0.625 FOV per second |
| Detector cadence / integration | At most 20 Hz / 0.025 seconds |
| Beam footprint | 0.05 high x 1/32 wide FOV |
| Startup / subsequent scan ceilings | 36 / 12 seconds |
| Target / minimum samples | 90% predicted mass / 4 readings |

The source reuses the existing continuous methane puff field, with its video
background and clipping disabled for an explicitly clean scalar-sensor fixture.
Synthetic sensor noise is added after fixed gain/offset. Same seed, position,
and acquisition time produce the same noise across policies. Generator wind
is passed as an explicitly labeled synthetic external-wind proxy, optionally
with declared bias. It is not presented as an estimated real wind capability.

## Mechanical planning and evaluation

The packing and speed adjustment described here are **existing recovery behavior**,
and are implementation gaps against `mass90_column_v1`. The frozen policy preserves
the selected region through mechanical conversion and rejects incompatible fixed
speed/cadence profiles. No runtime change is implied by this documentation.

Column paths choose a monotone horizontal order and the nearer vertical
endpoint, with axis-aligned initial and intermediate repositioning. Sweeps
cover contiguous vertical spans. Duration is the larger of travel time,
readout time, and exposure requirements. Point mode moves to each target and
dwells. Changing display resolution does not change any path or sample time.
Beam-swept, requested lattice, and sampled-footprint areas remain distinct.

The adaptive policy forms a smooth predicted-mass surface, selects the target
mass, and greedily packs column spans by additional mass per added physical
duration. A bounded loop updates the forecast horizon from the compiled path.
Final retained mass is rechecked at that path's own completion time. A missed
target and nonconvergence remain explicit. This is a heuristic candidate search,
not a proof of global minimum cost or global infeasibility.

Broad uncertainty, a uniform/zero mass surface, or the rescan baseline selects
age-based reacquisition. Oldest columns precede recently observed ones, with a
rotating deterministic tie order. It consumes the same physical budget. A
full-scene startup that exceeds the ceiling becomes an explicitly partial
reacquisition. A fixed corridor baseline uses declared geometry; source-free
runs do not receive the generator's source through that baseline.

Evaluation receives optional truth separately. It scores the actual executed
beam sweep at completion and at five times across its interval. The interval
metric describes the full executed sweep footprint against changing truth,
not continuous observation of every location. Support uses 8% of the true
frame maximum; mass uses all nonnegative signal. Missing or zero-total truth
has null metrics. Reports expose timing, actual expenditure, area fractions,
uncertainty, convergence, and coverage failures.

The development suite compares known-wind, biased-wind, and source-free cases
for two fixed noise seeds, against fixed-corridor and rescan policies. Common
budget ceilings do not equalize actual scan cost or scan counts. These are
descriptive development results, with no held-out advantage claim. The inspected
historical held-out benchmark remains `not_demonstrated`.

The measurement-only diagnostic starts each sweep from the same Gaussian
state and uses the same dynamics and declared source. One branch advances
without conditioning and then applies the historical raster likelihood to
its derived mass; the other conditions the scalar readings at acquisition
times. No scan path changes. Its JS distances are spatial-shape diagnostics
across different belief interpretations, not a calibrated probability comparison.

## Known model limitations

The source model has no signal decay term; advection can carry predicted signal
farther downstream than the synthetic generator's finite-lived puffs. The coarse
grid and spatial kernel can also smear thin plume structures, and expected
positive signal assigns mass to uncertain background. These can make scans
miss the real plume under the available time budget. The repaired API and
visualizations expose these failures; they do not establish calibrated dynamics
or uncertainty. No parameters are tuned against the historical held-out cases.

Legacy profiles are useful comparison evidence. Their normalized likelihood
arrays and report fields must not be silently reinterpreted as the new Gaussian
state or new report semantics.
