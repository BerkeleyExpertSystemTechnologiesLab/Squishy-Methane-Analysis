# Agent Guide

Completed engineering checkpoint: [evidence-gated focus](docs/reviews/evidence-gated-focus.md)
implements and evaluates the opt-in `evidence_gated_focus_loop_v3_candidate` from the
[handoff](docs/handoffs/evidence-gated-focus.md) and its
[context package](docs/handoffs/evidence-gated-focus-context.json), around the unchanged
calibrated estimator. Entry and retention use delivered-reading measurement-error
bounds only; broadening remains a charged return-home/complete full scan with no
intermediate confirmation geometry. 11 declaration/verification checks, 18 acceptance
tests, 104 replay-verified executions, four refinements below 1% and reviewed media
pairs are complete; the fixed-evidence shadow audit shows identical beliefs over 70,942
events. Behaviour is supported: false focus falls from 1425 to 0 in blank pairs, 849 to
9 in cessation with a completed post-change full scan every time, and 533 to 10 in
intermittency with .10-.95 s coarse reading reacquisition in an ongoing full scan;
the previously reported 26 s is the later focused action. Continuous, wind-error and weak cases are
bit-identical to the control. Reconstruction is `mixed`: blank J/T 91% higher and
cessation/intermittency 8.0-8.5% / 4.7-4.8% higher. The candidate stays opt-in; do not
treat the gate bounds as calibrated field confidence, do not suppress the remaining
false model signal, and note that source-activity inference is the smallest next
experiment, not implemented. On 2026-09-23 the user accepted this as a partial
research baseline. The [2026-09-29 cleanup](docs/reviews/evidence-gated-focus-cleanup.md)
closes the four timing, resumption-scoring, diagnostic-provenance and report-verdict
findings. 553 tests pass with five platform skips; all 104 original executions
replay with unchanged causal loss and metrics. Derived reports remain separate
from frozen evidence. Doubled intermittency still has no qualifying hit despite
566 post-resumption slots in an ongoing full scan; the frozen fresh-full-scan
eligibility criterion excludes it, so report this limitation explicitly.
Engineering completion is bounded to this cleanup, not hardware/interface readiness
or scientific superiority. Do not promote the candidate. Company submission is
authorized to `BerkeleyExpertSystemTechnologiesLab/Squishy-Methane-Analysis`,
branch `plume_scanner_new`, under the user's WinChurchill account. Preserve the
company repository's other projects and keep generated artifacts local.

Completed checkpoint: [scanner baseline calibration](docs/reviews/scanner-baseline-calibration.md)
adds opt-in charged vertical margins and a versioned Gaussian decay model around
the same lean loop. All 36 development and 72 held-out executions replay; 520 tests
pass (five Windows skips), and four evaluation refinements pass. Padding fails
the core-coverage criterion; retain zero margin. The frozen synthetic fit is
half-life 6 s, source width .035 FOV and rate .2829738193616142. J/T improves in all
24 held-out pairs, but all calibrated blank cases falsely focus and no calibrated
cessation case resumes full scanning. Enhancement-index MSE worsens in four pairs.
Do not promote the candidate or infer calibrated strength/uncertainty. Reference
configs are in `configs/scanner-calibration/`; media stay local. The smallest next
stage is evidence-driven source activity, not implemented. Representation error
and local drift remain open. Robotics/simulator selection remain deferred.

Completed checkpoint: the [focus-extent diagnosis](docs/reviews/focus-extent-diagnosis.md)
adds evaluator-only fixed-evidence probes and an overlapping-mechanism audit.
All 42 declared snapshots match public replay; 506 tests pass (five Windows skips).
The mean already has an excessive downwind tail; source input/persistence and
representation contribute, while uncertainty adds background eligibility. No
missing amplitude multiplier, nominal wind conversion mismatch or closure
enlargement was found in the inspected fixtures. Do not simply remove source
input: that also loses continuing signal. Runtime and policy remain unchanged.
That checkpoint proposed source/lifetime calibration with a representation check;
it is now tested in the separate round above. Historical `not_demonstrated`
findings remain.

Completed checkpoint: the [lean full-FoV loop](docs/reviews/full-fov-loop-implementation.md)
implements mandatory complete startup, signal-supported focus, otherwise charged
return home/full rescan, with fresh actual coverage per sweep. The independent
runtime shares inference, acquisition and mechanics; historical policies remain
behind a compatibility adapter. All 18 declared noise/mechanism runs pass causal
replay; 489 tests pass with five Windows symlink skips. Retained source-anchored
inference can falsely reactivate focus after cessation, even without added noise.
Its proposed fixed-path source/lifetime calibration is now tested separately above.
No new policy freeze or general reconstruction superiority is established.
For this work, the user explicitly
requests focused Git commits and pushes of the working branch to the configured
remote. Preserve generated/local artifacts; do not force-push or merge by default.

Completed checkpoint: the [threshold-bound v5 experiment](docs/specs/threshold-bound-scanning-v5.md)
implements a current-mean signal gate followed by a latent upper-quantile region,
with mean-only ablation and charged age scanning when signal is absent or focused
scanning is infeasible. It uses explicit zero-inflow transport and fixed estimators.
Preserve its separate versioned bindings and historical modes. The
[review](docs/reviews/threshold-bound-scanning-v5.md) separates mixed allocation
evidence from unreliable bounds. All 348 runs, refinement, public replay and
16 decoded inspection videos are complete; 473 tests pass with five Windows skips.
Fixed-mixture conclusions remain numerically unresolved. The smallest proposed
follow-up is fixed-path forecast/source/representation calibration, not implemented.
Do not infer calibrated confidence, mandatory shrinking or a new policy freeze.

Completed checkpoint: the [scan-value diagnosis](docs/specs/scan-value-diagnosis.md)
adds opt-in `legacy_clipped_v1` / `zero_inflow_v1` diagnostic transport bindings,
detached READY snapshots and evaluator-only forced-path continuations. Preserve
historical session decoding/hashes and production lifecycle checks. Diagnostic
truth stays outside inference/scoring. The [review](docs/reviews/scan-value-diagnosis.md)
and `SYSTEM_GOAL.md` record supported Gaussian value misranking, limited candidate/
continuation effects and unresolved mixture attribution. The next proposed stage
is fixed-path source/lifetime and forecast calibration; it is not implemented.
Do not infer a new policy freeze or reconstruction superiority from the repair.

This project is an active sensing prototype for reconstructing moving,
intermittent plumes under mechanical scan and acquisition constraints.
It asks:

> Given partial, timestamped point measurements and mechanical scan limits,
> where should the scanner look next to reconstruct the changing plume and
> ultimately estimate its strength?

Plume reconstruction and strength estimation are the eventual purpose.
Prediction and adaptive scan selection are the means: choose useful coverage
without spending so long scanning that plume motion makes the assembled image
stale or distorted. The present software is a research demonstrator; it does
not yet establish calibrated reconstruction or strength estimation.

## Eventual Goal

The eventual goal is a method for reconstructing plume signal over space and
time, supporting numerical estimation of plume strength with a remote point
sensor. The motivation is to reduce dependence on costly conventional OGI
equipment and human visual interpretation. Better point-sensor accuracy and
lower end-to-end cost are hypotheses to establish, not demonstrated facts.
The physical meaning and units of "strength" remain to be specified from the
sensor measurement model; signal intensity is not automatically emission rate.

The user's instrument premise is that the desired measurement capability is
too costly or impractical to deploy as a full detector array, so one remote
point sensor must acquire spatial information sequentially. That economic and
instrument constraint motivates the problem; single-point sensing is not an
arbitrary software restriction. Treat this as the premise of the intended
instrument, not a universal claim that expensive sensors cannot form arrays.
The direct task is choosing measurements for reconstruction of an evolving
field. Mechanical path planning supports acquisition. Bayesian/POMDP theory
can inform the method without turning the project into robot navigation or
requiring a general-purpose POMDP solver. Related-work assessment must include
adaptive scientific-instrument acquisition, not only mobile-robot planning;
application specificity alone does not establish novelty.

The intended sensing loop is:

1. Start from a broad prior, full-scene scan, known source corridor, or external
   detector, with the source assumption declared.
2. Consume retrieved beam-integrated scalar measurements with their acquisition
   and availability times, footprint, and fixed calibration.
3. Condition the correlated signal mean and covariance on that evidence.
4. Propagate that belief forward using a wind/advection estimate.
5. Expand uncertainty according to scan age, wind error, model mismatch, and
   plume spread.
6. Select the next scan region with the declared policy: expected positive-signal
   mass in v1-v3, or experimental reconstruction risk in v4, with mechanical
   scan cost and posterior uncertainty represented explicitly.
7. Reconstruct the changing plume from timestamped evidence and report useful
   coverage, acquisition-time spread, reconstruction error, and eventually
   strength-estimation error when suitable calibration and truth exist.

Spatial coverage and elapsed scan time are coupled: measurements at different
positions are acquired from different plume states. A shorter focused scan
may reduce temporal distortion, but missing plume signal can worsen both
reconstruction and strength estimation. Smaller regions and shorter scans are
not objectives in isolation; neither improvement nor a particular nonlinear
relationship is assumed. The reference time or time window of a reconstruction
must be explicit.

For a reference experiment, freeze the physical footprint, sampling spacing,
readout/integration settings, mechanical speeds, and scan-pattern rules.
Keep them modular and configurable between declared experiments. The selected
region may move, grow, remain stable, or shrink as evidence changes; repeated
trimming is not a requirement. Display settings must not alter the instrument.

The project should be pitchable as:

> adaptive scanning for reconstructing moving plumes and estimating their
> strength under mechanical and acquisition-time constraints.

## Frozen v1 specification and current implementation

The authoritative v1 contract is
[`docs/specs/reconstruction-policy-v1.md`](docs/specs/reconstruction-policy-v1.md),
with [`instrument_profile_v1`](docs/specs/instrument-profile-v1.md) required for
every experiment. The freeze covers reconstruction over physical time,
`beam_reconstruction_v1`, and `mass90_column_v1`. It does not select a numerical
hardware reference. The explicit `SensingController.from_session(...)` path now
implements these rules; legacy entry points retain their historical semantics.
[`docs/reviews/policy-v1-freeze.md`](docs/reviews/policy-v1-freeze.md) records the
original gaps and handoff; [the implementation review](docs/reviews/policy-v1-conformance.md)
records their resolution, verification and the 36-run negative development result.

The subsequent [mechanism-grounded v2 candidate](docs/specs/mechanism-grounded-policy-v2.md)
is a separately named experimental revision, not a replacement freeze. It adds
reportable-excess tracking, two-axis block surveillance and an explicit service
interval, with independent simulated and assumed noise. Its 40 x 64 profile is
tentative synthetic hardware. Preserve v1 and assess both reconstruction failure
and self-reinforcing focusing; the minimum reading count is not a coverage guarantee.
The completed v2 development study has 36 main runs plus seven diagnostics.
Coarse tracking executes, but loses to the fixed corridor in each case; doubled
results are mixed and often reflect fallback rather than tracking. The claim
remains `not_demonstrated`. A hard 90% feasibility gate, uncalibrated temporal
model, weak-signal threshold and incomplete scene refresh remain open issues.

The [persistent wind candidate](docs/specs/persistent-wind-motion-v3.md) implements
`beam_reconstruction_motion_v3_candidate` alongside these estimators. It retains
v2 policy IDs and settings; exact mixture expected excess and component-envelope
diagnostics must not be replaced by aggregate-Gaussian shortcuts. Its
[study review](docs/reviews/persistent-wind-prediction.md) records verified code,
failed 9/25/49-hypothesis refinement and the completed 84-record comparison.
Classification remains `numerically_unresolved`; treat its finest approximation
as provisional. Full-strength source-free map gains coexist with worse
enhancement-index MSE, and main adaptive mean loss increases. Changing local drift
is required next-stage work; it is not solved by learning this session-constant
bias. An inherited incoming-boundary clipping defect also needs correction and
validation in subsequent model work; it was preserved in this frozen comparison.

The [reconstruction-value v4 candidate](docs/specs/reconstruction-value-policy-v4.md)
changes allocation only. Its `value_selected` paths minimize model-predicted
whole-field physical-time risk over bounded repeated patterns; 90% coverage and
mean readiness are not execution gates. Keep small charged oldest-block service,
actual-visit startup and immutable active paths. Both unchanged estimators remain
available. The [implementation review](docs/reviews/reconstruction-value-policy-v4.md)
records 96 completed comparisons, accepted allocation/evaluation refinement,
20 decoded inspection videos and 430 passing tests (five Windows skips).
Gaussian main results are mixed (3/12 pairs improve beyond 1%, 9 worsen).
Fixed-mixture results remain numerically unresolved; doubled biased-wind mean
map loss doubles, while the full-strength source-free diagnostic improves.
The claim remains `not_demonstrated`. Do not infer reconstruction advantage from
lower predicted risk. Boundary treatment, local drift and calibration remain
subsequent model work. Compute time is separate; real-time feasibility is unproven.

The explicit v1 and historical recovery interfaces are described in `docs/sensing-core.md`.
Its Bayesian state is a correlated Gaussian signal mean and covariance;
the normalized expected positive-signal mass used to choose a 90% region is
a separate decision surface. It is not a map of independent occupancy
probabilities or a 90% guarantee of whole-plume containment. Legacy profiles
and their recorded results retain their original interpretation.

Reading assimilation follows acquisition time; choosing a new path waits for
the current sweep to finish. Physical scanner coordinates, detector cadence,
inference-grid resolution, source time, and video playback are separate inputs.

Shared operating assumptions and frozen-v1 requirements follow. These apply to
the explicit v1 implementation; legacy modes remain separately labelled:

- The field is represented as a 2-D intensity or belief grid.
- The scanner may observe only part of the field during one update.
- Measurements can be temporally smeared because a scan is not instantaneous.
- Physical scans execute sequentially. A new scan starts only after the prior
  scan completes, and predictions target the planned next-scan completion time.
- The planner consumes completed scan evidence by default. Full-frame evidence
  is an explicit baseline mode and must remain labeled as such.
- Motion is represented by a short-horizon 2-D vector.
- That vector may come from an external wind/anemometer proxy, image-derived
  motion, or a synthetic test fixture.
- The Bayesian loop is:

  ```text
  prior -> scan likelihood -> posterior -> predictive prior -> next ROI
  ```

- The methane scenario is the main proof target.
- Mechanical column sweeping is the relevant scanner model for the methane
  scenario: vertical pass time is separate from horizontal step time.
- Source-anchored, source-free, and full-scene-start modes must remain distinct.
- Legacy smooth credible-region masks retain their historical meaning. The
  frozen v1 policy selects expected positive-signal mass and preserves every
  selected cell through column closure; it does not require repeated trimming.
- Fixed physical sweep speed and commanded cadence must be compatible with
  sampling pitch and integration. A compiler must not silently slow the scanner,
  enlarge its beam, or discard selected columns to fit a tracking budget.
- A failed tracking candidate produces an explicit shortfall and, when feasible,
  a separately labeled, fully charged age-based reacquisition action.
- Reports should keep requested, mechanically swept, and sampled area separate.
  Actual-sweep probability-mass recall is a current coverage metric; attention
  overlap is diagnostic only. Coverage alone does not establish reconstruction
  fidelity or strength-estimation accuracy.

## What Is Implemented

The repository has:

- a callable sensing controller with version-one serializable scanner, reading,
  wind, scan-plan, and completion messages; correlated Gaussian signal inference;
  physical-lattice point/column paths; and a simulator/fake-scanner adapter,
- immutable v1 instrument/session configurations, transport-v2 scan plans with
  explicit shortfalls/no action, fixed-speed profile compilation, coverage-preserving
  `mass90_column_v1`, charged age-based fallback and actual-visit startup tracking,
- public-call causal trace replay, common-physical-time reconstruction and signed
  enhancement-index error, unavailable/incomplete truth handling, descriptive
  posterior coverage and a model-conditional simultaneous-band diagnostic,
- physical-clock and compressed-summary inspection videos, matched rendering
  and common-prior measurement diagnostics, and a development coverage/time suite,

- motion-vector estimation from coarse frames,
- uncertainty-bound attention boxes,
- smooth attention fields,
- single-pixel/raster scan simulation,
- mechanical column-sweep scan simulation,
- sparse scan likelihood construction,
- Bayesian posterior and predictive belief updates,
- smooth credible-region scan masks,
- source-anchored methane plume demos,
- source-free plume tracking demos with known wind,
- full-scene startup before focused tracking,
- time-addressable synthetic and finite-video sources that reject unavailable
  future samples instead of freezing the final frame,
- causal non-overlapping scan scheduling and uncertainty-driven reacquisition,
- explicit scan-evidence and full-frame-baseline modes,
- schema-v3 reports that score the actual next mechanical sweep and retain
  per-scan causal schedules, belief-integrity checks, and sampled-value
  variation evidence,
- JSON and video outputs for proof inspection,
- regression tests around motion, scan policy, Bayesian updates, and video
  behavior,
- a deterministic headless methane benchmark with causal policy injection,
  exact matched scan budgets, development/held-out manifests, non-oracle
  baselines, paired confidence intervals, and explicit negative-result
  reporting,
- strict recorded-input schema v3 with truth-free causal execution, canonical
  frozen calibration lineage, and evaluator-only optional truth, plus explicit
  schema-v4 raw truth-free media windows with parent/window hash lineage,
- schema-v5 recorded manifests with qualification-v2 and preparation-v4
  contracts for the frozen R0C engineering diagnostic,
- dataset-neutral no-overwrite accession audit, preparation, and suite-freeze
  commands with decoder/pixel hashes and independently verifiable companion
  lineage records, and
- paired recorded-suite role isolation across source media, calibration
  sources, and acquisition/calibration capture groups.

## What Is Still Missing

Do not imply these are already solved:

- production hardware validation of the explicit v1 implementation,
- calibrated posterior coverage and demonstrated reconstruction advantage,
- real sensor calibration,
- real pan-tilt or scanner hardware control,
- real footage validation at scale,
- true unknown-source search policy,
- source inference from first principles,
- calibrated gas concentration estimation,
- safety-critical leak detection guarantees,
- multi-source plume separation,
- spatially varying, rotating, or turbulent motion fields,
- end-to-end embedded deployment,
- robust failure recovery when the plume leaves the field of view,
- a trained real-world plume segmentation model,
- uncertainty calibration against measured wind and sensor noise.

The first held-out fair-benchmark result is `not_demonstrated`: the current
adaptive policy loses substantially to a fixed source corridor at equal
mechanical cost. Treat that as the current evidence baseline, not as a test to
weaken or tune away with held-out cases.

The recovered controller's 18-run development suite also remains
`not_demonstrated`: its fixed corridor outperforms adaptive tracking in the
tested conditions. The coarse signal model has no decay term and can propagate
signal beyond the synthetic plume's lifetime. These are visible model failures,
not evidence that the corrected software or videos establish performance.

The frozen-v1 36-run descriptive development suite remains `not_demonstrated`.
Coarse adaptive tracking candidates exceed the declared cycle ceiling and use
age fallback; the fixed corridor achieves lower reconstruction loss with slightly
less actual scan time. Narrow startup visits only 1.875% of the scene within 75 s,
so it does not test tracking advantage. Common ceilings are not exactly matched
expenditure. The no-decay model and uncalibrated posterior remain known failures.
Conformance, renderer correctness and numerical quadrature checks do not establish
superiority. Recorded-data expansion remains paused after implementation closeout.

The current source-free mode tracks plume belief without re-injecting the true
source, but it still assumes a wind/advection estimate. It is not full unknown
source discovery.

## Exclusion Principle

When extending the project, protect the core claim by excluding work that makes
the system look stronger but less honest.

Do not turn the project into a generic YOLO, ViT, or CNN detector. Neural vision
may be added as an optional discovery, reacquisition, or validation module, but
it must not replace the active scan planner as the core contribution.

Do not make full-frame instantaneous vision the default assumption for the
tracking loop. If a mode uses full-frame evidence, label it as full-frame
evidence and keep it separate from mechanically scanned evidence.

Do not claim source discovery when the code uses a known source, source corridor,
or source-anchored prior. Use precise names: source-anchored, source-free with
known wind, or full-scene startup.

Do not use synthetic ground truth, future frames, or hidden source parameters in
a mode that is supposed to represent source-free or real-data tracking.

Do not hide scan duration. A policy that covers the plume but takes too long is
not a successful focused scanner.

Do not let the ROI collapse just because the posterior became sharp. Preserve
minimum scan budgets, scan-age uncertainty, and model-mismatch margins.

Do not treat "ROI", "sweep extent", "sampled mask", "attention field", and
"true plume envelope" as interchangeable. Keep these separate in code, reports,
and visual labels.

Do not optimize only for pretty visual overlays. The proof depends on metrics:
coverage, hit rate, scan fraction, scan duration, uncertainty radius, KL/JS
divergence, and failure cases.

Do not present image-derived motion as a required capability. The preferred
motion input for a real system is external wind/advection sensing. Image-derived
motion is a fallback, cross-check, or demo input.

Do not replace explicit uncertainty with a black-box score. Any learned module
must feed likelihood, prior, uncertainty, or validation signals back into the
planner.

Do not over-claim compute comparisons. The current planner is CPU-light because
it avoids per-frame neural detection, but any latency claim must name the frame
size, hardware, scan mode, and whether the timing is compute latency or physical
scan time.

Do not expand scope into full CFD, 3-D plume physics, regulatory leak
quantification, or production safety guarantees unless the user explicitly asks
for that research direction.

## Development Rules

- Read `CHANGELOG.md` with `SYSTEM_GOAL.md` before substantive work. Maintain
  the changelog actively: add or update a dated entry in the same logical change
  for meaningful behavior, formulation/specification, interface, physical-profile,
  experiment/result changes, and completed work items. Summarize the change,
  reason, verification, limitations and relevant revisions/review links; group
  related commits into broad checkpoints rather than logging every small edit.
  Distinguish frozen specifications, implemented behavior and pending work, and
  preserve historical negative findings. Keep detailed evidence in linked reviews
  and active-work status in `SYSTEM_GOAL.md`.
- Keep the callable sensing controller independent of simulation, rendering,
  recorded-data access, filesystem operations, and evaluator truth. Its
  dependencies are signal inference, physical planning, and typed messages.
- Simulator and hardware adapters supply actual scalar readings, causal wind,
  and execution receipts. Rendered or expanded raster pixels are not extra
  independent sensor observations.
- Record the starting branch/commit and inspect existing changes before work.
  Use a focused `codex/` branch and explicit file staging for logical commits.
  Preserve unrelated work and historical evidence; do not rewrite history or
  push remotely unless requested. Keep generated videos and chat exports local.
- Record verification commands, producing commits, and artifact hashes at
  closeout. Keep engineering completion separate from scientific claim status.
- Preserve the frozen formulation/policy identifiers. Change algorithmic rules
  through a new named version; record explicit immutable instrument, model,
  policy, and evaluation configurations for each experiment.
- Keep a current-time filter distinct from retrospective smoothing. Never use a
  later-delivered reading to improve a past online reconstruction score.
- Confidence-bound allocation is an experimental appendage. The v1 diagnostic
  must not silently replace the 90% expected-mass baseline or add uncharged scans.
- Preserve the existing renderer unless a specific semantic defect requires a
  correction. A formulation or allocator change does not require a video redesign.

- Keep the Bayesian planner and mechanical scan model central.
- Prefer small, testable changes over broad rewrites.
- Preserve the distinction between source-anchored and source-free modes.
- Preserve the distinction between physical scan cost and render/video time.
- Add or update tests when changing scan policy, uncertainty behavior, report
  fields, or CLI semantics.
- Generated videos are local artifacts and are not committed. Regenerate the
  small canonical JSON example only when its demonstrated behavior changes.
- Keep experiment execution headless. Filesystem and OpenCV video-writing
  effects belong in adapters/observers, not in the causal runner.
- Keep `plume_focus.video` as a compatibility and CLI façade. New source,
  scanning, metric, report, or rendering behavior belongs in its focused
  module, and the experiment runner must not import the façade.
- Treat generated videos and reports as evidence artifacts, not source of truth.
- Keep README claims aligned with implemented modes and metrics.
- Use CPU-friendly NumPy/OpenCV paths unless a new dependency has a clear role.

## Evaluation Checklist

For any new experiment or pitch artifact, record:

- formulation/policy versions and immutable instrument/model/configuration hashes,
- common reconstruction reference horizon, spatial/time quadrature, causal
  integrated squared error (or unavailable truth), and receipt latency,
- scenario and startup mode,
- source assumption,
- wind or motion source,
- frame shape and effective FPS,
- scan mode and mechanical timing,
- mean and final scanned area fraction,
- initial, mean, maximum, and final sweep duration,
- actual-sweep support and probability-mass recall,
- actual-sweep intersection-over-union and success threshold,
- attention hit rate as a separately labeled diagnostic,
- uncertainty radius,
- timeline termination reason and evidence mode,
- KL/JS distribution metrics when available,
- known failure mode or caveat.

Judge scan selection by its contribution to reconstruction and, when calibrated,
strength estimation under the declared resource constraints. Report coverage
and physical scan time together, including temporal distortion and missed signal;
do not equate the smallest ROI with the best result.
