# Plume Focus Living System Goal

Last updated: 2026-09-29

Overall status: `active`

## Opt-in research branch release (2026-11-01)

`shulun/scan-order-study` packages the original C2 P0-V B/L_up comparison as
one independently runnable, preselected synthetic example. Both actions start
from the same history/pose/region and pay the frozen planning fees before their
own exposure schedules. See [the branch guide](docs/scan-order-study.md).
No default, gate, model or production controller change; no full-study,
real-time, hardware or field-validation claim follows from this release.

## Completed company submission (2026-09-29)

The user supplied the company repository and explicitly requested a new
`plume_scanner_new` branch and a clean push under their GitHub user, WinChurchill.
The reviewed package from `1bdf3c6`, with source preparation through
`e13995263f3a38a954a610a1cd8279767e9025e7`, is submitted under `PlumeScanner/`
on company main `94ed07fc96592b7cb073f59b88f9d98bd6e1ff37`. Initial company
submission commit `79f2d0137ca35950f82e09640b2487999147c6c1` is pushed and its
remote tip verified. GitHub resolves both author and committer to WinChurchill.
Fresh installation passes 554 tests/five skips, demo/replay/render commands work,
and all 271 package blobs match the source after normal text EOL conversion.
Frozen evidence bytes and other company projects are preserved. Local preparation
branch is `codex/company-submission`; the company branch is exactly
`plume_scanner_new`. Closeout notes follow in a documentation-only commit.
The [submission review](docs/reviews/company-submission.md) records verification.
No research experiment or default promotion starts from this transfer.

## Completed review and cleanup round (2026-09-29)

Start `98aeba97a9e0d18e77ac018bf1384a4c0b624e71` on
`codex/evidence-gated-focus`; cleanup branch `codex/evidence-gated-cleanup`.
Implementation commit `b1684f2` closes all four documented engineering findings.
553 tests pass with five platform skips; all 104 historical executions replay
with unchanged causal loss and original metrics. The separately hashed
[cleanup review](docs/reviews/evidence-gated-focus-cleanup.md) records corrected
derived summaries and renderer verification. Estimator, policy settings and
scientific claim remain unchanged. The company URL has now been supplied and the
separate submission above is authorized; no next experiment starts automatically.

Current stage: evidence-gated focus candidate engineering cleanup `complete`:
implemented, study executed, accepted as a research baseline, and four engineering
review findings closed (docs/reviews/evidence-gated-focus-cleanup.md; behaviour supported in tested
fixtures, reconstruction mixed, promotion withheld). Scanner baseline calibration `complete` (docs/reviews/scanner-baseline-calibration.md); prior focus-extent and overlapping-mechanism diagnosis `complete` (see docs/reviews/focus-extent-diagnosis.md). Prior lean full-FoV/focus implementation and staged mechanism/noise
verification `complete`; fixed-path source/lifetime and forecast calibration
implemented and tested with a partial outcome. Threshold-bound allocation test `complete` (Gaussian
mixed; bound reliability unsupported; fixed mixture numerically unresolved);
scan-value diagnosis complete; reconstruction-value acquisition
complete (Gaussian mixed; fixed mixture numerically unresolved); persistent
wind uncertainty experiment complete (numerically unresolved); mechanism/v1
implementation complete; recorded-data expansion paused.

Current claim status: `not_demonstrated`

## Evidence-gated focus (study executed; engineering review closed)

User decision on 2026-09-23: retain the current policy as a partial research baseline
and record its Git checkpoint without changing runtime behavior. That historical
checkpoint left four review findings: accept valid actual acquisition timing, count reacquisition within scans that
straddle signal resumption, separate evidence alpha from next-action diagnostics, and
correct the normal report producer's preservation verdict. The
[review](docs/reviews/evidence-gated-focus.md#open-review-findings-and-partial-checkpoint)
records reproductions and closure conditions. Eighteen focused tests passed during
review, but they miss these cases; the full suite was not rerun in that review.
The authorized 2026-09-29 cleanup now closes these findings with new regressions,
full-suite verification and separately derived reports. Source-activity inference
and robotics integration remain pending separate work.

The [external-agent handoff](docs/handoffs/evidence-gated-focus.md) and its
[context package](docs/handoffs/evidence-gated-focus-context.json) define a bounded
test of measurement-supported entry/retention gating around the unchanged calibrated
estimator. Broadening means a charged return-home/full-FoV scan; there is no new
intermediate scan geometry. Gate parameters are proposed, not validated.

The [candidate specification](docs/specs/evidence-gated-focus-v3.md) is implemented as
`evidence_gated_focus_loop_v3_candidate`; 11 declaration/verification checks, 18 new
acceptance tests and an inert-hook integration keep historical traces identical. All
104 declared executions (32 development, 72 held-out) completed and replay-verified,
the four declared refinement comparisons converge below 1%, and the media pairs decode
and leave execution hashes unchanged.

Behaviour is supported by the predeclared criteria: across both stages every blank
pair drops from 1425 false-focus decisions to 0, cessation from 849 to 9 with a
completed post-change full scan in every case (37.0-39.0 s coarse, 144.0-144.5 s
doubled), and intermittency from 533 to 10. Corrected scoring gives the coarse
candidate its first qualifying reading .10-.95 s after resumption, in the ongoing
full scan; the old 26 s score skipped that scan and described the later focused
action. Twelve candidate/control intermittency summaries change, while execution,
loss and criteria remain unchanged. The doubled intermittency case reports
`incomplete_full_scan_budget` because one full scan cannot fit in the remaining
120 s for a fresh full scan, but its ongoing full scan still contains 566
post-resumption slots and no qualifying hit. That missed opportunity is now explicit;
the predeclared eligibility rule still excludes this case. Continuous, supplied-wind-error and weak cases are bit-identical to the control
(+0.0000% J/T, equal core coverage), so the predeclared preservation tolerance holds
exactly and no behaviour is lost where signal is genuinely present.

Reconstruction does not improve. Blank J/T is 91% higher in both stages, cessation
8.0-8.5% and intermittency 4.7-4.8% higher, so the honest classification is
`mixed`: the operating behaviour is repaired while map error is not. The calibrated
estimator still recreates false peaks (105 mean-positive but evidence-unsupported
decisions; 20 candidate runs end with a false posterior peak, maximum final mean
0.680). Continuous preservation is verified independently by
`scripts/check-evidence-gated-focus-report.py`, which records the historical
report-label defect. The normal producer is repaired by the cleanup; the frozen
report is preserved. Nothing was promoted to default, no map
suppression, source switch-off, belief reset, margin change or resolution change was
added, and no reconstruction or strength superiority is claimed. Source-activity inference remains the smallest next
model experiment; a confirmation-geometry change is justified only by demonstrated
reacquisition failures, which these fixtures did not show. Historical results and
robotics deferral remain intact.

## Completed scanner baseline calibration

All 36 margin-development and 72 frozen held-out runs complete and pass public
replay with original causal loss; 520 tests pass with five existing Windows skips.
Four evaluation refinements pass below 1%. Historical modes and the renderer
layout remain intact; 12 videos decode and selected keyframes were reviewed.

Nonzero padding fails core coverage on the doubled profile, so retain zero margin.
The frozen 6 s half-life/.035-width/.2829738194-rate model lowers J/T in all 24
held-out pairs by 12–41% and improves doubled continuous core coverage to nearly
100%. It nevertheless sustains false focus in every blank case and never resumes
full scanning after cessation. Enhancement-index MSE worsens in four coarse
continuous/wind-error pairs. These findings support a reconstruction improvement
within the fixtures, not successful calibration of the complete loop or strength.

Keep the calibrated binding opt-in. The smallest proposed next stage is an
evidence-driven source-activity model that can stop injection when measurements
support absence without discarding continuing emissions. This is not implemented.
Representation sensitivity, changing local drift, posterior reliability and real
sensor calibration remain open. Robotics integration and simulator choice remain
deferred. The [review](docs/reviews/scanner-baseline-calibration.md) contains results,
frozen hashes, commands, limitations and local artifact locations.

## Completed focus-extent diagnosis

All 42 predeclared snapshots across the 18 saved runs pass exact public/evidence
replay. Independent conditioning checks confirm amplitude scaling. The excessive
region is mainly a long mean tail already present before uncertainty eligibility;
source injection and persistence contribute. Uncertainty adds background cells,
while closure adds none in these inspected plans. Wind conversion agrees exactly.
All 84 projection refinement checks pass. No duplicate assimilation/transport was
found; unused compatibility state and repeated derived maps are separate cleanup
opportunities. Runtime, renderer and allocation remain unchanged.

Verification: 506 passing tests, five existing Windows skips; 184 historical
artifacts preserved, 28 diagnostic figures, 91 hashed diagnostic artifacts.
The [review](docs/reviews/focus-extent-diagnosis.md) preserves negative source-off
results: removing injection can erase continuing signal and worsen forecast error.
That proposed fixed-path source/lifetime calibration is now completed separately
above. No allocator redesign was introduced in that round.

## Completed full-FoV loop and lean runtime

The user's clarified sequence requires a complete full-FoV scan before focus,
and a charged return to the initial pose followed by another complete scan when
signal support fails. This replaces age-block fallback in the intended new
demonstration and rules out prior-startup bypass. The
[inspected implementation plan](docs/reviews/full-fov-loop-plan.md) records the
state machine, shared-code extraction, independent minimal settings and a
zero-noise-first test. Full-rescan timing must not inherit a 12 s focus ceiling;
the doubled full scan alone takes 129.5875 s at current mechanics.

This loop correction precedes the previously proposed estimator-calibration
stage. The estimator, intended signal, profiles and renderer are retained.
Noise is a separate declared test variable. The independent runtime and shared
acquisition/compiler seams are implemented. All 18 declared zero/.01/.05 noise
executions pass public-call and causal-loss replay without retuning; 14 canonical
videos are decoded and inspected through recorded representative frames. Full
regression passes 489 tests with five platform skips. Historical behavior hashes
pass against `84cf70d`. Cessation exposes false re-focusing from the retained
source-anchored model, recorded in the [implementation review](docs/reviews/full-fov-loop-implementation.md).

## Completed threshold-bound test and handoff

Start `5bb58ee`, branch `codex/threshold-bound-scanning`. Implement the approved
[v5 protocol](docs/specs/threshold-bound-scanning-v5.md): signal evidence enables
focusing, upper bounds include plausible surrounding signal, and absent or
infeasible signal falls back to age-based scene scans. The estimator stays fixed;
measure calibration without fitting it. A mean-only ablation separates the
effect of uncertainty. All 288 short continuations and 60 full Gaussian sessions
are complete, with 348 coverage records, 152 acquisition-bound/reading diagnostics,
eight bound fixtures, short/full evaluation refinement and 16 decoded videos.
473 tests pass with five Windows symlink skips; eight representative public API
replays reproduce plans and losses. Historical artifacts and untracked output/
remain unchanged. Implementation checkpoints: `1149a13`, `67bca3f`, `f003270`,
`d967b87`; final revision and hashes are recorded in the local closeout artifact.

The [review](docs/reviews/threshold-bound-scanning-v5.md) records mixed allocation
and uncertainty benefits. Coarse short loss improves about 8.5–8.7% against v4,
but doubled loss worsens 2.5–12.5%. Upper bounds substantially underpredict
signal; blank gains belong to age fallback. Mixture conclusions retain the
unresolved 49-hypothesis limitation. No new policy is frozen and the claim stays
`not_demonstrated`. Next proposed work is a fixed-path forecast/source/lifetime
and representation calibration test, with exposure/propagation effects separated.
Changing local drift remains subsequent work; neither stage is implemented here.

## Completed scan-value diagnosis and next-stage handoff

Start 7bd9d54, branch codex/scan-value-diagnosis; implementation checkpoints
6b6218f through 28f5765. The [protocol](docs/specs/scan-value-diagnosis.md)
produced all eight frozen snapshots, 9,536 paired continuation records and 16 idle
references. Opt-in zero-inflow transport, detached READY snapshots, causal forced
paths and immutable-artifact auditing are implemented. Historical behavior and
untracked output/ are preserved. No new policy or motion model is frozen.

Final regression: 447 passed, five Windows symlink skips. Numerical arithmetic,
scoring/evaluation refinement, exposure ablations, paired evidence checks and
video inspection are complete. Eight Gaussian value-misranking contrasts are
supported; candidate/continuation limitations have narrower support. Boundary
correction contributes but is not a universal improvement. Mixture causality,
wind quadrature, strict blank-signal contrasts and propagation partition effects
remain explicit limitations. The [review](docs/reviews/scan-value-diagnosis.md)
contains results, reproducible commands, hashes and resolved/unresolved items.

The smallest proposed next stage is fixed-path source/lifetime and forecast
calibration: stopped source-free pulse versus continuing replenishment, known
wind, separate fitting and forecast evidence. This stage has not started.
Changing local drift remains required subsequent formulation work if structured
motion uncertainty survives those controls. Calibration does not solve it by
assumption. No broad new comparison suite is authorized by this handoff.

The actively maintained [CHANGELOG.md](CHANGELOG.md) summarizes broad historical
checkpoints and subsequent substantive changes. This file remains the source for
active work, completion gates and detailed evidence status; the changelog links
to it and to the relevant specification/review without replacing those records.

## Eventual purpose: user scope clarification

The eventual goal is plume reconstruction under resource constraints, in
support of numerical plume-strength estimation with a remote point sensor.
Adaptive coverage is the method for obtaining useful measurements of an
intermittent, moving plume within mechanical travel and acquisition limits.
This supersedes the earlier description that excluded reconstruction from the
project's core purpose; it does not claim reconstruction is already validated.

Each point is measured at its own acquisition time. A wide, slow scan can mix
different plume states into a distorted image. The hypothesis is that predicting
the useful region and scanning it sooner can improve reconstruction and strength
estimation by reducing temporal mismatch. Shorter scans may also miss signal:
the objective is useful reconstruction under constraints, not routine trimming
or maximizing static coverage alone. The tradeoff is coupled and need not be
linear; its form and benefit must be measured.

The practical motivation is to reduce the equipment and human interpretation
cost associated with conventional OGI. Superior point-sensor accuracy and lower
overall cost remain proposed advantages requiring comparative evidence.

The user further identifies scientific-instrument economics as the reason for
point sensing: the intended measurement capability is too costly or impractical
to replicate into a full spatial detector array. One sensor must therefore
trade acquisition time for spatial information. This is an instrument premise
to document and eventually validate, not an assertion that all expensive
sensors are intrinsically limited to one point. The direct optimization task
is measurement selection for reconstruction of a changing field; mechanical
routes are part of that acquisition cost. POMDP/Bayesian theory is available
background, and robot path-planning work is adjacent rather than a substitute
for an instrument-specific measurement and reconstruction objective. Include
adaptive scanning instrumentation in related work, and do not infer novelty
from the unusual application or claim that existing theory solves its details.

"Strength" needs an explicit observable, units, and calibration; no emission-rate
or concentration-accuracy claim follows from the current synthetic signal.
Future reconstruction evaluation must declare its reference time/window and
measure temporal distortion, missed signal, and estimation error alongside
coverage and charged time. The v1 specification freeze below now defines the
immediate reconstruction objective; calibrated physical strength remains open.

Keep the reference instrument's footprint, sampling spacing, cadence,
integration settings, mechanical speeds, and scan-pattern rules fixed while
comparing policies. Configuration remains modular between declared experiments.
The region may grow, move, remain stable, or shrink according to evidence.
Renderer presentation and playback are independent of physical behavior.

The completed recovery and historical negative results below remain evidence
for their stated configurations. The separately approved implementation stage
below has now completed code conformance and development evaluation. Existing
demonstrator gates are not sufficient to establish the broader reconstruction
and strength-estimation objective.

## Reconstruction-value acquisition (2026-09-15)

Status: `complete` as the bounded implementation/evaluation goal; start `51d1f63`,
branch `codex/reconstruction-value-policy`. Implemented the
[approved v4 acquisition candidate](docs/specs/reconstruction-value-policy-v4.md)
with unchanged Gaussian and mixture estimators, common-horizon reconstruction
risk, bounded candidate paths and one required oldest-block service visit.
Checkpoints `9ae89a8` and `65d6077` implement the policy and freeze the study;
`c29f811` and producing revision `ad801c6` complete lifecycle/numerical inspection.
The expanded regression suite passes 430 tests (five existing Windows skips).
All ten Gaussian/mixture reference decisions pass joint numerical checks; the
common 0.1 s / 32-history setting is frozen before execution. The earlier wind
quadrature limitation remains unresolved. All 36 new executions / 96 comparison
records are complete; 60 historical controls pass reuse checks. Five evaluation
mesh checks pass. Twenty videos fully decode, with selected frames from all ten
old/new cases reviewed. All 32 required service blocks receive actual readings;
149 of 193 value-selected sweeps execute below 90% predicted mass.

Gaussian main comparisons are mixed: 3/12 pairs improve beyond 1%, 9 worsen.
Fixed-mixture comparisons remain `numerically_unresolved`: 2/12 main pairs improve,
10 worsen, and doubled biased-wind mean map loss rises about 100%. Its full-strength
source-free diagnostic improves 27%; weak-signal map loss worsens. These descriptive
results do not establish general superiority. Whole-field map loss remains primary;
strength-index error is separate. Compute is offline and not charged to the synthetic
physical clock. The [v4 implementation review](docs/reviews/reconstruction-value-policy-v4.md)
records commands, artifact hashes, producing/closeout revisions and limitations.

Boundary treatment, changing local drift and estimator calibration remain subsequent
model work. No next stage starts automatically; recorded-data expansion stays paused.
Previous `not_demonstrated` findings remain intact.

## Persistent wind uncertainty (2026-09-14–15)

Status: `complete` as the bounded implementation/evaluation goal. Starting revision
`66ea099`, branch `codex/persistent-wind-prediction`. The
[persistent wind candidate](docs/specs/persistent-wind-motion-v3.md) retains the
existing v2 allocator, noise, sources and instruments. Its persistent mixture,
exact allocation mass, conservative bands and causal public-API replay are
implemented. The [completed review](docs/reviews/persistent-wind-prediction.md)
records all 72 main plus 12 supplementary comparisons, fixed-evidence diagnostics,
numerical checks and ten reviewed videos from unchanged-layout inspection.

Scientific classification is `numerically_unresolved`: both 9/25 and 25/49
hypothesis refinements failed, so the 49-hypothesis results are provisional.
Spatial/time refinement accepts 120 x 160 / 0.05 s. Main biased-wind adaptive mean
loss increases by 1.00% coarse and 2.47% doubled. Full-strength source-free map
loss improves by 37.4%, but enhancement-index MSE worsens. Weak/blank gains remain
in surveillance and also do not improve index MSE. Learned offsets do not validate
wind recovery. No superiority claim or new policy freeze follows.

Full regression at `bfdaecc`: 403 passed, five existing Windows symlink capability
skips. All 42 configuration pairs pass; all 36 fresh Gaussian baselines reproduce
historical execution hashes and loss exactly. Ten videos / 4,550 frames fully
decode, with selected frames visually reviewed. Compilation, independent v1
specification arithmetic, import boundaries and Git diff checks pass. The review
links reproducible commands, per-artifact producing revisions, hashes and the
final closeout record. Original evidence and untracked `output/` are preserved.

**Stored scope decision:** persistent wind error was this round. Changing local
drift is also needed and remains the next required motion-model stage. It was
deferred, not rejected or represented by learning a constant offset. An isolated
post-hoc probe identifies incoming-boundary clipping that can pin signal and
sustain artificial inflow; correction and validation are also required subsequent
model work. Neither effect has been isolated as the sole cause of study outcomes.
Persistent tails, model/posterior calibration, weak-signal handling, hard-90%-gate
fallback and incomplete refresh remain unresolved. Hardware/real-data expansion
and within-sweep replanning remain outside this completed goal. Recorded-data
expansion stays paused; the broader reconstruction/strength research goal is active.

## Mechanism diagnosis and experimental v2 (2026-09-13)

Status: `complete` as a bounded diagnosis/implementation/development checkpoint;
candidate remains experimental and claim `not_demonstrated`. Starting revision `a341e68`, branch
`codex/mechanism-grounded-revision`. The user requested diagnosis, a formulation
mapped to actual mechanisms, and repository corrections. Fixed-path diagnosis
and a separately named v2 candidate are implemented. The 36 main comparisons,
seven diagnostics, numerical checks and unchanged-layout inspection are complete.
See [the diagnosis/review](docs/reviews/mechanism-revision.md) and
[candidate specification](docs/specs/mechanism-grounded-policy-v2.md).
Preserve frozen v1 and historical negative results. Main-study noise and dynamics
remain unchanged; tentative physical resolution is explicitly unconfirmed hardware.
Data revision `af40fdd`; inspection tools/media revision `2125a1e`. Full regression:
382 passed, five Windows capability skips. Six videos / 2,724 frames fully decoded,
with representative physical progression, changing regions, blank signal and final
idle state visually reviewed. Configuration/artifact hashes are in the review.

Coarse tracking now executes but loses to the fixed corridor in every tested case.
Doubled results are mixed and mostly use fallback; full startup reaches only 52.5%
of the scene. The observed region grows and shrinks rather than recursively
retaining 90% of itself. Remaining work needs a separate plan: validate forecast
dynamics/retrieval assumptions, decide how weak signal and attainable coverage
should enter the objective, and address the hard-90%-failure-to-surveillance
transition. Do not claim these scientific/formulation issues are resolved or
silently change the frozen v1 rule to improve a demonstration.

## Frozen v1 implementation (2026-09-13)

Status: `complete` (bounded engineering goal; claim remains `not_demonstrated`).
Starting revision: `4a598acc38fb2ad8be0cfaf370c5955fa8ba275b`.
Branch: `codex/policy-v1-conformance`. Preserve local untracked `output/` artifacts.
Delivered explicit immutable profiles/fixed timing, coverage-preserving policy,
causal reconstruction evaluation, simulator/fake-scanner integration, existing-layout
inspection and verification. The frozen specifications were not changed. Final
implementation revision: `8de3a3e81aae1975ff366fe216478c3f1e3a01a2`.

All 36 synthetic comparisons and six quadrature fixtures completed from
`043549d`; the common 120 x 160 / 0.05 s mesh passed the refinement criterion
(maximum loss change 0.3772%). Coarse adaptive tracking uses fallback for all
post-startup decisions and loses to the fixed corridor. Narrow startup visits
only 1.875% of the scene, so it supplies no tracking-advantage comparison.
Actual expenditure is reported alongside common ceilings. Signal persistence
and posterior miscalibration remain unresolved; no dynamics tuning was added.

Full suite: **366 passed, five Windows symlink-capability skips**; independent
arithmetic, schema, dependency boundaries, compilation, link and diff checks
passed. Twelve videos (5,448 frames) fully decoded; representative coarse,
narrow, delayed-batch and overrun frames reviewed. Labelled videos from `698c793`
preserve the numerical traces; final implementation reproduces their operational
hashes. See [the implementation review](docs/reviews/policy-v1-conformance.md)
for focused revisions, commands, numerical evidence and artifact hashes.

This closes the approved implementation, not the broader scientific goal. Next
research work requires a separately scoped plan addressing the resource/model
limitations; no new allocation or dynamics version is activated here. Historical
evidence is preserved and recorded-data expansion remains paused.

## Formulation and scan-policy v1 freeze (2026-09-13 UTC)

Status: `complete` (specification and implementation handoff only).
Starting branch: `codex/sensing-core-recovery`.
Starting revision: `66f07700a754e5f9f25ae55d78f6423958b56d61`.
Specification revision: `ab80dc585c0972a1dc6c3a022bdedfd0e5e8b315`.

Freeze the causal reconstruction objective and Gaussian measurement model,
then the coverage-preserving 90% scan rule, with a complete instrument profile
required for each experiment. Do not select a numerical reference instrument.
Produce worked checks, an implementation-gap review, and aligned repository
guidance. Preserve the existing renderer and sensing code in this stage.

Delivered: [frozen formulation/policy](docs/specs/reconstruction-policy-v1.md),
required [instrument profile/schema](docs/specs/instrument-profile-v1.md),
[worked examples](docs/specs/freeze-v1-worked-examples.md), and
[audited implementation handoff](docs/reviews/policy-v1-freeze.md).
Nine independent arithmetic groups passed; dynamic sequential/batch inference
agreed within `2.22e-16`. The full existing suite passed 335 tests with five
Windows symlink-capability skips; schema validation, import-boundary regression,
compilation, local documentation links, and Git diff checks passed. No runtime
source, existing tests/examples, or renderer changed. No new performance run or
video inspection occurred.

Three current-code probes reproduced budget subset allocation (46% retained
from a selected 92% example), implicit speed reduction (0.5 to 0.25 FOV/s), and
a past-time predictive view containing a later-delivered reading. These remain
explicit implementation gaps; they do not invalidate the completed specification
freeze or establish that an existing benchmark used future evidence. Artifact
hashes, producing revisions, and reproduction commands are in the handoff review.

The historical held-out result and recovery development result remain
`not_demonstrated`. Recorded-data expansion remains paused. The next focused
implementation stage becomes active only when its execution begins.

## Sensing-core recovery (2026-09-13)

Status: `complete` (engineering). Implementation revision:
`e3e1b5e7f715a698398d7aedf340c058bf0d7ae3`.
Verification: 335 tests passed, 5 platform skips; compilation, diff checks,
installed CLI, JSON API example, complete video decodes, and artifact hashes
verified. Final artifacts were generated with clean working trees.

The 18-run development comparison remains negative: adaptive tracking-only
mass recall averaged 24.63% with known wind, 77.85% with the declared wind
bias/error, and 8.57% in source-free mode, versus 100%, 100%, and 91.23% for
the declared fixed corridor. Actual expenditure differs and is reported.
The source/advection model's excessive signal persistence and uncalibrated
uncertainty remain visible failures. This completes the approved engineering
recovery, not the broader scientific goal. No new stage is automatically active.

The final operational hash is
`8a2692a5b741801911ca5e1d04f5614e8943ca5cfca8ff85c7c3442201d0d449`.
Main and development evidence are under `outputs/sensing-recovery-v1` and
`outputs/sensing-recovery-development-v1`; their manifests/hashes and commands
are recorded in `docs/reviews/sensing-core-recovery.md`.

The user approved one bounded engineering goal covering the repository audit,
correlated signal inference, physically timed scans, a callable controller API,
simulator integration, faithful videos, and development comparisons. The
starting revision is `cb890205dce3cd2a0e848e67971dee4d96a2ae51`; the work branch
is `codex/sensing-core-recovery`. See `docs/sensing-core.md` for the contract
and `docs/reviews/sensing-core-recovery.md` for audit and verification evidence.
Historical held-out results remain frozen and must not be tuned against.
R0D-A stays planned; it is not the next active work item during this recovery.
Completion requires numerical, causal, mechanical, renderer, API, and repository
checks and reproducible development results, including negative results.
It does not complete the broader scientific goal or establish hardware calibration.

This file is the durable source of truth for the deliverable goal, completion
gates, development stages, hardcoded assumptions, decisions, and evidence.
`AGENTS.md` defines claim boundaries and development rules. `README.md`
describes the software that is currently available.

[`RECORDED_DATA_WORK_PLAN.md`](RECORDED_DATA_WORK_PLAN.md) is the subordinate
execution tracker for external-video accession, OGI-derived sensor emulation,
and spatial evaluation evidence. R0 now has a concrete public GasVid provider
folder and a local, ignored accession: 31 MP4 files passed the strict full
decode audit, but the audit remains rights-blocked because no explicit license,
derivative-use terms, or redistribution terms were established. Its R1
deterministic preparation/freeze engineering is complete on generated fixtures
through `48c7d8d`. R0C now supplies one rights-cleared development-only propane
diagnostic, but no untouched held-out partner or methane evidence. Parent Stage
4 and G3 therefore remain blocked.

Recorded workflow R0B remains externally blocked on owner permission and
authoritative assignment/interval provenance. Its bounded R0B-Q engineering
substage is complete at `54893a9`; no R0B engineering substage is active. Its
metadata-only qualification manifest/report, scoped
permission decisions, per-asset evidence states, stable `qualification_id`,
normalized data-root-relative `asset_root`, exact template/check CLI semantics,
and preparation-spec-v3 path/hash binding are implemented and verified. The
v3 `local-raw-pilot` is
bounded raw-window input only: no calibration source, structure suppression,
or spatial label is consumed. The canonical GasVid check is valid but
ineligible, so no interval decode, derivative, or pilot was run. No GasVid
pilot result, spatial evidence, Stage 4/G3 change, or claim-status change is
permitted until the required rights and provenance are verified.

Recorded workflow R0C is complete with result
`engineering_not_demonstrated`. One explicitly CC-BY-4.0 Utah State ground
controlled-propane video passed byte/decode audit, qualification, preparation,
freeze, headless execution, and renderer-equivalence checks without labels,
cleaning, wind, pseudo truth, policy tuning, or a dataset-specific runner
branch. The plumbing is causal and reproducible, but the raw requested mask
covered 89.54–100% of the image (mean 91.05%) across eight scans, and its
bounding ROI remained full-height and nearly full-width. Column stride made
the modeled swept fraction smaller, but no spatial truth exists to show plume
coverage. This is neither methane validation nor policy-performance evidence.

The next proposed bounded substage is R0D-A, a development-only measurement
separability diagnostic and calibration-source decision. It must leave the
scan policy and frozen R0C result unchanged. R0D-A is `planned`, not active;
see the subordinate tracker for its explicit end conditions.

The GasVid continuation is separate. The current inquiry draft must first be
aligned with the qualification checker's per-file group/split, interval, and
permission fields and given a hashed response/follow-up/closure protocol. Only
then should the user's identity and send authorization be requested. A
response may qualify the preregistered operation or close GasVid `blocked`; it
must not weaken the gate after outcomes are observed.

## Existing demonstrator contract

The contract and gates below describe the existing active-scan demonstrator.
They are retained as recorded milestones toward the broader purpose above;
their completion would not establish calibrated plume-strength reconstruction.

> Deliver a reproducible, CPU-friendly research demonstrator that converts
> causal partial mechanical scans and uncertain wind/advection measurements
> into an explicit plume belief, then selects the next mechanically feasible
> scan. On frozen recorded or independently generated methane data, it must
> preserve plume coverage with a smaller and faster scan than a full-scene
> sweep and demonstrate statistically credible improvement over declared
> non-oracle baselines at equal mechanical cost.

The primary proof target is source-anchored methane tracking from a known source
or source corridor. Source-free tracking is an experimental secondary mode. It
must not be presented as unknown-source discovery.

Research-demonstrator completion does not imply:

- calibrated gas concentration estimation;
- a production leak detector or safety guarantee;
- physical scanner or pan-tilt hardware control;
- first-principles source discovery;
- multi-source separation;
- spatially varying turbulence or full CFD;
- embedded or safety-critical deployment.

A valid negative benchmark can complete a development stage. It cannot complete
the overall system goal.

## Status and evidence rules

Stage statuses are:

- `planned`: accepted work that has not started;
- `active`: the one stage currently being executed;
- `complete`: engineering work and its stated evidence are complete;
- `not_demonstrated`: a valid evaluation completed without supporting its
  performance claim;
- `blocked`: completion requires missing external evidence or authority.

Claim statuses are independent:

- `supported`: every preregistered integrity and performance gate passed;
- `not_demonstrated`: the evaluation was valid but the performance gate failed;
- `invalid`: the evaluation cannot support either conclusion.

Only one stage may be `active`. Historical results are append-only. A later run
may supersede an earlier result, but it must not erase it.

Generated videos and full benchmark reports remain untracked. Record their
commands, manifest hashes, report hashes, model profile, and producing commit
here. Regenerate the small canonical JSON example only when its demonstrated
behavior or report contract changes.

## Current evidence baseline

Baseline commit: `ad12498`

Baseline branch: `codex/source-aware-plume-dynamics`

Baseline tests: 79 passing

Primary benchmark outcome: `not_demonstrated`

Recorded R0C implementation commit (historical baseline): `a8a65b0`

Recorded R0C tests at that commit: 307 passing, 5 Windows capability skips; `compileall` and
`git diff --check` passed

Canonical GasVid qualification:
`560f3bad93d3c5b72340ecf30ee174400ef1805b0ea9698c727b143b994c18ed`

Qualification check report:
`f330e64a26cfec5807bcc034190d5cdc8bb4c7a992f25a80ed11223cea519487`;
exit `2`, `eligible=false`, 36 blockers

R0C licensed asset SHA-256:
`da3f7702376eeaa6c2d6b37a835b874965d8f49b8b60f56227af8f42bd6b75fd`

R0C frozen suite canonical hash:
`c63650e9d89ad6467f8c632bedb84b1e6540315c42c07ceb56a86055fbd74479`

R0C headless/rendered/video SHA-256:
`08f973d09699d72af2526dcebb8ab4d9e08cc1c2ac75feaaa53c988291b3740e`,
`a554f9c2ce013d14e0f0c9cbd3558080d550bd040ac61a8f89702edec92beccb`,
and `8e1202a3e85a2fdcb21d6919899ae74244d8a194fbaa5405e990747ba9bdf27c`

R0C outcome: `engineering_not_demonstrated`

Stage 3 development-smoke report:
`ffe0d17c56df80fb2338ec387d37d31e4a6169d5c869f172602301391dabe36a`

Canonical JSON:
`0763ff6be26fa3d6caf46f0e92fe1626ae2ff1eb6b3246101d9c456173c2d6a6`

Established engineering evidence:

- scans execute sequentially and predictions target planned next-scan
  completion;
- completed scan evidence is the default planner input;
- full-frame evidence is an explicitly labeled baseline;
- finite sources reject unavailable future samples;
- scan allocation and benchmark comparisons use exact mechanical budgets;
- report schema v3 preserves the v2 requested-mask, mechanical-sweep,
  sampled-pixel, attention, and offline-truth boundaries while adding a
  per-scan causal schedule, sampled-value variation evidence, and explicit
  belief-integrity checks;
- source-aware known-source propagation emits along intermediate wind positions
  rather than translating the whole belief away from the source;
- source-free mode does not receive a hidden source or emission trail;
- the experiment runner is headless and independent of video writing;
- recorded manifest schema v3 strictly validates accession-audit lineage,
  capture groups, hashed media, clocks, sensor input, calibration, wind, source
  assumptions, counterfactual scanner settings, optional/lossless truth,
  method protocol, and model profile;
- schema v3 permits truth-free causal execution without deriving labels from
  sensor frames; schemas v1/v2 remain reproducible compatibility paths;
- explicit schema v4 binds a raw truth-free no-wind interval to its audited
  parent, preregistered selection metadata, exact source-frame/time range, and
  decoded-window hash while leaving schema v1-v3 payloads unchanged;
- the structure-suppressed sensor adapter uses fixed median/MAD calibration
  from a separate audited leak-off asset and does not fit evaluation frames;
- schema-v3 calibration is loaded from canonical bytes and bound to the exact
  source capture, preparation spec/profile, producing commit, fit algorithm,
  shape, sample count, and artifact hash;
- deterministic no-overwrite preparation and freeze commands produce
  truth-free manifests, preparation reports, suite manifests, and companion
  lineage records;
- audit schema v2 binds each compressed asset to both its OpenCV version/build
  signature and exact ordered decoded grayscale pixel hash; preparation and
  runtime reject decoded-content drift;
- audit validity enforces the runtime's minimum three-frame input contract;
- every non-excluded audited byte identity must be consumed as sequence media
  or calibration input, while duplicate bytes require an objective exclusion;
- freeze-record validation rechecks the audit, source/calibration bytes,
  preparation inputs/report, prepared manifests, and exact suite membership;
  suite validation separately decodes runtime media and checks sidecars;
- headless mechanical scan mode requests only the selected coordinates and
  timestamps from the sensor source;
- paired development/held-out validation rejects incompatible source-media,
  calibration-media, and acquisition/calibration capture-group roles;
- recorded frame and wind lookups use latest-not-after causal sampling;
- recorded truth is isolated behind an evaluation-only interface, and changing
  truth annotations cannot change the planned scan;
- individual hash-validated runs explicitly remain ineligible for the final
  claim until frozen-suite and independence review.

Frozen fair-benchmark evidence:

- development manifest:
  `914a31f920a379eae8e4ac11bdfa12cab57190aa71547826051c563243cbf758`;
- held-out manifest:
  `da4e87e3665973ca270905889b14919081b359aa0e47bb077c65ec924b896ecd`;
- source-aware held-out report:
  `a7766a91866156d01376685ba9d361c8a245324a2f0d4410fe18e7d95aa58f4c`;
- all 540 matched runs were present and timeline integrity passed;
- matched area and duration differences were exactly zero;
- adaptive minus fixed source corridor was `-0.5144`, paired-bootstrap 95%
  interval `[-0.5529, -0.4758]`.

Known failures:

- adaptive selection loses substantially to the fixed source corridor;
- exact wind can make the forecast too narrow and reduce coverage;
- uncertainty growth is uncalibrated;
- the measurement likelihood is a normalized intensity heuristic;
- persistent emission is uncalibrated and does not model intermittent
  visibility;
- current final evidence is synthetic and not independent of the project.

GasVid qualification baseline observed through 2026-07-31:

- the original GasVid/GasNet paper
  (https://svl.stanford.edu/gibson2/assets/Wangetal.2019.pdf), the VideoGasNet
  publisher record
  (https://www.sciencedirect.com/science/article/pii/S0360544221017643), the
  Stanford record, public provider folder, and linked download route expose no
  explicit dataset licence, local-processing permission, derivative grant, or
  redistribution grant found by the audit; `rights_status` stays `unverified`;
- the papers describe 31 24-minute grayscale recordings. VideoGasNet modeled
  only 25 after excluding the five 18.6 m recordings and the one 8.8 m
  recording. The local accession contains and validly decodes all 31, so the
  full accession must remain distinct from that model-specific subset;
- provider dates, shooting times, distance sweeps, and filename counts support
  five candidate capture groups and an 18/13 filename-to-pad hypothesis, but
  neither is authoritative filename-to-pad or split provenance;
- the original paper calls the initial class-0 segment non-leak/background but
  also lists a nonzero `5.3 g CH4/h` class-0 value. Provider `Leak 0`/`Leak 1`
  markers are therefore not authoritative semantic or duration labels;
- `MOV_2568` has a 157-second marker gap and `MOV_2582` a 360-second gap,
  rather than the nominal 180 seconds; neither is silently repaired or
  excluded; and
- the FGSTP primary paper (https://arxiv.org/abs/2505.00295) reports manual
  frame labels for a selected 14-video experiment (9 training, 5 testing), but
  an accessible mask/code accession, exact file/frame mapping, and reuse rights
  have not been established. Spatial labels are unnecessary for the first
  truth-free raw causal pilot.

## System completion gates

| Gate | Requirement | Status | Evidence | Remaining issue |
| --- | --- | --- | --- | --- |
| G1 Causal integrity | Sequential scans, completed evidence only, no future samples, no planning truth | `complete` | `2e9c210`, fair-benchmark integrity fields | Preserve in every later stage |
| G2 Modular configuration | Validated typed profiles own all behavior-changing model and policy values and reports serialize the resolved profile | `complete` | `c61bbfe`; 85 tests passing | Real-data calibration remains G3/G4 work |
| G3 Independent evidence | Frozen development and held-out recorded data or an independent generator | `blocked` | Stage 4A/R1 engineering through `48c7d8d`; R0B window-contract engineering `366730d`; R0B-Q blocked qualification; R0C rights-cleared propane development diagnostic, frozen suite canonical hash `c63650e...74479` | R0C has no held-out partner, methane identity, frame-synchronous wind, or spatial truth; GasVid remains unqualified; no frozen development/held-out evidence pair exists |
| G4 Uncertainty behavior | Wind, scan age, sensor error, and intermittency change uncertainty predictably; excessive uncertainty reacquires or terminates | `planned` | Reacquisition regression tests exist | Calibration and ensemble motion are missing |
| G5 Mechanical utility | At least one preregistered focused point reaches 0.90 actual-sweep mass recall while strictly smaller and faster than full scene | `not_demonstrated` | Current held-out benchmark | Adaptive medium/large-budget under-coverage |
| G6 Comparative evidence | At least 0.02 mean gain, positive 95% paired lower bound versus every competitive baseline, no negative budget mean | `not_demonstrated` | Held-out report hash above | Current adaptive-minus-corridor result is strongly negative |
| G7 Reproducibility | Clean checkout runs tests, development benchmark, held-out benchmark, JSON, and optional video through documented commands | `active` | Contract `523df93`; lineage/result `a8a65b0`; 307 tests passing and 5 Windows capability skips; R0C audit → qualify → prepare → validate → freeze → headless/rendered run completed with matching operational traces | Recorded Stage 4B and final clean-checkout acceptance remain unfinished; generated R0C reports/video are intentionally untracked and identified by hashes |

Overall completion requires every gate to be `complete` and the final claim
status to be `supported`.

## Stage ledger

### Stage 1 — Fair causal benchmark

Status: `complete`

Performance result: `not_demonstrated`

Evidence commit: `2e9c210`

Objective:

- create deterministic development and frozen held-out manifests;
- enforce causal policy inputs and exact matched scan cost;
- compare adaptive behavior with competitive non-oracle baselines;
- report negative evidence without weakening the claim gate.

End condition: complete. The benchmark is valid, but the adaptive policy loses
to the fixed source corridor.

### Stage 2 — Source-aware continuity and modular video path

Status: `complete`

Performance result: `not_demonstrated`

Evidence commits: `5e6d7a1`, `104f55c`, `ad12498`

Objective:

- separate plume dynamics from the video façade;
- preserve source-free behavior;
- correct detached known-source propagation;
- rerun the unchanged fair benchmark.

End condition: complete. Source continuity is corrected and the small-budget
adaptive recall improved from 1.04% to 8.54%, but comparative performance
remains unsupported.

### Stage 3 — Model Contract and Hardcode Removal

Status: `complete`

Evidence commit: `c61bbfe`

Verification: 85 tests passing; development smoke exact-cost and timeline
integrity passed with expected claim `not_demonstrated`

Objective:

- add one validated public `ModelProfile`;
- compose dynamics, measurement, uncertainty, and scan-policy settings;
- resolve named compatibility profiles for synthetic and file-backed inputs;
- pass the profile through the causal runner and benchmark;
- serialize all declared and physically resolved values;
- keep rendering-relative-density thresholds outside scan planning.

Implementation boundaries:

- preserve the Bayesian loop, mechanical scan semantics, and benchmark
  policies;
- retain `plume_dynamics_config` as a labeled compatibility override;
- do not tune behavior against held-out episodes;
- do not present compatibility values as real-data calibration.

Required tests:

- profile validation and serialization;
- explicit frame-rate resolution of uncertainty speeds;
- default behavior compatibility;
- profile/scenario mismatch rejection;
- full profile presence in video and benchmark reports;
- rendering threshold changes cannot change the next mechanical scan;
- complete existing regression suite.

Completed end conditions:

- no hidden behavior-changing literal remains in the causal runner or adaptive
  policy;
- the resolved report identifies units, provenance, and calibration status;
- `synthetic_demo_v1` reproduces the prior smoke-test and benchmark semantics;
- the canonical JSON demonstrates the profile contract;
- the full test suite passes.

Result: all Stage 3 engineering conditions passed. The compatibility profile
preserves the prior canonical smoke metrics: mean probability-mass recall
`1.0`, mean swept area fraction `0.1607`, mean scan duration `0.2571` seconds,
and mean JS divergence `0.112543`. These remain smoke-test values, not
performance evidence.

### Stage 4 — Real-Data Evidence Contract

Status: `blocked`

Stage 4A engineering status: `complete`

Recorded-data subordinate workflow: R0 and R0B `blocked`; R1 and R0B-Q
engineering `complete`; R0C `complete` with result
`engineering_not_demonstrated`; no bounded engineering substage active; see
[`RECORDED_DATA_WORK_PLAN.md`](RECORDED_DATA_WORK_PLAN.md)

Stage 4A starting branch: `codex/stage4-recorded-evidence-contract`

Stage 4A starting commit: `fbfbb78`

Stage 4A evidence commits: `2260d1b`, `1ac62f1`, `8f90aa8`, `1466faa`,
`f0748ac`, `484b063`, `ac38940`, `afacbdc`, `7eb26be`, `a074113`,
`b0eefe8`, `c8099af`, `4eb339f`, `067e06a`, `48c7d8d`, `366730d`,
`54893a9`, `523df93`, `a8a65b0`

Latest engineering verification: 307 tests passing and 5 Windows capability
skips; `compileall` and `git diff --check` passed;
development benchmark smoke result `not_demonstrated`; smoke report hash
unchanged at
`ffe0d17c56df80fb2338ec387d37d31e4a6169d5c869f172602301391dabe36a`

Completed Stage 4A implementation:

- strict typed sequence and suite manifests with canonical manifest hashes and
  asset SHA-256 verification;
- suite freeze timestamp, freeze provenance, and pre-change model commit;
- explicit acquisition, split, clock, normalization, calibration, wind,
  source, scanner, truth-annotation, and model-profile provenance serialized
  through the complete resolved manifest;
- constant-rate and timestamp-sidecar clocks with latest-not-after sampling;
- fixed normalization plus sensor and calibration provenance;
- causal missing, constant, and timestamped wind providers with explicit age,
  staleness, scan-derived fallback, and report counts;
- separate dense binary-support and weighted-truth evaluation sources;
- manifest-authoritative scanner, source assumption, and model profile settings
  with runtime override rejection;
- headless recorded execution plus an optional rendering adapter and dedicated
  `plume-focus-recorded` validation/run CLI;
- report sections for data provenance, truth isolation, and wind evidence;
- explicit labels separating engineering fixtures, unmanifested video, and
  provenance-checked recorded inputs;
- focused modules for manifest parsing, confined asset I/O, causal providers,
  public loading, workflow, and CLI behavior;
- acceptance tests covering timestamp clocks, available/stale/missing wind,
  scan-derived fallback, fixed normalization, weighted truth, known-corridor
  and source-free inputs, truth isolation, suite freezing, and optional video.
- schema-v3 truth-free causal input with unavailable spatial metrics rather
  than sensor-derived pseudo truth;
- canonical, platform-neutral fixed-calibration artifacts and strict
  calibration result records;
- deterministic no-overwrite `audit`, `prepare`, and `freeze` commands with an
  end-to-end generated-video fixture;
- objective disposition for every audited byte identity and typed
  non-performance exclusion reasons;
- companion suite-freeze records and cross-suite media/calibration/capture-role
  isolation;
- decoded grayscale pixel-stream hashes plus OpenCV build signatures for
  compressed-video reproducibility;
- required use of every non-excluded audited identity and objective handling
  of duplicate bytes; and
- strict companion freeze-record validation across preparation/freeze lineage,
  with runtime sequence assets covered by separate suite validation.

Recorded OGI engineering checkpoint:

- a dataset-neutral accession manifest and full hash/decode audit command
  produce an artifact explicitly labeled `not_sequence_evidence`;
- default sequence schema v3 links every sequence to the strict audit, exact
  source hash, capture group, sensor-input grade, and rights state; explicit
  v4 adds raw temporal-window lineage, while v2 remains a compatibility path;
- raw OGI and fixed-calibration structure-suppressed input remain distinct;
- structure suppression uses a separate audited leak-off clip, fixed per-pixel
  median/MAD calibration, typed polarity, and an immutable profile;
- temporal smoothing is rejected in profile v1 because the removed
  spatial-order-dependent batch filter was not a valid scanner response model;
- silver pseudo truth is lossless, protocol-hashed, evaluator-only, and always
  diagnostic; gold labeling remains subject to external review;
- recorded video currently supports only counterfactual virtual scanner
  evidence, never a measured-hardware label;
- sample-access regression tests ensure headless planner execution cannot
  silently use full evaluation frames;
- paired-suite validation rejects cross-role reuse of source media,
  calibration media, and acquisition/calibration capture groups.
- schema-v5 manifests separate generic plume planning from provider-backed gas
  identity; qualification-v2 binds purpose, operations, project assignments,
  provider facts, and content-hashed evidence; preparation-v4 binds whole or
  windowed raw assets; report-v3 records causal scan/sample/belief traces.

Unmet Stage 4B end condition: at least one independently recorded development
sequence and one untouched held-out sequence must run through the causal
engine; all planner inputs must have real acquisition provenance; suite hashes
must be frozen before later policy work. R0C supplies one rights-cleared,
prepared, frozen, and executed development-only propane diagnostic, but it has
no held-out partner and cannot substitute for methane evidence. Local GasVid
media is hash/decode-audited but remains unqualified. R0B awaits the
owner-inquiry decision path; Stage 5 is not activated.
Hash validation, public downloadability, paper-wide labels, and inferred
filename groupings must not be treated as proof of rights, spatial truth,
split provenance, or experimental independence.

The licensed R0C run removed the immediate reuse-rights obstacle for one
recorded-video engineering smoke. Its causal plumbing passed, while its raw
adaptive requested mask stayed near full-frame and the result was classified
`engineering_not_demonstrated`. It is controlled propane and has neither a
held-out partner nor spatial truth or frame-synchronous wind, so Stage 4B
remains unmet.

### Stage 5 — Uncertainty-Aware Dynamics and Measurement

Status: `planned`

Implementation:

- add an `AdvectionEstimate` containing mean motion, covariance or weighted
  hypotheses, timestamp, and provenance;
- retain `MotionEstimate` through a compatibility adapter;
- introduce modular dynamics and measurement protocols;
- propagate multiple motion hypotheses;
- model persistent emission as configurable and potentially intermittent;
- calibrate likelihood, negative evidence, dispersion, decay, and advection
  scale using development evidence only.

End condition: uncertainty perturbation, time-step consistency, source-free
leakage, and negative-evidence tests pass; every calibrated value identifies
its development evidence.

### Stage 6 — Robust Hybrid Scan Policy

Status: `planned`

Implementation:

- combine the strong fixed source corridor with adaptive posterior tracking;
- score predicted plume mass and information value per physical scan second;
- retain exact budgets, contiguous travel, minimum coverage, and explicit
  reacquisition;
- tune only on development data;
- keep constant-velocity, static-corridor, last-observed, zero-motion, and
  oracle policies separately labeled.

End condition: development evidence meets the preregistered mechanical-utility
gate, stress failures remain visible, and held-out evidence has not influenced
the policy.

### Stage 7 — Frozen Acceptance and Delivery

Status: `planned`

Implementation:

- run the unchanged system once on the frozen held-out suite;
- produce the report, confidence intervals, recall-cost curves, failure cases,
  hashes, and representative video;
- derive `supported`, `not_demonstrated`, or `invalid` from the frozen gates;
- finalize one-command reproduction and delivery documentation.

End condition: Stage 7 may close with any valid outcome. The overall system goal
closes only if the outcome is `supported`.

## Hardcode and calibration register

| Parameter or group | Classification | Declared owner | Units/provenance | Planning effect | Status |
| --- | --- | --- | --- | --- | --- |
| Source emission rate `0.24` | Calibration candidate | `ModelProfile.dynamics` | inverse seconds; legacy 2% update at 12 FPS | Yes | Explicit, uncalibrated |
| Emission spacing and source kernel | Numerical/model choices | `ModelProfile.dynamics` | pixels | Yes | Explicit, uncalibrated |
| Source-prior width, downstream fade, upwind tail, kernel weight, ROI weighting | Prior/model choices | `ModelProfile.dynamics` | pixels, frame fraction, weights | Yes | Explicit, uncalibrated |
| Foreground extraction ratio | Evaluation/measurement choice | `ModelProfile.measurement` | relative frame intensity | Full-frame motion and offline evaluation | Explicit, uncalibrated |
| Neutral likelihood, intensity percentiles, blur, observed floor/scale | Measurement choices | `ModelProfile.measurement` | relative likelihood and pixels | Yes | Explicit, uncalibrated |
| Wind error, spread, model and quantization radii | Uncertainty choices | `ModelProfile.uncertainty` | pixels/frame and pixels; resolved to pixels/second | Yes | Explicit, uncalibrated |
| Reacquisition diagonal fraction and diffusion mapping | Policy/model choices | `ModelProfile.uncertainty` | dimensionless and pixels | Yes | Explicit, uncalibrated |
| Credible mass, blur, minimum area, source radius | Scan-policy choices | `ModelProfile.scan_policy` | probability, pixels, area fraction | Yes | Explicit, uncalibrated |
| Column/row mass and minimum/maximum sweep size | Mechanical policy choices | `ModelProfile.scan_policy`, request overrides | probability and pixels | Yes | Explicit and reported |
| Force known source into scan | Source-anchored policy choice | `ModelProfile.scan_policy` | boolean | Yes | Explicit; true in compatibility profile |
| Initial corridor direction | Initialization prior choice | `ModelProfile.scan_policy` | pixels/frame | Yes | Explicit, uncalibrated |
| Static corridor center, widths, upwind decay | Benchmark baseline definition | `ModelProfile.scan_policy.static_corridor` | pixels | Baseline only | Explicit and reported |
| `0.45`, `0.22`, `0.08` attention cutoffs | Visualization/diagnostic | report/render constants | relative density after max normalization | No | Explicitly non-confidence |
| Success mass recall `0.90` | Acceptance rule | experiment request/report | probability mass recall | Evaluation only | Explicit |
| Claim improvement `0.02`, bootstrap `0.95` | Preregistered evidence gate | benchmark contract | probability-mass recall | Evaluation only | Frozen |
| Synthetic plume generation parameters | Benchmark fixtures | episode and source configs | episode-specific | Truth generation only | Separated from planner |
| Scanner timing and stride | Physical input or fixture | experiment request/source config | seconds, hertz, pixels | Mechanical feasibility | Explicit and reported |
| Recorded wind maximum age | Physical input/policy boundary | recorded sequence manifest | seconds and acquisition provenance | Yes; selects external wind versus scan-derived fallback | Explicit and reported |
| Recorded normalization bounds | Calibration/input transform | recorded sequence manifest | declared sensor units and calibration provenance | Yes; changes likelihood input | Explicit and reported |
| OGI background and noise maps | Calibration-derived sensor transform | runtime `FixedLeakOffCalibration`; schema-v3 `FrozenCalibrationAssetSpec` | canonical per-pixel median and normal-consistent MAD from a separate audited leak-off asset | Yes; changes measured evidence | Implemented and hash-bound; not fitted to external data |
| Calibration fit/artifact IDs | Reproducibility contract | `per_pixel_median_normal_consistent_mad_v1`, `plume_fixed_calibration_v1` | fixed algorithm and platform-neutral little-endian artifact codec | Yes through resolved calibration | Serialized in the calibration record; changing either requires a new version |
| OGI residual polarity and evidence scale | Sensor-emulation policy/calibration choices | immutable `SensorEmulationProfile` | `bright`, `dark`, or `two_sided_magnitude`; relative scale per noise sigma | Yes; changes measured evidence | Explicit, serialized, uncalibrated |
| OGI output range `[0, 255]` | Sensor interface contract | `uint8_relative_gas_evidence_v1` | non-negative relative evidence, not radiance or concentration | Interface only after profile mapping | Fixed and reported |
| Sensor temporal smoothing | Unsupported behavior | `SensorEmulationProfile` schema v1 | non-null values rejected | Would affect planning if implemented | Removed pending a stateful scanner-response model |
| Recorded truth grade and protocol | Evaluation evidence boundary | sequence schema v3 | `none` or lossless uint8 NPZ plus confined protocol path and SHA-256 | No; evaluator only | Truth-free metrics unavailable; silver diagnostic; gold still requires external review |
| Recorded scanner grade | Evidence boundary | classified sequence schemas | `counterfactual_virtual` only | Determines interpretation of mechanical metrics | Measured-hardware labeling rejected |
| Binary truth cutoff `128/255` | Evaluation format contract | `DenseTruthTimeline` for classified truth sidecars | 8-bit annotation encoding | No; evaluation only | Explicit; not a confidence threshold |
| Constant-FPS tolerance `max(0.05 Hz, 1%)` | Input-integrity tolerance | `recorded_assets.resolve_frame_clock` | hertz | No; validation only | Explicit |
| Decoded-frame hash `plume-focus-decoded-frame-stream-v1` | Input-integrity format | dataset-audit schema v2 | ordered grayscale dtype, shape, and exact pixel bytes | No; validation only | Versioned; decoder build signature is additional provenance, not pixel identity |
| Minimum recorded frames `3` | Input-integrity requirement | accession audit and `load_video_frames` | frames | No; validation only | Shared audit/runtime contract |
| Recorded profile `uncalibrated_video_v1` | Compatibility model choice | recorded sequence manifests | uncalibrated profile provenance | Yes | Explicit; not a real-data default |

No value in the compatibility profiles is considered suitable for final
real-data use merely because it is now configurable.

## Reproduction commands

Install and test:

```powershell
python -m pip install -e ".[dev]"
pytest
```

Generate the causal methane smoke artifact:

```powershell
plume-focus-video `
  --scenario methane `
  --max-scans 8 `
  --vertical-pass-seconds 0.005 `
  --horizontal-step-seconds 0 `
  --column-mass-fraction 1 `
  --row-mass-fraction 1 `
  --max-sweep-columns 320 `
  --output outputs/causal-methane.mp4 `
  --report outputs/examples/causal-methane.json
```

Run benchmark harness and evidence suites:

```powershell
plume-focus-benchmark --suite development --smoke
plume-focus-benchmark --suite development
plume-focus-benchmark --suite held-out
```

Validate and run independently supplied recorded data:

```powershell
plume-focus-recorded-data audit `
  --root <external-video-root> `
  --accession-manifest <accession-manifest.json> `
  --report <dataset-audit.json>
plume-focus-recorded-data prepare `
  --root <recorded-data-root> `
  --spec <preparation-spec.json> `
  --profile <preparation-profile.json> `
  --output-manifests <recorded-data-root>/manifests `
  --report <recorded-data-root>/metadata/preparation-report.json
plume-focus-recorded-data freeze `
  --root <recorded-data-root> `
  --preparation-report <preparation-report.json> `
  --preparation-report-sha256 <exact-byte-sha256> `
  --suite-id <suite-id> `
  --split <development-or-held-out> `
  --frozen-at-utc <explicit-utc-time> `
  --pre-change-model-commit <full-40-character-commit> `
  --freeze-provenance <provenance> `
  --output <frozen-suite-manifest.json> `
  --freeze-record <freeze-record.json>
plume-focus-recorded validate --manifest <sequence-manifest.json>
plume-focus-recorded validate --suite <frozen-suite-manifest.json>
plume-focus-recorded validate `
  --freeze-record <freeze-record.json> `
  --root <recorded-data-root>
plume-focus-recorded validate-pair `
  --development-suite <development-suite-manifest.json> `
  --held-out-suite <held-out-suite-manifest.json>
plume-focus-recorded run `
  --manifest <sequence-manifest.json> `
  --max-scans <count> `
  --report <report.json> `
  --output-video <optional-video.mp4>
```

## Decision log

| Date | Decision | Rationale | Rejected alternative |
| --- | --- | --- | --- |
| 2026-07-25 | Research demonstrator is the deliverable level | Matches available mechanics and evidence without implying a safety system | Production or safety claims without sensor/hardware evidence |
| 2026-07-25 | Known-source tracking owns the primary proof | It is the mature and honestly labeled project mode | Calling source-free tracking source discovery |
| 2026-07-25 | Keep one living goal file | Makes gates and negative evidence persistent across tasks | Unlinked per-stage plans that lose overall context |
| 2026-07-25 | Fair evaluation precedes performance tuning | Prevents the earlier high acceptance result from acting as proof | Tuning against a friendly deterministic demo |
| 2026-07-25 | Treat source continuity as a structural correction, not model validation | Continuous leakage requires source-connected belief, but real visibility may be intermittent | Claiming the corrected synthetic visual proves real physics |
| 2026-07-25 | Declare legacy values in named uncalibrated profiles | Reproducibility requires explicit values and provenance | Hiding scenario literals in the runner or calling them real defaults |
| 2026-07-25 | Relative-density contours cannot control scans | They are max-normalized display aids, not calibrated confidence | Tuning mechanical policy to make overlays look connected |
| 2026-07-25 | Treat hash-validated recorded input as provenance-checked, not independently proven | File hashes establish identity and consistency but cannot establish real acquisition or experimental independence | A boolean evidence-grade flag that lets an engineering fixture appear claim-ready |
| 2026-07-25 | Stop after Stage 4A until external sequences exist | Stage 4 requires real development and untouched held-out evidence before uncertainty calibration | Activating Stage 5 against only project-generated fixtures |
| 2026-07-26 | Reopen the Stage 4A engineering audit before accepting its closeout | The first closeout omitted provenance fields and did not exercise several promised schema branches | Treating passing narrow fixtures as proof of the complete recorded-data contract |
| 2026-07-26 | Model cleaner hardware evidence as a separate sensor-emulation layer | Raw OGI contains fixed scene structure and unstable pixels that the expected sensor interface may not contain | Treating OGI pixels as direct clean-sensor readings |
| 2026-07-26 | Reject temporal smoothing in sensor profile v1 | The removed batch EMA depended on spatial sample order and did not represent persistent hardware state | Keeping a superficially causal but physically ambiguous filter |
| 2026-07-26 | Require exact audit, calibration, truth-protocol, and split lineage | Behavior-changing external inputs must be confined, hashed, typed, and covered by rights and split checks | Trusting partial lookalike JSON or free-text evidence grades |
| 2026-07-26 | Limit recorded-video scanner evidence to `counterfactual_virtual` | Current runs sample a recorded field through a mechanical model and contain no measured hardware scan stream | Letting a manifest relabel modeled timing as hardware evidence |
| 2026-07-26 | Close recorded workflow R0 as `blocked` | Primary publications were verified, but no authoritative accession, terms record, or local GasVid asset was established | Treating a paper URL as data access |
| 2026-07-26 | Keep preparation truth-free and calibration independently frozen | Causal ingestion and sensor calibration must not depend on pseudo labels, evaluation outcomes, or runtime decoder behavior | Requiring truth or refitting leak-off video during sequence load |
| 2026-07-26 | Preserve sampled sensor emulation instead of emitting a planner video | A cleaner sensor abstraction should transform requested measurements without granting hidden full-frame access | Making a preprocessed full-frame video the default planner input |
| 2026-07-26 | Complete R1 engineering while leaving Stage 4/G3 blocked | Deterministic code and generated fixtures pass, but no independent external sequence or rights record exists | Promoting engineering reproducibility into evidence readiness |
| 2026-07-26 | Bind compressed-video use to exact decoded pixels | Version/build metadata alone does not prove that codecs emitted identical sensor values | Treating an OpenCV version string as decoded-data identity |
| 2026-07-26 | Require independently verifiable preparation/freeze lineage | A companion record must be revalidated against its source/derived bytes and paired with runtime suite validation before acceptance use | Treating a write-only lineage record as verified evidence |
| 2026-07-26 | Make accession-audit validity imply runtime frame sufficiency | An audit must not approve a clip that the recorded loader will reject immediately | Treating any nonempty decoded clip as runnable |
| 2026-07-26 | Keep suite publication distinct from freeze-record eligibility | The freeze record is explicitly non-evidence; the suite freezes references but carries no claim-status field | Reporting the suite itself as `not_sequence_evidence` |
| 2026-07-26 | Reopen R0 when a concrete GasVid provider route was established, while retaining the rights block | Access, file identity, reuse rights, and claim eligibility are separate facts | Continuing to call the data absent, or treating a public download as a license |
| 2026-07-31 | Add schema-v4 raw truth-free temporal windows without advancing the evidence claim | Long recordings need exact, preregistered intervals, but window identity and causal execution do not establish reuse rights, pad membership, spatial truth, or independent evidence | Running a guessed GasVid interval, silently slicing whole assets, or treating clip timing labels as plume truth |
| 2026-07-31 | Activate R0B-Q as a metadata-only qualification gate before GasVid preparation | The primary-source/provider audit found no dataset permission, while filename-to-pad mapping and native interval meaning remain inferred or contradictory | Creating frame derivatives first and documenting provenance afterward |
| 2026-07-31 | Preserve the full 31-file accession independently of VideoGasNet's 25-file modeled subset | The six VideoGasNet exclusions are model-scope decisions, not decode failures, rights decisions, or universal evidence exclusions | Silently redefining the accession around one paper's model input |
| 2026-07-31 | Keep reported FGSTP masks outside the raw-pilot prerequisite | https://arxiv.org/abs/2505.00295 reports a labeled 14-video experiment, but its mask/code accession, file/frame mapping, and reuse rights are not established; raw causal execution does not require spatial truth | Assuming a paper figure proves labels are available, or blocking truth-free execution on segmentation labels |
| 2026-07-31 | Treat qualification-check exit `2` as a valid blocked result | The canonical qualification passed structural, canonical-byte, audit, confined-path, and file-hash checks, while 36 declared evidence blockers correctly prevented pilot eligibility | Weakening the checker, promoting inferred assignments, or running a pilot despite `eligible=false` |
| 2026-07-31 | Close R0B-Q engineering and return R0B to an external block | Commit `54893a9` completes the fail-closed qualification machinery, but it cannot supply owner permission or authoritative assignments/intervals | Leaving an engineering stage active while waiting externally, or treating code completion as GasVid readiness |
| 2026-07-31 | Make the owner inquiry the next decision point | A recorded response can either qualify one raw truth-free no-wind pilot or establish that GasVid must be closed blocked and replaced | Running an unqualified pilot, waiting indefinitely, or weakening the evidence contract |
| 2026-07-31 | Supersede immediate sending with inquiry/checker alignment | The draft omits machine-required per-file group fields and lacks a hashed follow-up/closure protocol; sending it now could not deterministically clear the implemented gate | Asking the user to authorize correspondence that cannot satisfy the checker |
| 2026-07-31 | Plan R0C on the licensed USU ground propane OGI video, after removing the methane-only manifest assumption | Explicit CC BY 4.0 terms permit a real recorded-input engineering smoke while GasVid remains blocked; typed propane identity prevents the existing recorded schema from falsely labeling it methane | Calling a propane smoke methane evidence, adding a dataset branch, adding labels prematurely, or waiting idle on GasVid rights |
| 2026-07-31 | Freeze the R0C contract before media accession | Commit `523df93` binds provider gas identity, project preregistration, operation permissions, preparation lineage, causal traces, and renderer-independent reports before any selected media is downloaded or viewed | Inspecting the recording first and tuning scanner, normalization, or reporting behavior afterward |
| 2026-07-31 | Close R0C as `engineering_not_demonstrated` | The licensed recorded-input path is valid and causal, but eight unchanged raw-profile scans requested 89.54–100% of the image and there is no spatial truth to turn column-stride savings into a coverage claim | Calling plumbing success useful adaptive focusing, tuning the frozen run, or treating propane as methane evidence |
| 2026-07-31 | Plan R0D as a measurement-contract diagnostic, not a policy benchmark | Raw OGI carries scene structure and pixel/display instability unlike the expected cleaner sensor; the next uncertainty is signal separability, while policy tuning without a defensible measurement model would confound causes | Tuning ROI thresholds on R0C, frame-global cosmetic normalization, or claiming coverage without evaluator truth |

## Evidence log

| Date | Stage | Commit/artifact | Result |
| --- | --- | --- | --- |
| 2026-07-25 | Stage 1 | `2e9c210` | Fair benchmark completed; claim `not_demonstrated` |
| 2026-07-25 | Stage 2 | `104f55c` | Source-aware causal propagation integrated |
| 2026-07-25 | Stage 2 | held-out report `a7766a...58f4c` | Integrity passed; adaptive substantially lost to fixed corridor |
| 2026-07-25 | Stage 3 | `c61bbfe` | Typed profiles and report serialization complete; 85 tests passed |
| 2026-07-25 | Stage 3 | smoke report `ffe0d1...be36a` | Exact cost and timeline integrity passed; claim remained `not_demonstrated` |
| 2026-07-25 | Stage 3 | canonical JSON `c96c3b...8b054` | Profile disclosure added; prior smoke metrics preserved |
| 2026-07-25 | Stage 4A | `2260d1b` | Recorded manifest, causal clock/wind, isolated truth, report, and CLI contracts complete; 96 tests passed |
| 2026-07-25 | Stage 4A | canonical JSON `9438e5...e3a44` | Synthetic fixture and truth/wind evidence labels added; smoke metrics preserved |
| 2026-07-25 | Stage 4A | development smoke `ffe0d1...be36a` | Benchmark compatibility preserved; claim remains `not_demonstrated` |
| 2026-07-26 | Stage 4A audit | `1ac62f1` | Complete input provenance, focused recorded modules, schema guide, and expanded acceptance coverage; 104 tests passed |
| 2026-07-26 | Stage 4A audit | `8f90aa8` | Suite freeze time, provenance, and pre-change model commit made mandatory and reportable; 105 tests passed |
| 2026-07-26 | Recorded OGI engineering | `102f283`, `1466faa`, `484b063`, `f0748ac`, `ac38940` | Persistent workflow, fixed-calibration sensor emulation, strict accession audit, evidence schema v2, paired split checks, and causal sample-only execution implemented |
| 2026-07-26 | Recorded OGI verification | `.venv\Scripts\python -m pytest -q` | 151 passed, 1 skipped because Windows symlink creation was unavailable |
| 2026-07-26 | Recorded workflow R0 | No authoritative asset/rights record | `blocked`; no recorded sequence, benchmark result, or claim-status change |
| 2026-07-26 | Recorded R1 engineering | `afacbdc`, `7eb26be`, `a074113`, `b0eefe8`, `c8099af`, `4eb339f`, `067e06a`, `48c7d8d` | Canonical calibration, truth-free schema v3, deterministic prepare/freeze, decoded-pixel binding, preparation/freeze validation plus separate suite validation, audit/runtime validity alignment, explicit suite claim boundaries, CLI, and cross-role isolation implemented |
| 2026-07-26 | Recorded R1 verification | `.venv\Scripts\python -m pytest -q` | 208 passed, 1 skipped; audit → prepare → validate → freeze → freeze-record validation fixture passed; no external evidence run |
| 2026-07-26 | Recorded workflow R0 | local `data/recorded/assets/dataset-audit.json`, SHA-256 `246fc11ded363272bce0cec44d7cdfef3ebbab2bd637cdade474b16a1797a189` | 31/31 GasVid MP4s valid; 2,489,878,899 bytes and 696,065 decoded frames; `rights_blocked=true`; artifact is `not_sequence_evidence` |
| 2026-07-31 | Recorded R0B engineering | `366730d` | Provenance-bound raw media windows, exact parent-clock resolution, window/parent/selection hashes, symlink and alias rejection, deterministic preparation/freeze verification, truth-free two-scan execution coverage, and an unsent rights/metadata inquiry draft added; no GasVid derivative or pilot run |
| 2026-07-31 | Recorded R0B verification | `.venv\Scripts\python -m pytest -q` | 244 passed, 2 capability skips; `compileall` and `git diff --check` passed; Stage 4/G3 and claim status unchanged |
| 2026-07-31 | Recorded R0B-Q source audit | https://svl.stanford.edu/gibson2/assets/Wangetal.2019.pdf; https://www.sciencedirect.com/science/article/pii/S0360544221017643; public provider/route URLs in `RECORDED_DATA_WORK_PLAN.md` | No explicit dataset licence or local-processing/derivative/redistribution permission found; `rights_status=unverified`; full 31-file accession is distinct from the 25-file VideoGasNet modeled subset |
| 2026-07-31 | Recorded R0B-Q metadata audit | provider sheet `1869c151...db55`; accession audit `246fc11d...a189` | Five candidate capture groups and 18/13 filename-to-pad assignment remain strong inference; class-0 meaning is contradictory; `MOV_2568`/`MOV_2582` marker gaps are 157/360 seconds |
| 2026-07-31 | Recorded R0B-Q label-source audit | https://arxiv.org/abs/2505.00295, checked 2026-07-31 | FGSTP reports manual labels for 14 selected videos (9 train, 5 test), but mask/code availability, exact mapping, and rights are unverified; no label is required for the raw truth-free pilot |
| 2026-07-31 | Recorded R0B-Q engineering closeout | `54893a9`; `.venv\Scripts\python -m pytest -q`; `.venv\Scripts\python -m compileall -q src tests`; `git diff --check` | Qualification gate committed; 280 passed, 4 Windows symlink-capability skips; compile and diff checks passed; R0B remains externally blocked and no pilot ran |
| 2026-07-31 | Recorded R0B-Q qualification | `data/recorded/metadata/gasvid-qualification-v1.json`, SHA-256 `560f3bad93d3c5b72340ecf30ee174400ef1805b0ea9698c727b143b994c18ed` | Canonical enriched metadata for all 31 assets; rights, assignment, and interval evidence remain unresolved; no media decode, derivative, label, calibration, or pilot |
| 2026-07-31 | Recorded R0B-Q qualification check | `plume-focus-recorded-data qualification-check --root data/recorded --qualification metadata/gasvid-qualification-v1.json --report outputs/gasvid-qualification-v1.check.json`; report SHA-256 `f330e64a26cfec5807bcc034190d5cdc8bb4c7a992f25a80ed11223cea519487` | Exited `2`; `eligible=false`; 36 blockers: 31 inferred assignments, rights block, unresolved interval semantics, and three unknown pilot permissions; no pilot run |
| 2026-07-31 | Recorded fallback primary-source audit | Utah State DOI `10.26078/BQPE-JT65`; CSU DOI `10.25675/3.05018`; BASE Lab and CARB records in `RECORDED_DATA_WORK_PLAN.md` | No candidate combines explicit derivative-friendly rights, methane-confirmed temporal OGI, and Stage 4B split/truth provenance; USU C2 is selected only for a licensed propane engineering smoke |
| 2026-07-31 | Recorded R0C pre-accession contract | `523df9307246fb8365710f8ff2e9fd8784fbdd1c`; `.venv\Scripts\python -m pytest -q`; `.venv\Scripts\python -m compileall -q src tests`; `git diff --check` | Schema-v5 manifest, qualification-v2, preparation-v4, report-v3 causal traces, exact operation authorization, provider-backed propane identity, and renderer independence committed before media access; 307 passed, 5 Windows capability skips; no R0C media inspected or run |
| 2026-07-31 | Recorded R0C accession and freeze | `a8a65b0`; asset SHA-256 `da3f770...b75fd`; audit `58b91a7...8737a`; qualification `5c7c096...7b2c2`; preparation report `610da2b...7037`; suite canonical hash `c63650e...74479`; freeze record `d92403a...c0b6` | Provider MD5 matched; 2,067 decoded frames; qualification eligible; development-only `plume`/`propane` raw suite validated; no held-out or truth attached |
| 2026-07-31 | Recorded R0C execution | `a8a65b0`; headless report `08f973d...740e`; rendered report `a554f9c...eccb`; local video `8e1202a...f27c` | Eight sequential scans; changing samples; beliefs finite/non-negative/normalized; renderer-independent trace; requested-mask area mean 91.05%; result `engineering_not_demonstrated`; Stage 4/G3 and system claim unchanged |

## Update protocol

At the beginning of a stage:

1. read the latest `CHANGELOG.md` checkpoint and this file, then set exactly one
   stage to `active`;
2. copy its objective and end conditions into the bounded execution goal;
3. record the starting branch and commit;
4. confirm which evidence is frozen and may not be tuned against.

At stage closeout:

1. run the stated tests and reproduction commands;
2. record the exact result, including negative outcomes;
3. record the implementation commit and generated-artifact hashes;
4. update completion gates without weakening them;
5. update `CHANGELOG.md` in the same logical change with the checkpoint's actual
   outcome, verification, limitations and relevant revisions/review links;
6. set the next stage to `active` only when execution actually begins.

Use one bounded execution goal per stage. This file carries state between tasks;
an indefinite goal must not be used as a substitute for evidence-backed stage
closeout.
