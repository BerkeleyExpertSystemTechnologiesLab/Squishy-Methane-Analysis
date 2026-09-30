# Changelog

Broad project checkpoints, newest first. This is the maintained summary of what
changed, why it matters, and what was actually established. Detailed evidence
stays in the linked reviews; current work and completion gates stay in
[SYSTEM_GOAL.md](SYSTEM_GOAL.md); the mathematical contract stays in
[the frozen specification](docs/specs/reconstruction-policy-v1.md).

Historical entries were reconstructed on 2026-09-13 from Git and the repository's
conversation/video reviews. A commit date is not necessarily the date an idea or
uncommitted change originated. July commit dates use their original recorded
timezone; September entries use America/Los_Angeles, with UTC document dates
noted where different. No substantial June checkpoint was found in the inspected
records; that does not establish that no work exists elsewhere.

## 2026-09-29 — Submit the reviewed scanner with a usable quickstart

From `1bdf3c6` on `codex/company-submission`, prepare the reviewed package for
`BerkeleyExpertSystemTechnologiesLab/Squishy-Methane-Analysis`, branch
`plume_scanner_new`, under the existing repository's `PlumeScanner/` convention.
Rewrite the README around installation, a self-contained 40 s demo, replay,
rendering, API use, tests and supported claims; preserve older instructions in
`docs/legacy-usage.md`. Add a single-run CLI with exclusive output creation and
public-call replay, a complete source-anchored/known-wind coarse configuration,
and version-dispatched configuration decoding in the two-cycle example. The demo
uses the existing frozen estimator, instrument and gate settings; its 40 s horizon
is an engineering example, not the 360 s study. Configuration hash:
`2c43534d49a0596c3fbeae372c90d01333d3a42492ffc1e82ae924b02b276446`.
Ignore local output/temp directories explicitly.
Mark JSON files as `-text` so Git preserves frozen file-byte hashes across
platforms; no configuration values or frozen declaration are changed.
Submission verification and
remote-tip confirmation are recorded in the
[submission review](docs/reviews/company-submission.md). Generated evidence,
footage, caches, credentials and environments remain local. Company main and
other research directories are preserved; no policy promotion or research claim
change is included. Push uses the user's WinChurchill account and commit identity.

Source verification: 554 tests pass with five platform skips; compilation and
whitespace checks pass. The README's single-run command produces one full scan
and two focused scans, a replay-verified JSON record and the expected explicit
end-of-budget shortfall. The two-cycle example completes FULL then FOCUS.
The exported package also passes 554 tests/five skips in a fresh installation;
all 271 package blobs and README links are checked. Physical/summary videos
decode and inspected startup/focus keyframes retain their causal labels.
Clean historical hard-break spelling and redundant blank EOF lines for the
imported tree's whitespace check; no Python behavior or evidence bytes change.
Source preparation commits are `2ca861f` and `e139952`. Company submission
`79f2d0137ca35950f82e09640b2487999147c6c1` is pushed to `plume_scanner_new`,
with matching local/remote tips and GitHub author/committer WinChurchill verified.
This closeout follows as documentation only. Company main is not merged into or
modified by this submission; generated artifacts remain local.

## 2026-09-29 — Review and clean up the evidence-gated focus checkpoint

From `98aeba9`, work on `codex/evidence-gated-cleanup` addresses the four findings
in the [partial checkpoint](docs/reviews/evidence-gated-focus.md): stable slot
identity permits valid actual timing, resumption scoring includes readings inside
ongoing scans, compact per-action evidence provenance separates preceding bounds
from next-action alpha, and normal report generation preserves criterion verdicts.
Missing evidence now remains unavailable. Regression tests include entry/renewal
delays and overruns, causal and identity rejection, straddling scans, multi-action
serialization/rendering, supported/failed/unavailable reports and no-overwrite guards.
The [cleanup review](docs/reviews/evidence-gated-focus-cleanup.md) records verification
and separate local artifact provenance; historical results and freezes are preserved.
Implementation commit `b1684f2`; final full regression passes 553 tests with five
platform skips, focused checks pass 49, compilation and whitespace checks pass.
All 104 historical executions replay with unchanged causal loss and exact original
metrics; their hashes and freeze are preserved. The separately hashed derived report
fixes the continuous-preservation verdict and changes 12 intermittency summaries:
coarse candidate reading reacquisition is .10-.95 s in the ongoing full scan instead
of the later 26 s focused action. Doubled candidates still miss signal despite 566
post-resumption slots; the predeclared fresh-full-scan criterion excludes them and
the limitation is explicit. Two local renderer smoke videos decode fully, with
three frames inspected and the execution hash unchanged. The four engineering
findings are closed. The estimator and policy settings are unchanged;
the candidate remains opt-in and reconstruction remains mixed. Company transfer is
pending the repository URL; commits stay local as the user requested.

## 2026-09-23 — Retain evidence-gated focus as a partial checkpoint

From `063f4f3` on `codex/evidence-gated-focus`, record the user's decision to retain
the tested policy as a research baseline while marking engineering partially complete.
The [review](docs/reviews/evidence-gated-focus.md#open-review-findings-and-partial-checkpoint)
records four open findings: rejection of valid timing changes, missed reacquisition
inside an ongoing scan, mixed diagnostic alpha provenance, and the report verdict
overwrite. The prior read-only review passed all 18 gate tests and reproduced the
additional gaps with in-memory probes; it did not rerun the full suite. This checkpoint
changes documentation only, verified with `git diff --check`; existing execution
artifacts, historical results, runtime and local `output/` remain unchanged. No fixes,
policy promotion, source-model changes or robotics work are included. Commit and push
this status checkpoint on the existing focused branch without merging or rewriting it.

## 2026-09-19 — Implement and evaluate the evidence-gated focus candidate

Implemented and executed the opt-in `evidence_gated_focus_loop_v3_candidate` gate from
the [handoff](docs/handoffs/evidence-gated-focus.md), leaving the calibrated estimator,
transport, mechanics and renderer module untouched. The gate uses delivered scalar
readings only: Bonferroni measurement-error bounds at the frozen alpha/entry/retain
settings, fresh full-epoch evidence for entry, fresh per-sweep evidence for retention,
explicit hysteresis, and the unchanged charged return-home/complete-full-scan fallback.
The predicted upper-signal field still chooses where to focus. The
[specification](docs/specs/evidence-gated-focus-v3.md) records the statistic, states,
transitions, budget accounting, gate contract and non-goals; `full_fov_loop.py` gained
two inert hooks so historical modes keep identical traces; `evidence_gate.py`,
`evidence_gated_loop.py` and `evidence_gated_execution.py` add separately versioned
message, policy and session types with `SensingController.from_session` dispatch.

Verification: 11 declaration/verification checks pass (context-package byte hash, frozen
reference hashes, an independent normal-quantile comparison, the documented
observed-value cutoffs .35564779584685524 at 640 slots and .37111494274450096 at 2560,
planning-import isolation, configuration round trips for all 104 records, estimator
identity and truth lineage). Eighteen new acceptance tests; the full 543-test suite
passes, with five existing Windows symlink skips. This sandbox denies pytest temporary
directory listing, producing 171 fixture setup errors on this branch and on the
untouched base commit `00d8429` alike; those are environmental.

The fixed-evidence shadow audit of all 12 historical seed-101 histories shows beliefs
identical at every one of 70,942 replayed events (maximum absolute difference 0). On
identical evidence the gate would have left focus at the first completed full epoch in
every blank history, seconds after discontinuation in every cessation history and
immediately after every stop boundary in the intermittent histories (541 mean-positive
but evidence-unsupported decisions), while agreeing with the mean gate throughout every
continuous, wind-error and weak history. The longest support age carried into a decision
was 110.4 s, the declared staleness of a sequential full scan.

The 104 declared executions (32 development, 72 held-out; two profiles, 4 or 6 cases,
two policies, 360 s, identical estimator, mechanics and truth per pair) all completed,
replay-verified and refine below 1%. The 32 development records and 72 held-out records
move the predeclared criteria as follows. Blank: 1425 false-focus decisions become 0
with zero focus sweeps. Cessation: 849 false-focus decisions become 9, with a completed
post-change full scan in every case (37.0-39.0 s coarse, 144.0-144.5 s doubled after the
change). Intermittent: 533 false-focus decisions become 10, with genuine reacquisition
26.0-26.6 s after the coarse resume boundary; the doubled case correctly reports
`incomplete_full_scan_budget` because one 129.5875 s full scan cannot fit in the
remaining 120 s. Continuous, wind-error and weak cases are bit-identical in plan
sequence, J/T (+0.0000%) and core coverage, so the 5% / 5-point preservation tolerance
holds exactly. Reconstruction does not improve: blank J/T is 91% higher in both stages,
cessation 8.0-8.5% and intermittency 4.7-4.8% higher, with intermittent core readings
falling because focus stops during the background window. Classification is
`mixed`: behaviour repaired, map error not. The calibrated estimator still recreates
false source peaks: 105 decisions had a positive old mean gate with unsupported reading
evidence and 20 candidate runs ended with a false posterior peak (maximum final mean
0.680), so source-activity inference remains the smallest next model experiment. No map
suppression, source switch-off, belief reset, margin change or policy promotion was
introduced. Five inspection pairs rendered in the unchanged six-panel layout decode
(7,427 frames) and leave execution hashes unchanged, including one alternate 2 fps / 3x
playback; reviewed keyframes are recorded with the observed gate lines. The report
artifact's continuous-preservation label is a known reporting defect (an aggregate
`status` key overwrites the criterion label); `scripts/check-evidence-gated-focus-report.py`
recomputes all four held-out criteria from the 104 records and records the corrected
`supported` verdict in `checks/predeclared-criteria.json` without touching frozen
artifacts. The [review](docs/reviews/evidence-gated-focus.md) records the full tables,
refinement, media review, closeout revisions, claims, limitations and reproduction
commands. All 104 executions share the frozen producing code hash and the branch is
pushed to `origin` with the remote tip verified equal to local `HEAD`. Generated
evidence, videos and traces stay local under the ignored
`outputs/evidence-gated-focus/` tree.

### Implementation checkpoint before study execution (historical)

Implemented the opt-in `evidence_gated_focus_loop_v3_candidate` gate from the
[handoff](docs/handoffs/evidence-gated-focus.md), leaving the calibrated estimator,
transport, mechanics and renderer untouched. The gate uses delivered scalar readings
only: Bonferroni measurement-error bounds at the frozen alpha/entry/retain settings,
fresh full-epoch evidence for entry, fresh per-sweep evidence for retention, explicit
hysteresis, and the unchanged charged return-home/complete-full-scan fallback. The
[specification](docs/specs/evidence-gated-focus-v3.md) records the statistic, states,
transitions, budget accounting, gate contract and non-goals. `full_fov_loop.py`
gains two inert hooks so historical modes keep identical traces; `evidence_gate.py`,
`evidence_gated_loop.py` and `evidence_gated_execution.py` add separately named
message, policy and session types with `SensingController.from_session` dispatch.

Verification: 11 declaration/verification checks pass, including the context-package
byte hash, the frozen reference-configuration hashes, an independent normal-quantile
comparison, the documented observed-value cutoffs (.35564779584685524 at 640 slots,
.37111494274450096 at 2560) and a planning-import isolation scan. The fixed-evidence
shadow audit over all 12 historical seed-101 histories is complete: beliefs are
identical at every replayed event (maximum absolute difference 0), and on identical
evidence the new statistic would have left focus at the first completed full epoch in
every blank, cessation and intermittent history (coarse 228/126/81, doubled 48/36/22
unsupported decisions) while agreeing with the old mean gate in every continuous,
wind-error and weak history. The 32 development and 72 held-out executions are declared,
hash-frozen and unexecuted at this checkpoint. 18 new acceptance tests; the full
543-test suite passes (five existing Windows symlink skips); this sandbox denies
pytest's session-finish symlink cleanup on a workspace base temp directory, an
environment artifact that occurs after all tests have run.

## 2026-09-19 — Prepare external-agent evidence-gate handoff

Prepared a [self-contained implementation assignment](docs/handoffs/evidence-gated-focus.md)
and [machine-readable context](docs/handoffs/evidence-gated-focus-context.json)
from inspected implementation `2e16471`. The package contains both complete frozen
reference configurations and 72 historical held-out result summaries. It proposes
a reading-supported entry/retention gate with explicit hysteresis, conservative
measurement-error bounds, mandatory fresh full-rescan obligations and 104 new
paired executions. These are proposed settings, not implemented or calibrated
behavior. The estimator remains fixed to isolate the gate; source-activity inference
is deferred pending the operating-loop test. Robotics remains deferred. Verified
configuration/result extraction and documentation diff; no runtime code changed.

## 2026-09-19 — Tune and calibrate the scanner baseline

Completed from b5b48e6 on codex/scanner-baseline-calibration. The
[declared protocol](docs/specs/scanner-baseline-calibration.md) separates charged
vertical margins from fixed-evidence source/lifetime fitting and freezes a
72-execution held-out study. Historical behavior remains available; robotics
and simulator selection are deferred. Versioned margin/decay implementation and
the frozen pipeline landed in `07f844d`; `08ed40d` verifies an exact, faster truth
quadrature path. All 520 tests pass (five existing Windows skips), with historical
compatibility checks passing. The 36 margin runs fail the declared nonzero-margin
criterion: doubled-profile core coverage falls from about 86% to 51–53% as padding
consumes the time budget. Retain zero margin. The [review](docs/reviews/scanner-baseline-calibration.md)
records this negative result and all 72 frozen held-out results. The calibrated
model lowers J/T in 24/24 pairs by 12–41%, with improved continuous core readings,
but every blank case falsely focuses and no cessation case resumes full scanning.
Four enhancement-index MSE comparisons worsen. Keep the candidate opt-in;
evidence-driven source activity is the smallest next proposed stage, not implemented.
The locked fit selects a 6 s half-life, width .035 and rate .2829738194. During
held-out verification, detached no-decay queries were changed to the existing
vector forecast to avoid rebuilding unused covariance over long idle periods.
The original records and parameter freeze are preserved; versioned compatibility
evidence checks query error below 2e-15; all 108 saved executions pass replay.
All four evaluation refinements pass. Twelve videos decode with selected-frame
review; playback leaves trace hashes unchanged. Reference configs are committed;
generated evidence stays local. Historical `not_demonstrated` findings remain.

## 2026-09-18 — Diagnose focus extent and overlapping mechanisms

Completed evaluator-only round from 44be6a9 on codex/focus-extent-diagnosis.
The [declared protocol](docs/specs/focus-extent-diagnosis.md) prioritizes amplitude,
source injection, spatial representation, uncertainty and mechanical closure.
Saved evidence is frozen before isolated probes. Operating policy and historical
results remain unchanged; no calibration or superiority claim is added.

All 42 snapshots from 18 saved traces reproduce public inference exactly. The
mean already contains an excessive downwind tail; source input/persistence and
spatial representation contribute, with additional uncertainty-driven background
selection. Arithmetic amplitude scaling and nominal wind conversion are correct;
mechanical closure adds no cells in these inspected plans. Source-off replay also
loses continuing signal, so removing it is not a justified production fix.

The [review](docs/reviews/focus-extent-diagnosis.md) records the overlap audit,
falsification outcomes, numerical results and reproduction commands. All 84
projection checks pass; 506 tests pass with five existing Windows skips. Runtime
hashes and 184 historical artifacts are unchanged; 28 diagnostic figures stay
local. Implementation checkpoint 9e52042; final commits and remote confirmation
are recorded in local closeout. Source/lifetime calibration remains proposed next.

## 2026-09-16 — Implement and verify the lean full-FoV/focus loop

The [approved plan](docs/reviews/full-fov-loop-plan.md) now implements the user's
sequence: mandatory complete full-FoV startup, signal-supported focus, otherwise
charged return home and another complete full scan. Each rescan requires fresh
actual readings. Full scans use actual physical duration, not the 12 s focus
ceiling; initial scans take 33.575/129.5875 s for coarse/doubled profiles.

Independent minimal settings and plan messages replace inherited risk/mass
fields in this new binding. Shared acquisition and neutral physical compilation
retain inference, causal replay and detector accounting; historical policy
dispatch sits behind a compatibility adapter. No age-block substitute, prior
bypass, periodic service detour or belief reset is added. The public API example
and [implementation guide](docs/sensing-core.md) expose the lean module boundaries.

All 18 declared 360 s runs (two profiles × continuous/blank/cessation × simulated
noise 0/.01/.05, seed 0) pass exact public-call and causal-loss replay. Assumed
likelihood noise remains .05. Blank cases stay in full scanning at every noise
level; full/focus counts match across noise levels. Fourteen canonical videos
are decoded and inspected through documented representative frames, retaining
the existing layout. Labels separate added noise, uncertainty and normalized
excess; terminal idle is explicit. Playback changes preserve execution hashes.

Verification: 489 tests pass, five existing Windows symlink skips; compilation,
import boundaries and diff checks pass. Legacy/v1–v5 arithmetic fixtures retain
exact plan/posterior hashes against `84cf70d`. Start `84cf70d`, runtime `90fba10`,
observer/verification checkpoints `da531b6`, `9d506c6`, `d3a80c6`, evidence
checkpoint `6b5f796`; focused commits are pushed on `codex/full-fov-loop-plan`.
The [closeout review](docs/reviews/full-fov-loop-implementation.md) records commands,
numerical results, artifact hashes and final-revision provenance. Media and
preexisting `output/` remain local.

Remaining limitation: source-anchored inference can rebuild false signal after
cessation and reactivate focus, even at zero added noise. Long estimated tails,
uncalibrated bounds and reconstruction advantage remain unresolved. This round
establishes loop conformance, not a new policy freeze or superiority. Fixed-path
source/lifetime and forecast calibration are proposed next. Earlier negative
findings remain evidence for their original, separately preserved policies.

## 2026-09-16 — Complete signal-gated threshold-bound allocation test

Start `5bb58ee`, branch `codex/threshold-bound-scanning`. The approved
[v5 protocol](docs/specs/threshold-bound-scanning-v5.md) keeps the estimator fixed
and tests a current-mean signal gate followed by a fixed 0.15 / 95th-percentile
region. Budgeted partial selection remains explicit; absent/infeasible signal
uses age-based scene scanning. No mandatory shrinking or new policy freeze.
New versioned bindings preserve historical hashes and use corrected transport.
The frozen comparison contains 348 completed execution records.
Checkpoint `1149a13` implements the rule, versioned messages, exact mixture
quantiles and autonomous snapshot continuations. The manifest is frozen before
new outcomes. Initial regression: 466 passed, five Windows symlink skips;
additional adapter, prefix, startup and detached-bound checks extend acceptance.
Bound-reliability, reading-residual and saved-trace numerical observers do not
change inference or operational paths.
Observer checkpoint `67bca3f` adds bound and causal comparison diagnostics.
The full regression now passes 473 tests with five Windows symlink skips.
Independent v1/risk arithmetic, compilation and import-boundary checks pass.
Coverage auditing and comparison plots are evaluator-only additions.
The Gaussian results are mixed: short coarse improvements over v4 coexist with
doubled-profile deterioration; the mean-only ablation also gives mixed evidence.
Saved-state upper bounds substantially underpredict actual signal in nonblank
fixtures. The [review](docs/reviews/threshold-bound-scanning-v5.md) records these
limitations separately. Sixteen videos are decoded with representative keyframe
review; playback changes preserve the operational trace hash. All 348 runs,
348 coverage records, 152 planned-bound/reading reports, eight bound fixtures,
14 short/full contrast refinement reports and public replay checks are complete.
Mixture results remain numerically unresolved under the fixed 49-hypothesis
approximation. No new policy freeze; historical `not_demonstrated` findings stand.
Checkpoints `f003270` and `d967b87` add audits, inspection and documentation;
local `outputs/threshold-bound-v5/closeout.json` records final revision/hashes.
Smallest proposed follow-up: fixed-path forecast/source/representation calibration,
with exposure and propagation numerical effects separated. It is not implemented.

## 2026-09-16 — Complete scan-value diagnosis and opt-in boundary correction

Start 7bd9d54 on codex/scan-value-diagnosis; implementation checkpoints
6b6218f through 28f5765. Added versioned zero-inflow transport, detached READY
snapshots, evaluator-only forced paths and resumable, hash-checked attribution.
Historical sessions, renderer layout, instrument settings and output/ remain
unchanged. Final audit compares causal wind fields across their configuration and
message wrappers; this metadata correction does not change any experiment.

All eight frozen snapshots are complete: 9,536 paired continuation records plus
16 idle references. Gaussian value misranking is supported in eight of 12 tested
fixture/transport contrasts; added geometry helps in one and continuation matters
in two. Boundary contribution is path-dependent, including negative effects.
Mixture concentration remains causally unresolved under shared model mismatch
and failed wind quadrature. The strict corrected blank-signal contrast also
remains numerically unresolved. Historical `not_demonstrated` findings stand.

Verification: 447 tests pass, five Windows symlink skips; independent v1/v4
arithmetic, compilation and diff checks pass. All ten evaluation refinements
accept the shared mesh. Exposure ablations retain their winners; propagation
partition sensitivity is reported. Ten videos (466 frames) were decoded and
reviewed with unchanged operational hashes across playback settings. Summary
labels explicitly state one frame per sweep. Local manifests record revisions,
configuration/evidence hashes, coverage, physical spending and offline compute.

Next proposed work is a bounded source/lifetime and forecast-calibration check
on fixed paths, before any new local-motion formulation or allocator. No new
policy is frozen. See [the completed review](docs/reviews/scan-value-diagnosis.md)
for the loop explanation, hypothesis table, numerical limits and reproduction.

## 2026-09-15 — Begin scan-value diagnosis and isolated transport correction

Start `7bd9d54`, branch `codex/scan-value-diagnosis`. The approved diagnosis keeps
the v4 objective/instrument fixed and tests transport, value prediction, candidate
limitations, continuation approximation and mixture concentration separately.
Opt-in diagnostic sessions identify corrected zero-inflow transport explicitly;
historical configurations and default propagation retain their original meaning.
Detached READY snapshots and mechanically validated forced-path evaluation make
the same evidence/actions inspectable without giving truth to the controller.
Initial verification: 27 existing risk tests and 10 transport/snapshot checks pass.
The study and formal attribution are pending; no new motion model, allocator or
policy freeze is implied. See `docs/specs/scan-value-diagnosis.md`.
Checkpoint `6b6218f` implements the opt-in intervention and freezes eight actual
v4 decision snapshots. Full regression: 441 passed, five Windows skips; additional
delay/overrun and zero-mixture checks pass. An exact Gaussian risk cache reuses
identical covariance/schedule calculations across noise seeds; mixtures remain
uncached. Independent checks verify cached/uncached equality and detached output.

## 2026-09-15 — Complete reconstruction-value acquisition experiment

Start `51d1f63`, branch `codex/reconstruction-value-policy`. The approved
[candidate](docs/specs/reconstruction-value-policy-v4.md) replaces the hard mass
execution gate with expected physical-time reconstruction risk over feasible
repeated scan patterns. Both unchanged estimators are included in stages; map
loss stays primary. Small charged oldest-block service replaces periodic
full-budget surveillance. Preserve v1-v3, historical evidence and negative results.
The first implementation (`9ae89a8`) adds bounded candidate generation, Gaussian/mixture
risk, versioned plans, causal delivery assumptions and actual-visit service credit.
Independent risk/propagation, transport and delayed lifecycle tests pass; the
full suite passes 418 tests with five existing Windows capability skips.
Checkpoint `65d6077` freezes the 96-record declaration and historical hashes.
Further exact scorer/caching optimizations pass 430 regressions with five existing
Windows skips. All ten Gaussian/mixture reference decisions pass joint time/history
checks; freeze 0.1 s integration and 32 histories before the closed-loop study.
The saved 12.25 s candidate has feasible alternatives under 12 s. Delayed-delivery
and overrun inspection preserves the renderer layout and operational hash.
Gaussian main comparisons are mixed: two-seed mean map loss improves about 4-5%
in biased-wind cases but worsens about 5-15% in known-wind/source-free cases.
All 36 new runs and 96 comparison records are complete at producing revision
`ad801c6`; 60 historical controls pass configuration/evidence compatibility checks.
Five evaluation mesh checks pass, and 20 inspection videos fully decode with
selected frames reviewed. All 32 required service blocks complete; 149 of 193
value-selected sweeps execute below 90% predicted mass.
Gaussian classification is `mixed` (3/12 main pairs improve, 9 worsen). The fixed
mixture remains `numerically_unresolved` (2 improve, 10 worsen): doubled biased-wind
mean map loss rises about 100%, while the full-strength source-free diagnostic
improves 27%. Weak-signal cases remain problematic. Model-predicted risk is not
measured reconstruction quality, and compute time does not establish real-time use.
The [completed v4 review](docs/reviews/reconstruction-value-policy-v4.md) records
commands, producing/closeout revisions, hashes and resolved/unresolved issues.
Boundary correction, changing local drift and calibration remain subsequent model
work. Claim status remains `not_demonstrated`; no new policy freeze.

## 2026-09-14–15 — Complete persistent wind experiment

**Engineering checkpoint complete; hypothesis `numerically_unresolved`; claim
`not_demonstrated`.** Started at `66ea099` on
`codex/persistent-wind-prediction`. The [candidate specification](docs/specs/persistent-wind-motion-v3.md)
implements persistent conditional Gaussian fields and likelihood weights while
preserving v2 allocation, mechanics, source assumptions, threshold and noise.
Exact mixture expected excess and conservative component-envelope diagnostics
replace Gaussian-only reporting in this candidate. Frozen v1 and historical
negative findings remain intact; no new policy freeze.

The [completed review](docs/reviews/persistent-wind-prediction.md) records the
predeclared 72 main plus 12 diagnostic comparisons, fixed-evidence forecasts,
controlled transport fixtures, configuration/evidence checks and artifact hashes.
Main biased-wind adaptive mean J/T worsens by 1.00% coarse and 2.47% doubled.
Full-strength source-free map loss improves by 37.4%, but enhancement-index MSE
worsens. Weak/blank map gains remain surveillance-only and also worsen index MSE.
Learned wind corrections do not recover the nominal generator correction;
likelihood concentration can reflect model mismatch. Two noise seeds provide
descriptive evidence only; common ceilings do not imply identical expenditure.

Both 9/25 and 25/49 hypothesis refinements fail; 49 hypotheses remain provisional.
Spatial/time refinement accepts 120 x 160 / 0.05 s. All 42 configuration pairs
pass, and all 36 fresh Gaussian baselines exactly reproduce historical execution
hashes and loss. Ten videos / 4,550 frames fully decode; selected physical-time
and completed-sweep frames were visually reviewed. The six-panel renderer is
preserved, with only a necessary mixture uncertainty subtitle. Existing signal
color scales clip high values; raw inference and scoring retain them.

Independent batch calculations, delayed replay, zero-covariance compatibility,
fake-scanner/public replay, configuration round trips and regression pass.
Full suite: 403 passed, five existing Windows symlink capability skips. Independent
specification arithmetic, compilation and Git diff checks pass. Offline SciPy
CDF/bilinear evaluation optimizations preserve reference arithmetic and execution
hashes; compute cost remains substantial and is not physical scan time. Reporting
corrects available blank-truth background weight and explicitly identifies the
inherited whole-sweep geometric miss proxy as distinct from time-weighted capture.

Inference/API checkpoint `81e3192`; frozen declaration `c9d71e0`; fixed-evidence
clock correction `ba4b1eb`; offline evaluation `0201240`; last core/reporting
change and full regression `bfdaecc`; later commits add observers and evidence
reviews. The linked review and local closeout provenance record final revisions,
verification commands and producing revisions. Generated artifacts stay local;
the original 301-file evidence index and historical untracked `output/` are preserved.

**Stored next-stage decision:** changing local drift is required, not solved by
learning persistent error. A post-hoc isolated probe also finds inherited
incoming-boundary clipping that pins signal and can sustain artificial inflow.
It is preserved in the frozen comparison and requires subsequent correction and
validation. Persistent tails, posterior/model calibration, weak signal, hard-budget
fallback and incomplete refresh remain open; recorded-data expansion stays paused.

## 2026-09-13 — Diagnose mechanism mismatch and test experimental v2

Started at `a341e68` on `codex/mechanism-grounded-revision`. Video review exposed
full-height fallback despite two-axis scanner capability, and uncertain background
dominating the zero-threshold mass rule. Fixed-path ablations implicate process
uncertainty/weighting; removing injected reading noise does not fix the scan.

The [mechanism-grounded candidate](docs/specs/mechanism-grounded-policy-v2.md)
and [diagnosis](docs/reviews/mechanism-revision.md) connect measurement, reportable
signal, physical segment geometry and causal reconstruction. A separately named
policy adds two-axis surveillance, periodic service and an explicit synthetic
reporting threshold; actual noise and assumed R are independently configured.
The tentative 40 x 64 physical profile retains a 20 Hz detector and declares its
slower acquiring speed. No renderer redesign, v1 rewrite or lowered main-study
noise/Q. The development comparisons and weak/blank-signal stress tests are complete.
Full regression currently passes 382 tests (five Windows capability skips),
including expansion/movement, outside-region surveillance and delayed delivery.
The per-run focusing audit records area sequences, minimum-size visits and age
distribution; the 90% rule is never applied recursively to the previous ROI.
Implementation `af40fdd` produced 36 main comparisons and seven diagnostics.
Two-axis tracking executes in the coarse fixture, but the fixed corridor still
has lower reconstruction loss in its three cases. Doubled-profile results are
mixed and mostly use fallback. No adaptive superiority or new policy freeze.
Inspection tools reproduce saved executions, and an evaluator-only projection
check distinguishes finite spatial resolution from online estimation error.
Media/tools revision `2125a1e`: six videos, 2,724 fully decoded frames, selected
physical-time and summary frames visually reviewed. Coarse selection expands and
contracts; that does not guarantee useful reconstruction or scene refresh. The
hard 90% feasibility gate can still send a resource-limited scanner into background
surveillance. Temporal model calibration, weak-signal handling and that decision
remain unresolved; no main-study parameters were tuned to remove the negative result.
Closeout removes a redundant compiled-cell area diagnostic that could be mistaken
for actual area after partial execution; the executed-footprint metric and all
scientific comparison values are unchanged. A finite-source regression covers it.

## 2026-09-13 — Implement frozen v1 and complete development inspection

**Engineering checkpoint complete; adaptive advantage `not_demonstrated`.**
Started at `4a598ac` on `codex/policy-v1-conformance`; final implementation
`8de3a3e`. The frozen mathematical and scan-policy specification is unchanged.

- Bound explicit immutable instrument/session settings to the controller. The
  separate profile compiler preserves physical footprint, pitch, cadence,
  integration and speeds, charging motion, acquisition, settling and recovery.
- Implemented `mass90_column_v1` with exact mass-prefix selection, cell-preserving
  column closure, bounded completion forecasts and visible constraint failures.
  Oversized tracking selections remain intact; age fallback is separately
  labelled and charged. Full startup requires actual visits to every cell.
- Added versioned public-call traces and causal replay for common-time
  reconstruction and signed-enhancement-index error. Delivery latency, idle
  periods, overruns and unavailable/incomplete truth remain explicit. The
  simultaneous-band diagnostic does not change allocation.
- Added explicit synthetic coarse/narrow CLI configurations and a fake-scanner
  example. Preserved the renderer layout and legacy modes. Limited rendering
  corrections cover causal timestamps, actual motion, units, profiles and idle
  status; old and labelled videos have identical operational traces.
- Completed 36 descriptive comparisons and six quadrature checks. The common
  120 x 160 / 0.05 s evaluation mesh passed (maximum refinement change 0.3772%).
  Coarse adaptive tracking uses age fallback and loses to the fixed corridor;
  narrow startup visits only 1.875% of the scene. Common ceilings do not mean
  exactly matched expenditure. Model persistence and uncertainty calibration
  remain unresolved.

Focused checkpoints: A `c1ced3b`, B `37bb8c7`, C `d4d3d44`, D `4dbfa8e`.
The pre-acceptance fixed-corridor coverage-gate correction is `043549d`, which
produced the final numerical data. Preliminary interrupted artifacts are kept
separately. Labelled inspection/validation is `698c793`; final API immutability
and inspection provenance are `8de3a3e`. No dynamics tuning or held-out tuning.

Verification: **366 tests passed, five Windows capability skips**; nine
independent arithmetic groups, schema/import boundaries, compilation, links and
Git diff checks passed. All 12 videos / 5,448 frames decoded and representative
coarse, narrow and delayed-delivery/completion frames were reviewed. Historical
recovery operational hash is unchanged. See the
[implementation review](docs/reviews/policy-v1-conformance.md) for commands,
producing revisions, numerical results, artifact hashes and unresolved issues.
The [sensing guide](docs/sensing-core.md), [status](SYSTEM_GOAL.md), README and
agent guide now distinguish implemented v1 from legacy behavior. User `output/`
artifacts are preserved; generated media remain local, no remote push occurred,
and recorded-data expansion remains paused.

## 2026-09-13 — Adopt this changelog

**Documentation checkpoint.** Added the May-to-present history and made changelog
maintenance part of repository practice. Future substantive changes and completed
work items must update this file alongside the relevant code or documentation.
Read it with the system goal before starting work; preserve historical and
negative findings when adding new entries.

Starting revision: `aac8c87`, branch `codex/sensing-core-recovery`. This entry is
recorded in the changelog-introduction commit, identifiable with
`git log --oneline -- CHANGELOG.md`. Checks: referenced revisions and local links
verified; `git diff --check` passed. No runtime changes or new experiment; the
existing test-suite result below was not rerun for this documentation change.

**Next checkpoint, pending:** implement the frozen v1 contract using the
[audited handoff](docs/reviews/policy-v1-freeze.md#focused-implementation-handoff).
The implementation stage has not started. Recorded-data expansion remains
paused and the adaptive performance claim remains `not_demonstrated`.

## 2026-09-12 — Freeze reconstruction formulation and scan policy v1

**Specification checkpoint; runtime implementation pending.** Revisions
`ab80dc5` (specification) and `aac8c87` (verification/closeout). The documents use
the corresponding UTC date, 2026-09-13.

- Clarified the purpose: reconstruct an evolving plume under acquisition and
  mechanical time constraints, eventually supporting a physically defined
  strength estimate. The immediate target is a time-referenced map of **inferred
  beam-integrated concentration enhancement**; it is not automatically a 3-D
  concentration field, plume mass, or emission rate.
- Froze `beam_reconstruction_v1`: correlated Gaussian conditioning of retrieved
  scalar readings, explicit footprint/calibration assumptions, acquisition versus
  availability time, causal motion prediction, and reconstruction error evaluated
  over a common physical horizon.
- Froze `mass90_column_v1`: select 90% of expected positive-signal mass, preserve
  selected cells through legal column closure, charge the complete path, and
  expose constraint failures. Repeated trimming is not required. Reacquisition
  is a separately labeled and charged age-based action; replanning waits for
  completion of the current sweep.
- Required an explicit `instrument_profile_v1` for every experiment. Beam size,
  physical pitch, cadence, integration and mechanical speeds remain fixed within
  comparisons. No numerical reference hardware was selected. Inference/display
  grids remain separate from instrument settings.
- Specified a posterior coverage-bound diagnostic while retaining UCB and other
  allocation extensions as separate experiments. A 90% expected-mass target is
  not a 90% probability of enclosing the whole plume.

Nine independent worked-check groups passed. Dynamic sequential and independent
batch conditioning agreed within `2.22e-16`. The existing suite passed **335 tests
with five Windows symlink-capability skips**; schema, dependency-boundary,
compilation, link, diff and artifact checks passed.

The audit reproduced current-code gaps: budget-based subset allocation, implicit
sweep-speed reduction, and a past-time predictive view containing later-delivered
evidence. Profile binding, the revised allocator, causal reconstruction evaluation,
and the confidence diagnostic remain implementation work. Sensing code and
renderer were unchanged; no new performance experiment or video inspection ran.

References: [formulation/policy](docs/specs/reconstruction-policy-v1.md),
[instrument profile and schema](docs/specs/instrument-profile-v1.md),
[worked examples](docs/specs/freeze-v1-worked-examples.md),
[review and handoff](docs/reviews/policy-v1-freeze.md), and
[research synthesis](docs/research/adaptive-beam-plume-formulation.md).

## 2026-09-12 — Audit May/July divergence and recover the sensing core

**Engineering checkpoint; adaptive advantage not demonstrated.** Work started
at `cb89020`; audit `081278e`, core/API `fc5657f`, adapters/videos `33a4a4c`,
final implementation `e3e1b5e`, and closeout `66f0770`.

- Replaced the legacy per-sweep percentile likelihood in the new sensing path
  with a finite correlated Gaussian signal model, fixed synthetic calibration,
  timestamped scalar conditioning, and separate expected positive-signal mass.
  This was an explicit replacement model, not a claim to have recovered correct
  May Bayesian code. Legacy profiles and evidence were preserved.
- Added the callable `plan_next` / `ingest_readings` / `finish_scan` controller,
  versioned serializable messages, simulator/fake-scanner adapters, acquisition
  ordering, duplicate handling, and actual execution receipts.
- Separated physical scanner geometry from display resolution and charged
  acquisition, initial repositioning and skipped-column travel. Added physical-time
  and compressed decision-summary views and controlled rendering/measurement
  replays; corrected measurement-obscuring outlines.

Verification recorded **335 passed, five platform skips**, CLI/API examples,
artifact hashes, full video decodes, and representative frame inspection.
The 18-run development comparison remained `not_demonstrated`: the fixed corridor
outperformed adaptive tracking in the tested conditions. Actual expenditure
differed and was reported. Excessive signal persistence and uncalibrated
uncertainty remained visible model failures.

The review supported a change in research emphasis, but not a wholesale rollback:
May already contained measurement and scan-timing defects, while July added
useful causal scheduling and evaluation. Video differences included playback
compression, changed operating conditions, and rendering defects. The subsequent
v1 freeze above addresses the user's concerns about recurring trimming, physical
settings, and reconstruction purpose; this recovery alone did not settle them.

References: [scope/history review](docs/reviews/2026-09-12-scope-review.md),
[video evidence](docs/reviews/2026-09-12-video-evidence.md), and
[recovery verification/results](docs/reviews/sensing-core-recovery.md).

## 2026-07-24–2026-08-01 — Causal experiments, benchmarks, and recorded-data infrastructure

**Several engineering checkpoints; broader scientific claim not demonstrated.**

- `a49558d` saved source-free tracking WIP and project instructions whose origin
  included May discussions. It is not evidence that all of that work began in July.
- `607b6f7` through `d167f1d` introduced causal, non-overlapping scans, finite-input
  handling, actual-sweep evaluation, a headless runner and focused modules;
  `plume_focus.video` became a compatibility/CLI facade.
- Fair benchmark work through `2e9c210`, followed by source-aware propagation
  and evidence through `ad12498`, made comparison and source assumptions explicit.
  The historical held-out result was **`not_demonstrated`**: the adaptive policy
  lost to a fixed source corridor under that benchmark's cost definition. Later
  critiques of allocation, operating regime and travel accounting limit its
  interpretation; they do not erase the result.
- `c61bbfe` added reportable model profiles and `fbfbb78` established the living
  system goal. The work expanded into recorded-input contracts, sensor emulation,
  calibration lineage, accession, preparation, freezing and qualification,
  beginning with `2260d1b` and continuing through July.
- `523df93`, `a8a65b0`, and `cb89020` captured the licensed R0C propane/OGI
  diagnostic contract, execution and evidence closeout. It was an engineering
  diagnostic under different sensing conditions, not calibrated methane point
  sensing or evidence of superiority. Its result was
  `engineering_not_demonstrated`; the recorded checkpoint had **307 passing tests
  and five platform skips**. GasVid qualification/permission issues remained open.

This period improved experimental honesty and preserved useful infrastructure,
but did not resolve the original per-reading neighbor-update model. The legacy
measurement path still relied on per-sweep normalization and normalized spatial
weights. Recorded-data expansion was later paused to recover the sensing core.

References: [historical review](docs/reviews/2026-09-12-scope-review.md),
[system evidence and stage ledger](SYSTEM_GOAL.md), and
[recorded-data work history](RECORDED_DATA_WORK_PLAN.md).

## 2026-04-29–2026-05-18 — Original May project checkpoint

**Concept and prototype checkpoint; not a validated Bayesian or mechanical model.**
Saved snapshot: **`1fcca43`, 2026-05-04**. Later May changes were not all committed
then; some were captured by the July 24 WIP checkpoint `a49558d`.

The original scope combined sparse single-detector observations, a changing plume,
wind/motion information, spatial uncertainty and a constrained next scan. The
intended cycle was prior -> measurement likelihood -> posterior -> motion
prediction -> next region. The user's mechanical clarification was vertical
passes with horizontal movement between passes and a beam pixel that might not
be enlarged. That column-sweep intent predates July.

The May snapshot contained Bayesian-named likelihood/weight updates, wind
propagation, belief-mass region selection, mechanical column budgets and scan
videos. The videos included visibly accumulating measurements, an inspection
behavior worth preserving. Subsequent May discussions requested source-free
tracking with known wind, questioned how a single reading should update nearby
unsampled pixels, and added project boundaries/exclusion principles. Those
discussions are part of the May checkpoint even when the code was saved later.

The important unresolved issues were already present: per-sweep percentile
normalization did not support the intended single-reading likelihood; unsampled
neighbors did not receive the claimed direct measurement influence; normalized
spatial weights were not independent plume-presence probabilities; and the old
loop could overlap physical scans and score attention instead of the actual next
sweep. Smooth pictures and short videos did not establish causal reconstruction.

May established the research direction, narrow-beam/column-sweep intent and
interactive scan-inspection expectation. It did **not** establish complete
independent detector cadence/integration specifications, a calibrated strength
observable, or adaptive superiority. The later Gaussian formulation is a declared
replacement, while the current 90% rule preserves the broad mass-selection idea
without treating historical percentages as a containment guarantee.

Evidence: [repository/chat reconciliation](docs/reviews/2026-09-12-scope-review.md),
[historical video review](docs/reviews/2026-09-12-video-evidence.md), and
[May versus later instrument assumptions](docs/specs/instrument-profile-v1.md#historical-mapping).
The linked reviews identify local conversation exports and media; those artifacts
remain local rather than being committed into this changelog.

## Maintaining this file

- Read the latest checkpoint and pending work together with `SYSTEM_GOAL.md`
  before substantive work. Do not infer that a frozen specification is implemented.
- Add or update an entry in the same logical change for meaningful behavior,
  mathematical/specification decisions, interfaces, physical assumptions,
  experiments/results, or completed work items. Group related commits into broad
  checkpoints; omit routine formatting and repeated unchanged status.
- Record the date, concrete change and reason, whether it is specification or
  implementation, verification/evidence, material limitations, and relevant
  revisions or review links. For the entry's own commit, Git file history provides
  the revision; do not add self-referential hash-only commits.
- Label pending work as pending. Preserve negative findings and original operating
  conditions; append a dated correction when historical interpretation changes.
  Keep full logs, artifact hashes and raw evidence in their existing review/local
  artifact locations, with links here. Update the system goal when active work or
  claim status changes; this changelog summarizes history rather than replacing it.
