# External coding-agent handoff: test an evidence gate for focused scanning

Status: implementation assignment prepared on 2026-09-19; NOT implemented or validated.
Read this document and the sibling `evidence-gated-focus-context.json` before coding.
They are self-contained task context for an agent with a checkout of this repository
and no conversation history. The JSON contains complete reference configurations,
baseline results and provenance. Summary statistics are not sufficient to reconstruct
raw observations: regenerate missing traces as instructed below.

Context package SHA256 at preparation:
`61a97b0916e094bf92760a968a5ff3dfdc51d02a8eb23b5b038494a08eaa53d6`.
This is a byte hash of the prepared file; Git line-ending conversion can alter it.
If bytes differ, inspect parsed content and Git provenance before accepting reuse;
do not treat an unexplained mismatch as compatible. Historical source byte hashes
are provenance of the original Windows artifacts, not universally portable digests.

## Assignment and completion meaning

Implement and evaluate one opt-in policy, `evidence_gated_focus_loop_v3_candidate`:

**Complete full-FoV scan -> reading-supported focus -> loss of reading support ->
charged return home and another complete full-FoV scan.**

Separate two decisions: evidence controls whether to focus; the existing predicted
upper-signal field controls where to focus. A focused region can grow, shrink, move
or remain stable. Broadening in this round means returning to the complete full-FoV
scan. Do not introduce an intermediate padded confirmation region or a third geometry.
The full scan itself confirms absence or finds displaced signal.

Keep the calibrated Gaussian estimator unchanged. Record its continuing false source
predictions explicitly. Source-activity inference remains the next model experiment;
do not implement it simultaneously, switch source injection off using this gate,
reset belief, or suppress false signal in the displayed map. This isolates whether
an observation-based gate repairs operating behavior independently of map calibration.

Completion means verified implementation, the bounded experiment, honest results and
committed/pushed documentation. It does NOT require positive scientific findings.
Do not silently tune the rule to make the experiments pass or promote it to default.

## Repository and collaboration contract

- Repository: `C:\Users\05mus\Documents\New project 2`; Git remote
  `https://github.com/WinChurchill/plume-focus.git` (`origin`). Other checkout paths
  are acceptable. Use repo-relative paths in code, not this machine's absolute path.
- Inspected implementation base: `2e16471aa06fe5cbb59ecb2c945402cb9c2233e1` on
  `codex/scanner-baseline-calibration`. This handoff is a documentation-only descendant
  on `codex/evidence-gate-handoff`. Inspect current branch, HEAD, status and intervening
  diff. Start `codex/evidence-gated-focus` from the checkout containing this handoff;
  do not reset away intervening work. Record the actual starting revision.
- Read `AGENTS.md`, `CHANGELOG.md`, `SYSTEM_GOAL.md`, the calibration review/spec and
  `docs/sensing-core.md`. This assignment supersedes the earlier proposal to implement
  source-activity inference immediately: first isolate the new mode gate.
- Preserve pre-existing untracked `output/`. Generated study evidence belongs under
  ignored `outputs/evidence-gated-focus/`; never commit videos, generated traces or
  conversation exports. Commit compact frozen configs and source documentation.
- User has authorized focused Git commits AND pushes to origin. Stage explicit files;
  no force push, history rewriting, merge, deletion of unrelated artifacts or blanket
  `git add .`. Confirm branch tracking and remote tip at closeout.
- Windows/PowerShell is the reference platform. Use `.\.venv\Scripts\python.exe`, set
  `OPENBLAS_NUM_THREADS=1`; use bounded workers, and do not kill unrelated processes.
  No subagents unless separately authorized. No new dependency without a concrete need.
- Update changelog, status and implementation guide at each meaningful checkpoint.
  Preserve historical `not_demonstrated` outcomes and exact historical config decoding,
  identifiers and hashes. Do not rewrite old results to the new interpretation.

## Problem, evidence and limitations

The instrument retrieves a beam-integrated gas scalar sequentially. It is not an
instantaneous camera or a point concentration sensor. Reconstruction over physical
time is the objective; enhancement-index error is a separate outcome, not emissions.

Calibration completed 36 development and 72 held-out 360-second runs. All 108 replay;
520 tests pass with five Windows symlink skips. Four evaluation refinements change
loss by only .114-.363%; inference-grid probes still show 1.1-3.9% sensitivity.

Frozen calibration: half-life 6 seconds, source width .035 FOV, source rate
.2829738193616142 synthetic units/s, vertical margin zero. Against the old model,
J/T improves 11.60-40.76% in all 24 held-out pairs. Mean J/T across cases is
.0105393083 old / .0086905663 calibrated / .0116284374 repeated full scan.
The full-scan comparator used the OLD estimator, so it does not isolate allocation.
Common horizons were not exactly matched spending.

Doubled continuous core coverage improves from 82.08/84.57% to nearly 100%; actual
core readings increase from 347/353 to 537/529. Coarse core readings increase from
251/268 to 434/418. Wider-scene age worsens. Four enhancement-index MSE comparisons
worsen: coarse continuous 17-18%, coarse wind error 52-54%.

Every calibrated blank case falsely focuses; coarse has 228/238 focused sweeps,
doubled 48/48, and each run only one full scan. None of four cessation cases begins
a full scan after cessation. Near-zero readings coexist with a positive source peak.
The propagation code adds configured source input every step; conditioning never
learns source activity. The existing focus gate tests the posterior maximum > .15.
Thus a recreated peak can keep the gate on despite observations supporting absence.
This is a specific model/decision interaction, not proof of an assimilation bug.
Blank failures also occur without simulated noise.

Padding alone failed: doubled development core coverage drops from about 86% to
51-53% because vertical padding consumes time otherwise spent on horizontal extent.
Keep zero padding. Do not resume margin tuning, raise resolution or add minimum height.

## Frozen physical and model settings

Use the full serialized references in the sibling JSON or
`configs/scanner-calibration/{coarse,double}-frozen-candidate.json`. Do not rely on
constructor defaults. They include the source realization generator, wind and calibration.

| Setting | Coarse | Doubled |
|---|---:|---:|
| Physical lattice | 20 x 32 | 40 x 64 |
| Footprint y x x, normalized FOV | .05 x .03125 | .025 x .015625 |
| Cadence / capacity | 20 / 20 Hz | 20 / 20 Hz |
| Integration | .025 s | .025 s |
| Vertical acquiring/transit speed | 1 FOV/s | .5 FOV/s |
| Horizontal speed | .625 FOV/s | .625 FOV/s |
| Settling | 0 | 0 |
| Initial complete full scan | 33.575 s | 129.5875 s |

Both profiles are synthetic, not hardware calibration. Coordinates are normalized
FOV `(y,x)`, home `(0,0)`, seconds from session origin. Focus ceiling 12 s including
declared delivery/receipt delay; minimum four readings. Full scans use their complete
physical duration, never the 12-second focus ceiling. Remaining experiment duration
still limits legal actions. Never truncate a full-scan obligation into focus.

Model: 12 x 16 grid; gain 1, offset 0, assumed detector noise .05; initial mean 0,
initial std .2, correlation length .08 FOV, process std .01/sqrt(s), diffusivity
.00005, max propagation step .5 s, explicit `zero_inflow_v1`. Preserve the calibrated
decay/source values above. Forecast eligibility remains the .95 latent upper quantile
at the midpoint of the focus ceiling, threshold .15. Actual simulated noise is
separately declared and must not leak into gate/inference configuration.

## Exact experimental gate (new proposal, not an established calibrated detector)

The gate uses actual delivered scalar readings, not posterior maxima, predicted
source activity, inferred raster pixels, evaluator truth or wind-derived pseudo-readings.
Use the same fixed calibration as assimilation:

`x_i = (reading.value - offset) / gain`, `s = sensor_noise_std / gain`.

For an opened physical plan with M scheduled scalar acquisitions, freeze M before
execution. On successful finish, form bounds for each unique delivered reading:

`z = NormalQuantile(1 - alpha_action/(2*M))`

`L_i = x_i - z*s`, `U_i = x_i + z*s`.

These are Bonferroni measurement-error bounds on each clean exposure response,
conditional on the declared residual-noise model. They are NOT calibrated field
posterior bounds, instantaneous full-scene bounds, session-wide error guarantees or
tests of whether the emission source exists. No independence of neighboring signal
values is required for the union bound. Correlated/biased detector errors still need
valid marginal error assumptions; test rather than assert those for real instruments.
Do not average spatially different readings as repeated observations of one value.

Freeze `alpha=.05`, entry level `.15`, retain level `.10`. The lower retain level is
explicit hysteresis, not a changed map reporting threshold. These settings are a
predeclared candidate, not fitted calibration. Flag the conservative penalty with
large M: weak signal may fail entry. Do not remove multiplicity correction silently.

State transitions:

1. Start with mandatory FULL coverage obligation. No prior bypass. Maintain fresh
   actual-cell coverage for each full-scan epoch; wide footprints do not invent visits.
2. At complete FULL epoch, enter focus only if at least one accepted `L_i > .15`
   from that epoch. Otherwise start a new charged return-home/FULL epoch. Historical
   support from previous epochs cannot reopen focus.
3. After a fully executed focused sweep, retain focus only if at least one fresh
   `L_i > .10` from that sweep. Otherwise start a fresh return-home/FULL obligation
   at the next decision. Do not add a grace loop or wait for the predicted mean to fall.
4. If support exists but the existing focused compiler finds no legal focused region,
   use the same charged FULL fallback. A gate never overrides physical feasibility.
5. Once FULL is required, complete its fresh coverage before focus may resume,
   even if intermediate readings become positive. At session end, return explicit
   `incomplete_full_scan_budget` if the remaining time cannot fit the obligation.
6. Aborted/partially executed FOCUS cannot renew focus; conservatively require FULL.
   Partial FULL preserves missing cells and resumes the existing coverage obligation.
   A missing receipt does not close a scan. Invalid/conflicting evidence must reject
   atomically without changing gate or acquisition state.

For a full epoch spanning multiple partial physical plans, action p=1,2,... receives
`alpha_action = .05/[p*(p+1)]`; this sums to at most .05 across that epoch. Persist
the union of accepted evidence bounds and actual coverage across those plans. A
normal one-plan FULL uses .025. Each focused sweep uses `.05`; no session-wide
guarantee is claimed. Allocate the action budget only when a nonempty plan opens,
not on repeated queries/no-action results. Use M scheduled slots even after partial
execution or missing readings. Validate one accepted observation per scheduled slot
for this gate. Extra/unmappable readings may remain valid inference evidence under
the existing API but cannot provide gate support; record a gate-contract violation
and conservatively request FULL. Do not weaken production ingestion validation.

Magnitude check: at assumed noise .05, entry from a normal complete FULL requires
at least one calibrated observed value above approximately .35565 (640 readings)
or .37111 (2560 readings). The underlying signal level being tested is still .15;
the difference is the conservative noise/multiplicity allowance. This is a potentially
important weak-signal failure mechanism, not a hidden new physical threshold. Record
the effective observed-value cutoff in diagnostics. The .10 retain level is a proposed
engineering setting, not a statistically fitted optimum.

Diagnostic evidence states: `supported` if a lower bound crosses the applicable
level; `below_level_on_observed_footprints` if all upper bounds are strictly below
it; otherwise `inconclusive`; empty evidence is `unavailable`. The last three all
request FULL, but must remain distinct labels. A focused sweep cannot prove whole-
scene absence. Comparisons use strict > for support; equality is not support.

Full-scan evidence is sequential and can be stale at completion. Record supporting
reading age explicitly. Do not relabel it current-time detection. A signal observed
early in a long full scan may justify one subsequent focus attempt; that focus must
renew support using its own evidence. This is a declared limitation to test, not
permission to reuse that early reading indefinitely. Anticipated real hardware may
need a separate confirmation mechanism later; do not add it in this round.

## Engineering boundaries and implementation sequence

Existing seams:

- `signal_belief.py` and `calibrated_scanner.py`: Gaussian propagation, scalar
  conditioning and calibrated decay. Leave their mathematics/configuration unchanged.
- `full_fov_loop.py`: FULL/FOCUS lifecycle, current mean gate, predictive region and
  longest feasible prefix. Extract only minimal shared hooks if needed; historical
  defaults must retain identical traces. Do not copy the entire controller.
- `acquisition.py`: active plan, atomic evidence identities, acquisition-order replay,
  `_finish_evidence`, `_close_scan`. `_pending` is cleared on close; collect gate
  evidence only after receipt validation succeeds and before close, without breaking
  transactional failure behavior. Do not count replayed assimilation events twice.
- `physical_scan.py`: legal paths/cadence/travel; reuse unchanged.
- `controller.py`: `SensingController.from_session(...)` dispatch.
- `full_fov_execution.py`, `calibration_execution.py`: configuration/plan decoding,
  simulator, causal replay. Add typed versioned bindings alongside historical ones.
- `calibration_evaluation.py`, `calibration_truth_grid.py`, counterfactual utilities:
  evaluator-side loss and detached forecasts; truth never enters gate/planner.
- `scripts/render-full-fov-demo.py`: existing panels. Only necessary mode/gate/units
  labels; no renderer redesign, no map suppression.

A. Write the candidate specification, immutable settings/session/plan transport and
pure gate-statistics function. Use separately named messages/schema identifiers.
Preserve the estimator formulation ID. State includes full epoch evidence, action
index, gate decision provenance, coverage and support ages; support snapshots/clones
and serialization/replay. Freeze selected confidence parameters explicitly.

B. Integrate the state transitions without changing `plan_next`, `ingest_readings`,
`finish_scan`. Preserve immutable active paths and actual timing. A mode decision
changes only at completed-scan boundaries; repeated read-only queries are inert.
Streaming and batch delivery of identical available evidence give the same finished
gate and belief; different delivery schedules can change online error legitimately.

C. Add resumable declaration/verification/execution/reporting commands with checksums.
Freeze manifests before viewing new experiment outcomes. Record each producing
revision and code/config hashes. Do not permit reuse merely because filenames match.

D. Execute the declared experiments, numerical refinement and visual checks. Write a
review with separate claims for behavior, reconstruction and remaining model failure.
Commit each checkpoint with changelog/status updates, push the branch, verify remote.

Plan/receipt-linked diagnostics must include old mean-gate value, new evidence state,
entry/retain level, alpha allocation, M/n, max L/max U, support reading identities,
acquisition/availability/age, epoch, requested and swept cells, fallback reason,
actual completion and whether full coverage is incomplete. No NaN/Infinity JSON;
use explicit unavailable/null. Do not call gate support `source_active_probability`.

## Bounded experimental procedure

### 1. Fixed-evidence attribution before closed-loop comparisons

Use the 12 saved calibrated seed-101 histories (two profiles x six scenarios) if
locally available and hash-compatible. Otherwise regenerate from the complete
reference configurations and existing deterministic source/noise adapter. Historical
outputs are local evidence, not a mandatory external dependency. Do not fabricate
missing raw readings from summary statistics.

Replay identical readings and delivery events, audit the old mean gate versus the
new per-sweep evidence statistics. This is a shadow decision audit, not an executed
alternative and not an estimate of alternative reconstruction loss. Verify that the
new gate cannot change belief on identical evidence. Report mean-positive/evidence-
unsupported disagreement, supporting-reading ages and prospective transition times.
For historical repeated FULL epochs, use their actual epoch/plan identities.

### 2. Development/mechanism executions: 32 records

Two profiles x four cases (continuous, blank, cessation at180, intermittent active
[0,120) and [240,360)) x two simulated noise values (0,.05), seed0 x two policies
(unchanged calibrated mean-gated loop, new evidence-gated loop). Duration360 s.
Source phases use the existing seed0 reference generator convention. No policy
parameter search. Noise-free controls must be examined before blaming stochasticity.

### 3. Fresh held-out executions: 72 records

Two profiles x six cases x three fresh seed/noise pairs x two policies.
Cases: continuous known wind; continuous supplied-wind error (.003,-.004) FOV/s;
weak continuous at .3 amplitude; blank with the same asset-location assumption;
cessation180; intermittent [0,120),[240,360). Seed/noise pairs are (201,.01),
(202,.05),(203,.05). Source phases are the first three draws from
`default_rng(seed).uniform(0,2*pi,3)`, assigned in the existing order. Keep truth,
noise coupling and mechanics common between each pair; paths get their own readings.
These seeds provide descriptive results, not population confidence claims. Noise and
source phases are not fully crossed, so do not infer an isolated noise effect here.

Both policies use exactly the same frozen calibrated estimator. No old-estimator
full-scan comparator is needed for this gate attribution. Do not expand the suite
automatically. Baseline104 executions (32+72) are new unless exact input/evaluation
compatibility of an existing record is verified. Up to12 historical shadow histories
may need regeneration, separately identified; they are not fresh held-out evidence.

Declare held-out manifests before development outcomes. Freeze all code/configuration
that can change operational behavior before held-out execution. Fixes discovered
there require explicit invalidation and rerun of affected paired records; no quiet
retuning. Use source-window truth only in the adapter/evaluator. No source activity
or cessation times in controller inputs. Identical session hashes across scenarios
except declared causal wind inputs help audit this boundary.

### 4. Metrics and interpretation

Primary: common-horizon causal J/T, including startup, idle, travel and delivery.
Separate signed enhancement-index error/MSE, actual spending, core readings,
geometric core coverage, ages, gate switches, FULL completion and return-home cost.
Bright core remains clean signal > max(.15, .5*instantaneous field maximum).
Focus-only core coverage is unavailable when there is no focus, not 100%; report
whole-horizon core sampling too so avoiding focus cannot win by dropping denominators.

For cessation/intermittency report truth-change-to-first-FULL latency and separate
time to completed FULL; supporting observation age; unavailable due to remaining
budget; and whether the first fully background epoch refocuses. A FULL straddling
cessation is not an all-background scan. Distinguish intended FULL with insufficient
time from actually executed FULL. Measure reacquisition latency after signal returns.

Report persistent posterior peaks separately even when the gate correctly refuses
focus. Compare old/new means on identical evidence and actual closed-loop means on
their respective evidence; do not claim that a changed scan path repairs source
inference. Source-injection correction is explicitly out of scope.

Use 120x160/.05s loss quadrature; refine four new-gate cases (both profiles,
continuous and blank) to240x320/.025s, then480x640/.0125s only if changes exceed
max(1e-10,1% reference loss). If a decisive cessation effect is near tolerance,
refine that comparison too. Unresolved comparisons remain unresolved at the limit.

Predeclared scientific criteria (not conditions for declaring implementation done):
- Behavioral support: noise-free blank never focuses; after unsupported focused
  evidence the next legal action is FULL; completed all-background FULL does not
  immediately refocus in noise-free controls. Report every noisy false entry rather
  than treating per-action alpha as a session guarantee.
- Useful operation in tested held-out cases: no blank false-focus action; every
  cessation case with sufficient remaining time executes a post-cessation FULL;
  every intermittent case with sufficient scan opportunity reacquires genuine
  signal. If any fail, classify behavior as mixed/unsupported with exact cases.
- Preserve continuous operation: per-profile aggregate continuous known/error-wind
  J/T increases by no more than5% against the calibrated control and continuous
  core geometric coverage decreases by no more than5 percentage points. Missing
  focus coverage does not pass. Report weak-signal failures separately, never hide
  them inside full-strength averages. These are engineering tolerances, not proof
  of superiority or statistical significance.
- Call reconstruction improved only beyond numerical error and1% comparator loss;
  otherwise describe equivalent/mixed/worse. Behavior can improve while J/T worsens.

## Acceptance tests and completion contract

Required automated checks:
1. Independent gate arithmetic, one/many readings, calibration gain/offset, equality
   at boundaries, negative/zero values, hysteresis, multiplicity and full-epoch
   alpha accounting. Test noise-free and known-Gaussian measurement fixtures;
   do not assert empirical uncertainty calibration from a small plume suite.
2. Posterior-positive/reading-zero cannot enable focus; strong evidence can enable
   the gate even when old mean gate fails (geometry may still be infeasible).
3. Complete startup, fresh full epoch, partial coverage, no repeated credit from
   duplicates or expanded display pixels, no indefinite renewal by stale evidence.
4. Ordered/delayed batches, duplicate idempotence, atomic conflict/receipt rejection,
   partial/zero execution, overruns, cross-scan cadence, immutable active plan,
   closed-scan evidence rejection, and no-action near the terminal horizon.
5. Same-evidence estimator equivalence; unchanged old paths/hashes; serialized
   session/plan/gate-state round trips; fake-scanner/public simulator replay.
6. Truth/import isolation, detached-query immutability and display/playback independence.
7. Full regression suite, import checks, compileall and `git diff --check`.

Reference commands (implement the new driver, do not present nonexistent commands as
already working):

```powershell
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase declare
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase verify
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase shadow
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase development --workers 2
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase freeze
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase heldout --workers 2
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase report
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts
git diff --check
```

Phases must be resumable, immutable/hash-checked and clearly distinguish missing,
failed, incomplete and completed work. Reporting verifies all104 execution records,
public-call/loss replay, numerical checks and artifact hashes, or fails with a precise
inventory. Do not silently drop failed records or reuse historical code-hash waivers.

Inspect existing-layout physical-time and compressed-summary pairs for new-gate
coarse/doubled continuous, coarse blank, doubled cessation and coarse intermittent
(five pairs). Decode all frames; record the actual keyframes reviewed, including
gate exits, home travel, full completion and re-entry. One alternate playback test
must preserve operational hashes. Review original/candidate mode timelines together.

Final deliverables: versioned code/configs/tests; candidate specification;
`docs/reviews/evidence-gated-focus.md`; synchronized CHANGELOG/SYSTEM_GOAL/AGENTS and
implementation guide; frozen manifests, numerical tables, traces, reviewed media,
hash inventory, exact reproduction commands, start/final/producing revisions,
focused commits and verified push. Report engineering completion independently of
behavioral usefulness, reconstruction and strength outcomes. Leave candidate opt-in.

Handoff the smallest next test: source-activity inference if maps remain false despite
correct mode recovery; gate sensitivity if conservative bounds lose real signal;
scan confirmation geometry only if stale/full-scan evidence causes demonstrated
reacquisition failures. No automatic broad redesign or robotics integration.

Out of scope: vision models, UCB/risk allocation redesign, mixture-estimator extension,
new spatial drift, covariance inflation, source-weight resetting, source inference,
new margin/resolution tuning, hardware/network drivers, robotics or simulator choice.
