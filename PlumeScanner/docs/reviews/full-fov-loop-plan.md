# Full-FoV startup and rescan: lean implementation plan

Status: implemented and verified; see the [closeout review](full-fov-loop-implementation.md).
User clarification supersedes
the next-step recommendation to start with estimator calibration. Inspected
revision: `876df4df9ca1e6bf51f8f409e208e533d3dd619d`; planning branch:
`codex/full-fov-loop-plan`. Existing `output/` and completed studies are preserved.
The user requests focused commits and remote pushes during this work. Push the
working branch after verification; do not force-push, merge, or include local
generated media and experiment artifacts.

## Required behavior

Every run begins with a complete physical scan of the full FoV from the declared
starting pose. Only after its actual readings and completion receipt may the
controller enable focus. After a completed focused scan, supported signal keeps
focus enabled; absent support causes a charged return to the starting pose and
another complete full-FoV scan. A focused region may move, grow, remain stable
or shrink. The belief is retained across both modes; returning home is not a
belief reset. No oracle source location or truth is added.

Retain the current strict inferred-mean signal gate above 0.15 and latent upper
0.95-quantile region as the initial focused-selection rule. This keeps the
operating-loop correction separate from another threshold or estimator change.
Use Gaussian inference first. Mixture integration is deferred until the loop
has been demonstrated; its historical implementation remains available.

The new demonstration has no prior-startup bypass, age-block fallback, periodic
service detour or risk ranking. Those remain identifiable historical policies.
A mean gate with an empty/infeasible focused region requests a full rescan with
an explicit reason. Budget exhaustion is a distinct incomplete/no-action result.

```text
FULL SCAN (return home first when necessary)
  -> actual full completion -> evaluate signal
       supported -> FOCUSED SCAN -> actual completion -> evaluate signal
       absent    -> FULL SCAN (return home first)
```

The API lifecycle READY/SCANNING is separate from FULL/FOCUS operating mode.
Assimilation continues during a scan; neither mode nor its committed geometry
changes before completion. Partial execution cannot satisfy full-scan completion.
Use a fresh coverage epoch for every full scan, not the all-time visit mask.
Track a pending full-scan path/cursor and missing actual samples if interrupted;
resume that obligation without enabling focus. Invalid/missing evidence remains
missing. Returning from an interrupted action uses the actual receipt pose.

## What the audit found

| Current code | Mismatch or unnecessary coupling | Focused correction |
|---|---|---|
| `sensing_policy_v5.survey` | Age-ranked blocks from the current pose replace full rescanning | Explicit return-home plus full-FoV action |
| Doubled run manifests | Prior startup can focus after partial coverage | Full-scan startup required for the new entry point |
| Controller `startup_complete` | All-time visits are insufficient to certify a repeated sweep | Per-full-scan actual completion state |
| `PolicyConfigV5(PolicyConfigV4)` | Inherits risk integration/history and mass settings | Independent minimal loop settings, composed with model/instrument settings |
| `ScanPlanV4(ScanPlanV3)` | Carries unused risk fields and inactive `target_mass=0.90` | Neutral physical plan plus small mode/threshold diagnostics |
| Controller version checks | Concrete policy generations enter lifecycle methods | Resolve a policy implementation once in a compatibility factory |
| Physical compiler | Useful motion math returns a policy-laden v2 message | Extract a neutral path result; old wrappers preserve exact historical messages |

The controller already isolates truth, rendering and filesystem access. Its
acquisition-order replay, duplicate handling, delivery clocks and detector
accounting are useful safeguards, not clutter to delete.

## Lean repository representation

Keep a single implementation of each essential responsibility:

| Responsibility | Owns | Reuse / change |
|---|---|---|
| Signal belief | Mean, covariance, propagation and conditioning | Existing `signal_belief` and distribution projection |
| Loop policy | Full/focus mode, threshold decision, full-scan obligation | Small independent module with plain immutable settings |
| Physical planning | Return path, column passes, exposures, cadence and duration | Existing `physical_scan` math behind neutral path output |
| Acquisition controller | Active action, reading identities, clocks, actual pose and receipts | Extract/reuse current lifecycle logic, without policy-version branches |
| Adapters and observers | Simulation, traces, evaluation and existing panels | Keep outside runtime decision logic |

The public operations remain `plan_next`, `ingest_readings`, `finish_scan`.
New configuration composes instrument, estimator and loop settings; it does not
subclass the v4/v5 experiment hierarchy. A versioned factory maps old session
types to their existing behavior. Do not move or delete old modules wholesale.
Do not build a generic plugin framework or another duplicate controller.

Represent return-home and the following full scan as one ordered physical plan
with a non-acquiring motion prefix. This avoids adding a separate hardware
operation merely to demonstrate the loop. The trace labels return versus
acquisition, and charges both. A caller cannot teleport the initial pose.
Motion-only interruption receipts still need honest state handling; they do not
complete the full-scan obligation.

A new plan contains mode, reason, physical path/timing, gate value, qualifying
cells and omissions. No mass target, risk candidate array or surveillance state
belongs in its normal payload. Historical wrappers keep those old fields intact.

## Timing and noise controls

Current full-FoV paths from home take 33.575 s (20x32) and 129.5875 s (40x64),
before any later return-home travel. Both use 20 readings/s; the doubled profile
has 2,560 physical readings. A 75 s run cannot establish doubled full startup.

Separate the 12 s focused-scan ceiling from full-scan execution time. A full scan
receives its actual compiled time plus return/detector/receipt costs, subject to
the hard remaining session duration. If it cannot finish within that duration,
report incomplete startup/rescan; never substitute age blocks or claim a full
scan. Use a declared longer horizon (initial fixture: 360 s) to observe repeated
full scans in both profiles. Keep speed, cadence, footprint and signal unchanged.

Start with simulated measurement noise zero, while explicitly retaining the
existing assumed likelihood noise 0.05 and dynamics. This is a mechanism check,
not a claim of a perfectly calibrated noiseless estimator. Once the sequence
passes, repeat with simulated noise 0.01 and 0.05 and the same assumed model.
Record both quantities independently; do not silently reduce process uncertainty
or fit the estimator. Compare clean exposure response, added noise and retrieved
reading numerically. Preserve the renderer layout and fixed intensity scales;
the normalized excess diagnostic must not be presented as sensor noise.

## Bounded implementation order and acceptance

1. Extract neutral physical-path and acquisition-lifecycle seams with historical
   plan/trace equivalence checks. Existing fixtures, message decoding and hashes
   must survive. No estimator change.
2. Add the minimal independent full/focus loop binding, per-sweep completion and
   charged home return. Write state-transition and interrupted-execution tests.
3. Run zero-noise continuous-plume and blank-source demonstrations on both
   profiles from time zero. Add a separately labelled signal-cessation control
   to expose the focus-to-full transition; do not alter the main plume fixture.
   If persistent inferred signal prevents that transition, report it rather than
   resetting the belief. Test deterministic gate crossings independently.
4. After the loop passes, repeat these declared fixtures at noise 0.01/0.05.
   Keep settings fixed across noise comparisons. No new broad superiority suite.
5. Review physical-time video from initialization through full completion and
   subsequent decisions. Check return travel, every actual visit, fixed path,
   scalar readings versus clean responses, zero/focus/full mode labels and end
   conditions. Report J/T and spending descriptively, after mechanism conformance.

Required safeguards: no early focus, no age fallback, full-scene rescan after gate
failure, no belief reset, no uncharged return, no missing-evidence completion,
causal delayed delivery, idempotent duplicates, actual overruns, read-only queries,
fixed mechanics, display independence, public API/fake-scanner agreement and
historical regression/import/diff checks. Keep tests and data outside the lean
runtime; retain historical negative findings. Update status/changelog together.

The exit result is a readable implementation of the user's actual scan sequence
and an inspectable noise ablation. It does not promise improved reconstruction,
calibrated bounds, real hardware operation or a new policy freeze.
