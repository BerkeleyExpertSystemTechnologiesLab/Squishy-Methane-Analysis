# Formulation and scan-policy v1 freeze review

Date: 2026-09-13 UTC (2026-09-12 local). Status: **complete — specification freeze**.
This stage freezes specifications and prepares implementation; it does not change
the runtime estimator, controller, allocator, renderer, simulator, or evaluator.

Starting branch: `codex/sensing-core-recovery`.
Starting revision: `66f07700a754e5f9f25ae55d78f6423958b56d61`.
Specification producing revision: `ab80dc585c0972a1dc6c3a022bdedfd0e5e8b315`.
The following closeout commit changes only this review and system status.
Its final revision is recorded in the local `outputs/policy-v1-freeze/closeout.json`
after commit; obtain it from history with
`git log -1 --format=%H --grep="docs: record policy v1 freeze verification"`.
The starting worktree already contained user-discussion edits to `AGENTS.md`,
`README.md`, and `SYSTEM_GOAL.md`, two research Markdown memos, and local PDF
artifacts under `output/`. Relevant discussion/research is preserved in this
documentation work; generated PDFs, videos, and chat exports remain local.
The starting diff and copies are retained under `.tmp/policy-v1-freeze/` locally.

## Frozen decisions

| Decision | Result |
| --- | --- |
| Scientific purpose | Reconstruct the evolving projected beam-integrated enhancement field, eventually supporting a defined physical strength estimate |
| Primary score | Causal integrated squared reconstruction error over a common physical horizon; signed enhancement and fixed quadrature; latency/startup/idle time included |
| Estimator | Correlated finite Gaussian state, footprint likelihood, timestamped conditioning, declared linear advection/diffusion and process noise |
| Reference hardware | None selected; complete explicit instrument profile required for every experiment, immutable within comparisons |
| First tracking rule | `mass90_column_v1`: select 90% expected positive-signal mass, preserve selection through legal column closure, charge the full path |
| Repeated trimming | Not a requirement; no previous-mask shrink factor or routine budget-based deletion of selected cells |
| Unmet budget/coverage | Explicit tracking shortfall; a separately labeled whole-column age schedule may execute within the same budget |
| Replanning | After completion only; readings can update inference during a fixed path |
| Confidence bounds | Defined simultaneous-band diagnostic; UCB and supplementary allocation remain separate experiments |
| Renderer | Preserve current implementation; later edits only for demonstrated label/semantic defects |
| Evidence status | Historical held-out and recovery development results remain `not_demonstrated`; recorded-data expansion paused |

Authoritative files: [formulation/policy](../specs/reconstruction-policy-v1.md),
[instrument contract](../specs/instrument-profile-v1.md),
[structural schema](../specs/instrument-profile-v1.schema.json), and
[worked checks](../specs/freeze-v1-worked-examples.md). Research memos provide
background, not an alternative normative contract. Existing behavior remains
documented separately in [the recovery guide](../sensing-core.md).

## Repository and conformance audit

`corrected` below refers to documentation/specification changes in this stage.
`implementation required` means the current code must not claim v1 conformance.

| ID | Evidence at the starting implementation | Correction or handoff | Status |
| --- | --- | --- | --- |
| D1 | Guide loop still described a likelihood map and smooth credible-region masks without separating legacy and Gaussian paths | Define retrieved scalar, joint Gaussian state, expected-mass decision surface, reconstruction objective, and legacy labels in `AGENTS.md`, README, system goal, and package description | corrected |
| D2 | `docs/sensing-core.md` was the apparent authoritative contract while describing current packing/speed behavior | Keep it as the recovery implementation guide; link frozen v1 and explicitly identify nonconformance | corrected |
| D3 | System status said no bounded stage was active; recorded R0C commit/tests were labeled current | Record this bounded specification stage and distinguish historical recorded-pipeline results from current verification | corrected |
| P1 | `sensing_policy._allocate` greedily fits a subset of desired columns using gain per incremental cost | Add a named v1 selector/compiler path preserving the original prefix; bounded self-completion forecasts, candidate log, shortfall, explicit fallback; preserve existing recovery behavior under its original profile | implementation required |
| P2 | `physical_scan.compile_scan` takes max of travel, readout, and exposure durations, reducing actual speed for inconsistent inputs | Require commanded cadence plus detector capacity and actual speed; validate pitch-speed-cadence equality; explicitly charge transit and settling without changing sweep speed | implementation required |
| P3 | `ScannerCapabilities` has constructor defaults and only normalized-FOV coordinates; no bound retrieval convention, calibration identity, commanded cadence, or profile hash | Add required profile validation and run binding through an explicit session wrapper/transport version; retain current message decoding and CLI compatibility for recovery runs | implementation required |
| P4 | Reacquisition triggers uniform mass or wind error; no explicit overdue-age trigger. Current column priority uses the newest selected-cell timestamp (`max`), not the oldest row | Implement specified oldest-cell/rotation ordering, whole-column additions, stale trigger, visit semantics, startup completion, and explicit no-legal-action; verify behavior when data are missing | implementation required |
| T1 | `controller.predictive_view` can be requested before latest evidence availability while incorporating that evidence | Reject such causal-view queries or expose them only as a separately labeled retrospective API; reconstruct online histories by causal delivery replay | implementation required |
| T2 | Open-scan replay, idempotence, path lock, finite source handling, receipts and actual overrun have existing tests | Preserve them; add cross-scan cadence validation, partial-startup semantics, and common-time evaluator checks. Receipt availability must remain distinct from physical completion | retain and extend tests |
| M1 | `signal_belief` already uses correlated Gaussian conditioning, fixed gain/offset, footprint quadrature and explicit dynamics | Keep estimator mathematics; bind beam-integral units/calibration/approximation configuration. Audit edge clipping, residual independence, and projected-wind assumptions for each profile | partial implementation; calibration open |
| M2 | `mass_on` uses equal quadrature, a hardcoded tiny floor, and returns normalized mass only | Make mass floor and weights explicit, retain total expected index, and define unavailable ratios; support fixed declared quadrature for geometry and evaluation | implementation required |
| E1 | `sensing_evaluation.summarize_run` reports sweep coverage and five temporal coverage probes, not common-time reconstruction loss | Add evaluator-only causal replay and the frozen `J`/`J/T` metric, common horizon, fixed mesh/weights, latency/idle/startup accounting and null truth | implementation required |
| E2 | No joint posterior retained-fraction bound is reported | Add the specified deterministic simultaneous-band diagnostic, separate `gamma` and `delta`, null/uncertified cases and modeled-grid/time limitations | implementation required |
| V1 | Existing sensing tests check measurement visibility, labels, truth absence, physical/summary views and playback independence | Retain renderer. Feed new quantities through existing panels/report labels only when required; do not redesign videos for this freeze | retained; no rendering changes |
| B1 | Controller/inference/planning/message modules import only their core group; simulator/evaluator/renderer sit outside | Existing AST boundary test passes. Preserve model/messages -> policy/mechanics -> controller; adapters use controller; evaluator truth never enters core | retained |
| G1 | Broad prior/source-free/anchored assumptions and negative evidence exist, but new research purpose could be mistaken for completed implementation | Freeze purpose and version distinction; retain negative results and pause recorded expansion | corrected |

The audit does not assert that uncertainty calibration, plume lifetime,
physical strength, or adaptive superiority has been solved. The current
source-free model can carry signal too far downstream; the chosen Gaussian
transport class remains a declared approximation whose failure should be visible.

## Reproduced implementation gaps

Run the read-only [probe script](../../scripts/probe-policy-v1-gaps.py) against the
starting implementation (unchanged by this documentation stage):

```powershell
.\.venv\Scripts\python.exe scripts/probe-policy-v1-gaps.py --output outputs/policy-v1-freeze/current-gap-probes.json
```

The 4 x 4 fixture's selected corner cells contain 92% mass. The complete legal
path takes 9.5 s. Under a 6 s budget, the existing allocator returns selected
indices `[0,12]`, spends 4 s, and retains 46%. It already reports the final mass
shortfall through the normal planning path; the defect relative to the freeze is
the automatic subset allocation, not an accusation that the current code labels
46% as 90%.

A second probe specifies 0.5 FOV/s vertical speed and 1 Hz readout on four rows.
The current compiler executes the full-height pass at 0.25 FOV/s. The new profile
instead rejects this inconsistent fixed-speed/cadence combination.

Finally, after a reading acquired at 0.55 s arrives at 1.55 s, current
`predictive_view(0.55)` returns a state containing it. This does not prove an
existing benchmark leaked that reading; it demonstrates an API hazard that the
new causal reconstruction evaluator must not use.

## Focused implementation handoff

The next implementation stage is **planned, not active**. Use the frozen
specification as its acceptance contract. It can be one bounded implementation
goal with focused commits, without a broad renderer or repository rewrite.

1. **Profile and time semantics.** Implement profile structural/semantic validation
   and immutable run binding, distinguish commanded cadence from capacity, reject
   silent speed substitution, and close the causal-view timing gap. Preserve old
   message versions/CLI profiles explicitly; do not silently upgrade historical
   recordings or choose a numerical hardware reference.
2. **Coverage-preserving policy.** Add `mass90_column_v1` alongside recovery's
   allocator. Implement selection tolerance, closure/order, bounded forecast
   iteration, own-completion validation, diagnostic candidates, and fully charged
   whole-column age fallback. Confirm stable, expanding, displaced, and shrinking
   regions; no automatic trimming under unchanged evidence.
3. **Evaluation and confidence.** Add the causal common-time reconstruction
   evaluator and the defined bound diagnostic. Keep truth in the evaluator and
   observations in the controller. Preserve acquisition versus availability and
   actual execution deviations in replay; evaluate prior/startup/idle periods too.
4. **Development comparison and inspection.** With complete named synthetic
   profiles, compare the frozen adaptive policy, static corridor, and simple
   rescan under common horizons and budget ceilings. Use continuous signal,
   known wind, wind error, source-free startup and finite-source cases. Compare
   rendering alone with identical traces; compare allocation using common source
   realizations and mechanics, with actual path-dependent readings. Review existing
   physical-time/summary views with only necessary semantic/label corrections.
5. **Acceptance and closeout.** Run numerical, causal, mechanical, serialization,
   adapter, renderer and dependency tests plus the full suite. Record actual
   expenditure, error, shortfalls, uncertainty behavior, artifact hashes, producing
   revisions and unresolved model failures. Preserve negative findings and leave
   hardware, real-data superiority, within-sweep replanning and calibrated emissions
   outside the goal.

Minimum new checks: full 90% selection survives closure; an over-budget selected
region is not quietly replaced by a tracking subset; narrow footprints/pitch and
speed remain fixed; skipped-column travel and reacquisition are charged; cycling
forecasts retain diagnostics; no meaningful signal produces an unavailable mass
ratio; oldest unvisited cells affect reacquisition; partial full-scene startup is
not mislabeled complete; delayed evidence cannot improve past online error;
batch/stream completed inference still agrees; actual overruns and between-scan
readout spacing are respected; null truth remains null; changing display/FPS
leaves operational traces and evaluation quadrature unchanged.

An implementation stage is complete when these definitions work and are
inspectable. Adaptive advantage remains an experimental result; do not weaken
conformance or change the objective to obtain one.

## Verification and artifact record

Verification results and producing revisions are recorded below.
No new benchmark or video was generated or visually reviewed in this stage.
Existing renderer regression tests ran; historical video inspection remains in
[the recovery evidence](sensing-core-recovery.md) and
[the video review](2026-09-12-video-evidence.md).

Reproducible commands from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/check-policy-v1-spec.py --output outputs/policy-v1-freeze/worked-checks.json
.\.venv\Scripts\python.exe scripts/probe-policy-v1-gaps.py --output outputs/policy-v1-freeze/current-gap-probes.json
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src tests scripts/check-policy-v1-spec.py scripts/probe-policy-v1-gaps.py
git diff --check
```

The independent structural-schema check uses `jsonschema` in the existing system
Python environment, not a new project/runtime dependency:

```powershell
python -c "import json; from pathlib import Path; from jsonschema import Draft202012Validator; p=Path('docs/specs'); s=json.loads((p/'instrument-profile-v1.schema.json').read_text()); Draft202012Validator.check_schema(s); Draft202012Validator(s).validate(json.loads((p/'instrument-profile-v1.example.json').read_text())); print('schema and example valid')"
```

The schema test also rejects an unknown field and a missing required acquisition
object. Cross-field cadence/speed/integration checks are independently exercised
by the arithmetic script; JSON Schema alone does not validate those relations.

### Closeout results

| Check | Result |
| --- | --- |
| Independent worked arithmetic | Nine groups passed: instrument consistency, scalar/correlation/zero/repetition, sparse footprint, dynamic batch equivalence, stable/changing selection, full closure/budget, age schedule, confidence interpretation, causal latency loss |
| Sequential vs augmented-joint batch | Maximum absolute difference `2.220446049250313e-16`; positive posterior eigenvalues in the worked dynamic case |
| Current implementation probes | Three gaps reproduced; no claim of repaired code |
| Full existing suite | `335 passed, 5 skipped in 22.85s`; skips concern file/video/directory symlink capability on Windows |
| Import boundary | Existing `test_controller_dependency_boundary` passed within the full suite; no source dependency changes |
| Schema | Draft 2020-12 metaschema and example valid; unknown and missing required fields rejected; system Python already had `jsonschema` |
| Compilation | `compileall` passed for `src`, `tests`, and the two added verification scripts |
| Documentation and Git | Local Markdown links and code-fence balance checked; staged/unstaged diff checks passed; `src`, `tests`, and `examples` identical to starting revision |
| Generated evidence | Two small local JSON checks plus a verification manifest; no new media/performance artifacts; prior PDFs remain untracked |

Raw local artifact SHA-256 values on the verification environment:

| Artifact | SHA-256 |
| --- | --- |
| `outputs/policy-v1-freeze/worked-checks.json` | `9d0010811c18c2559474c1b39bf988a50a2819899506ffe5ecac0b80df20bf11` |
| `outputs/policy-v1-freeze/current-gap-probes.json` | `affeca5b790acc28c657741b18eb87d6af5cbd371d8c7929a88dae23493835e7` |
| `docs/specs/reconstruction-policy-v1.md` | `cd4cf513ac617fa25963e0635e56b6a72081f1f859c4fd8bfeef4ad38da5af03` |
| `docs/specs/instrument-profile-v1.schema.json` | `ca02a111421caca7b88c60430aa67253cf3c578344d545a2d13fbcbbc9c03601` |
| Example instrument canonical JSON | `5d7d3de1d969eb1098cfd6bde553707d1f4fec27b472e629976b4cbd82039e42` |

Other specification/script raw hashes are in local
`outputs/policy-v1-freeze/verification-manifest.json`. Raw file hashes depend on
checkout line endings; the example profile's canonical JSON digest is separately
identified. The source documents and scripts are committed, generated evidence
remains local, and nothing was pushed remotely.

Resolved in this stage: reconstruction purpose and causal score; retrieved-scalar
semantics; covariance versus scan mass; complete per-experiment instrument
settings; coverage-preserving selection and explicit fallback; confidence-bound
scope; versioned handoff and aligned guidance. Remaining: the implementation gaps
in the table, physically characterized profiles, calibration and model adequacy,
and whether adaptive scans improve reconstruction. No scientific claim status
was upgraded. Recorded-data expansion stays paused. The next implementation
stage is planned and has not begun.
