# Mechanism diagnosis and experimental v2 revision

Status: bounded diagnosis, implementation, development comparison and visual
review complete; candidate remains experimental. Started on 2026-09-13 from `a341e68113c74bbdf6eb8b1069941f60fd150d7e`,
branch `codex/mechanism-grounded-revision`. User `output/` files are preserved.
Frozen v1, its 36-run evidence and `not_demonstrated` finding remain unchanged.

## Diagnosis

The user's video inspection exposed a mismatch between executable tracking
capabilities and actual demo behavior. The compiler supports both horizontal
selection and vertical endpoints. After startup, however, every v1 adaptive action
was a full-height fallback. Its first action scanned 10 full columns in 11.55 s;
the fixed corridor scanned 21 columns at 40% height in 10.15 s. Column scanning
does not imply full-height scanning. The old result tests those implemented rules,
not successful two-axis focusing of the plume.

The second concern was noise accumulating as scan weight. At the first post-startup
12 s forecast in coarse known-wind seed 0, 78.79% of expected positive mass lies
where truth is below 1% of the scene peak, and 37.63% of total expected mass is
the uncertainty excess over the positive posterior mean. These are different
quantities: the background figure also includes persistent/inaccurate predicted
signal. Truth is used only to diagnose and score, never supplied to the controller.

Fixed-startup-path interventions separate causes:

| Intervention | Background mass fraction | Uncertainty excess fraction | Compiled tracking time |
|---|---:|---:|---:|
| Recorded v1 | 78.79% | 37.63% | 33.55 s |
| No injected reading noise; same assumed R | 79.44% | 40.61% | 33.55 s |
| Injected and assumed reading noise divided by 10 | 79.59% | 36.98% | 32.15 s |
| Q set to zero, same evidence | 66.27% | 3.65% | 5.95 s |
| Prior standard deviation divided by four | 79.69% | 32.45% | 29.75 s |
| Source input removed, same evidence | 93.71% | 83.38% | 33.55 s |

The zero-Q intervention is not a calibration or proposed fix. Its snapshot map
error is slightly worse than the recorded model, and low covariance can become
overconfident. The test identifies the interaction of globally injected process
uncertainty with zero-threshold expected-positive weighting. Reducing the actual
simulated detector noise alone does not resolve the scan failure. Injected noise
and assumed R were tied to one setting; the new explicit configuration separates them.

Beyond these two points, the audit identifies:

- An any-cell-overdue trigger can monopolize later decisions with background
  service. A short service interval is not proof that the whole scene is refreshable.
- Selecting scattered cells and closing their column spans can substantially
  enlarge a candidate: the first v1 prefix covers 71.4% of physical cells, while
  its closure covers 97.5%. The compiler is preserving selection as specified;
  this is a consequence of the decision surface and action geometry.
- The 40 x 64 lattice requested for testing has four times as many physical
  readings as 20 x 32. At 20 Hz, a complete acquiring raster takes 128 s plus
  travel; a 75 s full-startup demonstration is infeasible. It is tentative hardware,
  not an instruction to enlarge display pixels or accelerate the detector.
- The source generator drops old puffs after a finite lifetime; the Gaussian
  evolution/source input does not encode that lifetime. The persistent tail and
  posterior calibration failures remain. Simply inserting a fitted decay to
  improve this video would confound the allocation diagnosis.
- A bounded forecast initialized at the maximum cycle budget is not a global
  search over all useful short sweeps. Source-free, weak-signal and wind-error
  cases must expose failures; the revision does not claim global optimality.
- The planner does not recursively retain 90% of the previous ROI; it selects
  from the full predicted lattice each time. Selective evidence can nevertheless
  reinforce a misplaced small region. New checks require expansion/movement when
  the belief changes and actual surveillance outside the previous tracking region.
  Reports expose size sequences, minimum-size visits and stale/unvisited cells.

## Revised formulation mapped to code

The [experimental v2 specification](../specs/mechanism-grounded-policy-v2.md)
keeps signed Gaussian reconstruction and the causal public API. It changes the
actionable signal convention, two-axis surveillance geometry and service cadence.
It links each operation to the physical instrument and to the specific role of
Bayesian inference/experimental design. It is not another freeze of unknown
hardware assumptions.

The numerical reporting threshold is fixed before closed-loop comparison at
three times the declared blank-reading standard deviation: 0.15 synthetic units.
This is not a measured gas detection limit. Actual and assumed reading noise,
prior and Q remain unchanged in the main study. The expected-excess signal above
that threshold is distinct from uncertainty and from full-signal reconstruction
loss. Without a predicted mean above threshold, the policy surveys instead of
calling normalized uncertainty tails successful tracking. A fixed corridor does
not need adaptive signal evidence to execute its declared region.

New `sensing_session_v2`, `sensing_policy_v2` and `sensing_execution_v2` modules
provide explicit configuration and a separate decision rule. Existing controller,
compiler, simulator/replay, evaluator and six-panel renderer are reused. Legacy
configuration/messages and v1 policy mathematics remain available. New rendering
labels disclose the reportable threshold; physical motion and panel layout are retained.

## Development checks and reproduction

Main comparisons are predeclared in `configs/sensing-v2/development.json`:
two profiles x known-wind/wind-error/source-free x three policies x seeds 0/1.
The coarse profile uses full startup; the tentative doubled profile uses explicitly
declared prior startup. Policies share startup within each comparison. A separate
doubled full-startup test preserves the resource failure. Supplementary tests cover
zero-threshold segments, full-height surveillance, 2/4-sigma thresholds, a weak
source-free plume and a blank source-free scene. They are diagnostics, not an
opportunity to select the best result or threshold afterward.

```powershell
.\.venv\Scripts\python.exe scripts/diagnose-sensing-mechanisms.py outputs/policy-v1-conformance/development/coarse-known-wind-mass90_column_v1-seed-0/trace.json --output outputs/mechanism-revision/diagnosis/fixed-path-verified.json
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run-sensing-v2-development.py --output outputs/mechanism-revision/development --workers 2
.\.venv\Scripts\python.exe -m plume_focus.sensing_cli --run-config configs/sensing-v2/coarse-full-startup.json --output-dir outputs/mechanism-revision/coarse-inspection --render
.\.venv\Scripts\python.exe examples/sensing_controller_v1.py --run-config configs/sensing-v2/coarse-full-startup.json
```

A preliminary smoke run is retained separately from the accepted comparison.
Use a fresh output root for reruns; preserve the existing study and inspection files.

## Completed numerical comparison

Data-producing revision: `af40fdd`, with a clean tracked worktree. The two
representative quadrature checks accept 120 x 160 / 0.05 s against 240 x 320 /
0.025 s; relative loss differences are 0.1993% and 0.2820%. This is a development
check of those paths, not a universal quadrature guarantee. All 43 runs complete.

Mean causal `J/T` over noise seeds 0 and 1 (lower is better):

| Profile/startup | Case | Candidate | Fixed corridor | Age rescan |
|---|---|---:|---:|---:|
| 20 x 32, full startup | Known wind | 0.0189982 | 0.0174079 | 0.0193782 |
| 20 x 32, full startup | Wind error | 0.0198201 | 0.0178615 | 0.0188620 |
| 20 x 32, full startup | Source-free | 0.0219534 | 0.0209902 | 0.0235561 |
| 40 x 64, prior startup | Known wind | 0.0194493 | 0.0214048 | 0.0203252 |
| 40 x 64, prior startup | Wind error | 0.0248844 | 0.0198071 | 0.0198071 |
| 40 x 64, prior startup | Source-free | 0.0259362 | 0.0305371 | 0.0259362 |

The coarse candidate loses to the fixed corridor in every case. The doubled
candidate wins the known-wind comparison but loses with wind error. Its source-free
result is exactly age surveillance: **zero tracking sweeps**. The doubled fixed
corridor often cannot fit its full geometry, so that comparator also falls back;
do not interpret the policy labels as proof that distinct tracking rules executed.
Two noise seeds and one fixed source realization do not establish superiority.

Mean actual charged scan seconds, in candidate / corridor / rescan order:

| Profile | Known wind | Wind error | Source-free |
|---|---|---|---|
| Coarse | 74.875 / 74.975 / 74.975 | 74.950 / 74.975 / 74.975 | 74.900 / 74.925 / 74.975 |
| Doubled | 74.738 / 74.838 / 74.963 | 74.612 / 74.963 / 74.963 | 74.963 / 74.813 / 74.963 |

The horizon is 75 s for every run. These are common ceilings, not exactly equal
expenditure. Source-free still receives a wind estimate. Cross-profile comparisons
also change startup assumptions and cannot isolate the effect of resolution.

Supplementary seed-0 diagnostics:

| Change from coarse known-wind candidate unless stated | J/T | Tracking sweeps |
|---|---:|---:|
| Main 3-sigma threshold | 0.01904924 | 9 |
| Zero-threshold mass with segment surveillance | 0.01944012 | 0 |
| 2-sigma threshold | 0.01797898 | 7 |
| 4-sigma threshold | 0.01924941 | 11 |
| Full-height surveillance, same threshold | 0.01904862 | 9 |
| Weak source-free signal x0.1 | 0.00036173 | 0 |
| Blank source-free scene | 0.00012743 | 0 |
| Doubled full-scene startup | 0.02024357 | 0 |

The 2-sigma setting is not selected afterward as a new default. The full-height
surveillance diagnostic spends 74.175 s versus the main run's 74.825 s. The weak
case performs surveillance without tracking; that is a limitation for weak-plume
monitoring, not proof that weak gas is absent. Blank truth metrics with a zero
denominator stay unavailable. The doubled full-startup run visits 52.5% of physical
cells and remains explicitly incomplete; its full scene needs 129.5875 s.

## Does the policy keep shrinking?

No recursive ROI multiplier is present. Every decision selects from the entire
predicted physical lattice. Within one decision the bounded forecast iteration
often moves from a budget-length horizon to a shorter scan; it stops after at
most five iterations and does not restrict the next cycle to that smaller ROI.

The coarse known-wind seed-0 tracking selections contain
`27, 105, 32, 60, 29, 49, 30, 44, 35` physical cells: four expansions and four
contractions, with no four-reading-floor hits. The corresponding acquiring
vertical extents average 0.068--0.202 FOV and tracking sweeps take 2.35--6.75 s.
Surveillance later takes 161 readings outside the previous tracking region.
Seed 1 also expands four times and contracts four times. Tests separately require
stable selection for stable state, expansion, movement and outside-region service.

These checks rule out monotonically shrinking behavior in the inspected runs,
not self-reinforcing errors in general. Coarse seed 0 still ends with 13.28% of
cells older than 60 s and maximum age 66.375 s. Several completion-time searches
report a cycle or lack of a stable selected set. Oscillating area, frequent
tracking and high predicted coverage are not reconstruction-success metrics.

## Spatial representation diagnostic

An evaluator-only least-squares projection of true snapshots onto the existing
bilinear 12 x 16 basis achieves MSE 0.00320--0.00382 at the seven declared times.
The actual online estimate has MSE 0.01624--0.02289 there. A diagnostic 24 x 32
basis lowers the oracle error further, to 0.00133--0.00214, but the experiment's
estimator is unchanged. Thus finite spatial resolution contributes error, while
it cannot by itself account for most of the observed snapshot error. The gap also
includes causal information limits, timing and model/estimation error; this test
does not attribute it all to a software defect. Snapshot projections with truth
are not online estimates or a converged common-horizon performance comparison.

## Inspection and remaining decisions

Six videos contain 2,724 fully decoded frames. Rendering reproduces each saved
operational hash. Representative physical and summary frames were visually
reviewed, including transit before readings, accumulation at 20 Hz, different
vertical extents, the doubled profile's smaller footprints, completion/idle and
blank-signal unavailable mass. The existing renderer changes only four added and
two removed lines for policy/threshold labels; its layout is retained.

Local inspection artifacts:

- [Coarse physical time](../../outputs/mechanism-revision/development/coarse-full-startup-known-wind-threshold90_segment_v2-seed-0/physical.mp4)
  and [compressed summary](../../outputs/mechanism-revision/development/coarse-full-startup-known-wind-threshold90_segment_v2-seed-0/summary.mp4).
- [Tentative 40 x 64 physical time](../../outputs/mechanism-revision/development/double-prior-startup-known-wind-threshold90_segment_v2-seed-0/physical.mp4)
  and [compressed summary](../../outputs/mechanism-revision/development/double-prior-startup-known-wind-threshold90_segment_v2-seed-0/summary.mp4).
- [Region-size sequence](../../outputs/mechanism-revision/development/focusing-sequence.png).

The videos also confirm remaining failures. At 36.825 and 43.575 s the coarse
predicted/estimated signal has a long tail beyond evaluator truth. In the doubled
run, the first 688 readings cover upper background while the declared source is
lower: the oversized tracking candidate fails the hard 90% gate and age service
takes over. Reporting that failure honestly does not make it a good reconstruction
action. Full-field coverage, signal weighting, and the best affordable action
remain different quantities.

Resolved in this candidate: coupled simulated/assumed reading noise; compulsory
full-height fallback as the only service primitive; continual any-cell-overdue
triggering; missing explicit doubled profile; and missing size/age/outside-region
diagnostics. Stable/growing/moving selections and fixed paths are verified. V1,
its physical compiler and Gaussian dynamics are preserved, not silently revised.

Unresolved: temporal persistence and posterior calibration; weak plume sensitivity;
hard-90%-failure-to-background-service behavior; bounded forecasting that can miss
short feasible actions; fresh whole-scene coverage at the doubled resolution;
real beam/retrieval/wind calibration; and demonstrated reconstruction advantage.
An explicit best-effort action with its actual attainable coverage is one design
option for a future version; it must not be disguised as a successful 90% plan or
routine trimming. A future formulation should compare executable actions by their
effect on reconstruction, while keeping uncertainty-driven discovery separately
accounted for. These are decisions for the next scoped plan, not claims implemented
by this candidate. Hardware, networking, emissions and real-data expansion remain out.

## Verification and provenance

Revisions: diagnosis `ced244f`; implementation/data
`af40fdd1d45df01a7ddfb7e4c3f2df07680ade59`; inspection and spatial diagnostic
`2125a1e2b1194d23c4aec23522a1d6ea33d510a5`. Final documentation revision is recorded
in the local closeout record. No remote push; user `output/` files are retained.

```powershell
.\.venv\Scripts\python.exe scripts/render-sensing-v2-inspection.py outputs/mechanism-revision/development
.\.venv\Scripts\python.exe scripts/inspect-sensing-v1-artifacts.py outputs/mechanism-revision/development
.\.venv\Scripts\python.exe scripts/diagnose-reconstruction-basis.py outputs/mechanism-revision/development/coarse-full-startup-known-wind-threshold90_segment_v2-seed-0/trace.json --output outputs/mechanism-revision/diagnosis/basis-verified.json
.\.venv\Scripts\python.exe scripts/check-policy-v1-spec.py --output outputs/mechanism-revision/v1-worked-checks.json
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src tests scripts examples
git diff --check
```

**382 passed, five Windows symlink-capability skips.** This includes strict import
boundaries, v1/v2 decoding, evidence replay, geometry/clock invariance, public API
example/simulator equality, and observer-only playback. Nine independent v1
arithmetic groups still pass; dynamic sequential/joint-batch disagreement is
`2.220446049250313e-16`. Saved v1 operational SHA-256 remains
`6da071ca6d2f0408b0885163a4c95a4395978300c3015f4b59506ad9892c4ded`.
Compilation and Git diff checks pass. Matplotlib could not write its optional
user font cache in the sandbox; both plot files rendered and were verified.

SHA-256 anchors (complete media and trace hashes are in the artifact manifest):

| File | SHA-256 |
|---|---|
| `configs/sensing-v2/coarse-full-startup.json` | `b5cb22e32b9dbd7747a456eabaf9249f009d9d0ebc25ec0e26218e487bc36ead` |
| `configs/sensing-v2/double-prior-startup.json` | `6abd7ffa59daf5fba45d5e2d8eb586c9487e7be496f18719738b5cb65474793d` |
| `configs/sensing-v2/development.json` | `37529854bf6fd2bd00d78e91532b696651534aa71c03fcc1d06e8f7a928a4c7b` |
| `development/original-data-artifact-manifest.json` | `33e575597eb08a8597711a29a1effcf2842294c8259780131abb94d6909cb40d` |
| `development/artifact-manifest.json` | `2636b8cacccc251ac340f17647c29380eb6d438bf05ceb5d1b3245a3ec415f36` |
| `development/evaluation-freeze.json` | `48af504553b08d7df73b20e26795a096cb7db6548f0d313254b55e0055b4d1a4` |
| `development/development-summary.json` | `7ee6fd3b106beae794c464b79ea7d37ec01185714056b763c3086d2d24aefe53` |
| `development/inspection-manifest.json` | `dd1bfd3ee38a3d0bff36e9369b24b297e560b15cb79206e852154c1e0f7240fd` |
| `diagnosis/fixed-path-verified.json` | `9cea9ca3b3f5ca00d79495d34b748e09c3480d75a3a13349c26e109e4e7dbbe4` |
| `diagnosis/basis-verified.json` | `bebe7c578da675a8b23617c5126d748dda05524adda7177e952b3e799cc0ea04` |

Paths beginning `development/` or `diagnosis/` are under local ignored
`outputs/mechanism-revision/`. The manifest preserves separate data/rendering
revisions and original evidence. Results are descriptive development findings;
all previous `not_demonstrated` records remain intact.

Closeout removed the redundant `physical_swept_fraction` auxiliary field: it
counted compiled cells, which can overstate execution after early completion.
Use `swept_fraction` for actual footprint area and visit metrics for actual
readings. The original complete-execution reports are retained with their producing
revision; none of the comparison values above use the removed auxiliary field.
A finite-source partial-execution regression verifies actual area and incomplete truth.
