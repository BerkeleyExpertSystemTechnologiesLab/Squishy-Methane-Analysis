# Comprehensive code review and bounded numerical repairs

Local review branch: `codex/comprehensive-audit-20261029`.
Base: `5cac633f47001d3aa455f8b6f3cc00b11c550ed3` on `plume_scanner_new`.
Scope: all 271 original package files (195 Python files, 51,024 lines), with
file inventory/AST checks, manual subsystem and critical-path review, complete
regression, and disclosed analytic/synthetic mechanism checks. Static coverage
is not a claim that every source line has been semantically proved correct.

## Repairs

- Reject nonfinite truth/predictions/statistics before loss clipping: an undefined
  map loss must never become zero through `max(0, NaN)`.
- Reject duplicate primary results, undeclared/missing budgets and unequal episode
  sets before paired inference. Reject duplicate JSON fields and NaN/Infinity.
- Fix shape-dependent diffuse-plume queries; validate analytic methane source
  parameters, empty batches, normalized pointing and finite delivery settings.
- Preserve existing run directories at legacy sensing CLI entry points. Existing
  recorded atomic no-overwrite writers remain unchanged.
- Restore the misplaced recorded-qualification assertions to their owning test;
  supporting symlinks now exercises the same checks without a NameError.
- Validate serialized Gaussian covariances, including symmetry and positive
  semidefiniteness. Add v2 READY snapshots for the three exact built-in full-FOV,
  calibrated and evidence-gated controller types, preserving full obligations and
  gate provenance. Keep the shared acquisition engine independent of snapshots.
- Guard unsupported historical counterfactual/lifecycle combinations and mismatched
  snapshot/configuration sessions instead of silently treating them as valid paths.

## Explicit numerical/scene extensions

The [numerical contract](../specs/numerical-audit-v2.md) specifies optional tensor
Gauss/composite-exposure integration, exact bilinear rectangle weights,
`scheduled_signal_puffs_v3` (birth-gated cessation), `half_open_v2` puff survival,
`event_split_midpoint_v2` time integration and a resolved source-phase seed helper.
Frozen source, estimator and policy bindings retain their defaults. The refined
sensor does not silently change the frozen midpoint inference likelihood.

For the previously disclosed column-4/sample-11 exposure at 10.8 s, the legacy
sensor returns 0.2034084715; v2 returns 0.1669462788450262 and a finer numerical
reference 0.1669462788450255. The legacy difference is +21.8407% for this selected
fixture, not an estimate of field sensor error or policy bias. An independent
closed-form Gaussian footprint check exposes a -55.9791% four-point error and
verifies high-order integration. Birth-gated cessation retains the old puffs
after shutoff; whole-field disappearance remains a separately named stress.

## Verification and evidence limits

The original tree in the normal Windows environment passes 558 tests and has one
existing NameError in the symlink test. An earlier restricted-sandbox run has 101
hard-link permission failures; approved normal execution resolves that environment
restriction without weakening the writers. The patched suite passes **615 tests**,
including **56 new cases**. Targeted analytic, lifecycle, covariance, malformed-input
and no-overwrite checks pass. All 203 Python files compile in memory, all 97 runtime
modules import and all 13 config JSON files parse (typed run configs also decode).
The old 107 frozen source blobs and eleven protected local code/result artifacts
match their original hashes. Both derived figures were visually inspected.

Runtime: Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, headless OpenCV 5.0.0.93,
pytest 9.1.1, Matplotlib 3.11.2; one process and BLAS thread. The final verification
record identifies the producing local commit, commands, hashes and logs. There
was no package installation, remote push, new performance study or hardware run.
The 104 historical study executions were not rerun in this round; their earlier
replay claim remains historical. No policy is promoted and `not_demonstrated`
remains the overall comparative claim.

The empirical 2-D signal is not mass normalized or physically calibrated. The
12x16 model has a representation limit; retained known-source input can sustain
signal after absence. Current decay-aware evaluator routing and wind-unit
conversion are correct in the checked fixtures. Cadence/speed coupling is a
declared column-scan contract. Existing repeat/replan/idle diagnostics are prior
functionality. A custom recovery/P2 state and forced-action continuation adapter,
a coherent new sensor/likelihood binding and independent physical evidence remain
separate tasks. Source-activity inference, general changing turbulence, posterior
calibration and hardware integration are not implemented by these repairs.
