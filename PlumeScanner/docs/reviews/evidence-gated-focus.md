# Evidence-gated focus

Current status (2026-09-29): **engineering cleanup complete**, retained as an opt-in
research baseline. The [cleanup review](evidence-gated-focus-cleanup.md) closes the
four findings documented at the 2026-09-23 partial checkpoint. All 104 historical
executions replay with unchanged loss and original metrics; corrected derived
summaries remain separate. Coarse reading reacquisition is .10-.95 s in the ongoing
full scan rather than the old 26 s later focused action. Reconstruction stays mixed;
no calibrated field confidence, interface readiness or superiority is claimed.

Started from `00d8429` on `codex/evidence-gated-focus` and implemented in
`e4f5e6a`. The [candidate specification](../specs/evidence-gated-focus-v3.md) and
[study manifest](../../configs/evidence-gated-focus/study.json) were written before
execution; the held-out manifest was frozen before any held-out execution. The
candidate remains opt-in. Engineering completion, behavioral usefulness and
reconstruction outcome are reported separately.

## Implementation

`evidence_gate.py` holds the pure statistic and versioned evidence messages;
`evidence_gated_loop.py` holds the session, plan and scanner;
`evidence_gated_execution.py` holds the run binding. `full_fov_loop.py` gained two
inert hooks (no-op for historical modes) and `SensingController.from_session`
dispatches on the new session type. The calibrated estimator, transport, physical
compiler and renderer module are unchanged. Gate support uses delivered readings
only; the predicted upper-signal field still chooses the focused region.

Eleven declaration/verification checks pass: context-package byte hash
`61a97b09...`, frozen reference-configuration hashes, historical summary hashes,
declared full-scan durations (33.575 s / 129.5875 s), configuration round trips and
digests for all 104 records, estimator identity between control and candidate,
controller dispatch, an independent normal-quantile comparison, the documented
observed-value cutoffs, a planning-import isolation scan, gate-state round trip, and
a truth-lineage check against the historical calibration quadrature cache. Eighteen
new acceptance tests cover gate arithmetic, hysteresis, multiplicity, contract
rejection, causality, delivery equivalence, serialization and dispatch. The full
543-test suite passes; this sandbox denies pytest temporary-directory listing, which
produces 171 fixture setup errors on both this branch and the untouched base commit
`00d8429` (verified in a separate worktree), so those are environmental.

## Fixed-evidence shadow audit

All twelve historical calibrated seed-101 histories were re-audited on identical
readings. Belief is bit-identical between the control and the gated session at every
event of every history (maximum absolute difference 0 over 70,942 events), because
the gate never touches the estimator.

| History | Scans | Mean-gate-positive decisions | Evidence-unsupported | Prospective divergences | First divergence |
|---|---:|---:|---:|---:|---:|
| coarse continuous | 110 | 109 | 0 | 0 | - |
| coarse wind error | 117 | 116 | 0 | 0 | - |
| coarse weak | 185 | 184 | 0 | 0 | - |
| coarse blank | 229 | 228 | 228 | 228 | 33.575 s |
| coarse cessation | 177 | 176 | 127 | 126 | 183.375 s |
| coarse intermittent | 152 | 151 | 81 | 81 | 123.375 s |
| double continuous | 26 | 25 | 0 | 0 | - |
| double wind error | 27 | 26 | 0 | 0 | - |
| double weak | 41 | 40 | 0 | 0 | - |
| double blank | 49 | 48 | 48 | 48 | 129.5875 s |
| double cessation | 44 | 43 | 37 | 36 | 193.5625 s |
| double intermittent | 37 | 36 | 22 | 22 | 137.8125 s |

The gate would have left focus at the first completed full epoch in every blank
history, a few seconds after discontinuation in every cessation history, and
immediately after the stop boundary in every intermittent history, while agreeing
with the mean gate throughout every continuous, wind-error and weak history. The
longest support age carried into a decision was 110.4 s (median 2.2 s), which is the
declared staleness limitation of a sequential full scan rather than a current-time
detection claim.

## Behaviour in the declared 104 executions

Two profiles x (4 development + 6 held-out) cases x (2 noise levels or 3 seed/noise
pairs) x 2 policies, 360 s each. Both policies use the identical frozen calibrated
estimator, mechanics and truth field; only the mode gate differs.

| Stage | Case | Pairs | False-focus decisions (control to candidate) | Focus sweeps (control to candidate) |
|---|---|---:|---:|---:|
| development | continuous | 4 | 0 to 0 | 268 to 268 |
| development | blank | 4 | 568 to 0 | 568 to 0 |
| development | cessation | 4 | 337 to 4 | 447 to 114 |
| development | intermittent | 4 | 209 to 4 | 374 to 126 |
| held-out | continuous | 6 | 0 to 0 | 406 to 406 |
| held-out | wind error | 6 | 0 to 0 | 436 to 436 |
| held-out | weak | 6 | 0 to 0 | 684 to 684 |
| held-out | blank | 6 | 857 to 0 | 857 to 0 |
| held-out | cessation | 6 | 512 to 5 | 680 to 173 |
| held-out | intermittent | 6 | 324 to 6 | 575 to 192 |

Predeclared behavioral criteria:

- **Noise-free blank never focuses:** supported. Both development blank pairs make
  zero focus decisions; the noisy blank pairs also make zero, so no false entry
  occurred even with noise. The per-action alpha is not treated as a session
  guarantee.
- **Unsupported focus forces a full scan:** supported. No focus sweep was followed by
  another focus sweep without fresh supported evidence
  (`focus_after_focus_without_support` = 0 in all 52 candidate executions), and no
  gate-contract violation occurred.
- **All-background full scan does not immediately refocus:** supported. Every
  candidate transition out of a completed background epoch went to the charged
  return home and another complete full scan.
- **Cessation recovers:** supported. Every cessation case with sufficient remaining
  time completed a post-change full scan: coarse 37.0-39.0 s and doubled
  144.0-144.5 s after the change, with the first full start 2.7-13.7 s after it.
- **Intermittency reacquires:** supported for the coarse profile. The corrected
  2026-09-29 scorer finds the first qualifying reading .10-.95 s after the 240 s
  boundary within the ongoing full scan. The frozen report's 26 s score skipped
  that scan and instead described the later focused action. The doubled profile
  remains outside the predeclared fresh-full-scan eligibility rule: 120 s remain,
  but a complete full scan needs 129.5875 s. Its ongoing full scan nevertheless
  contains 566 post-resumption slots and no qualifying hit, an explicit limitation
  beyond the fresh-scan budget shortfall.
- A full scan that straddles a change is not counted as an all-background scan, and
  intended-but-infeasible full scans are reported separately from executed ones.

## Reconstruction outcome

| Stage | Case | Summed J/T control | Summed J/T candidate | Ratio | Actual core readings |
|---|---|---:|---:|---:|---:|
| development | continuous | .061417 | .061417 | 1.000 | 1950 to 1950 |
| development | blank | .005051 | .009653 | 1.911 | 0 to 0 |
| development | cessation | .033315 | .035977 | 1.080 | 673 to 673 |
| development | intermittent | .042570 | .044572 | 1.047 | 1161 to 524 |
| held-out | continuous | .086333 | .086333 | 1.000 | 2950 to 2950 |
| held-out | wind error | .055642 | .055642 | 1.000 | 3157 to 3157 |
| held-out | weak | .007055 | .007055 | 1.000 | 4789 to 4789 |
| held-out | blank | .007585 | .014479 | 1.909 | 0 to 0 |
| held-out | cessation | .047034 | .051020 | 1.085 | 1028 to 1028 |
| held-out | intermittent | .059993 | .062853 | 1.048 | 1766 to 821 |

- **Continuous operation is preserved exactly.** For every continuous, supplied-wind
  error and weak case the candidate traces the same plans as the control, so the
  per-profile held-out aggregate J/T change is +0.0000% (coarse .08242095,
  doubled .08900541) and geometric core coverage is identical (difference
  0.0000 points, all defined). This satisfies the predeclared 5% / 5-point tolerance
  trivially and is an equivalence result, not an improvement.
  The frozen report artifact labels this criterion `unsupported` because the
  aggregate's own `status` key overwrites the criterion label when the two dicts are
  merged; its recorded detail already contains the exact preservation values.
  `scripts/check-evidence-gated-focus-report.py` recomputes all four held-out
  criteria directly from the 104 records and writes
  `checks/predeclared-criteria.json`, which records the corrected verdict
  `supported` and the one-line explanation. The subsequent cleanup repairs the
  normal producer and separately recomputes change metrics; the frozen report
  remains unchanged.
- **Blank, cessation and intermittency reconstruction is worse.** Blank J/T is 91%
  higher in both stages: refusing to focus on a phantom peak spends the session on
  complete scans and pays map error under the current loss. Cessation worsens
  8.0-8.5% in aggregate (individual pairs +7.5% to +9.7%) and intermittency 4.7-4.8%
  (+3.8% to +5.7%). Intermittent core readings fall because the candidate
  deliberately stops focusing during the background window.
- **Classification:** `mixed`. Reconstruction is equivalent on full-strength
  continuous and weak cases, with one comparison (blank) far beyond the 1%
  comparator tolerance in the wrong direction. No reconstruction improvement is
  claimed, and no strength estimation is addressed.

## Remaining model failure

The calibrated estimator still recreates a source peak from model injection even
when the readings support absence. Across the 52 candidate executions, 105 decisions
had a positive old mean gate while the reading evidence was unsupported (up to 9 in
one run, all in blank/cessation/intermittency cases), and 20 runs ended with a false
posterior peak (maximum final posterior mean 0.680 with clean readings near zero).
The gate suppresses the *action* those peaks would have caused; it does not repair
the map, and no map suppression, source-input switch-off or belief reset was added.
This is the specific model/decision interaction the handoff described, now measured
under both policies.

## Numerical refinement

Four new-gate development cases (both profiles, continuous and blank, noise-free)
were re-evaluated on identical recorded evidence at 240 x 320 / .025 s. All four
converged below the declared max(1e-10, 1%) tolerance of the reference loss, so the
480 x 640 / .0125 s stage was not entered:

| Case | J at 120 x 160 / .05 s | J at 240 x 320 / .025 s | Change | Tolerance |
|---|---:|---:|---:|---:|
| coarse continuous | 5.29826624 | 5.27907447 | .019192 (0.362%) | .052791 |
| coarse blank | .84980740 | .85086692 | .001060 (0.125%) | .008509 |
| doubled continuous | 5.75599402 | 5.73708891 | .018905 (0.329%) | .057371 |
| doubled blank | .88476797 | .88616034 | .001392 (0.157%) | .008862 |

The cessation and intermittency paired changes (+8.0 to +8.5% and +4.7 to +4.8%)
are far above the 1% near-tolerance rule, so no additional comparison was refined
and those remain at the reference quadrature. Values are recorded in
`outputs/evidence-gated-focus/refinement/`.

## Media review

Five inspection pairs (coarse and doubled continuous, coarse blank, doubled
cessation, coarse intermittent) were rendered in the existing six-panel layout as
physical-time and one-frame-per-scan compressed-summary videos; all frames decode and
the renderer leaves the execution records byte-identical. One additional alternate
playback render (2 fps, 3x) preserves the same operational hash. The only layout
addition is one gate line (evidence state, frozen slots and delivered count, maximum
lower bound, effective observed-value cutoff, entry/retain levels, action alpha,
fallback reason), which applies only to plans that carry gate fields.

All 7,427 rendered frames decode. Frames actually inspected, with their times: coarse
continuous 0.0 / 35.25 / 180.0 / 360.0 s plus the 33.575 s and 359.925 s summary
frames; doubled continuous 64.75 / 129.5 / 133.75 / 153.25 s plus 129.5875 s; coarse
blank 0.0 / 33.5 / 34.25 / 121.5 s plus 33.575 s and 349.925 s; doubled cessation
129.5 / 133.75 / 180.0 s plus 324.363 s; coarse intermittent 16.75 / 35.25 / 180.0 s
plus 193.775 s and 359.925 s; and the alternate 2 fps / 3x playback frame at 36.775 s.
At 34.25 s in the coarse blank run the frame reads "mean max 0.357 |
signal_not_supported" with a RETURN HOME action, an inconclusive gate line (max L
-0.2056 against cutoff 0.3556) and no white support in the evaluator panel. The
doubled cessation run ends on the explicit "IDLE / incomplete full-scan budget"
banner after "fallback focus_not_renewed". The continuous runs read "evidence
supported" with max L 1.2520 (coarse) and 1.7624 (doubled, cutoff 0.3711) and
unchanged focus paths. Reviewed keyframes, hashes and observations are recorded in
`outputs/evidence-gated-focus/media/visual-review.json`; the rendered media index is
`outputs/evidence-gated-focus/media/index.json`.

## Claims and limitations

- Engineering: **complete for the bounded cleanup** - implementation, study,
  refinement and media are present; the four findings below were subsequently
  closed with 553 passing tests and all-record replay in the linked cleanup review.
- Behavior: **supported** for the predeclared blank, cessation, intermittency and
  continuous-preservation criteria (recomputed by
  `scripts/check-evidence-gated-focus-report.py`), with the doubled intermittency
  reacquisition explicitly unavailable because one full scan cannot fit in the
  remaining 120 s.
- Reconstruction: **mixed** - exact equivalence on continuous/wind-error/weak, worse
  on blank, cessation and intermittency. Behavior can improve while map error worsens.
- Strength estimation: not addressed by this round.
- The gate bounds remain measurement-error bounds conditional on the declared noise
  model; they are not calibrated field confidence, and the multiplicity penalty means
  a settled reading above the .15 entry level can still be inconclusive.
- Weak-signal behavior was identical to the control in these fixtures; no weak-signal
  failure was observed, which does not establish that the conservative cutoff is safe
  for weaker or noisier instruments.

## Open review findings and partial checkpoint

Historical findings, all closed by the [2026-09-29 cleanup](evidence-gated-focus-cleanup.md).
The original reproductions below are preserved; the corrected producer and derived
reports do not overwrite frozen artifacts. The user accepted this partial checkpoint on 2026-09-23. The read-only review inspected
`063f4f3`, ran the 18 gate tests successfully and reproduced additional gaps with
in-memory probes. It did not rerun the full regression suite. None of the following
fixes was implemented by that documentation/Git checkpoint.

1. **P2: actual acquisition timing.** `gate_slot_map` in
   `src/plume_focus/evidence_gated_loop.py` requires timestamps to match planned
   acquisitions within 1e-9 seconds. Shifting strong readings and the receipt by
   .1 seconds is accepted by inference but produces `gate_contract_violation` and
   FULL instead of FOCUS. Separate stable scheduled-slot identity from actual
   timestamps. Closure requires regression coverage for legal delayed acquisition
   and actual overruns while preserving identity/causal validation.
2. **P2: reacquisition during an ongoing scan.** `change_metrics` in
   `scripts/run-evidence-gated-focus.py` skips scans beginning before signal resumes.
   A 200-260 s scan containing a genuine reading at250 after resumption at240 reports
   no reacquisition instead of10 s. Evaluate individual acquisition times across
   overlapping scans and reconsider opportunity accounting. The inspected doubled
   held-out traces contain no post-resumption hits, so this finding does not itself
   overturn their saved outcome. Add a straddling-scan regression and regenerate
   affected derived reports with explicit provenance, preserving old artifacts.
3. **P2: mixed diagnostic provenance.** `_decorate_plan` reports next-action alpha
   alongside preceding-evidence counts/bounds/cutoffs. A first focused plan reports
   alpha .05 for entry bounds computed using .025. Separate evidence parameters
   from upcoming-action allocation; test that reported evidence reproduces its
   cutoff and remains consistent through serialization and rendering.
4. **P2: report verdict overwrite.** In the driver's `criteria`, expanding the
   aggregate dictionary overwrites `supported` with `complete`, which then becomes
   `unsupported`. The separate closeout checker corrects the interpretation but
   does not fix normal report generation. Correct the producer and add passing,
   failing and unavailable-data regressions; retain the original frozen artifact.

The policy's tested behavior improvement remains useful; its reconstruction outcome
remains mixed. Persistent false source signal is separate pending model work, not
repaired by these engineering fixes. The candidate stays opt-in, with no new policy
freeze, interface-readiness claim or robotics integration. Start revision for this
documentation checkpoint: `063f4f3673605ce5c4a0c6dbe3898bcb4f636eed`; verification is
`git diff --check`, followed by a focused commit/push on `codex/evidence-gated-focus`.

## Study closeout (historical; current partial status above)

- Start revision `00d8429` (the handoff checkout on `codex/evidence-gate-handoff`).
- Study commits on `codex/evidence-gated-focus`: `e4f5e6a` (implementation),
  `6d677fa` (findings and renderer label), `2de8430` (frozen driver, dependency
  contract, closeout criteria check, documentation).
- All 104 executions carry one producing code hash,
  `e6d6b22d0634390dbcb686ea7a7b92e2177f9bb2b987ab22c36d377daf64a5b4`, which equals
  the worktree hash at closeout; every held-out record references the same freeze file
  hash, so no record was produced by unfrozen code.
- Verification at closeout: 372 tests pass and 0 fail; 171 fixture setup errors are the
  sandbox temporary-directory restriction and reproduce on the untouched base commit.
  `python -m compileall -q src scripts`, the import probe and `git diff --check` are
  clean.
- Branch `codex/evidence-gated-focus` is pushed to `origin` and verified with
  `git ls-remote` at closeout: the remote tip equals local `HEAD` and the branch
  tracks `origin/codex/evidence-gated-focus`.
- Local evidence under the ignored `outputs/evidence-gated-focus/` tree: 104 execution
  records, 104 verification records, 5 gate traces, 4 refinement records plus an index,
  14 videos with a media index and a visual-review record, the frozen declaration and
  held-out manifest, `freeze.json` and the hash inventory. The pre-existing untracked
  `output/` tree was preserved.

## Reproduction

```powershell
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase declare
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase verify
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase shadow
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase development --workers 2
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase freeze
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase heldout --workers 2
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase report --workers 2
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase refine
.\.venv\Scripts\python.exe scripts/run-evidence-gated-focus.py --phase media
```

Generated evidence, videos and traces stay under the ignored
`outputs/evidence-gated-focus/` tree; only the study manifest and source
documentation are committed. `--workers` uses bounded child processes with inherited
stdio because this sandbox blocks named pipes, so a `ProcessPoolExecutor` cannot run
here; the sequential path (`--workers 1`) is equivalent and resumable.
