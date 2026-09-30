# Reconstruction-value allocation implementation and study

Status: bounded implementation/evaluation goal complete, 2026-09-15. Gaussian
classification: `mixed`; fixed-mixture classification: `numerically_unresolved`.
Scientific claim remains `not_demonstrated`; this is not a new policy freeze.
Start: `51d1f633f8720d3cbcf14d61d2b2c7a27e201e5b` on
`codex/reconstruction-value-policy`; no intervening source changes were present.
The existing untracked `output/` artifacts are preserved. New generated evidence
lives under ignored `outputs/reconstruction-risk-v4/`.

## Implemented mechanism

The [approved specification](../specs/reconstruction-value-policy-v4.md) now has
separate immutable policy/session bindings, transport-v3 plans and unchanged
`plan_next`, `ingest_readings`, `finish_scan` operations. The candidate generator
uses the full physical lattice, preserves requested cells through segment closure,
and evaluates every prefix until acquiring alone exceeds the budget. It compares
at most 14 paths; it does not prove global optimality or global infeasibility.

The controller retains actual-visit startup and credits required service only
after complete actual block readings. A tracking path can have less than 90%
predicted mass and still be valid. Hypothetical repeats include repositioning,
detector recovery, delivery and receipts. Only the first sweep executes.

The Gaussian scorer integrates projected conditional covariance using the exact
bilinear spatial Gram matrix. Mixture scoring analytically averages conditional
within-component variance and samples possible evidence for component disagreement.
Each history keeps one generating wind hypothesis and updates all filter hypotheses.
No aggregate-Gaussian replacement or component pruning is used.

Cached sparse transition operators reproduce the existing propagation arithmetic;
subtraction roundoff is canonicalized at 1e-12 s, below the causal tolerance.
The trace identity `tr(G A P A^T) = tr(A^T G A P)` avoids dense query-only covariance
copies. A bounded pure-decision cache includes complete component moments/weights,
wind, session, visit ages, pose and lifecycle inputs. It reuses immutable results
during public-call replay without bypassing evidence validation or plan comparison.
Neither optimization changes the runtime estimator, its dynamics class or boundary
convention. Compute time is separate from mechanical scan time.

## Verification and numerical declaration

Checkpoints: `9ae89a8` initial implementation; `65d6077` study declaration and first
scorer optimization; `c29f811` lifecycle/observer verification; `ad801c6` numerical
freeze and producing code. The full suite at the producing revision passed 430 tests with five existing
Windows capability skips. Added checks include independent joint/scalar risk,
model-consistent histories, projected covariance, mixture learning, zero-motion
equivalence, actual delayed delivery, duplicates/conflicts, immutable scans,
partial service, repeated-pattern travel and full-field candidate movement.

All five Gaussian reference decisions pass 0.1-to-0.05 s refinement within 1%,
with the same winner. Their maximum absolute risk changes range from approximately
3.4e-6 to 2.9e-4. All five mixture decisions also pass time and 32/64-history
checks. Joint checks freeze 0.1 s and 32 histories for the full execution study.
The earlier 49-hypothesis
wind-quadrature failure remains unresolved regardless of allocation convergence.

The frozen declaration contains 96 records: 60 verified historical controls and
36 new risk-policy runs. Historical configuration SHA-256:
`d7c2ae7cbf9a3bb5d6640760d651970a8a507354b8bb114495117a9d7a8cbcc0`.
Source, instrument, estimator, calibration, noise and evaluation are checked
before control reuse. Comparison seeds remain descriptive; they do not establish
general superiority. The common numerical configuration was frozen before any
comparative new execution.

All 36 new run reports identify producing revision
`ad801c66c873f7e1b72d538e6185fd32eec87c7d`. The final manifest and
`outputs/reconstruction-risk-v4/verification.json` record the closeout revision
separately from that execution revision. Local artifact hashes include:

| Artifact | SHA-256 |
|---|---|
| Source study declaration | `caa03145844faee6fb03c374d67336b82f788261ed7f7e41cbd41b6a0273dede` |
| Bound execution configurations | `a97471775bf1a638fde3cf5bd0fa33532fce076da45d8c954e617a7855cbb7df` |
| Numerical freeze | `68f627835ad911542bbfe5771c397b1be1dac41f1743a8435a29e723908cee14` |
| Descriptive comparison summary | `0fab49304f661fbdd993150feaacc028e4c1a9b43eac53f3e7d71a33c3407d97` |

The local `artifact-manifest.json` records file hashes and runtime versions;
`inspection-manifest.json` and `visual-review.json` bind videos and inspected
frames to execution hashes. Historical controls are byte-checked against their
declared source inputs. They are reused evidence, not 60 new executions.

## Reproduction

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check-policy-v1-spec.py
.\.venv\Scripts\python.exe scripts/run-reconstruction-risk-study.py --output outputs/reconstruction-risk-v4 --phase declare
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe scripts/run-reconstruction-risk-study.py --output outputs/reconstruction-risk-v4 --phase refine --workers 3
.\.venv\Scripts\python.exe scripts/run-reconstruction-risk-study.py --output outputs/reconstruction-risk-v4 --phase run --workers 6
.\.venv\Scripts\python.exe scripts/summarize-reconstruction-risk-study.py outputs/reconstruction-risk-v4
.\.venv\Scripts\python.exe scripts/finalize-reconstruction-risk-study.py outputs/reconstruction-risk-v4
.\.venv\Scripts\python.exe scripts/check-reconstruction-risk-spec.py
.\.venv\Scripts\python.exe scripts/inspect-reconstruction-risk-lifecycle.py outputs/reconstruction-risk-v4/engineering-lifecycle
git diff --check
```

The execution runner writes exact delivery traces before evaluation and creates
evaluation refinements and unchanged-layout physical/summary views for declared
representative runs. Manifests contain operational/configuration/artifact hashes
and producing revisions. The separate 8-second delayed-batch/overrun engineering example is generated and
visually inspected. Its serialized fake-scanner sequence exactly matches the
simulator (`bebe969bc652a6f83ed6e838bab44b9b78c7cdb1aacc6fdc57a214cc955a65de`).
It shows no delivered measurements before batch availability, all 28 first-sweep
readings at t=2.425 s, and planned 1.825 s versus actual 2.025 s completion.
This is API inspection, not one of the comparative reconstruction records.
The 20 main/diagnostic comparison videos (9,084 frames) fully decode. Selected physical-time and
completed-sweep frames from all ten old/new cases were visually reviewed and hashed.
The blank case has noisy readings and no true plume, yet its normalized excess
panel can be bright from uncertainty tails. That diagnostic is not a plume
detection, absolute signal map or execution gate. The existing layout is retained.

Local reviewed examples:

- Coarse Gaussian: [physical time](../../outputs/reconstruction-risk-v4/development/coarse-full-startup-wind-error-threshold90_segment_v2-seed-0-baseline-risk/physical.mp4), [compressed summary](../../outputs/reconstruction-risk-v4/development/coarse-full-startup-wind-error-threshold90_segment_v2-seed-0-baseline-risk/summary.mp4).
- Doubled mixture negative result: [physical time](../../outputs/reconstruction-risk-v4/development/double-prior-startup-wind-error-threshold90_segment_v2-seed-0-motion-risk/physical.mp4), [compressed summary](../../outputs/reconstruction-risk-v4/development/double-prior-startup-wind-error-threshold90_segment_v2-seed-0-motion-risk/summary.mp4).
- Blank mixture: [physical time](../../outputs/reconstruction-risk-v4/development/diagnostic-source-free-biased-scale-0-seed-0-motion-risk/physical.mp4), [compressed summary](../../outputs/reconstruction-risk-v4/development/diagnostic-source-free-biased-scale-0-seed-0-motion-risk/summary.mp4).
- Delayed-batch/overrun API example: [physical time](../../outputs/reconstruction-risk-v4/engineering-lifecycle/physical.mp4).

## Completed comparison results

The completed Gaussian main comparisons are mixed. These are two-seed means of
causal whole-field `J/T`; lower is better. All 96 execution/evaluation records
are available, and all five evaluation refinements and the final media audit pass.

| Profile / case | Old adaptive | Risk acquisition | Change |
|---|---:|---:|---:|
| Coarse / known wind | 0.018998 | 0.020663 | +8.76% |
| Coarse / wind error | 0.019820 | 0.018879 | -4.75% |
| Coarse / source free | 0.021953 | 0.023388 | +6.53% |
| Doubled / known wind | 0.019449 | 0.022283 | +14.57% |
| Doubled / wind error | 0.024884 | 0.023843 | -4.19% |
| Doubled / source free | 0.025936 | 0.027196 | +4.86% |

For the fixed 49-component mixture, coarse biased-wind mean loss changes from
0.020019 to 0.019190 (-4.14%), but doubled biased-wind loss changes from 0.025498
to 0.051045 (+100.19%). Zero-uncertainty known-wind/source-free controls reproduce
the Gaussian results. Gaussian main pairs improve beyond 1% in three cases and
worsen in nine; mixture pairs improve in two and worsen in ten. This is not a
general reconstruction advantage. The finite-mixture observations remain
numerically unresolved with respect to the earlier wind-hypothesis quadrature.

The completed source-free biased-wind diagnostics use the coarse profile and
prior startup. Two-seed mean map losses currently available are:

| Estimator / signal scale | Old adaptive | Risk acquisition | Change |
|---|---:|---:|---:|
| Gaussian / 1 | 0.021956 | 0.022113 | +0.71% |
| Gaussian / 0.1 | 0.00031725 | 0.00033083 | +4.28% |
| Gaussian / 0 | 0.00009610 | 0.00010671 | +11.04% |
| Fixed mixture / 1 | 0.013747 | 0.009985 | -27.36% |
| Fixed mixture / 0.1 | 0.00030705 | 0.00032936 | +7.26% |
| Fixed mixture / 0 | 0.00007287 | 0.00007147 | -1.92% |

In the last row, enhancement-index MSE also falls, from 0.001289 to 0.00004929.
The blank mixture mean conceals opposite seed effects: seed 1 worsens despite
the improving mean. These cases change source mode and startup together relative
to the main coarse case; they do not isolate the cause of the different outcome.
Two noise seeds remain descriptive.

In these new mixture diagnostics, final effective hypothesis counts are 1.0 for
both full-strength seeds, 2.66-4.46 for weak signal and 4.60-4.89 for blank signal.
All 49 component states and log weights are retained. Concentration of those
weights does not validate recovery of the physical wind, including in blank scenes.

Gaussian expected squared-error risk depends on conditional covariance, not
signal brightness. Reading values affect the excess-based proposal geometry,
but do not directly enter the Gaussian score for a fixed path. Thus the policy
can leave a bright, locally well-observed plume to measure uncertain background.
This is a consequence of the declared objective under the retained model, not
evidence of calibrated uncertainty or better actual reconstruction.

Selected physical/summary frames verify that behavior. The coarse Gaussian
57.475 s summary completes a mostly blank right-side scan while truth remains
left. The doubled Gaussian 52.362 s summary completes a lower-region path with
8.2% predicted mass. Both cases execute valid sub-90% alternatives. Historical
videos show both focused target-met scans and charged off-plume reacquisition.
The original six panels, physical clock, acquisition progression and compressed
summary labels remain intact. Both Gaussian evaluation refinements pass: relative
loss changes are 0.336% coarse and 0.219% doubled.

The fixed corridor still has lower two-seed mean loss in five of the six
profile/case groups for each estimator. The exception is doubled source-free execution, where
risk acquisition beats the corridor but loses to old adaptive acquisition.
Actual scan time differs slightly (74.6125-74.9875 s across all records); common ceilings do not
make those expenditures exactly matched. Compute time is separate and the
49-component planner is an offline experiment, not a demonstrated real-time
instrument controller. Simulated physical time excludes computation and assumes
zero decision latency; runtime timing is reported separately. For the first coarse
mixture run, simulation takes 2,333.6 s and causal evaluation 615.7 s for a 75 s
physical horizon under the six-worker study execution.
The host reports an AMD Ryzen 9 9955HX3D, Windows 11 and 16 logical processors
available to Python; each worker uses one OpenBLAS thread. These are observed
elapsed times during concurrent execution, not a controlled latency benchmark.
The latent field has 12 x 16 cells and 49 retained components in biased cases.

Map and enhancement-index outcomes can disagree. The Gaussian doubled biased-wind
mean map loss improves 4.19%, while index MSE rises from 0.001395 to 0.002574.
These remain separate outcomes; neither is calibrated physical emission error.

Across 36 new runs, independent trace checks find 193 value-selected sweeps,
including 149 below 90% predicted mass. All 32 required service blocks complete
with actual readings; no service block is reported unreachable in these runs.
Partial and unreachable service remain separately exercised by tests. Each run
ends with one explicit no-action decision when its remaining budget cannot fit a
candidate. The path audit verifies budgets, minimum readings, physical cadence,
requested-cell preservation and the configured risk tie rule.

Seed-0 biased-wind inspection cases illustrate the coverage tradeoff below.
Values are old adaptive -> risk acquisition. Ages summarize visited cells;
the stale fraction also includes unvisited cells. Near-zero means evaluator-only
clean scalar <= 1e-12, with startup included; it is not a new allocation threshold.

| Profile / estimator | Median age (s) | Stale/unvisited | Near-zero readings |
|---|---:|---:|---:|
| Coarse / Gaussian | 36.50 -> 20.65 | 5.16% -> 0% | 59.46% -> 79.24% |
| Doubled / Gaussian | 40.85 -> 29.85 | 60.59% -> 65.04% | 77.28% -> 69.93% |
| Coarse / mixture | 46.55 -> 20.45 | 13.75% -> 0.16% | 55.12% -> 79.56% |
| Doubled / mixture | 43.20 -> 41.37 | 64.41% -> 63.44% | 72.23% -> 69.35% |

Fresher or broader measurements do not by themselves establish better plume
reconstruction. The risk objective, candidate family and refresh protocol change
together here; this is not an isolated ablation of the objective alone.

Actual time components are retained per run. For example, the coarse Gaussian
biased-wind seed 0 spends 8.975 s traveling and 66 s acquiring; the doubled
mixture counterpart spends 14.1375 s traveling and 60.85 s acquiring. These
synthetic executions have zero settling and receipt latency by declaration.

The recorded prediction/error comparison also shows the limitation of the score:

| Decision | Predicted repeated-pattern risk | Realized later-replanning error integral |
|---|---:|---:|
| Coarse Gaussian biased wind, seed 0, t=33.575 s, H=12 s | 0.006797 | 0.232370 |
| Doubled mixture biased wind, seed 0, t=0 s, H=36 s | 0.232807 | 0.829743 |

The hypothetical and realized continuations differ, so these pairs are not an
unbiased calibration test. They do show why the planner's own risk score cannot
stand in for measured reconstruction quality.

The first completed doubled biased-wind mixture run is a material negative result:
seed 0 map loss rises from 0.025969 to 0.050736, despite index MSE falling from
0.002657 to 0.001596. Its first scan lasts 34.6625 s and leaves essentially all
weight on bias `(-0.011544, -0.023668)` FOV/s. Subsequent selected paths retain
roughly 0.01%-0.84% predicted mass. The old policy eventually concentrates on a
different horizontal bias, `-0.011544` FOV/s. This is an observed interaction
between evidence collection and the unchanged estimator, not evidence that either
posterior is calibrated. The study does not isolate which retained dynamics,
boundary or calibration assumption causes this failure.

- All 36 new runs and 96 comparison records are complete. Gaussian classification
  is `mixed`; fixed-mixture classification remains `numerically_unresolved` because
  of the inherited wind quadrature limitation, with mixed finite-model outcomes.
- The saved 35.9875 s decision now has 13 legal alternatives. Its minimum-risk
  candidate takes 12.0 s with predicted mass 0.9062, versus the saved 12.25 s
  rejected candidate. Other candidates retain less than 90%; no mass gate is
  applied. This fixed-evidence diagnostic changes candidate construction and
  allocation together; it is not evidence of better executed reconstruction.
- All five evaluation refinements pass. Relative loss changes are 0.336% / 0.035%
  for coarse Gaussian / mixture, 0.219% / 0.143% for doubled Gaussian / mixture,
  and 0.120% for the blank mixture. The common 120 x 160 / 0.05 s mesh is retained.
- Independent v1/v4 arithmetic, full regression/import-boundary checks, compilation
  and Git diff checks pass. The final verification record distinguishes producing
  source revision from the documentation/postprocessing closeout revision.

Resolved within this goal: the experimental hard coverage/readiness gate;
full-FOV candidate generation and bounded risk scoring; causal delivery/receipt
accounting; actual-visit refresh credit; versioned public integration; and complete,
inspectable comparison evidence. Eight zero-uncertainty Gaussian/mixture pairs
have exactly matching events, executions and reconstruction loss.

Unresolved: general reconstruction superiority, the earlier 49-hypothesis wind
quadrature, calibrated posterior risk, persistent model signal and boundary
behavior, changing local drift, weak-signal robustness, full-scene refresh and
real-time computation. The small refresh rule promises a charged opportunity,
not complete scene coverage within 30 seconds. Hardware calibration, emissions,
drivers, networking and replanning within a sweep remain outside this goal.

Changing local drift, inherited incoming-boundary treatment and calibration remain
subsequent model work. This experiment does not fix them. Lower model-predicted
risk may still favor unhelpful background measurements; such outcomes must be
reported rather than hidden. Predicted repeated-pattern risk and actual later
replanning have different continuations, so their comparison is diagnostic rather
than an unbiased risk-calibration test.
