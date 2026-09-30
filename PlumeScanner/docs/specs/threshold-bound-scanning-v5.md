# Signal-gated threshold-bound scanning candidate

Status: implementation/study complete; see the [review](../reviews/threshold-bound-scanning-v5.md).
Start `5bb58ee`, branch
`codex/threshold-bound-scanning`. This is an allocation experiment, not a freeze.

At a completed-scan decision, focus only if the current mean on the physical
lattice exceeds 0.15 somewhere. A declared prior can support prior-startup;
subsequent support is labeled delivered-posterior. No fresh-hit requirement is
added. A false persistent mean can therefore keep the gate open; measure it.

Forecast independently from the current state to t+B/2 using the latest causal
wind. Select cells whose latent-signal 95th percentile is >=0.15; rank by that
value, oldest actual visit and deterministic rotation. Gaussian bounds use the
bilinearly projected covariance. Mixtures invert their actual weighted CDF,
including deterministic atoms, without changing any of the 49 component states.
These are model-conditional pointwise bounds, not whole-scene confidence claims.
Detector noise is not an extra bonus. The mean-only ablation replaces both
eligibility and ordering with aggregate mean, keeping the same current-mean gate.

Enumerate every ranked prefix until acquisition alone rules out all longer ones.
Choose the longest physically feasible prefix, preserving closure and the four
reading minimum; travel is not assumed monotonic. Include the oldest reachable
0.2-height service block when 30-second service is due. Charge delivery/receipt
allowances in feasibility. The cutoff never rises to hide budget failure.
When the gate is closed, eligibility is empty or no prefix fits, execute the
existing budgeted age-block scene scan. No legal action is returned only when
neither this focused family nor default block scan fits. Actual visits govern
startup/service credit; the path remains fixed while evidence is assimilated.

The new session/run/plan bindings are separate from v1-v4. All new runs explicitly
use zero_inflow_v1. Estimator, source/lifetime, noise, instrument and layout stay
fixed. Plan messages report gate provenance, eligibility, selected/unvisited
cells, physical closure, service and fallback. At-acquisition bounds are detached
observer diagnostics from the saved decision state; they do not alter execution
or introduce a second gate. Completion bounds are included in the new plan.
The inherited `target_mass` field remains a compatibility value for the shared
physical-plan schema; v5 never uses it as a target or feasibility test. Consumers
must use `selection_criterion`, the signal gate and the threshold selection status.
`predicted_mass_recall` is descriptive coverage, not a success criterion.

## Frozen experiment

Reuse eight corrected-transport snapshots from the completed scan-value study.
Freeze historical/configuration/snapshot hashes before new outcomes. Six Gaussian
snapshots x five policies x eight future-noise seeds = 240 twelve-second runs.
The policies are upper threshold, mean threshold, existing v4 risk, declared
corridor and age rescan. Two mixture snapshots x upper/mean/risk x eight seeds
= 48 runs. All actions are autonomous under the substituted binding; preserve
the full inference/lifecycle state and record its parent snapshot identity.

Six Gaussian source/configuration fixtures x the same five policies x seeds 0/1
= 60 complete 75-second sessions. Preserve startup modes, 36/12-second ceilings,
source realization and time/position noise coupling. Different paths receive
their own readings. No post-result expansion: total 348 execution records.
Corridor and age controls preserve their existing v2 rules (including service
semantics), with only the explicit transport binding changed; report this when
interpreting baseline contrasts. Mean/upper share every rule except the surface.

On saved states, evaluate bounds at 0/1/3/6/12 seconds; signal/background, age
and gradient strata; gate false positives/negatives; missed reportable signal,
background selection and forecast error. Sensitivity is quantiles .90/.95/.99
and thresholds .10/.15/.20, declaration-only, never tuning execution defaults.
Latent truth and noisy exposure-reading residuals are different diagnostics.
Neighbor pixels are not independent calibration trials.

Primary outcome is causal J/T. Also report signed enhancement-index error,
travel/acquisition/wait/latency, actual visits, background sampling, omissions,
gate transitions, fallback/service and compute time separately. Eight-seed paired
intervals are descriptive; two-seed sessions do not establish superiority.
Material effects exceed max(1e-10,1% of comparator J) and numerical error.
Start evaluation at 120x160/.05 seconds; refine representative upper-bound
executions from all six Gaussian and both mixture fixtures to 240x320/.025,
then 480x640/.0125 if needed. Check timing sensitivity without retuning.

Keep source/lifetime mismatch, event-partition sensitivity, exposure approximation,
weak-signal gates and unresolved wind quadrature explicit. No calibration,
covariance inflation, padding, component resets or new local drift is implemented.
Close with tests, paired evidence, reviewed existing-layout videos, hashes and
separate classifications of allocation usefulness and uncertainty reliability.
