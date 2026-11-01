# Fixed-evidence source-location prior experiment v1

Local exploratory experiment authorized by Xilin on 2026-10-28. Starting code:
`5cac633f47001d3aa455f8b6f3cc00b11c550ed3`; branch
`codex/source-prior-fixed-evidence`. No production controller or policy is changed.

## Frozen evidence and scope

Reuse the complete original 40 s continuous synthetic quickstart history, hash
`9545e6c6f9495f082ca845eafecf2d5cdef82ff19160d6eb2f8481aff9cedf10`:
708 streaming readings, one full and two focused scans. Original configuration,
timestamps, physical actions and readings are identical for every method. This
history was selected by a known-source controller; comparisons are conditional
on that history, not a demonstration of source discovery or closed-loop utility.
All priors are supplied at t=0. The accurate center is deliberately oracle-informed,
not estimated by Nakul/Ian or another detector. No source handoff is simulated.

## Declared methods

All methods share the original 12x16 latent field, initial covariance, fixed wind,
source width/rate, decay, observation noise and measurement footprint. Compare:

1. Uniform source-location density over the normalized unit field of view.
2. Gaussian regional prior centered at (y=.547945205479452, x=.15047021943573669).
3. Same regional prior centered at (y=.447945205479452, x=.3304702194357367).
4. Delta prior at the accurate center (the original demo baseline).
5. Delta prior at the biased center.

Both regional priors have independent standard deviations .06 FOV on each axis.
They are truncated only at the field-of-view boundary; their support is not a
hard bounding box. The offset is (-.10,+.18), Euclidean norm .2059126 FOV.
These parameters are fixed before inspecting comparative results.

Continuous priors use midpoint quadrature on a common 33x33 then 65x65 grid.
Delta priors are evaluated at their exact positions, not snapped to this grid.
Relative change in causal J below 1% for every method is the declared source-grid
stability criterion. A failed check triggers a separately saved 129-grid followup,
not replacement of the original results. Passing this check alone does not prove
convergence of source uncertainty, latent-grid representation or physical modeling.

## Inference and causality

Each source hypothesis maintains a conditional Gaussian field. On each reading,
update hypothesis weights by its pre-update predictive Gaussian likelihood, then
condition each field. Hypotheses are static source locations with fixed strength.
Their conditional covariance is identical on a fixed schedule and is shared
exactly. Aggregate variance includes both within-component and between-component
terms. No posterior mean source is substituted for the mixture.

The bank has no truth access. Truth is used only to set the labeled experimental
conditions and in the evaluator. The runner verifies the original controller's
public plan replay, compares the accurate-point component against its internal
belief at every event, and rejects delayed/out-of-order or batched histories.
Loss is evaluated using evidence already delivered at each evaluation time;
forecasts never modify the inference state. Evaluation uses the original spatial
120x160/time .05 s quadrature and includes unused time up to 40 s.

## Metrics and limitations

Primary metric: causal time-averaged field squared error J/T. Also retain the full
loss trace, per-reading prequential RMSE/log predictive density/95% coverage,
source posterior weights, mean location, mass within .05 FOV of the true location,
and full-scan/focused-scan/final field snapshots. Snapshot field coverage uses the
exact mixture CDF on the declared 20x32 inspection grid, not Gaussian moment
intervals. Coverage is descriptive: added simulator noise is zero whereas the
assumed likelihood noise is .05, and only one synthetic history is available.

The original source input is a point-sampled Gaussian peak amplitude, not a
conserved emission total. Keeping its implementation unchanged permits exact
baseline replay but produces discrete integral differences across positions due
to grid alignment and boundary truncation. Report these differences explicitly.
No claim that source priors differ only in total physical emission is justified.
No source activity, source rate or wind inference is added. Numerical and physical
model mismatch may produce biased, overconcentrated source posteriors even when
field reconstruction improves.

Reproduce from PlumeScanner with `PYTHONPATH=src OPENBLAS_NUM_THREADS=1`, using
`scripts/compare-source-priors.py --config configs/source-prior-study/v1.json
--output outputs/<fresh-directory>`. Use `refinement.json` for the followup.
The output directory must not exist. Generated artifacts stay local.
