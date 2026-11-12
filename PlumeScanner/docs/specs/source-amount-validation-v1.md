# Source-amount reconstruction, independent-scene and closed-loop validation v1

Freeze the accompanying configuration before runs. Starting branch is
`codex/source-prior-fixed-evidence`, revision `bf0500b`. No remote upload.

## Fixed clock and truth

The earlier checkpoint metric is not J/T. This study declares a .0125 s clock:
the same zero-inflow transport, cell-integrated in-FOV source, 6 s decay and
Gaussian kernel/noise are applied at every fixed tick, independently of scanning.
Latent signal is held constant between ticks. J/T is the exact time average of
spatial MSE for this discrete-clock model over [0,40), not a continuous physical
diffusion integral. Also subsample .025/.05 s on the same trajectory as a scoring
sensitivity diagnostic, without regenerating truth or changing inference.

Generate shared zero-source Gaussian trajectories from separate initial/process
RNG streams and deterministic unit-input trajectories for each source. Scale only
the latter by .5/1/1.5. Hash measurement noise by seed, acquisition tick and
position, so path changes cannot change the latent environment. Acquisition and
availability must lie on the clock; exposure midpoint, footprint and .0125 s
availability delay retain the declared matched sensor approximation. All original
physical motion, footprint, cadence and integration costs remain unchanged.
Inference propagates the same tick dynamics. Never assimilate a reading before
its availability: acquisition-time filtering becomes visible only at delivery.
Predictions used for scoring are detached from live filter state. No smoothing.

This is new evidence, not bit-identical reproduction of the parent histories.
Compare matched grids 12x16 and 18x24 with the same Q0, source/width/wind and physical
schedule. Noise kernel is evaluated on each grid. This checks dependence on the
finite matched model; it does not isolate representation error against a common
continuous physical truth or validate boundary physics.

## Independent scenes and estimators

Development uses the already explored source position and four old seeds plus
a mean control. Holdout uses four predeclared new positions (two interiors,
near inflow and near outflow) and four new seeds. No tuning, selection or prior
recentring after inspection. All .5/1/1.5 amounts are retained. Every estimator
gets the same readings on the fixed historical path. Location prior is uniform
over the unit FOV, never augmented with truth. Compare fixed Q0, joint amount
uniform on [.25,1.75] Q0, and true-amount location controls.

Compute amount integration on bounded, likelihood-dependent intervals, with a
documented bound on omitted mass. This is numerical acceleration, not a changed
prior. Verify against independent scalar integration and direct candidate filters.
Run source/amount orders 129/64 and 257/128. Require per-history/method J/T and
forecast MSE changes <max(1% finer score,1e-6), final source-mean shift <.002 FOV,
and amount-mean shift <.01. Retain failed criteria; declare a finer run if needed.
Convergence of these summaries is not convergence of complete posterior densities.

Before outcomes were inspected, a performance-only change uses 16-point amount
integration for moment queries whose retained-interval log-likelihood variation
is <=4, verified against full 128-point moments on 2,000 varied statistics to
1e-12. CDF/component exports retain the full declared order. The first attempt
was interrupted after 200 readings without final scores; keep its code/log and
reuse its unchanged truth arrays. No scene, prior or criterion is adjusted.
The second performance attempt also stopped before final outcomes. Weighted
source inputs are now computed in a shared dense/sparse batch instead of repeated
indexed matrix copies; equivalence is independently checked to 1e-14. Preserve
both incomplete attempts and their producer snapshots. These changes alter
computation only, not inference formulas or scientific choices.
The third attempt completed the 12x16, 129/64 level. A successor reuses that
entire result byte-for-byte and evaluates independent histories with four local
worker threads. Threaded/scalar posteriors are bit-identical in tests. Incomplete
fine-level work remains archived. Parallel execution changes neither cases nor
arithmetic within a history.

## Reconstruction and uncertainty outcomes

Primary outcome: per-history physical-clock J/T. Secondary: final and +6/+12 s
field MSE with no readings after 40 s. Save pointwise mixture PIT, variance,
squared-error/variance and 95% central-interval membership at the five declared
validation footprints at 20/40/46/52 s. Use actual mixture CDF, not mean +/-1.96 SD.
Report signed source displacement and amount errors only as diagnostics. Counts
over correlated points/times are descriptive, not independent calibration samples.
Do not infer whole-field 95% coverage from pointwise counts.

The hypothesis is weakened if heldout joint J/T or mean forecast errors do not
improve at both mismatched amounts on each grid, if nominal J/T worsens >2%, or
if inferred amount absolute error fails to improve over fixed Q0. Publish all
these checks and failures; do not tune away a source/amount tradeoff. Assess
overconfidence through predictive misses and error/variance, not location alone.

## Closed-loop execution

After fixed-path engineering checks and numerical refinement, run both fixed and
joint beliefs through the same existing evidence-gated scanner code, settings and
physical compiler on a 60 s budget. Scientific negative results are retained and
bar a superiority claim; the loop remains an exploratory test. Use two declared
heldout positions, three amounts, two seeds and the 12x16 model (24 executions).
The two methods face the identical precomputed truth and keyed noise. Their paths
and thus their readings may diverge; do not call them identical-data comparisons.

The adapter exposes actual finite-mixture marginal quantiles to the existing
policy, pruning at most 1e-12 component mass for numerical projection. No Gaussian
moment replacement for policy quantiles or evaluator PIT. Record all charged
motion, sampling, unavailable budget, startup and mode decisions. At plan time,
prediction uses only delivered readings. Save actual events/receipts and replay
them to verify beliefs/metrics, independently of rendering. Compare J/T on the
same 60 s clock, continued forecasts, physical costs and reading counts; smaller
scan regions alone are not success. Keep original policy and baseline source
files untouched. Physical voltages, emission units and real-time computational
feasibility remain unvalidated. Renderer is an observer, not evidence generation.

Before closed-loop outcomes, separately declare `source-amount-path-audit-v1.json`:
cross-replay both measurement histories under each fixed estimator. Holding one
estimator constant isolates the value of changed measurements from better belief
updates. Report the full 2x2 J/T matrix and post-startup MSE. If delivered readings
are identical, reuse the verified own-method scores and explicitly record that
there is no demonstrated acquisition change. Shadow replays never choose actions.

Before closed-loop outcomes, accelerate amount-moment queries using the analytic
truncated-normal integral of exp(a L-a^2 K) in well-conditioned cases, with the
declared quadrature fallback elsewhere. Fixed-path and numerical-refinement runs
retain their quadrature implementation. Tail/normalizer/moment and multi-source
weight tests compare the analytic path with full 256-point integration. Actual
CDF and policy quantiles retain full declared component quadrature. This changes
numerical evaluation, not the prior, likelihood or hypothesis family. Cache the
posterior mean input only between readings; invalidate it on new evidence.
For large predictive mixtures, evaluate the same component CDF in bounded-memory
chunks rather than allocating the full component-by-image tensor. Quantile
brackets and bisection tolerance match the dense implementation; tests compare
CDF, moments and quantiles directly. This is not a Gaussian moment projection.
