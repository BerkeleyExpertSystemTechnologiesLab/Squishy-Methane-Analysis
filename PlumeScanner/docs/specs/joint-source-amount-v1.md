# Matched source location and amount comparison v1

## Question

Does fixing the wrong source-input amount move inferred source locations, and
can joint location/amount inference improve reconstruction on the same readings?
Freeze `configs/source-prior-study/joint-source-amount-v1.json` before execution.

## Truth and evidence

Use the historical 40 s/708-reading *schedule*, positions, footprints and wind;
generate new values with a matched discrete Gaussian field. Every method sees
identical values within a realization. Amounts .5/1/1.5 times the reference total
are the only scenario difference. Width .035, half-life 6 s, supplied wind,
12x16 grid and all P0/Q/R assumptions stay fixed. Source input is explicitly
`cell_integrated_fov_normalized_v1` in both generator and inference. The scale
Q0=.002305094376073324 is synthetic input per second, not emission mass.

Run a zero-realized-perturbation mean control and ten shared seeds per amount.
Keep the inference covariance/noise assumptions in the mean control; it is not a
probabilistic calibration draw. Gaussian states may be negative. Subtract each
scenario's deterministic realization to verify identical realized random field
and reading perturbations; changing source amount must not scale noise.

The explicit matched clock advances at acquisition and delivery, as well as
plan/finish/end boundaries. A reading is assimilated at its acquisition reference
time but becomes visible only at delivery; the posterior is then propagated to
delivery for scoring. No scores are retroactively changed. The .0125 s
acquisition-to-availability interval is retained. This newly declared partition
is used on both sides; it is not bit-identical to prior matched-control evidence.

## Estimators

All use the same broad uniform source-position prior, field dynamics and noise.

1. Location only, amount fixed at Q0.
2. Joint location and amount; a uniform prior on amount/Q0 in [.25,1.75],
   independent of position, integrated with Gauss-Legendre quadrature.
3. Location only with true scenario amount supplied as an oracle control.

The true source position is not inserted into quadrature or supplied to methods.
It is used by the evaluator and exact-source verification only. Keep conditional
Gaussian fields separate; do not substitute a mean location/amount into one
nonlinear source kernel. Aggregate field variance includes within-component and
between-component terms.

For source input e(s) at Q0 and multiplier a, the affine Kalman representation
gives mean m+B[a e(s)] and log likelihood c+a L(s)-a^2 K(s). Reuse these sufficient
statistics without changing the model. Verify against small direct source banks,
including a full-history mean and stochastic realization.

Bounded likelihood screening is numerical quadrature acceleration. For each
location, maximize the quadratic likelihood over the declared amount interval;
this is an upper bound on its amplitude-integrated likelihood. Retain locations
within 50 log units of the maximum and explicitly bound omitted posterior mass.
If that bound exceeds 1e-10, widen to all locations. Retain the uniform global
prior; no truth-dependent search box is permitted.

## Evaluation and refinement

Primary metric: mean latent-field MSE at delivery of every 20th reading plus
reading 708. Score all methods on the same checkpoints. Save scan-end and 40 s
final fields separately. This checkpoint average is not physical-time J/T and
does not score a new acquisition policy.

At 40 s stop supplying readings, continue matched truth and forecasts to 46/52 s,
and report forecast field MSE. At each horizon evaluate the exact mixture CDF for
the latent footprint signal at (.55,.25), using the fixed instrument footprint;
these are evaluator-only validation targets, not extra instrument observations.
Counts over ten fixed-geometry seeds are descriptive, not a calibration guarantee.
Save mean/std maps, amount posterior summaries, source displacement and mass near
truth. Lower source error alone is not success.

Run source/amount quadrature 65/65, 129/129, then 257/257. For adjacent levels
check source-mean shift <.002 FOV, amount-mean shift <.01 multiplier, and each
history/method's primary field MSE difference <max(1% of finer score, 1e-6).
Do not loosen failed criteria. If the final comparison fails, retain unresolved
results or declare a separate refinement before interpreting gains. These checks
do not establish convergence of the complete density or all interval endpoints.

Report paired field-error changes and their signs for the same ten seeds at each
amount. Report regressions in the nominal case and failures as well as gains.
Truth/source parameters, physical schedule and original evidence bytes stay fixed
within their declared roles. This is a conditional single-geometry inference
experiment, not source discovery or general adaptive-sensing superiority.
