# Persistent wind uncertainty candidate

Status: implemented; declared development evaluation and inspection complete,
2026-09-15. Hypothesis classification: `numerically_unresolved`; see the
[study review](../reviews/persistent-wind-prediction.md). Formulation:
`beam_reconstruction_motion_v3_candidate`. This is not a replacement freeze.

## Decision and limits

This round tests persistent wind error. Changing local drift is also required,
but is explicitly deferred to the next motion-model stage. Learning a constant
wind bias cannot eliminate unpredictable future plume deformation.

Hypothesis: plausible persistent transport offsets improve nearby-future plume
prediction and causal reconstruction under the existing scanner constraints.
Keep v2 policy identifiers/rules, threshold 0.15, process/observation/prior noise,
diffusion, sources, mechanics and renderer layout. Keep the no-decay failure,
hard 90% gate and incomplete refresh visible. No noise or lifetime tuning.

An isolated post-hoc probe also finds incoming-edge clipping in the inherited
transport: an initial boundary value can remain pinned and supply persistent
inflow. This round preserves that transport for comparison and records the defect
in the [review](../reviews/persistent-wind-prediction.md). Boundary treatment needs
subsequent correction; it is distinct from decay and changing local drift.

## Finite conditional model

Let velocity be the causal nominal wind plus a session-constant bias. Its declared
zero-mean Gaussian prior covariance has units normalized FOV squared / seconds
squared. It is independent of the initial field; the old error_speed remains a
separate scheduler bound, not a standard deviation or calibrated distribution.

Use tensor Gauss-Hermite quadrature (initial order 3 per axis), transformed by
the symmetric positive-semidefinite covariance square root. Identical offsets
are merged. This is a finite approximation to the continuous bias prior. Given
each offset, use the unchanged Gaussian field dynamics and scalar conditioning.
Update log hypothesis weights with the predictive reading likelihood before
conditioning. Do not prune, temper, resample or reset weights between scans.
New causal nominal-wind messages affect subsequent transport, retaining the bias.

Aggregate mean is sum(alpha_k mu_k); covariance is
sum(alpha_k [P_k + (mu_k - mu)(mu_k - mu)^T]). Expected excess is
sum(alpha_k E[(X_k-a)+]), then multiplied by cell area and normalized once.
Never evaluate this nonlinear expectation from an aggregate Gaussian instead.
Keep the v2 aggregate-mean readiness test and coverage-preserving mass prefix.

Coverage diagnostics use the outer envelope of all component Gaussian bands,
including components with numerically tiny weights. Conditional cellwise union
bounds imply the envelope bound for the mixture; this is conservative, model
conditional, and does not certify calibration or the continuous bias model.
The diagnostic does not affect allocation.

## Evaluation frozen before new results

Use the existing 36 v2 main configurations and both estimators (72 records).
Known-wind and source-free known-wind controls have zero bias covariance. The
biased-wind case has diag(0.01^2, 0.01^2); this is a synthetic assumption.
Add 12 coarse-prior-startup source-free biased-wind diagnostics: scales 1, .1,
0; seeds 0/1; both estimators, shared declared source-free corridor. Retain
75 s, ceilings 36/12 s, common source/noise coupling and fixed instruments.

Fixed-evidence diagnostics use coarse/doubled biased-wind adaptive seed-0 v2
traces, at the first post-scan, middle and last executable decisions (deduplicate
indices), with horizons 0/1/3/6/12 s. These are hypothetical forecasts and actions,
not new executions of the saved path. Truth remains evaluator-only.

Compare orders 3/5 on both biased-wind adaptive seed-0 profiles and controlled
compact-field forecasts. Compare 5/7 if differences exceed max(1e-10, 1%) in J
or integrated expected excess (controlled fixtures use forecast mean excess).
Use one common coarser accepted order; if unresolved at 7, retain order 7 as
the declared finest-tested approximation and mark numerical convergence incomplete.
Apply existing spatial/time refinement, starting 120x160 / .05 s, with at most
two refinements. Insufficient convergence never becomes a superiority claim.

Report J, J/T, signed spatial-mean error, actual spending, expected excess,
background weight, fallback, ages, inferred bias and hypothesis concentration.
Preserve source/profile/startup limitations and distinguish estimator gains
from adaptive-versus-corridor gains. Two reading-noise seeds are descriptive.

## Acceptance and references

Check independent conditional batch calculations; mixture mass/moments; diagonal,
correlated and zero bias covariance; local forecast spreading; FOV exits without
renormalization; duplicate/delayed replay; fixed paths/actual clocks; round trips;
fake-scanner agreement; old regressions and unchanged playback-independent traces.
Review paired coarse/doubled physical and summary videos plus a blank diagnostic.

Gaussian conditioning: https://gaussianprocess.org/gpml/chapters/RW2.pdf
Gaussian sums: https://skoge.folk.ntnu.no/prost/proceedings/ifac2008/data/papers/2138.pdf
These support the inference construction, not performance of this scanner.
