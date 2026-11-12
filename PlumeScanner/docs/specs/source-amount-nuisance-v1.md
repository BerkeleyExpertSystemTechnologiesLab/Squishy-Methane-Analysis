# Source amount under wind and decay mismatch v1

Frozen before execution on 2026-11-12, starting from `f93b2a4` on
`codex/source-amount-nuisance-diagnostic`. This is a local, fixed-path diagnostic,
not a policy change or a physical emission-rate experiment.

## Question and comparisons

With true source input fixed at Q0, does joint location/amount inference explain
wind or decay mismatch by reporting a different amount? Compare the same scalar
readings with (1) nominal dynamics and fixed Q0, (2) nominal dynamics and joint
location/amount inference, and (3) true wind/decay dynamics and fixed Q0. The third
is an information control, not a zero-error target or an estimator told location.
The fixed-Q0 and true-amount controls coincide; do not duplicate them.

One matched case and wind-speed/half-life multipliers .5, .75, 1.25 and 1.5
are declared. Truth changes one named dynamics parameter; inference for the first
two methods stays nominal. The single source stays active, at the same location
throughout each history; width and in-FOV-normalized total input stay fixed.

Use the existing .0125 s fixed clock, zero-inflow transport, cell-integrated source,
12x16 field grid, .035 FOV width, nominal 6 s half-life, optical footprint and
708-reading/40 s schedule. Forecast without new readings at +6 and +12 s; the
source remains on. The inherited schedule was selected with source information:
this does not evaluate unknown-source search. No scanning policy is executed.

## Single-factor noise convention

Hold initial covariance and the **discrete per-tick process innovation covariance**
at nominal values in all conditions and all methods. For half-life mismatch only
transition A and within-step source response change; process noise is not
additionally rescaled by decay. Wind changes likewise affect A. Each condition
uses the same initial standard-normal draws, per-tick process draws and keyed
sensor noise within a seed. Realized field noise differs after propagation by
different A; do not claim identical realized process noise across conditions.
The information control uses the correct A/source response and the same fixed Q.
This deliberately separates decay dynamics from the Q rescaling in the old study.
The matched case must reproduce the declared nominal generative distribution.

## Cases, isolation and saved evidence

Development: original source, deterministic mean and seeds 20261101/20261102.
Evaluation: two new prespecified positions, each a deterministic mean and four
new seeds. Nine conditions x 13 histories = 117 histories. Evaluation aggregates
exclude deterministic controls. The same four random trajectories are reused
across evaluation positions and conditions; eight combinations are not eight
independent physical environments. No tuning on evaluation results.

Location prior is uniform on the complete FOV, with no truth-inserted candidates.
Amount prior is uniform on [.25,1.75] Q0, matching the earlier candidate. Oracle
parameters enter only the explicitly labeled information control. Reference
truth and source-relative footprints are evaluator-only. At each condition,
all methods share readings, coordinates, footprint, acquisition and availability
times. Persist immutable truth/noise, scalar evidence, physical-time errors,
snapshot fields, filter sufficient statistics, posterior diagnostics, config,
code snapshots and SHA-256 manifests. Never overwrite a completed run.

## Outcomes and numerical gates

Primary: causal monitoring J/T on [0,40), +6/+12 s map MSE, and joint amount
mean/central-95% interval at 40 s relative to true 1 Q0. Also report paired signs,
source errors, amount support-edge mass and 95% predictive membership/PIT at five
fixed global and two source-relative footprints at 20/40/46/52 s. Report global
and source-relative diagnostics separately. Coverage counts are descriptive,
correlated, pointwise quantities, not whole-field calibration guarantees.

Run source/amount quadrature 129/64 then 257/128. For every history/method require
location-mean shift <=.002 FOV; amount mean and interval endpoint shifts <=.01 Q0;
J/T and forecasts change <=max(1e-6,1% of finer MSE); predictive PIT shifts <=.01.
Membership flips for predictive or amount intervals remain explicitly unresolved
until a finer run stabilizes them. Retain all initial failures. Refine affected
histories to 513/256, then 1025/512 if needed; never relax thresholds. Numerical
failure limits the affected claim; it is not grounds to omit that condition.
The latent field clock/grid is the declared finite model, not a continuum claim.

Use analytic amount moments for event updates only after independent quadrature
tests; snapshot diagnostics use the actual quadrature mixture. Check causal
delivery scoring, direct known-source filtering, evidence replay, conserved
input, matched-method identity, information-control parameter isolation and
sampling invariance. Repeat diagnostic endpoints without new readings.

No scientific pass/fail rule requires joint inference to improve. A lower map
error with amount shifted away from 1 supports error compensation, not accurate
source recovery. An interval excluding 1 under known true Q0 diagnoses misleading
amount certainty within these cases. Correct-parameter controls can have higher
realized MSE; they are not guaranteed per-history lower bounds. Support saturation,
nominal regressions and negative cases remain visible. Hardware calibration,
multi-source/activity inference, external physics, improved acquisition and
real-time feasibility remain untested.

Deliver three figures (reconstruction/forecast error, inferred amount, predictive
coverage), machine-readable paired summaries, a worked mean-control map, and a
review explaining which source-strength claims survive nuisance mismatch.

## Publication reproduction entry (added after frozen execution)

`--fresh-schedule` regenerates the unchanged baseline's schedule rather than
requiring the ignored historical record. The canonical SHA-256 of ordered
position/acquisition/availability/footprint/integration fields must be
`01d7eb58bb9ecd82994af62ca990526a413896e48e0ba68edeb33fc86c36d09a`.
Baseline reading values are not reused as nuisance evidence. The generated
record/configuration receive new file hashes, with the archived hash retained as
lineage. This entry does not alter the frozen record, conditions, priors, scoring
or inference; `--smoke` remains an engineering-only reduced experiment. Unit tests
construct a short synthetic schedule and require no historical output files.
