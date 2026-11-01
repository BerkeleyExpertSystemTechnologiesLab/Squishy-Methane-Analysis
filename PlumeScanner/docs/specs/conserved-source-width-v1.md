# Conserved discrete source-width control v1

Question: did the prior width-at-fixed-peak experiment's localization error arise
mainly from spatial shape or from the changed injected signal amount? Keep the
inference model fixed and separate the two factors in the truth generator.
Starting branch `codex/source-prior-fixed-evidence`, HEAD
`5cac633f47001d3aa455f8b6f3cc00b11c550ed3`; preserve all prior local work and evidence.

## Frozen design

For the fixed true source location define its discrete source vector

    e_sigma(q)[i] = q * exp(-||grid_position[i] - source||^2 / (2*sigma^2))
    I_sigma(q) = mean_i e_sigma(q)[i]
    r(k) = I_(k*sigma0)(q0) / I_sigma0(q0)

The grid covers unit normalized area with equal cells, so its spatial integral is
its arithmetic mean. Baseline sigma0=.035, q0=.2829738193616142, grid=12x16.
Use k=.5,.75,1.25,1.5, with three conditions for each k and a shared baseline:

| Arm | Width | Peak rate | Total discrete source input |
|---|---|---|---|
| Matched baseline | sigma0 | q0 | I0 |
| Fixed peak / joint | k*sigma0 | q0 | r(k)*I0 |
| Fixed total / shape | k*sigma0 | q0/r(k) | I0 |
| Amount only | sigma0 | q0*r(k) | r(k)*I0 |

This is a 2x2 shape-by-amount design for each width. The fixed-peak arm exactly
reproduces the prior width histories; the amplitude-only arm has the same total
input as that arm with baseline shape. Fixed-total conserves the actual discrete
injection, NOT a continuous Gaussian formula proportional to sigma squared.
Conservation includes the source's current grid alignment and boundary truncation.
It does not enforce constant evolving field mass: transport, boundary outflow and
decay still act. It is not physical methane mass/second or a new calibrated source
strength definition. Total injection conservation need not preserve pointwise
measurements or observed SNR when spatial shape changes.

All inference hypotheses retain the original source width, peak-rate convention,
wind, diffusion, half-life, prior field covariance and measurement model. Truth's
source position, time history, dynamics and noise remain fixed. Only truth sigma
and/or q change as declared. Normalization uses known simulator truth for this
evaluator experiment, not information supplied to the estimator.

Use the same 40 s fixed path, all 708 reading positions/footprints/acquisition and
delivery times, and the same five source priors. Each of 13 conditions uses a
zero-perturbation mean control and seeds 20261028–20261037: 143 histories, of which
55 matched/fixed-peak histories must exactly reproduce prior evidence/truth.
Initial/process/measurement RNG states must match after every event within each
seed. Gaussian noise is unchanged in amplitude; it is not scaled with source q.
Signed Gaussian fields remain intentional. No replan, strength inference or
adaptive-policy comparison is introduced.

## Analysis and predeclared checks

Primary metrics: final source posterior-mean error/direction, posterior mass
within .05 FOV and true-position 95% HPD membership. Secondary: final latent field
MSE, not time-integrated J/T. Point-prior source errors are prescribed controls.
Report all three distribution priors, deterministic and stochastic outcomes.

For per-seed source error (and separately squared error), calculate

    shape effect = E(shape,I0) - E(baseline)
    amount effect = E(baseline shape,r*I0) - E(baseline)
    joint effect = E(shape,r*I0) - E(baseline)
    amount removal at changed shape = E(shape,r*I0) - E(shape,I0)
    interaction = joint - shape effect - amount effect

Save per-seed contrasts, means, standard errors and sign counts. Nonlinear errors
need not add; do not force a percentage split or call these universal causal
contributions. A fixed geometry/ten-seed comparison cannot establish general
superiority or calibrated source uncertainty.

Evaluate 257 and 513 source grids for every arm, with the same grids for paired
contrasts. Retain original thresholds: source-mean shift < .002 FOV, absolute
.05-radius mass change < .02, unchanged 95% HPD membership. Save failures and
report unresolved verdicts explicitly. No automatic extra grid is predeclared in
this round. Summaries using fine grids do not imply every density or HPD rank has
converged; small effect differences near numerical/stochastic uncertainty remain
exploratory.

Require relative discrete-input error <5e-14 for the targeted amount in every
history. Test both interior and boundary source positions, input-vector identities,
unchanged other parameters, unit multiplier and invalid inputs, alongside earlier
generator/inference tests. Two subnormal floating-point ULPs are allowed only in
vector-tail identity tests; the integrated amount tolerance remains unchanged.
Assert equal inference precision before caching likelihood quadratic terms.
Reproduce previous source posteriors at shared available grids within 1e-10.

Save configuration, original evidence hash, generated observations/latent truth,
normalization factors, exact likelihoods, posterior arrays, paired summaries,
refinement results, code/environment, plots and logs with SHA-256 manifests.
No slides, production-policy changes, commits or remote pushes.
