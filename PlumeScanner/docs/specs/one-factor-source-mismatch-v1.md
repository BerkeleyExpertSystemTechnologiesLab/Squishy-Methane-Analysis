# One-factor source-location mismatch control v1

The user requests separately varying wind speed, source width and decay time to
identify which can recreate source-location displacement. This is a fixed-path,
evaluator-only sensitivity experiment. No policy, default model or original
record is changed. Starting HEAD is `5cac633f47001d3aa455f8b6f3cc00b11c550ed3` on
`codex/source-prior-fixed-evidence`; existing local study changes are preserved.

## Design frozen before execution

`configs/source-prior-study/one-factor-mismatch-v1.json` declares 13 conditions:
matched plus each of three factors at truth/assumed multipliers .5, .75, 1.25, 1.5.
Each condition uses a deterministic mean control and the same ten seeds
20261028–20261037 as the previous matched experiment (143 generated histories).
All five source priors are retained and receive identical evidence within a
history. Exact affine likelihoods evaluate final source quadrature; there is no
mixture collapse or estimator tuning.

Only the named truth-generator parameter changes:

| Factor | Inference assumption | Truth variations |
|---|---|---|
| Wind speed | velocity (y,x)=(-.001095890410958904,.03836990595611285) FOV/s | multiply entire vector; preserve direction |
| Source width | sigma=.035 FOV | .0175, .02625, .04375, .0525 |
| Decay time | half-life=6 s | 3, 4.5, 7.5, 9 s |

The source position, peak injection rate, initial distribution, kernel, diffusion,
measurement noise, reading footprint, timestamps and frozen 40 s path are fixed.
The 708 acquisition/delivery times, including .0125 s latency, remain unchanged.
Inference always sees the original supplied wind and original model parameters.
No detector output or truth parameters enter inference.

Changing width with fixed peak rate also changes total discrete injection. This
is explicitly a width-at-fixed-peak test, not a conserved-emission geometry test.
Changing decay changes both deterministic persistence and the original model's
decay-scaled process covariance. Thus "one parameter" does not imply just one
physical effect. Wind direction error is not tested. The factor ranges are
exploratory relative perturbations, not empirically estimated sensor uncertainty;
equal percentage changes do not imply equally plausible or equally severe errors.

Pair each seed across all conditions. Verify identical RNG state after every
advance/observation to ensure the same underlying standard normals are consumed,
even if transition subdivisions could have changed. Abort on a mismatch instead
of silently labeling unaligned draws as common random numbers. Random process
increments may have different magnitudes under decay perturbation, as intended.
No seeds are selected after results. Signed Gaussian truth is retained; no
positivity clipping or physical methane realism claim.

## Measurements and comparisons

Primary: final source posterior-mean error/displacement, seed-paired error and
vector changes relative to matched, probability within .05 FOV, and true-location
95% HPD membership. Report mean, range, number of increased paired errors and
Monte Carlo SE of paired error changes. These are descriptive statistics over
ten realizations at one fixed source/path, not a population calibration claim.

Two descriptive flags are declared before execution:

- large error: source posterior mean distance >= .05 FOV;
- original-like displacement: x displacement >= +.05 FOV and abs(y) <= .02 FOV.

This distinguishes similar distance from similar direction. The original history
had approximately +.063 x displacement. These flags are diagnostics, not formal
hypothesis tests or proof of the original simulator's cause. Compare deterministic
mean controls as well as random realizations; signal/noise confounding under weak
or strong source inputs remains visible.

Secondary: final latent field MSE on the original 12x16 grid. No physical-time
quadrature is computed, so do not call it J/T or compare directly with the initial
study's integrated field score. Source-point errors are prescribed and cannot be
updated by data; they are sanity controls, not localization methods.

## Numerical and engineering verification

Evaluate 129 and 257 source grids for every history. If a history fails the
existing source-summary check, additionally run 513 for that history; preserve
both the failure and refinement. Conditions: mean shift < .002 FOV, absolute
.05-radius mass change < .02 and unchanged true-location HPD membership. Report
any unresolved case rather than relaxing thresholds. Selected summaries passing
does not certify convergence of the full posterior density or HPD rank.

Affine conditional mean and likelihood are exact for fixed Gaussian dynamics.
All histories have identical inference covariance and likelihood quadratic term;
check precision matrices exactly equal before reusing cached source-grid
quadratics. Different readings change linear/constant terms. The previous matched
scalar evidence, latent truth and 257 source posterior means must reproduce.

Test isolated parameter changes, unchanged inference wind/acquisition metadata,
common RNG alignment, and reproducibility, alongside the existing generator,
affine-likelihood, bank and calibration tests. Preserve original evidence SHA-256.
Save every generated reading history, truth array, truth/inference parameters,
likelihood statistics, posterior grid weights, final means/scores, refinement
history, code/environment, logs, figures and file manifests. Generated artifacts remain local.
