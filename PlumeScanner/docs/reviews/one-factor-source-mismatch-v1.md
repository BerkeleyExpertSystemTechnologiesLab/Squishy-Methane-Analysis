# One-factor source mismatch study — 2026-10-28

Implemented and executed on `codex/source-prior-fixed-evidence` with starting HEAD
`5cac633f47001d3aa455f8b6f3cc00b11c550ed3`, preserving previous uncommitted studies.
Production model/policies and historical evidence remain unchanged. No slides,
commits or remote pushes. See the [protocol](../specs/one-factor-source-mismatch-v1.md).

## Main finding

None of the declared single-factor perturbations reproduces the original large
source-position displacement (~.063 FOV). All stochastic runs for the three
location-distribution priors remain below the predeclared .05 FOV large-error
threshold and fail the original-like direction/magnitude flag. This is a bounded
negative result for the tested magnitudes, source and path, not evidence that
these factors cannot cause large errors elsewhere.

All three factors can change localization and degrade posterior coverage. It is
important to distinguish average distance, the average displacement vector,
deterministic bias, and noisy localization scatter.

## Design

13 conditions: matched, plus true wind speed, source width or half-life separately
at .5, .75, 1.25 and 1.5 times the estimator's unchanged assumption. Each condition
uses one zero-perturbation mean control and ten preregistered seeds, for 143
histories. Every history uses the same 40 s, 708-observation path and acquisition/
delivery timestamps. Five source priors share exactly the same evidence per
history. At each event, RNG states are checked equal across conditions within a
seed, establishing alignment of the underlying random draws.

Truth wind changes magnitude only; inference wind remains unchanged. Width is
changed at fixed peak source rate. Half-life is 3, 4.5, 7.5 or 9 s, versus the
assumed 6 s. Prior centers, strength, noise parameters, footprint and latent grid
stay fixed except for the mathematical consequences of the named truth parameter.

## Broad-prior results

All source errors are distances in normalized (y,x) coordinates. The paired delta
is each seed's error minus its matched-control error, then averaged. HPD counts
are approximate quadrature diagnostics; a range marks unresolved grid membership,
not a statistical confidence interval. Other priors are fully saved in JSON.

| Truth condition | No-noise error | 10-seed mean error | Seed range | Mean paired error delta | Truth in 95% HPD |
|---|---:|---:|---:|---:|---:|
| matched | 0.00255 | 0.00678 | 0.00265–0.00944 | +0.00000 | 10/10 |
| wind_speed_0p5 | 0.01241 | 0.01085 | 0.00818–0.01403 | +0.00407 | 0–1/10 |
| wind_speed_0p75 | 0.00656 | 0.00549 | 0.00303–0.00704 | -0.00129 | 10/10 |
| wind_speed_1p25 | 0.00723 | 0.01447 | 0.00802–0.02184 | +0.00769 | 4/10 |
| wind_speed_1p5 | 0.01499 | 0.02197 | 0.01527–0.03145 | +0.01519 | 0/10 |
| source_width_0p5 | 0.00711 | 0.02601 | 0.01746–0.03563 | +0.01923 | 0/10 |
| source_width_0p75 | 0.00616 | 0.01832 | 0.00597–0.02927 | +0.01154 | 0/10 |
| source_width_1p25 | 0.00672 | 0.00893 | 0.00351–0.01221 | +0.00215 | 4/10 |
| source_width_1p5 | 0.00959 | 0.01132 | 0.00712–0.01415 | +0.00454 | 0/10 |
| half_life_0p5 | 0.02713 | 0.02412 | 0.01458–0.03453 | +0.01734 | 0/10 |
| half_life_0p75 | 0.01089 | 0.00964 | 0.00110–0.01775 | +0.00286 | 5/10 |
| half_life_1p25 | 0.00703 | 0.01005 | 0.00192–0.01483 | +0.00327 | 6/10 |
| half_life_1p5 | 0.01012 | 0.01283 | 0.00547–0.01787 | +0.00605 | 1/10 |

## Interpretation

- Wind 1.5x: broad-prior mean error .02197, increasing in all ten paired seeds.
  Mean displacement is (y=-.00320,x=+.01335), consistent in x direction with the
  original error but much smaller. Wind .75x slightly improves mean distance on
  this finite seed set; the small difference is exploratory and not established
  beyond stochastic/numerical uncertainty. Error need not appear monotonic in a
  small fixed-geometry experiment.
- Width .5x: largest broad-prior mean distance among these conditions (.02601),
  but deterministic mean-control error is only .00711. Mean displacement vector
  is (-.00928,+.00110), whose norm is much smaller than mean distance. Noise-driven
  variation contributes substantially; do not describe .026 as a uniform fixed
  directional bias. Errors increase in all ten paired seeds.
- Half-life .5x: deterministic mean-control error is .02713 and stochastic mean
  error .02412. Its stochastic mean x displacement is -.01589, opposite the original
  +x error. It creates displacement but does not reproduce the original pattern.
- Accurate regional priors give similar trends. Biased regional priors generally
  add some error but still stay below .05 in every stochastic history. Accurate
  and biased point locations are fixed by construction (0 and .20591 source error)
  and are sanity controls, not localization results.
- Source uncertainty can fail before error becomes large: e.g. wind 1.5x, width
  .5x or half-life .5x put truth outside all ten broad-prior 95% HPD sets. The
  matched control includes truth in all ten. This is sensitivity evidence, not
  general calibration estimated from ten fixed-geometry samples.

Width is not a pure geometry-only change here. Relative discrete input totals for
width multipliers .5/.75/1.25/1.5 are .5712/.6994/1.4878/2.1249. The original
point-sampled peak-amplitude injection on a 12x16 grid is preserved. The especially
narrow source is under-resolved relative to the latent cells; signal level and
spatial sampling effects accompany width changes. Half-life changes the original
model's decay-scaled process covariance along with persistence. Equal relative
parameter changes are exploratory and are not equally plausible physical errors.

## Verification and unresolved numerical checks

31 targeted tests pass, including isolated truth changes, unchanged inference
wind/timestamps, RNG alignment/reproducibility, independent generator moments,
affine likelihood equivalence, source-bank behavior and existing calibration.
All eleven matched histories reproduce previous saved scalar evidence and latent
truth exactly, and their 257-grid posterior means match within 1e-10. Inference
likelihood precision matrices match exactly across all histories, validating reuse
of the cached quadratic term. Linear/constant terms change with evidence.

Every history runs 129 and 257 grids; 19 histories additionally run 513 under the
predeclared fallback. All final source-mean and .05-radius probability changes
pass their frozen tolerances (max .0019461 FOV and .00083383 respectively).
**The complete refinement criterion does NOT pass globally.** Two histories
(three prior-specific verdicts) still change 95% HPD membership at 257->513:

- wind .5x, seed 20261031: broad and accurate-region; coverage count should be
  treated as 0–1/10 for each, rather than a resolved 0/10;
- width 1.25x, seed 20261029: biased-region; see the resolution bounds in derived JSON.

Their source-mean shifts are about 1.2e-6–3.1e-6 FOV. Thus these are unresolved
HPD boundary classifications, not unresolved large shifts of source means.
Initial failures are retained; thresholds are not loosened. Final source summaries
use 257 or 513 as recorded per history, not a silently uniform grid. The complete
posterior density and HPD rank are not claimed converged. The raw comparison's
coverage lines use finest-grid counts; use the derived resolution bounds when
quoting ambiguous counts. The derived error-only figure avoids that ambiguity.

## Scientific limits and next useful discriminator

This isolates sensitivity within the estimator's Gaussian forward-model family.
It does not identify the cause of the original synthetic simulator discrepancy,
which may involve stronger changes, multiple simultaneous discrepancies, spatial
representation, observation operators or plume shape. Source-location-only
inference can compensate for wrong nuisance parameters and become overconfident.
The fixed path was originally selected using source knowledge. No closed-loop
policy, multiple source locations, real methane data or physical emission rate is
validated.

A width-at-conserved-discrete-input control would separate geometry from the
changed signal amount observed here. Directly comparing the original simulator's
field/observation operator with the matched generator is also more discriminating
than attributing the original error to any one factor from this result alone.
Neither followup is implemented in this round.

## Artifacts and reproduction

All results are local under `outputs/2026-10-28-one-factor-mismatch-01/`:

- `config.json`, original evidence/configuration, environment and code snapshots;
- per scenario/seed: generated evidence, latent truth, actual truth parameters,
  RNG signatures, likelihood statistics, posterior weights/field means and scores;
- `refinement.json` per history retains every refinement verdict;
- `summary.json` has paired differences, all priors, magnitudes and directions;
- `derived/summary-with-resolution-bounds.json` exposes ambiguous HPD counts;
- `derived/source-error-comparison.png`, raw comparison and displacement figures;
- logs, report/protocol and SHA-256 manifests.

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/plume-study-mpl .venv/bin/python -u scripts/one-factor-source-mismatch.py --config configs/source-prior-study/one-factor-mismatch-v1.json --output outputs/<fresh-directory>
MPLCONFIGDIR=/private/tmp/plume-study-mpl .venv/bin/python scripts/summarize-one-factor-mismatch.py --input outputs/<fresh-directory> --output outputs/<fresh-directory>/derived
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_mismatch_control.py tests/test_model_consistent_control.py tests/test_source_likelihood_quadrature.py tests/test_source_prior_belief.py tests/test_scanner_calibration.py
```

The frozen config refers to the previous matched-study artifacts for exact replay
checks; preserve those inputs when reproducing. This study makes no runtime claim.
