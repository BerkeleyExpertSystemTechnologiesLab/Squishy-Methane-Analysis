# Fixed-evidence source prior study — 2026-10-28

Implemented and executed locally on `codex/source-prior-fixed-evidence`, starting
from `5cac633f47001d3aa455f8b6f3cc00b11c550ed3`. No production policy changed;
no commits or remote pushes were made. This is an exploratory estimator study,
not evidence of general superiority or a source-localization success.

## Question and protocol

How does source-location information affect the current field belief under exactly
the same measurements? Follow [the protocol](../specs/source-prior-fixed-evidence-v1.md)
and `configs/source-prior-study/v1.json`. Five priors receive the same 708 readings
from the existing 40 s continuous, zero-added-noise demo. The accurate regional
and point priors use oracle location information explicitly. Regional priors have
standard deviation .06 FOV and retain soft support over the entire FOV. The biased
center is displaced by (-.10,+.18) in normalized (y,x).

`SourcePriorBank` holds a conditional Gaussian field per fixed source hypothesis.
Conditional covariance is shared exactly; each prior's source probabilities are
updated using pre-assimilation predictive likelihoods. Forecasts and evaluation
are causal. The source rate, width, wind, decay, sensor model and physical history
are unchanged. Source-activity and strength inference are not implemented.

## Results

Lower J/T means lower causal, time-averaged field MSE. The 129x129 source grid gives:

| Prior | J/T | Change vs accurate point | Prequential reading RMSE | Final source-mean error (FOV) |
|---|---:|---:|---:|---:|
| Broad uniform | .01652363 | -0.27% | .14318 | .06285 |
| Accurate region | .01611498 | -2.74% | .14241 | .06285 |
| Biased region | .01642077 | -0.89% | .14309 | .06289 |
| Accurate point / original | .01656866 | reference | .14793 | 0, prescribed |
| Biased point | .02514605 | +51.77% | .18613 | .20591, prescribed |

The biased regional prior has about 34.70% lower field error than the biased point
prior on this history. It permits observations to revise the initial position
assumption; the point prior has no support at alternative locations.
The accurate region has about 2.47% lower J than the broad prior. These are single
descriptive comparisons, not statistically established gains.

The three distribution priors ultimately concentrate near (y=.543, x=.213), while
the generator source is (y=.547945, x=.150470). At 40 s, probability within .05 FOV
of the generator source is only approximately 7e-11 to 1.5e-10. Thus a better field
prediction is accompanied by a biased, strongly concentrated source posterior.
Model mismatch, discretization and unmodeled errors are plausible explanations;
this experiment does not isolate their contributions.

Final 95% marginal field interval coverage on the 20x32 inspection grid is about
83.1–83.3% for mixtures, 82.0% for the accurate point, and 75.9% for the biased point.
Per-reading predictive interval coverage is about 86.2–87.4%. These numbers are
descriptive, not validated coverage guarantees: one history, correlated field
pixels, synthetic/model mismatch, and no added noise despite assumed sigma=.05.

## Verification and refinement

- 21 targeted checks passed: new mixture tests and original calibration tests.
- Independent original filters match the bank for multiple sources/updates.
- Sequential source weights match an independent joint Gaussian likelihood.
- Total variance, detached forecasts and invalid-input/time checks passed.
- Every original public plan replays. Accurate-point mean max difference:
  1.2212453270876722e-15; shared covariance max difference: 0.
- Accurate-point J differs from saved baseline by 4.44e-16.
- All 708 reading identities/timestamps/values are shared; original file hash unchanged.
- 33 to 65 candidates per axis: mixture J changed 2.11–2.20%, FAIL at 1% tolerance.
- Separate 65 to 129 followup: J changed 0.081–0.097%, PASS at the same tolerance.
- J stability does not establish source posterior convergence. Effective support
  changes from about one to two cells and mean location shifts about .006 FOV.
- Reviewed comparison, source-posterior and final-field figures. No slides created.

## Limits and next discriminating experiment

The fixed evidence was acquired by a known-source policy. It includes one full
scan followed by two source-informed focused scans, so it is not an unbiased
source-discovery benchmark. All priors are available at startup; no detector
handoff or evidence-fusion dependence is tested.

The original Gaussian source input uses a peak rate and point-sampled injection,
not conserved integrated physical emission. At order 65, prior-expected discrete
input integrals are .0020861 (broad), .0021601 (accurate region), .0021780 (biased
region), .0023051 (accurate point) and .0022751 (biased point). Boundary truncation
and alignment with the latent grid explain numerical variation. We preserve the
original implementation for exact replay and do not silently renormalize it.

Before attributing a paper-level benefit to source information, isolate why the
posterior moves away from the generator source: use a separately labeled
model-consistent observation control on the same path, followed by latent-grid
and source-input discretization checks. Keep the present mismatched synthetic
history as a reference. Then evaluate independent histories and closed-loop scans.

## Artifacts and reproduction

Local ignored output folders:

- `outputs/2026-10-28-source-priors-01/`: frozen 33/65 configuration and results.
- `outputs/2026-10-28-source-priors-refinement-01/`: frozen 65/129 followup.

Each contains the copied evidence/configuration, provenance/environment, source
snapshots, per-reading prediction trace, causal loss trace, posterior NPZ arrays,
three PNG figures, refinement verdict and SHA-256 manifest. The source bank and
runner snapshots describe uncommitted code on top of the starting commit.

From PlumeScanner:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/plume-study-mpl .venv/bin/python scripts/compare-source-priors.py --config configs/source-prior-study/v1.json --output outputs/<fresh-initial-directory>
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/plume-study-mpl .venv/bin/python scripts/compare-source-priors.py --config configs/source-prior-study/refinement.json --output outputs/<fresh-refinement-directory>
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_source_prior_belief.py tests/test_scanner_calibration.py
```
