# Signal-gated threshold scanning: implementation and evidence

Status: implementation and bounded study complete. No new policy freeze.

Started at `5bb58ee6bd200f7fb168dc61f8a9e6fb69ba0458` on
`codex/threshold-bound-scanning`. Operational implementation: `1149a13`;
bound/comparison observers: `67bca3f`; coverage/audit checkpoint: `f003270`.
The [frozen experimental protocol](../specs/threshold-bound-scanning-v5.md)
implements the approved signal gate and fixed intensity cutoff. The estimator,
source, noise, profiles, renderer layout and squared reconstruction objective
were preserved. All new comparisons use explicit zero-inflow transport.

## What changed

The planner searches the entire physical lattice every cycle. Supported current
mean signal enables focusing; the forecast upper bound determines eligibility
and priority. A fixed cutoff can produce a stable, growing, moving or shrinking
region. Prefix selection respects motion, acquisition and receipt timing, reports
omissions, and preserves requested cells through closure. Age-based scanning
continues when the gate is off or a focused action cannot execute. A due service
block remains charged and receives credit only from actual readings.

Separate session/run and transport bindings preserve old decoding and hashes.
Exact mixture quantiles retain all 49 hypotheses; they do not use aggregate
Gaussian confidence formulas. Mean-only and upper-bound policies share every
other rule. Corridor and age controls retain their historical v2 service rules;
their contrasts therefore do not isolate the uncertainty term.

## Gaussian allocation evidence

The following are relative reductions in mean causal loss from upper-threshold
scanning. Positive values favor upper-threshold scanning. Short runs use eight
paired future-noise seeds and a 12-second horizon; full sessions use two seeds
and 75 seconds. Common ceilings are not identical expenditure.

| Fixture | Short vs v4 risk | Short vs mean-only | Full vs v4 risk | Full vs mean-only |
|---|---:|---:|---:|---:|
| Coarse known wind | +8.52% | -1.01% | +2.21% | -0.58% |
| Coarse biased wind | +8.73% | -2.16% | +2.79% | -0.23% |
| Doubled known wind | -2.53% | +13.03% | -3.86% | +0.52% |
| Doubled biased wind | -12.46% | +5.99% | -4.63% | +1.24% |
| Source-free full signal | -1.09% | 0.00% | -4.97% | -0.01% |
| Source-free blank | -4.37% | 0.00% | +43.20% | 0.00% |

After descriptive paired intervals, a 1% material threshold and numerical
contrast error, short coarse improvements versus v4 and doubled deteriorations
are supported in these fixtures. Against mean-only, uncertainty helps both
doubled cases and hurts coarse biased wind; the coarse known-wind difference
straddles the material threshold. Source-free comparisons with v4 remain
unresolved; upper and mean short runs agree there. Full-session percentages are
descriptive, not two-seed significance claims. Blank absolute losses are small.

Absolute full-session upper-threshold results (mean of seeds 0/1):

| Fixture | Causal J/T | Signed mean-index error | Charged seconds |
|---|---:|---:|---:|
| Coarse known wind | 0.016888 | +0.005298 | 74.950 |
| Coarse biased wind | 0.016978 | +0.002196 | 74.900 |
| Doubled known wind | 0.016608 | -0.000385 | 74.950 |
| Doubled biased wind | 0.018025 | +0.007649 | 74.937 |
| Source-free full signal | 0.021721 | -0.008115 | 74.975 |
| Source-free blank | 0.000045917 | -0.000386 | 74.975 |

Thus allocation usefulness is **mixed**, and uncertainty's incremental value is
**mixed**. This is not evidence of general reconstruction superiority.

The 48 mixture continuations are complete. In the doubled biased-wind fixture,
upper-threshold loss is 49.39% below v4 but only 0.18% below mean-only. In the
source-free full-signal fixture it is 0.26% above v4 and 16.09% above mean-only.
All mixture conclusions remain **numerically unresolved** because the fixed
49-hypothesis wind approximation did not pass its earlier quadrature study.
An allocation contrast within that approximation does not remove the limitation.

## Bound reliability and the decision loop

At a six-second forecast, approximately 91–100% of actual above-threshold cells
exceed the Gaussian nominal 95th-percentile bound in the five nonblank saved
fixtures. The two mixture fixtures have roughly 42% and 73% exceedances. These
are spatially dependent, fixed-truth frequencies, not independent calibration
trials or whole-plume containment probabilities. Nevertheless, they show the
declared bounds are poor operational upper predictions for these fixtures.
The nominal interpretation is not validated by attractive mean maps or by an
allocation gain. Blank signal has no signal-stratum trials: that value is null.

The mechanism is observable: coarse focusing repeatedly samples the signal
region instead of the background favored by v4. On doubled fixtures, service
travel and focused measurements do not consistently improve whole-field loss.
The eligible region expands as well as contracts; no recursive 90% crop exists.
The seed-0 source-free full session alternates focus and fallback, with four
gate-negative decisions while truth still contains reportable signal. Blank
sessions stay in age scanning. The large blank full-session gain is entirely a
fallback result and cannot be credited to focused uncertainty allocation.
Gate diagnostics also include initial/startup decisions, which must not be
mistaken for failed focused decisions.
No false-positive mean gate occurred in these Gaussian full-session runs.
That is evidence only for the declared fixtures, not a guarantee that persistent
false mean signal cannot prevent fallback elsewhere. In blank videos the
normalized excess diagnostic may still appear bright because tiny masses are
normalized; the gate remains off and that panel does not drive v5 allocation.

Sensitivity to .90/.95/.99 quantiles and .10/.15/.20 thresholds is recorded only
on saved states. Defaults were not tuned. Planned-acquisition/completion latent
bounds and delivered noisy reading residuals are separate evidence. Exposure
integration, inherited propagation partition sensitivity, source/lifetime
mismatch and unresolved wind quadrature remain limitations.

Noisy reading residuals tell a complementary story. For seed-0 short upper runs,
mean squared standardized predictive residuals are 23.4–42.7 in the four
source-anchored Gaussian fixtures, versus 0.96 in blank signal. The source-free
full-signal Gaussian run is near 1.04 while missing the plume: it mainly samples
background. Agreement on acquired readings therefore does not certify an
accurate unsampled field. Mixture values are 32.2 (doubled) and 5.0 (source-free).
These are descriptive, selection-dependent diagnostics, not independent tests
with a chi-squared calibration claim.

## Verification and artifacts

Full regression: **473 passed, five Windows symlink skips**. Independent v1 and
risk arithmetic, import-boundary tests, compilation and Git diff checks pass.
The fake-scanner example reproduces the simulator's first two plans after JSON
normalization. New tests cover exact mixture/atomic quantiles, gate crossings,
zero-variance equivalence, no noise bonus, changing regions, non-monotonic travel,
partial startup, service credit, receipt ceilings, duplicates, delayed replay,
overruns, snapshot purity, serialization and misleading metadata rejection.

Local artifacts live under `outputs/threshold-bound-v5/`; untracked `output/`
and all historical evidence remain untouched. The declaration records input,
configuration, snapshot and historical trace SHA-256 identities before outcomes.
Run records include producing revision and exact delivery batches. Earlier
observer artifacts name the then-current base checkout while observers were
developed; observer source was subsequently committed at `67bca3f` and
`f003270`. Operational policy code was committed before the runs.

All **348 executions**, **348 completion-coverage records**, **152 planned-bound
and reading-residual reports**, and **eight bound/sensitivity fixtures** are
complete. All eight representative upper-threshold short executions pass the
240x320 / 0.025-second refinement check; loss changes are 0.066–0.268%, below
the 1% criterion. The base 120x160 / 0.05-second mesh remains the common reported
mesh. Numerical checks concern quadrature of the fixed model and trace; they
do not validate the model or the 49-hypothesis wind approximation.
All eight short and six full-session seed-0 comparison refinements pass at the
first finer level. Numerical changes in loss contrasts are at most 0.262% of
comparator loss in short checks and 0.095% in full checks. Descriptive intervals
still leave close comparisons unresolved. Eight representative public-API replays
(five Gaussian policies and three mixture policies) reproduce plans and losses;
all 348 records pass input lineage, physical path, acquisition, delivery, receipt
and common-horizon auditing.

Sixteen videos were fully decoded: paired upper/risk physical-time and summary
views for coarse Gaussian, doubled Gaussian and doubled mixture, plus blank
upper views and a separate playback variant. Representative start, acquisition,
and completion keyframes were visually reviewed. Scalar readings, signal mean,
uncertainty, normalized excess diagnostic, actual path and age remain separate.
The 8-fps/1x versus 5-fps/2x comparison has an identical execution SHA. The
default 0.0125-second delivery lag is sub-frame; exact delay semantics are
checked in the trace and tests. Inspection details are in `inspection/review.json`.

Reproduction commands are in the [implementation guide](../sensing-core.md).
Run `python -m pytest -q`, `scripts/check-policy-v1-spec.py`,
`scripts/check-reconstruction-risk-spec.py`, `python -m compileall -q src tests
scripts examples`, and `git diff --check` in the declared virtual environment.
`OPENBLAS_NUM_THREADS=1` is used throughout. Execution-producing revisions are
`1149a13` (316 records), `67bca3f` (24) and `d967b87` (8); the operational policy
was unchanged across those checkpoints. Final revision, verification commands
and the full artifact-manifest digest are recorded in local `closeout.json`.

SHA-256 anchors:

| Artifact | SHA-256 |
|---|---|
| Frozen declaration | `bb167751c2da854e20029891ca5b4995618f3e449edb781a994829e2007abe5a` |
| Final comparison summary | `cbf76d20658dc350f813bdf16f367c1bc50f68508275f116aed2fba1b504b815` |
| Media review | `71da552587548389f1b6d4724237da806ba8c0f82f161fd94e0057c12d941c2c` |
| Public replay verification | `6e8fcd13023597eec378662ffb40d79cdd02c69831c740547d754387df172615` |

Resolved: versioned threshold allocation, exact mixture quantiles, explicit
partial/fallback semantics, causal snapshot substitution, physical accounting,
repeatable numerical/reporting checks and faithful inspection. Unresolved:
general reconstruction advantage, reliable uncertainty coverage, source/lifetime
and spatial representation mismatch, exposure/propagation sensitivity, local
drift, mixture quadrature and real-time feasibility. No hardware claim is made.

## Smallest justified follow-up

Use fixed executed paths to test source/lifetime and forecast calibration before
another allocator redesign. Separate mean underprediction, uncertainty coverage,
coarse-field/footprint representation error, and exposure/propagation numerical
effects on held-out forecast evidence. These are competing explanations, not
causes isolated by this allocation study. Keep
the threshold policy available as an allocation control, with the mean-only
ablation. Do not claim that uncertainty padding would repair these results.
Changing local drift remains separate required motion-model work; this round
does not implement it or authorize a further comparison suite.
