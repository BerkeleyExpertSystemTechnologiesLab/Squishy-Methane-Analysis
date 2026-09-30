# Scan-value diagnosis

Status: complete, 2026-09-16; engineering conformance and scientific attribution
are reported separately. Start `7bd9d54`, branch `codex/scan-value-diagnosis`.
Final verified implementation checkpoint: `28f5765`; the subsequent documentation
closeout revision is recorded in the local `closeout.json`.
The [protocol](../specs/scan-value-diagnosis.md) fixes the experiments before results.

## What the loop represents

An actual footprint reading is evidence about a retrieved beam-integrated scalar.
It is not a frame or an independent observation of every painted display pixel.
The mean is a current signal estimate. Covariance describes both marginal
uncertainty and how observations affect other cells. Expected positive excess
combines mean and marginal uncertainty above a reporting threshold; its
normalization is a relative scan mass, not a probability of whole-plume presence.
V4 candidate risk is expected time-integrated squared reconstruction error under
the declared model; measured reconstruction loss uses evaluator-only truth.

Conditioning is mu+ = mu- + Ph(y-h^T mu-)/(h^T Ph+r). Gaussian initial spatial
correlation gives a nearby influence, which subsequent evidence and dynamics
modify. An expected high reading need not raise the mean; an unexpected zero can
lower it. There is no additional Gaussian bump added to every high-value pixel.
For fixed Gaussian dynamics/noise, covariance does not depend on the reading
value. This is correct conditioning and can nevertheless be an inadequate model
of where plume reconstruction errors will develop.

A small unknown displacement gives, to first order, delta x approximately
-grad(mu)^T delta v * delta t. Thus uncertain displacement around signal gradients
is different from homogeneous additive field noise. This local linearization is
an explanatory mechanism, not a new implemented motion model or a complete
description of turbulent beam-integrated plume evolution. The persistent mixture
represents a session-constant wind offset; learning it cannot establish absence
of changing local deformation. A concentrated wrong family may remain confident.

The physical-time signal/uncertainty panels follow delivered evidence. The mass
panel is the forecast saved when the active path was selected. Playback, plotted
support and evaluator truth do not supply evidence. Fixed display saturation can
conceal amplitude discrepancies; loss uses unclipped values.

| Loop stage | Implemented operation and visible quantity | Diagnosis belongs here |
|---|---|---|
| Startup | A physical sweep acquires scalar readings at different times. The delivered-measurement panel accumulates those readings; the estimate is a filtered common-time field, not that raw mosaic. | Startup age and temporal distortion are real instrument costs. They are not evidence of missing Gaussian neighborhoods. |
| Correlated update | `signal_belief` applies the innovation through `P h`. A surprising positive reading can raise nearby means; a zero can lower them. A repeatedly expected high reading need not add more signal. | Calibration, footprint assumptions and fixed covariance determine influence. No independent positive bump is pasted around bright pixels. |
| Future prediction | Advection/diffusion/source/process noise propagate the field. The motion mixture retains one session-constant bias per component and updates its likelihood weight. | H1 boundary transport; H5 whether the available motion family still predicts well after concentration; event partition and exposure confounders. |
| Candidate construction | V4 proposes full-FOV excess, covariance-reduction and age rankings, applies physical closure and service requirements, and checks executable duration. | H3: a useful legal geometry may be absent. The candidate set is not restricted to the preceding ROI. |
| Risk ranking | The scorer predicts whole-field squared loss over the common physical-time horizon, under repeated candidate patterns and declared delivery timing. | H2: model risk may misrank actual value. H4: repeating the region may misrepresent replanning after the first sweep. |
| Execution | Only the first sweep is executed; the public controller assimilates actual delivered evidence and permits a new decision after completion. | Mechanical time, actual visits, receipt latency and service credit must remain causal. Forced diagnostic actions are labeled separately. |

For a correctly specified linear-Gaussian model, expected posterior squared
error is the projected covariance trace. That result does not say that a fixed
physical plume realization has zero model bias. Under misspecification, the
actual error can include spatially structured bias that the declared covariance
does not represent. Candidate generation can respond to the measured mean while
the Gaussian **final value score** remains independent of reading values for a
fixed schedule. This is the distinction between correct conditioning arithmetic
and a physically useful uncertainty model.

The full-field posterior persists between scans. V4 constructs candidates over
the full FOV and has no 90% execution gate; it does not discard or renormalize
away an unscanned tenth of the posterior each cycle. Repeated acquisition of a
small region can still leave other areas stale, but `0.9**n` is not the update
mechanism in the experiments diagnosed here.

The persistent mixture can introduce signal-dependent disagreement through
different transported means. It does not guarantee that those means span future
deformation. All 49 component states/log weights remain present in this round;
an effective hypothesis count near one is an observation, not by itself proof
that a weight update is erroneous or justification for resetting weights.

## Literature and transfer assumptions

| Primary source | Established contribution | Transfer limitation here |
|---|---|---|
| [Rasmussen and Williams, chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf) | Correlated Gaussian conditioning; with fixed kernel/noise, posterior variance is independent of observed target values. | The algebra does not validate our plume covariance, boundary, source or dynamics assumptions. |
| [Kandel et al., FAST](https://www.nature.com/articles/s41467-023-40339-1) | Instrument-integrated adaptive microscopy estimates reconstruction improvement and accounts for routing. | A material-image utility predictor does not validate dynamic plume prediction or our mechanical profile. |
| [Tang et al., 2026](https://proceedings.mlr.press/v300/tang26d.html) | Examines prediction-error amplification under misspecified Bayesian active learning and nonrepresentative acquisition. | Its analysis motivates checking model-dependent acquisition; it is not a proof that our failures share one cause or a ready-made plume policy. |
| [Scheel et al., 2026](https://amt.copernicus.org/articles/19/2343/2026/) | Beam-path methane reconstruction combines transport/state estimation; wind-model errors can exceed reported uncertainty. | Reflector geometry, source restrictions and retrospective smoothing differ from our causal angular-map task. |

Primary pages rechecked during this round. The existing
[research memo](../research/adaptive-beam-plume-formulation.md) remains background;
no literature performance claim is transferred to this instrument.

## Causal attribution standard

Startup age and finite cadence constrain every policy. Observed poor scans alone
do not distinguish absent candidates from misranked candidates. Likewise, a
model-consistent numerical test verifies arithmetic without establishing physical
calibration. Compare identical candidate continuations and common truth, and
separate fixed-evidence transport changes from changes in data collection.
"Production candidate" in this diagnosis means membership in the frozen
historical candidate family. Corrected transport scores that same geometry;
the experiment does not claim that a fresh unrestricted corrected-model decision
would generate exactly that family. After the forced first sweep, the replanning
continuation uses ordinary causal v4 candidate generation.

The completed study contains 9,536 paired continuation records and 16 idle
references from all eight declared snapshots. Six Gaussian snapshots use 18–20
frozen candidates; two mixture snapshots use only their declared two/three-path
subsets. Every horizon is 12 physical seconds. Future-noise counts are 8, 16 or
32 according to the predeclared bounded extension rule. No further extension is
requested; close winners at the limit remain unresolved. These are descriptive
comparisons on selected development fixtures, not independent confirmatory tests
or a general superiority claim.

| Hypothesis | Completed evidence | Conclusion and surviving falsification condition |
|---|---|---|
| H1 boundary transport | Independent edge controls confirm artificial inflow. Identical-evidence/actions comparisons show 92 material improvements and eight worsenings by paired sampling intervals across the tested paths. | A path-dependent contribution is established; these counts are descriptive, including the provisional mixture. Correction is not universally beneficial. A remaining ranking failure after correction cannot be assigned to this defect alone. |
| H2 model value misranking | Eight of 12 Gaussian fixture/transport contrasts pass the paired-effect and numerical checks. Useful production candidates often already exist. | Supported in these contrasts. Adequate held-out forecast/residual calibration with persistent ranking failure would contradict calibration as the primary remaining explanation. |
| H3 missing candidates | One supported Gaussian contrast: doubled known-wind, historical transport. Added geometry lowers J by 0.012002 relative to the best production candidate. | Limited support, not the dominant explanation here. No family limitation is assessed from the deliberately restricted mixture subset. No improving tested extension does not establish global optimality. |
| H4 repeated-pattern continuation | Two supported Gaussian contrasts, corrected known-wind coarse/doubled; action regret under replanning is 0.008894 / 0.002507 in J. | A secondary contributor in those fixtures. Other equal empirical winners do not prove equivalent continuation values. |
| H5 premature mixture concentration | Both snapshots concentrate near one effective component; the learned horizontal bias has the wrong sign, and plume-path residuals exceed background-path residuals. | Causal attribution unresolved: shared model mismatch and failed 49-component wind refinement remain. Concentration itself is not a reason to reset weights. |

Representative **Gaussian** repeated-pattern results below use the same initial
evidence and physical truth within each comparison. Values are mean J/T; the
alternative is the best realized production candidate, selected retrospectively
for diagnosis. This is not an executable oracle policy. Spending is reported
per candidate in `diagnosis-summary.json`; common ceilings are not equal spending.

| Fixture / transport | Future seeds | Predicted winner J/T | Better available production path J/T |
|---|---:|---:|---:|
| Coarse known wind / historical | 32 | 0.022806 | 0.015949 |
| Coarse known wind / corrected | 8 | 0.015448 | 0.014446 |
| Doubled biased wind / historical | 32 | 0.059783 | 0.017459 |
| Source-free full signal / corrected | 8 | 0.022337 | 0.020015 |

The corrected coarse/doubled biased-wind Gaussian empirical winners already
match the best tested production paths. Thus the repair changes the explanation
in some fixtures without resolving all known-wind/source-free failures. Blank
corrected contrasts remain numerically unresolved: the finest refinement changes
the relevant seed-0 contrast by 6.06e-7, exceeding its 5.28e-7 tolerance. Sampling
uncertainty and numerical uncertainty are kept separate. Historical negative
results and the 49-hypothesis quadrature limitation remain valid evidence limits.

## Engineering and numerical controls

The opt-in correction zeros individual out-of-domain advection contributions
without renormalizing the remaining stencil. Tests cover both velocity signs,
diagonal transport, nonnegative substochastic operators, outflow and runtime/risk
agreement. A source/noise/diffusion-free incoming horizontal edge impulse starts
with mean integral 0.0625. At 30 seconds the historical operator has approximately
0.988, while the corrected operator has 0.009539. The remaining corrected tail is
finite-grid numerical diffusion; this is not an exact continuum exit solver.

Independent joint Gaussian conditioning agrees with sequential conditioning to
floating-point tolerance. In 20,000 model-consistent histories, posterior MSE is
0.0289304 versus analytic 0.0289747 (0.153% difference). Existing independent v1
and v4 specification arithmetic also passes. These tests establish algebraic
consistency, not calibration to the synthetic plume generator.

Empty-event propagation partitions can change projected forecast variance by
10.54%. Therefore query purity alone does not establish a partition-independent
continuous-time model. Six representative Gaussian candidate comparisons with a
0.05-second propagation limit do **not** show material selected-candidate regret;
the largest observed regret is 4.89e-6. Historical propagation partitioning is
preserved and remains a numerical limitation.

All six representative Gaussian risk-grid checks accept 0.1 seconds against
0.05. Four targeted mixture checks accept 32 histories against 64 and 0.1 seconds
against 0.05; the history-count differences are zero at these concentrated
snapshots. This does not resolve the earlier 49-component wind-quadrature failure.
All ten Gaussian/mixture evaluation checks accept the shared 120 x 160 mesh and
0.05-second spacing against 240 x 320 / 0.025 seconds. Positive attribution
contrasts additionally receive a saved-path, seed-0 finer-grid check; this is
separate from paired future-noise uncertainty.

Exposure integration versus the estimator's midpoint approximation reaches
0.59 assumed-noise standard deviations RMS in these fixtures. Four fixed-path
Gaussian ablations preserve the same best realized candidate when midpoint
readings replace exposure averages while keeping additive noise, paths and
timing fixed. This limits that discrepancy's explanatory role for those rankings;
it does not certify every exposure approximation. Independent compact-signal
controls show exact stationary agreement and finite moving-signal discrepancy.

An evaluator-only best 12 x 16 bilinear projection has an irreducible error floor
of about 0.00330–0.00337 per second in the nonblank fixtures. This floor is common
to candidates at a given horizon and cannot explain a ranking reversal. It also
does not explain all the observed forecast error: corrected Gaussian 12-second
whole-field MSE is about 0.0165–0.0226, with much larger errors in the true signal
region. These fixed-realization discrepancies motivate calibration tests; they
are not posterior-coverage estimates from an independent population sample.
The best-grid projection uses instantaneous evaluator truth and is an oracle
lower bound, not a performance target known to be attainable by this scanner.
Its gap includes missing/stale/noisy evidence as well as model error.

The final full regression checkpoint is **447 passed, 5 skipped**. The skips concern
file/video/directory symlinks unavailable under this Windows execution profile.
Controller import-boundary checks are included. Compilation and `git diff --check`
pass. Generated traces, videos and reports remain local; no historical artifacts
or untracked `output/` files are rewritten.
Continuation wall times are recorded separately under concurrent offline load;
they exclude fixture setup and initial scoring. Mixture replanning can take
minutes for a 12-second physical diagnostic horizon. Compute is not charged to
the synthetic clock under the preserved study convention. This round makes no
real-time hardware-performance claim.

## Mixture concentration and observable mismatch

Both selected historical mixture runs retain 49 components, but their effective
hypothesis count falls from 9.398 (weighted quadrature prior, not a uniform prior)
to approximately one by the first completed sweep. The inferred bias is
(-0.011544, -0.023668) FOV/s. The generator's declared transport minus the supplied
nominal wind is (-0.012, +0.015) FOV/s. Thus the learned horizontal correction has
the opposite sign and yields almost stationary effective transport. This is an
evaluator-only comparison; the true velocity is never supplied to inference.

Concentration does not establish its own causal role. The source-free fixtures
have continuing emissions in the generator but no source input in the estimator.
The source-anchored model also lacks the generator's finite puff lifetime.
Near-stationary learned transport may compensate for mismatched source/lifetime
assumptions rather than identify physical wind. That explanation is plausible
from the configuration and learned bias; this round does not isolate it by
changing those assumptions. It is a reason to test model consistency before
resetting or flattening weights.

First-sweep predictive reading diagnostics use the exact component mixture
density and its scalar second moment. In doubled biased wind (seed 0), the
expected-excess alternative has mean squared standardized residual 56.925 under
historical transport and 62.161 under corrected transport. The model-preferred
background path has 0.849 in both. For source-free full signal the corresponding
values are about 3.66 for the excess path, 0.715 for the preferred covariance
proposal, and 5.18 for the corridor. These are descriptive residual diagnostics,
not Gaussian confidence tests on a mixture. Background-only validation could
conceal the discrepancy in the unobserved plume. No likelihood tempering,
component pruning, covariance inflation or weight reset is applied.

For the doubled mixture, the available 5.925-second excess proposal covers 77.1%
of actual excess at its first completion, while the preferred 11.975-second
background path covers none. Their eight-seed repeated J values are 0.33958 versus
0.60304 under historical transport and 0.30415 versus 0.59380 under correction.
This is strong descriptive evidence of an unhelpful score in the fixed mixture,
but remains numerically unresolved for the intended wind prior. Different paths
receive their own scalar readings; they do not share fabricated observations.

## Verification, artifacts and reproduction

Focused commits: `6b6218f` (transport/snapshots), `b15107e` (lifecycle/cache),
`85b61bf` (attribution/refinement), `297af65` (mixture/inspection), `2a4b3c0`
(formal audit/resume), `35c7152` (mismatch evidence), `28f5765` (reporting guards).
The local artifact manifest records each producing revision and file SHA-256;
resumes verify historical trace, declaration, configuration, snapshot and path
identities. The original `6b6218f` saved Gaussian continuation reproduces its loss
exactly under final replay. No historical configuration hashes are rewritten.

The command sequence and opt-in API are in [the implementation guide](../sensing-core.md).
Additional closeout commands, from the repository root:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest -q -rs
.\.venv\Scripts\python.exe scripts/check-policy-v1-spec.py
.\.venv\Scripts\python.exe scripts/check-reconstruction-risk-spec.py
.\.venv\Scripts\python.exe -m compileall -q src scripts examples tests
.\.venv\Scripts\python.exe scripts/scan_value_diagnosis_observers.py outputs/scan-value-diagnosis --phase contrasts
.\.venv\Scripts\python.exe scripts/summarize-scan-value-diagnosis.py outputs/scan-value-diagnosis
.\.venv\Scripts\python.exe scripts/plot-scan-value-diagnosis.py outputs/scan-value-diagnosis
.\.venv\Scripts\python.exe scripts/audit-scan-value-diagnosis.py outputs/scan-value-diagnosis --final
git diff --check
```

Local evidence is under `outputs/scan-value-diagnosis/`: `declaration.json`,
`diagnosis-summary.json`, `audit.json`, `artifact-manifest.json`, `closeout.json`,
fixture-level numerical/coverage/residual records, and separate plots. The ten
inspection videos all decode (466 frames total); physical-time progress and
compressed summaries were visually reviewed for paired coarse background scans
and doubled mixture failures. The existing six-panel layout is unchanged.
Summary labels explicitly say one frame per sweep. Playback at 8 fps/1x and
12 fps/2x preserves the execution hash. `inspection/review.json` records the
reviewed files and hashes. Videos show actual delivery/completion and charged
travel; they do not certify posterior calibration.

Resolved: isolated boundary implementation defect in the opt-in mode, detached
causal diagnostic execution, paired-path evidence integrity, numerical arithmetic,
and inspectable attribution records. Unresolved: physical model calibration,
partition-dependent propagation, the strict blank-signal contrast, wind-mixture
quadrature, general candidate completeness and changing local deformation.
The historical `not_demonstrated` claim remains unchanged.

## Next-stage handoff

The smallest justified next stage is **transport/model calibration, beginning
with source and lifetime consistency**, on fixed evidence and paths. Keep the
corrected transport as an explicit experimental reference. Do not freeze a new
policy or change the uniform reconstruction objective to make the present scores
look better. Candidate and continuation findings should remain available as
secondary explanations rather than be discarded.

Use a bounded identification check: one source-free advected pulse whose source
has actually stopped, and one continuously replenished plume, with known wind,
the existing instrument/noise settings and the same small saved path set. Separate
best-grid representation error, common-time mean forecast error and observation
residual variance. Keep fitting evidence separate from the forecast interval used
to judge calibration. The immediate question is whether the declared source and
transport assumptions can explain the mean before assigning unexplained errors
to local wind deformation. These are proposed follow-up controls, not additional
completed experiments in this round.

If source/lifetime consistency removes the structured residual and risk-ranking
failure, implement only the supported calibration/model correction under a new
identifier. If accurate mean forecasts still leave gradient- and age-dependent
uncertainty unexplained, then conduct the separate formulation round for changing
local displacement/deformation. That is the route from the user's neighborhood
intuition to a calibrated predictive mechanism. It does not require arbitrary
Gaussian padding or relaxing learned weights.

Falsification: if fixed-path forecast calibration is adequate within the declared
tolerances but action regret persists specifically under replanning, prioritize
the measured continuation limitation. If a corrected-model production family still
lacks an improving legal path, prioritize that demonstrated candidate extension.
Unsettled blank-signal rankings and fixed-mixture quadrature must remain explicit
unresolved tests; neither warrants another broad superiority suite.
