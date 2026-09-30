# Scan-value diagnosis and transport intervention

Status: implementation and study active. Start `7bd9d54` on
`codex/scan-value-diagnosis`. This is a diagnostic experiment, not a policy freeze.

Follow startup -> correlated conditioning -> prediction -> executable candidates
-> risk ranking -> actual evidence. Keep uniform physical-time squared map error,
instrument settings, sources, noise, v4 allocation and the renderer fixed.

H1: boundary defects materially contribute to failure. Correct zero exterior
inflow and compare fixed evidence/actions; unchanged losses/rankings weaken H1.
H2: forecast uncertainty misranks measurements. Compare each candidate's predicted
and actual repeated-pattern loss; agreement weakens H2.
H3: useful candidates are absent. Test deterministic local/shifted alternatives;
no improvement weakens H3 only within that family.
H4: repeated-pattern continuation misrepresents early replanning. Compare the
same first sweep with causal replanning; equivalent rankings weaken H4.
H5: persistent-mixture concentration removes useful uncertainty. Inspect selected
failures and residuals; concentrated accurate forecasts/rankings weaken H5.
Concentration itself does not establish error. Multiple causes may coexist.

The model stores a mean and spatial covariance, not independent occupancy
probabilities. Conditioning acts through covariance times the measurement
innovation. There is no separate Gaussian bump added to each bright pixel.
Motion-mixture disagreement represents constant wind-bias uncertainty, not
unpredictable local deformation. The mean, marginal standard deviation and
normalized expected-excess panels summarize different quantities; the v4 mass
panel is a saved decision forecast and diagnostic, not the risk surface.

## Explicit intervention and compatibility

Diagnostic session/run types bind `legacy_clipped_v1` or `zero_inflow_v1`.
The corrected advection uses zero exterior cell contributions without weight
renormalization; clipping only makes indices safe. Runtime and scorer use the
same transport identifier in cache keys. Observation interpolation, source,
diffusion and process noise remain unchanged. Historical payloads/hashes/defaults
stay unchanged. Public acquisition operations retain their validation.

READY snapshots serialize the complete detached inference/lifecycle state.
Forced-path diagnostics are separate from autonomous historical replay and
validate paths by recompiling from actual pose and detector timing. All truth
and filesystem access belongs to evaluator/adapter modules. Historical evidence
may be replayed under another transport without claiming its policy chose those
actions. Online scoring incorporates only evidence already delivered.

## Frozen study protocol

Select eight seed-0 v4 snapshots: Gaussian coarse/doubled known/biased wind,
Gaussian coarse prior-source-free full/blank, mixture doubled biased and coarse
prior-source-free full. Use the first decision after a completed sweep, after
actual startup when required, with >=12 seconds remaining. Freeze IDs/input hashes
before new outcomes. Missing fixtures remain unavailable. Horizon is 12 seconds.

Freeze production geometry from legacy inference. Gaussian extensions are 3/6 s
compact rectangles around the excess maximum, four one-inference-pitch shifts
of the 3 s rectangle, previous region and corridor; deduplicate to <=22 paths.
Rectangles grow in ascending normalized Chebyshev distance from the anchor,
with deterministic cell-order ties; retain the longest feasible closed rectangle.
Apply the existing mechanical closure and required service block without trimming.
Each transport evaluates the same geometry after identical historical evidence.
Test repeated patterns, first-sweep-then-replanning, and idle. Use the same source
realization/time-position noise coupling with 8 future-noise seeds, extending to
16/32 only for unresolved winners. Mixture follow-up executes only the production
winner, best expected-excess production proposal and feasible corridor.

Model-consistent independent arithmetic checks are per schedule, not substitute
shared-truth comparisons. Audit empty-event propagation partitioning and exposure
versus midpoint observations before attributing failures to decision theory.
Risk refinement: .1/.05/.025 s; mixture histories 32/64/128 as needed.
Evaluation starts 120x160/.05 s with representative doubled/halved checks.
Material effects exceed max(1e-10, 1% of reference loss); unresolved numerical or
paired sampling differences cannot establish candidate rankings. Confidence
intervals are descriptive for these fixed source realizations, not population
or multiple-comparison-adjusted discovery claims. Preserve 49-hypothesis limits.

No changing local drift, covariance inflation, arbitrary padding, weight resets,
new allocation or large comparison suite is authorized by this experiment.
