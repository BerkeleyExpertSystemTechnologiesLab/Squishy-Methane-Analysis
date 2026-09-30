# Reconstruction-value acquisition candidate

Status: implemented and evaluated experimental candidate, 2026-09-15; not a new
policy freeze. The [completed review](../reviews/reconstruction-value-policy-v4.md)
records conformance, mixed Gaussian outcomes and numerically unresolved mixture
outcomes. Starting revision
`51d1f633f8720d3cbcf14d61d2b2c7a27e201e5b`, branch
`codex/reconstruction-value-policy`. Policy: `reconstruction_risk_segment_v4_candidate`.

## Objective and boundaries

Minimize expected whole-field integrated squared reconstruction error over a
common future horizon equal to the current time ceiling. Enhancement-index error
remains a separate outcome. Compare executable repeated scan patterns, execute
only their first sweep, then decide again from actual evidence. This is bounded
lookahead, not global optimization. There is no mandatory shrinkage or retained
mass fraction. Retain four minimum readings, fixed physical profiles and full
selected-cell preservation through mechanical closure.

Stage the existing Gaussian estimator followed by the unchanged persistent-wind
mixture. Preserve assimilation, noise, source assumptions, dynamics and v1-v3
semantics. Boundary clipping and changing local drift remain required subsequent
model work. No vision model, hardware/networking or within-sweep replanning.

## Candidates and service

Rank the full physical lattice by midpoint expected reportable excess,
single-reading covariance-reduction proposal score, and oldest actual visit.
For each ranking enumerate prefixes for ceilings B/4, B/2, 3B/4 and B, retaining
the longest feasible prefix. Stop enumeration only when acquisition time alone
exceeds the ceiling. Include the legacy executable proposal and declared corridor;
deduplicate to at most 14 paths. Aggregate mixture covariance is permitted only
for the proposal score, never as a replacement mixture risk.

At a decision at least 30 seconds after the last credited service completion,
require one oldest reachable 0.2-FOV-height survey block; rotate deterministic
age ties. This check does not interrupt an active sweep or guarantee a visit
exactly every 30 seconds. Include the block in candidate geometry and
charge its entire compiled cost. If none fits, report service unmet and continue
considering other legal actions. Credit only actual complete block visits.
Preserve full-scene startup until all physical cells were actually visited.

Recompile the same core region from subsequent predicted poses within the common
horizon, applying the same service rule and detector recovery. Include expected
delivery/receipt timing. Propagate without new evidence over the final remainder.
An idle prediction is a diagnostic reference, not a replacement for a legal action.
Select minimum risk, then shorter duration and deterministic cell order on ties.
Threshold 0.15 and achieved 90% coverage are diagnostics, not execution gates.

## Risk and numerical checks

Use the evaluator's bilinear spatial projection and uniform cell-area weights.
For Gaussian fields, integrate projected conditional covariance. For the mixture,
analytically average the within-component covariance contribution and sample
possible reading histories for expected between-component variance. Each history
has one persistent generating hypothesis; update all filtering hypotheses.
Precompute covariance/linear mean operators and vectorize histories. Use stable,
deterministic common random numbers, retaining every component and log weight.

Let `G` be the evaluator's bilinear map from latent cells to evaluation points,
`W = I / n` its uniform spatial averaging, and `A = G^T W G`. For integration
nodes `u_l` and intervals `dt_l`, the Gaussian implementation scores

\[
R(S) \simeq \sum_l \Delta t_l\,\operatorname{tr}(A P_l^S).
\]

Here `P_l^S` incorporates only readings delivered by `u_l`, propagated to that
time using the existing estimator. A fixed schedule makes those conditional
covariances independent of unknown reading values. Travel and waiting leave
intervals without new evidence; repeated readings use the already-conditioned
covariance, so redundancy is accounted for. `projection_gram` and `score_pattern`
in `sensing_risk` implement this expression; it is not mean latent-grid variance.

For a mixture with current weights `alpha_k`, deterministic conditional component
covariances `P_kl`, and hypothetical delivered evidence `Y`, the score is

\[
R(S) \simeq \sum_l \Delta t_l\left[
\sum_k \alpha_k\operatorname{tr}(A P_{kl})
+ E_Y\sum_k \alpha_{kl}(Y)
 (\mu_{kl}(Y)-\bar\mu_l(Y))^T A
 (\mu_{kl}(Y)-\bar\mu_l(Y))\right].
\]

The first term uses current weights because expected posterior component weight
equals its current weight, and each component's covariance is deterministic for
the fixed schedule. The second term uses posterior weights and means separately
for each sampled history. Draw one generating component per history, retain it
throughout the horizon, and generate sequential readings from that component's
conditional predictive likelihood. The hypothetical filter updates every
component. Nested histories and common random numbers compare candidates without
requiring different executed paths to receive identical measurements.

Read-only quadrature queries do not become propagation or assimilation events.
The time mesh includes delivery and receipt boundaries. These expressions are
model-conditional Bayes risks for a fixed repeated pattern; they are neither
guarantees about simulator truth nor an adaptive multi-step optimum.

Start event-aware time quadrature at 0.1 s, refine to 0.05 then 0.025 s if needed.
Start with 32 histories, refine to 64 then 128. Require risk changes and selected
candidate regret within max(1e-10, 1%) on predeclared representative decisions.
Freeze one common setting before comparisons; use the finest setting and report
numerically unresolved if bounded refinement fails. Zero-covariance mixtures
must agree with Gaussian scoring without Monte Carlo error. Keep biased-wind
49-hypothesis settings fixed and preserve their earlier convergence limitation.

## Declared evidence and completion

72 main records: two profiles, three existing cases, both estimators, threshold90 /
corridor / risk policies, seeds 0 and 1. Another 24 records: coarse prior startup,
source-free biased wind, signal scales 1 / 0.1 / 0, both estimators, old/new adaptive
policies and seeds 0/1. Verify compatibility before reusing 60 historical controls;
36 new risk runs are required. Preserve 75 s, 36/12 s ceilings, physics/noise and
120x160 / 0.05 s evaluation, with representative evaluation refinement.

Acceptance includes independent small-model risk calculations, delayed evidence,
read-only queries, actual overrun/partial service, the saved 12.25 s / 12 s failure,
candidate geometry, fake scanner/replay/serialization and regression boundaries.
Review unchanged-layout physical and summary videos for both profiles and blank
signal. Report predicted risk versus realized error, map/index error, actual cost,
ages, coverage and service. No superiority claim follows from lower predicted risk.
Close with numerical classifications by estimator, reproducible commands, hashes,
producing revisions and synchronized changelog/status/implementation documentation.
