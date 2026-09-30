# Focus extent diagnosis

Declared 2026-09-18, starting revision `44be6a9`, branch
`codex/focus-extent-diagnosis`. This is an evaluator-only diagnosis, not a policy
change. Keep the lean full-FoV loop, Gaussian estimator, zero-inflow transport,
physical profiles, 0.15 threshold and 0.95 quantile unchanged operationally.

Freeze saved evidence hashes and decision event indices before probing. Use all
18 lean runs: first decision after completed startup, first at/after 180 seconds,
and (cessation only) first decision after the first post-180 full rescan. Deduplicate;
missing events remain unavailable. Zero-noise results precede noisy confirmation.

Trace delivered readings, current mean, future mean, upper bound, requested,
closed and sampled regions. Unknown locations are not measured zeros. Evaluate
truth at the matching physical time; saved clean readings describe exposures.
Measure amplitude, area, wind-aligned extent, connected peak component and a
supplementary half-maximum bright core. No relative threshold replaces 0.15.

Hypotheses and falsification:

* Amplitude implementation: independent scalar innovation checks at readings
  0/.1/.3/1 must match. Agreement contradicts a missing amplitude multiplier.
* Source contribution: compare identical-evidence source-off replay and future-only
  source-off prediction. Unchanged extent limits source attribution.
* Representation: separately halve correlation length, double inference resolution
  and double the diagnostic footprint on controlled reading probes. Unchanged
  errors within resolution tolerance limit this explanation.
* Spread overlap: future-only diffusion-off, process-noise-off and both-off probes.
  Existing covariance remains. Compare effects on mean and uncertainty separately;
  a narrower bound is not evidence of better calibration.
* Uncertainty eligibility: same-posterior mean versus upper-bound masks; report
  added background and captured true signal. If mean is already broad, uncertainty
  cannot explain that mean width.
* Timing/persistence: 0/1/3/6/12 second forecasts plus focused midpoint/completion;
  moving packets and continuous sources with/without fading and finite lifetime.
  Nominal velocity remains velocity, not velocity change.
* Mechanical closure: requested versus swept cells, full-mask compiled cost and
  infeasibility. Equality rules out closure enlargement for that decision.

Audit mechanisms by affected state, physical assumption and active call path.
Classify distinct roles, overlapping assumptions, actual duplicated implementation
and inactive compatibility code. Do not remove safeguards or historical policies.

Alternative-model replay follows recorded paths using the shared acquisition
engine, never requires reproducing historical policy decisions, and must first
match public-controller replay under unchanged assumptions. No truth enters
inference. Detached queries must not mutate evidence or operational state.

Outputs: immutable declaration, resumable per-run records, controlled probes,
separate diagnostic figures, representative doubled-mesh checks, existing video
timestamp links, hashes, review and smallest justified handoff. No new autonomous
execution matrix, estimator tuning, arbitrary Gaussian padding or policy freeze.
Differences below one physical pitch or mesh error remain unresolved. Preserve
historical `not_demonstrated` results.
