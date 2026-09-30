# Evidence-gated focus loop v3 candidate

Status: implemented opt-in candidate. Not a default, not a calibrated detector, not
promoted by any result in this round. The estimator is the frozen calibrated Gaussian
model; this specification changes only the mode decision.

Policy identifier: `evidence_gated_focus_loop_v3_candidate`.
Control policy for the study: `full_fov_margin_loop_v2_candidate` (the frozen
calibrated mean-gated loop), with identical estimator, transport, mechanics and truth.

## Separated decisions

Two decisions are deliberately separated:

1. **Whether to focus** is decided by delivered scalar readings through the evidence
   gate defined below. It never consults the posterior, the source model, the
   planner, wind-derived pseudo-readings or evaluator truth.
2. **Where to focus** remains the existing calibrated predicted upper-signal field at
   the midpoint of the focus ceiling (latent `.95` upper quantile against the
   `.15` reporting threshold). No allocation change is introduced.

A focused region may grow, shrink, move or remain stable. Broadening in this candidate
means the charged return home followed by another complete full-FoV scan. There is no
intermediate padded confirmation region and no third geometry.

## Frozen settings

| Setting | Value |
|---|---:|
| Gate `alpha` (full-epoch budget) | `.05` |
| Full-epoch action alpha, action `p = 1, 2, ...` | `alpha / (p (p + 1))` |
| Focused-sweep action alpha | `.05` |
| Entry level | `.15` |
| Retain level | `.10` |
| Focus ceiling | `12 s` including declared delivery/receipt delay |
| Minimum focused readings | `4` |
| Vertical margin | `0` |
| Estimator formulation | `beam_reconstruction_decay_v1_candidate` |
| Transport | `zero_inflow_v1` |

These are predeclared candidate parameters, not fitted calibration and not a
statistically optimized detector. The lower retain level is explicit hysteresis, not a
changed map reporting threshold.

## The statistic

For an opened physical plan with `M` scheduled scalar acquisitions (frozen before
execution), and calibration `x_i = (reading.value - offset) / gain` and
`s = sensor_noise_std / gain`:

```
z = NormalQuantile(1 - alpha_action / (2 * M))
L_i = x_i - z * s
U_i = x_i + z * s
```

The bounds are Bonferroni **measurement-error** bounds on each clean exposure response,
conditional on the declared residual-noise model. They are not calibrated field
posterior bounds, not instantaneous full-scene bounds, not a session-wide error
guarantee, and not a test of whether an emission source exists. Neighbouring signal
values need no independence for the union bound; correlated or biased detector errors
would still require valid marginal error assumptions, which must be tested rather than
asserted for real instruments. Spatially different readings are never averaged as
repeated observations of one value.

`M` stays at the frozen scheduled slot count even after partial execution or missing
readings, so the correction stays conservative. Effective observed-value cutoffs at the
assumed noise `.05` and action alpha `.025` are:

| Scheduled slots | Observed value a single reading must exceed |
|---:|---:|
| 640 (coarse full scan) | `.35564779584685524` |
| 2560 (doubled full scan) | `.37111494274450096` |

The underlying signal level being tested is still `.15`; the difference is the
conservative noise and multiplicity allowance. This is a potential weak-signal failure
mechanism, not a hidden physical threshold. Each plan records the applicable cutoff.

## Diagnostic evidence states

For one applicable level:

- `supported`: at least one accepted `L_i > level` (strict; equality is not support),
- `below_level_on_observed_footprints`: every accepted `U_i < level`,
- `inconclusive`: neither of the above,
- `unavailable`: no accepted reading.

The last three all request a full scan but remain distinct labels. A focused sweep can
never prove whole-scene absence; only a completed full scan can support that statement.

## State transitions

1. Every session starts with a mandatory FULL coverage obligation. There is no prior
   bypass. Fresh actual-cell coverage is maintained per full-scan epoch; wide
   footprints do not invent visits.
2. At a complete FULL epoch, focus is entered only if at least one accepted
   `L_i > .15` comes from **that** epoch. Otherwise a new charged return-home/FULL
   epoch starts immediately. Historical support from earlier epochs cannot reopen focus.
3. After a fully executed focused sweep, focus is retained only if at least one fresh
   `L_i > .10` comes from that sweep. Otherwise a fresh return-home/FULL obligation
   starts at the next decision. There is no grace loop and no waiting for the predicted
   mean to fall.
4. When support exists but the existing focused compiler finds no legal focused region,
   the same charged FULL fallback is used. A gate never overrides physical feasibility.
5. Once FULL is required, its fresh coverage must complete before focus may resume,
   even if intermediate readings become positive. If the remaining session time cannot
   fit the obligation, the session ends with `incomplete_full_scan_budget`.
6. An aborted or partially executed FOCUS cannot renew focus; FULL is conservatively
   required. A partial FULL preserves missing cells and resumes the coverage
   obligation. A missing receipt does not close a scan. Invalid or conflicting evidence
   rejects atomically without changing gate or acquisition state.

Full-scan evidence is sequential and can be stale at completion. Support ages are
recorded explicitly and the early reading that justified one focus attempt must be
renewed by that focus's own evidence. This staleness is a declared limitation to test,
not permission to reuse an early reading indefinitely.

## Action budget accounting

- A normal one-plan FULL epoch uses `alpha_action = .05/2 = .025`.
- A FULL epoch spanning partial physical plans gives action `p` the alpha
  `.05 / (p (p + 1))`, whose partial sums stay below `.05` for the epoch.
- Each focused sweep uses `.05` with no session-wide guarantee.
- The action budget is allocated only when a nonempty plan opens; repeated read-only
  queries and no-action results allocate nothing.
- A full epoch persists the union of accepted evidence bounds and actual coverage
  across its plans.

## Gate contract

Each delivered reading must map to exactly one scheduled slot of the open plan
(the stable `scan_id:slot_index` reading identity, scan identity and reported
position are checked), and a partially
executed action reports fewer delivered slots than the frozen schedule. Unmappable,
duplicated or conflicting readings are recorded as a gate-contract violation: their
bounds provide no support and the gate conservatively requests FULL. Extra readings
remain valid inference evidence for the estimator under the existing public API; the
production ingestion validation is not weakened.

Actual acquisition midpoints need not equal planned midpoints. The shared
acquisition layer validates exposure start, single-detector cadence, delivery
availability and actual completion. Gate evidence retains the actual acquisition
and availability times, including valid delays and execution overruns.

## Integration seams

- `full_fov_loop.py` gains two inert hooks, `_on_plan_opened` and
  `_on_scan_finished`. Historical modes leave them empty and keep identical traces.
- `evidence_gate.py` holds the pure statistics and the serializable
  `GateReadingEvidence`, `GateActionEvidence`, `GateDecision`, `GateTransition`
  and `EvidenceGateState` messages.
- `evidence_gated_loop.py` holds `EvidenceGateSettings`, `EvidenceGatedSession`,
  `EvidenceGatedPlan` and `EvidenceGatedScanner`. The scanner subclasses the
  calibrated scanner and reads no evaluator or simulator module.
- `evidence_gated_execution.py` holds `EvidenceGatedRunConfig`; `decode_config`,
  `decode_plan` and `SensingController.from_session` dispatch on the versioned
  message and policy identifiers alongside the historical bindings.
- Mode decisions change only at completed-scan boundaries. Streaming and batch delivery
  of identical available evidence produce the same finished gate and belief; different
  delivery schedules may legitimately change online error.

## Diagnostics

Every returned plan carries the old mean-gate value (`gate_max_mean`), the diagnostic
evidence state, the applicable level, the action alpha and cumulative epoch alpha, the
frozen `M` and delivered slot count, the maximum lower and upper bound, the effective
observed-value cutoff, supporting reading identities with acquisition, availability and
age, the full epoch, requested and swept cells, the fallback reason, the last action's
execution completeness, and whether full coverage is incomplete. Gate support is never
called a source-active probability and no non-finite number is written; unavailable
values are explicit nulls.

The 2026-09-29 cleanup adds optional `gate_evidence_actions`, one compact record
per action contributing to the preceding decision: scan/epoch/action identities,
alpha, frozen and delivered counts, multiplier and observed-value cutoff. The
existing `gate_alpha_action`, `gate_action_index` and `gate_epoch_alpha_allocated`
describe the prospective next action; evidence counts and maximum bounds describe
the preceding decision. The scalar `gate_observed_value_cutoff` describes the last
evidence action, not a common cutoff for a multi-action full epoch. The renderer
labels these distinctions. Legacy plans decode with an empty evidence-action tuple;
public replay checks all their recorded fields and verifies new provenance when
the field is present. Policy settings and decisions on unchanged historical
evidence remain unchanged.

## Explicit non-goals

- No source-activity inference, no source-injection switch-off, no belief resetting and
  no suppression of remaining false model signal in the displayed map.
- No change to the calibrated estimator, transport, decay or source parameters.
- No margin tuning, resolution change, minimum-height rule or padded confirmation
  geometry.
- No UCB/risk allocation redesign, mixture extension, new drift model, covariance
  inflation or hardware/robotics work.
- Not a default policy, not a calibrated detector, and not evidence of reconstruction
  superiority.
