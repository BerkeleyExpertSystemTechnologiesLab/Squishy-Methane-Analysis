# Numerical audit extensions (2026-10-29)

Status: implemented, opt-in engineering references. Historical experiment,
instrument, policy and estimator identifiers remain frozen. These utilities do
not establish a policy advantage or physical methane calibration.

## Signal, sensor and likelihood are separate contracts

`SyntheticMethaneLeakConfig` defines an empirical 2-D signal-puff fixture.
Its peak decays while its width grows; its spatial integral is not normalized
as methane mass. `clean_signal=True` removes display background and clipping;
it does not produce ppm or kg/s. Gaussian puffs are a valid modeling family,
but the present amplitude, shimmer, lifetime and wind meander are synthetic.

`FakeScanner(..., quadrature=None)` preserves the historical four-ray spatial
average and three-time Simpson exposure. That is an exact definition of a
surrogate sensor, with no general finite-beam accuracy guarantee. The optional
`SensorQuadratureV2` uses normalized tensor Gauss-Legendre spatial averages
and composite temporal Gauss-Legendre integration. It follows the moving pose,
clips the rectangle to the FOV, splits known path/source boundaries, and keeps
the same acquisition/availability clocks and seed/time/position noise keys.
Default orders are 16 in each spatial axis and 8 in time, with at most .01 s
temporal panels. A per-reading limit prevents excessive node allocation.

```python
from plume_focus.sensor_quadrature import SensorQuadratureV2
from plume_focus.sensing_simulation import FakeScanner

profile = SensorQuadratureV2()
scanner = FakeScanner(source, noise_std=0., quadrature=profile)
execution = scanner.execute(plan)
# Record profile.to_dict() with any future experiment using this adapter.
```

The frozen session runners still construct the historical adapter. They do not
silently switch experiments to v2. The current inference likelihood remains its
declared four-point, exposure-midpoint approximation. Using the refined adapter
with that filter deliberately introduces an observation-operator mismatch; it
must be labeled and tested, not called an upgraded coherent estimator.
`exact_rectangle_observation_weights` supplies a separate analytic reference
for the clamped bilinear latent field, integrating piecewise-linear basis
functions at every grid knot. A future versioned inference binding must use a
consistent spatial/exposure likelihood before making calibration claims.

## Cessation and endpoint conventions

The historical `ControlledSignal` and `WindowedSignal` gate the whole field.
They remain disappearance fixtures, including for old cessation-named records.
`ScheduledPuffSource` (`scheduled_signal_puffs_v3`) gates births by explicit
half-open time windows, retains previously emitted puffs, and uses the new
`half_open_v2` lifetime convention. Puff survival is `0 <= age < memory`;
this removes the isolated double-counted expiry endpoint at aligned birth times.
Finite memory and empirical amplitudes remain toy assumptions.

```python
from plume_focus.scheduled_puff_source import ScheduledPuffSource
from plume_focus.video_sources import SyntheticMethaneLeakConfig

config = SyntheticMethaneLeakConfig()
source = ScheduledPuffSource.from_phase_seed(
    config, phase_seed=6101,
    # Explicit prehistory warm-up, then stop injecting at 75 s.
    emission_windows=((-config.memory_frames/config.fps, 75.),),
)
# Save source.configuration(): all resolved phases, window and source version.
```

The phase seed sets the generator's three actual phases. It is independent of
FakeScanner's detector-noise seed. Three seeds or phases do not establish
independent environmental validation worlds or a held-out generalization claim.

## Evaluator-only time integration

`TruthQuadrature` retains `temporal_rule="left_endpoint_v1"` by default.
`temporal_rule="event_split_midpoint_v2"` uses interior midpoint nodes, splits
explicit breakpoints and retains the partial final interval. For saved traces,
provide all evidence-delivery/receipt changes as well as relevant source
discontinuities; otherwise a cell may straddle a causal-estimate change.
Refine step, phase and event splitting before interpreting small differences.
Common truth-squared quadrature errors can cancel in paired losses, so their
absolute size alone does not invalidate a policy ordering.

Truth and predicted statistics must be finite. `max(0, NaN)` is not permitted
to turn an undefined loss into zero. Counterfactual configuration and snapshot
sessions must agree; nonpositive horizons and unsupported lifecycles fail early.

## READY snapshots and scope of old diagnostics

`DecisionSnapshot` v1 and its serialized bytes retain historical semantics.
Version 2 additionally supports exact built-in FullFovController,
CalibratedScanner and EvidenceGatedScanner types. It records the session, decay
binding, moments, wind, actual visits, delivered identities, current missing-full
mask, return-home obligation, epoch, full-scan count, gate evidence/decision,
alpha accounting, transitions and decision-evidence provenance.

```python
from plume_focus.controller import SensingController
snapshot = controller.decision_snapshot()  # only with no active scan
independent = snapshot.fork_controller(SensingController)
```

These are detached *decision* states; the preceding scan's playback event list
is not a second historical trace. No source/truth is captured. Unknown recovery
subclasses must supply an explicit state adapter, so extra budgets cannot be
silently omitted. The existing `run_counterfactual` still supports historical
policy sessions only. A full-FOV/evidence/P2 forced-action adapter remains a
separate integration task; v2 snapshots alone do not implement it.

## Correctness and output integrity repairs

Invalid generator fields, nonfinite pointing and delivery times fail before
source queries. Empty clean-source batches are supported. The legacy diffuse
plume generator now gives identical values for flattened and reshaped queries.
Paired statistics reject duplicate results, undeclared/missing budgets and
unequal episode sets. Recorded manifests reject duplicate keys and NaN/Infinity.
Legacy sensing CLI run entry points refuse nonempty output directories before
computation; within-run progress files can still update deliberately. Existing
recorded no-overwrite/hard-link protections are unchanged.

Analytic Gaussian averages, independent piecewise integration, polynomial
exposures, discontinuous-time integration, seed determinism, partial/full/gated
snapshot continuations and malformed inputs are regression fixtures. They are
not atmospheric, hardware, posterior-calibration or policy-superiority evidence.
