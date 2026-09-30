"""Shared acquisition lifecycle; no allocation, simulation or historical-policy imports."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from plume_focus.sensing_contracts import CausalWind, SensorReading, ExecutionReceipt, ScanPlan, Position, finite
from plume_focus.signal_belief import BeliefView, GaussianSignalBelief
from plume_focus.signal_distribution import PredictiveDistribution

@dataclass(frozen=True)
class AssimilationEvent:
    reading_id: str
    acquired_at_seconds: float
    available_at_seconds: float
    belief: BeliefView


class AcquisitionController:
    def _initialize_acquisition(self, capabilities, model, source_position, instrument=None):
        self.capabilities, self.model = capabilities, model
        self.source_position = source_position
        self._belief = GaussianSignalBelief(model)
        self._wind = CausalWind()
        self._active: ScanPlan | None = None
        self._base = None
        self._pending: dict[str, SensorReading] = {}
        self._received: dict[str, float] = {}
        self._seen: dict[str, SensorReading] = {}
        self._last_sampled = np.full(capabilities.lattice_shape, -1.0)
        self._last_available = 0.0
        self._last_pose: Position | None = None
        self._scan_count = 0
        self._rotation = 0
        self._events: list[AssimilationEvent] = []
        self._instrument = instrument
        self._last_detector_reading: SensorReading | None = None
        self._last_survey_completed = 0.0

    @property
    def visit_times(self) -> np.ndarray:
        return self._last_sampled.copy()

    @property
    def active_plan(self) -> ScanPlan | None:
        return self._active

    @property
    def assimilation_events(self) -> tuple[AssimilationEvent, ...]:
        return tuple(self._events)

    def belief_view(self) -> BeliefView:
        # A delivered past reading updates historical state; current knowledge
        # is forecast to the last availability time before being exposed.
        belief = self._belief.clone()
        belief.advance(max(belief.time_seconds, self._last_available), self._wind, self.source_position)
        return belief.view()

    def predictive_view(self, at_seconds: float) -> BeliefView:
        """Read-only prediction using only the evidence already delivered."""
        if at_seconds < self._last_available - 1e-10:
            raise ValueError("Causal view precedes incorporated evidence availability; replay delivery events for past views.")
        belief = self._belief.clone()
        belief.advance(at_seconds, self._wind, self.source_position)
        return belief.view()

    def predictive_moments(self, at_seconds: float, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Causal projection for adapters/evaluators; no truth or retrospective access."""
        if at_seconds < self._last_available - 1e-10:
            raise ValueError("Causal view precedes incorporated evidence availability.")
        belief = self._belief.clone()
        belief.advance(at_seconds, self._wind, self.source_position)
        return belief.moments_on(shape)

    def predictive_distribution(self, at_seconds: float, shape: tuple[int, int], *, fast_projection=False) -> PredictiveDistribution:
        """Detached marginals; consumers may calculate mass/bands without truth inputs."""
        if at_seconds < self._last_available - 1e-10:
            raise ValueError("Causal distribution precedes incorporated evidence availability.")
        belief = self._belief.clone()
        belief.advance(at_seconds, self._wind, self.source_position)
        if hasattr(belief, "distribution_on"):
            return belief.distribution_on(shape, fast_projection=fast_projection)
        mean, variance = belief.moments_on(shape)
        return PredictiveDistribution(at_seconds, np.ones(1), mean[None], variance[None])

    def motion_diagnostics(self) -> dict | None:
        return self._belief.diagnostics() if hasattr(self._belief, "diagnostics") else None

    def predictive_aggregate_moments(self, at_seconds, shape):
        """Read-only mixture moments for loss-only quadrature refinement."""
        if at_seconds < self._last_available - 1e-10:
            raise ValueError("Causal moments precede incorporated evidence availability.")
        belief = self._belief.clone()
        belief.advance(at_seconds, self._wind, self.source_position)
        return belief.aggregate_moments_on(shape) if hasattr(belief, "distribution_on") else belief.moments_on(shape)

    def ingest_readings(self, readings: tuple[SensorReading, ...] | list[SensorReading], *, available_at_seconds: float) -> tuple[AssimilationEvent, ...]:
        finite("delivery time", available_at_seconds)
        if available_at_seconds < self._last_available - 1e-10:
            raise ValueError("Delivery calls must be chronological.")
        trial = dict(self._pending)
        new = []
        for reading in readings:
            prior = self._seen.get(reading.reading_id) or trial.get(reading.reading_id)
            if prior is not None:
                if prior != reading:
                    raise ValueError("Conflicting duplicate reading identity.")
                continue
            if self._active is None or reading.scan_id != self._active.scan_id:
                raise ValueError("Reading does not belong to the open scan.")
            if reading.acquired_at_seconds - reading.integration_seconds / 2 < self._active.started_at_seconds - 1e-10:
                raise ValueError("Reading precedes scan start.")
            if reading.available_at_seconds > available_at_seconds + 1e-10:
                raise ValueError("Reading is not yet available.")
            if self._instrument is None and not np.isclose(reading.integration_seconds, self.capabilities.integration_seconds):
                raise ValueError("Reading integration differs from declared scanner capabilities.")
            trial[reading.reading_id] = reading
            new.append(reading)
        ordered = sorted(trial.values(), key=lambda r: (r.acquired_at_seconds, r.reading_id))
        sequence = ([self._last_detector_reading] if self._instrument is not None and self._last_detector_reading else []) + ordered
        rate = self._instrument.acquisition.detector_capacity_hz if self._instrument else self.capabilities.readout_hz
        for left, right in zip(sequence, sequence[1:]):
            minimum = 1 / rate
            if self._instrument:
                minimum = max(minimum, (left.integration_seconds + right.integration_seconds) / 2)
            if right.acquired_at_seconds - left.acquired_at_seconds < minimum - 1e-10:
                raise ValueError("Readings exceed the single detector readout cadence.")
        if not new:
            self._last_available = available_at_seconds
            return ()
        # Validation completes before any mutation.
        replay = bool(self._pending) and min(r.acquired_at_seconds for r in new) < self._belief.time_seconds - 1e-10
        self._pending = trial
        self._received.update({r.reading_id: available_at_seconds for r in new})
        if replay:
            self._belief = self._base.clone()
            # Keep earlier availability snapshots for faithful playback. New
            # replay states become available now, not at past acquisition times.
            work = ordered
        else:
            work = sorted(new, key=lambda r: (r.acquired_at_seconds, r.reading_id))
        produced = []
        for reading in work:
            self._belief.assimilate(reading, self._wind, self.source_position)
            event = AssimilationEvent(reading.reading_id, reading.acquired_at_seconds,
                                      max(available_at_seconds if replay else self._received[reading.reading_id], reading.available_at_seconds),
                                      self._belief.view())
            produced.append(event)
            self._events.append(event)
        self._last_available = available_at_seconds
        return tuple(produced)

    def _finish_evidence(self, receipt: ExecutionReceipt) -> BeliefView:
        if self._active is None or receipt.scan_id != self._active.scan_id:
            raise ValueError("Receipt does not match an open scan.")
        if receipt.completed_at_seconds < self._active.started_at_seconds:
            raise ValueError("Completion precedes scan start.")
        if receipt.received_at_seconds < self._last_available - 1e-10:
            raise ValueError("Completion receipt precedes delivered evidence.")
        if any(r.acquired_at_seconds + r.integration_seconds / 2 > receipt.completed_at_seconds + 1e-10 for r in self._pending.values()):
            raise ValueError("A reading lies after actual scan completion.")
        self._belief.advance(receipt.received_at_seconds, self._wind, self.source_position)
        for reading in self._pending.values():
            iy, ix = np.minimum((np.array(reading.position_yx) * self.capabilities.lattice_shape).astype(int), np.array(self.capabilities.lattice_shape) - 1)
            self._last_sampled[iy, ix] = max(self._last_sampled[iy, ix], reading.acquired_at_seconds)
        if self._pending:
            self._last_detector_reading = max(self._pending.values(), key=lambda r: r.acquired_at_seconds)
        self._seen.update(self._pending)
        self._last_pose, self._last_available = receipt.final_position_yx, receipt.received_at_seconds

    def _close_scan(self):
        self._scan_count += 1
        self._active = None
        self._pending = {}

    def _open_scan(self, plan):
        self._active = plan
        self._base = self._belief.clone()
        self._pending, self._received, self._events = {}, {}, []
        self._last_available = plan.started_at_seconds
