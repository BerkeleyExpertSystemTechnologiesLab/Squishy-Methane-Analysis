"""Headless frozen-v1 simulator and serializable controller-call trace."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from plume_focus.controller import SensingController
from plume_focus.sensing_contracts import CausalWind, ExecutionReceipt, Position, SensorReading, finite, position
from plume_focus.sensing_messages_v2 import ScanPlanV2, ScanRequestV1
from plume_focus.sensing_session import RequiredMessage, SensingSessionV1, configuration_hash
from plume_focus.sensing_simulation import CleanMethaneSource, FakeScanner, SensorSource, SensingScanTrace
from plume_focus.video_sources import SyntheticMethaneLeakConfig
from plume_focus.signal_belief import BeliefView, positive_expectation


@dataclass(frozen=True)
class SyntheticSourceV1(SyntheticMethaneLeakConfig, RequiredMessage):
    end_time_seconds: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "source", tuple(float(v) for v in self.source))
        object.__setattr__(self, "wind_velocity", tuple(float(v) for v in self.wind_velocity))
        if len(self.shape) != 2 or any(type(n) is not int or n < 2 for n in self.shape):
            raise ValueError("Synthetic source shape must have two dimensions >= 2.")
        for value in (self.fps, self.base_sigma, self.emission_interval, self.memory_frames):
            finite("source scale/time", value, positive=True)
        if self.end_time_seconds is not None:
            finite("source end time", self.end_time_seconds)

    def build(self) -> CleanMethaneSource:
        fields = asdict(self)
        end = fields.pop("end_time_seconds")
        return CleanMethaneSource(SyntheticMethaneLeakConfig(**fields), end_time_seconds=end)


@dataclass(frozen=True)
class ReconstructionGridV1(RequiredMessage):
    shape: tuple[int, int]
    step_seconds: float
    spatial_quadrature: Literal["uniform_cell_area"]
    temporal_quadrature: Literal["left_endpoint"]

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "step_seconds", float(self.step_seconds))
        if len(self.shape) != 2 or any(type(n) is not int or n < 1 for n in self.shape):
            raise ValueError("Evaluation shape must contain positive integers.")
        finite("evaluation time step", self.step_seconds, positive=True)
        if self.spatial_quadrature != "uniform_cell_area" or self.temporal_quadrature != "left_endpoint":
            raise ValueError("Unsupported declared evaluation quadrature.")


@dataclass(frozen=True)
class SensingRunConfigV1(RequiredMessage):
    session: SensingSessionV1
    source: SyntheticSourceV1
    wind: CausalWind
    seed: int
    duration_seconds: float
    budget_seconds: float
    initial_budget_seconds: float
    delivery: Literal["stream", "batch"]
    delivery_delay_seconds: float
    completion_overrun_seconds: float
    evaluation: ReconstructionGridV1

    def __post_init__(self):
        for name in ("duration_seconds", "budget_seconds", "initial_budget_seconds"):
            finite(name, getattr(self, name), positive=True)
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("delivery_delay_seconds", "completion_overrun_seconds"):
            finite(name, getattr(self, name))
            object.__setattr__(self, name, float(getattr(self, name)))
        if type(self.seed) is not int or self.seed < 0 or self.delivery not in ("stream", "batch"):
            raise ValueError("Invalid seed or delivery mode.")
        if self.wind.available_at_seconds > 0:
            raise ValueError("This synthetic runner requires a causal wind estimate available at startup.")

    def to_dict(self):
        result = super().to_dict()
        nested = self.session.to_dict()
        nested.pop("schema_version")
        nested.pop("message_type")
        result["session"] = nested
        return result

    @property
    def capabilities(self):
        return self.session.instrument.capabilities()

    @property
    def model(self):
        return self.session.model

    @property
    def policy(self):
        return self.session.policy.policy_id

    @property
    def start_full_scene(self):
        return self.session.startup_mode == "full_scene"


@dataclass(frozen=True)
class ControllerCallV1(RequiredMessage):
    operation: Literal["plan_next", "ingest_readings", "finish_scan"]
    at_seconds: float
    position_yx: Position | None
    request: ScanRequestV1 | None
    wind: CausalWind | None
    plan: ScanPlanV2 | None
    readings: tuple[SensorReading, ...]
    receipt: ExecutionReceipt | None

    def __post_init__(self):
        finite("API event time", self.at_seconds)
        if self.operation == "plan_next":
            if self.position_yx is None or self.request is None or self.plan is None or self.readings or self.receipt is not None:
                raise ValueError("A decision event needs pose, request, and returned plan.")
            position(self.position_yx)
            if abs(self.plan.started_at_seconds - self.at_seconds) > 1e-10:
                raise ValueError("Decision time does not match plan start.")
        elif self.operation == "ingest_readings":
            if any(x is not None for x in (self.position_yx, self.request, self.wind, self.plan, self.receipt)):
                raise ValueError("A delivery event contains only readings and delivery time.")
        elif self.operation == "finish_scan":
            if self.receipt is None or self.readings or any(x is not None for x in (self.position_yx, self.request, self.wind, self.plan)):
                raise ValueError("A completion event contains its receipt.")
            if abs(self.receipt.received_at_seconds - self.at_seconds) > 1e-10:
                raise ValueError("Completion event time must equal receipt availability.")
        else:
            raise ValueError("Unsupported controller operation.")


@dataclass(frozen=True)
class SensingRunV1:
    config: SensingRunConfigV1
    scans: tuple[SensingScanTrace, ...]
    events: tuple[ControllerCallV1, ...]
    termination_reason: str
    startup_complete: bool
    visited_fraction: float

    @property
    def source_assumption(self):
        return "source_free_with_provided_wind" if self.config.session.source_position is None else "source_anchored_with_provided_wind"

    @property
    def declared_source_position(self):
        return self.config.session.source_position

    def operational_payload(self):
        return {"schema_version": 2, "artifact_type": "sensing_execution_trace",
                "configuration": self.config.to_dict(), "configuration_hashes": self.config.session.hashes,
                "run_configuration_hash": configuration_hash(self.config.to_dict()),
                "termination_reason": self.termination_reason, "startup_complete": self.startup_complete,
                "visited_fraction": self.visited_fraction, "events": [e.to_dict() for e in self.events],
                "executions": [{"scan_id": s.plan.scan_id, "executed_plan": s.execution.executed_plan.to_dict(),
                                "exhausted": s.execution.exhausted} for s in self.scans]}

    @property
    def operational_hash(self):
        return configuration_hash(self.operational_payload())


def run_sensing_v1(config: SensingRunConfigV1, *, source: SensorSource | None = None) -> SensingRunV1:
    # Import only when the module is initialized, avoiding configuration cycles.
    from plume_focus.sensing_execution_v4 import ControllerCallV2
    from plume_focus.sensing_session_v4 import SensingSessionV4
    from plume_focus.sensing_execution_v5 import ControllerCallV3
    from plume_focus.sensing_session_v5 import SensingSessionV5
    call_type = (ControllerCallV3 if isinstance(config.session,SensingSessionV5) else
                 ControllerCallV2 if isinstance(config.session, SensingSessionV4) else ControllerCallV1)
    source = source if source is not None else config.source.build()
    controller = SensingController.from_session(config.session)
    scanner = FakeScanner(source, noise_std=getattr(config, "simulated_noise_std", config.model.sensor_noise_std), seed=config.seed,
                          gain=config.model.gain, offset=config.model.offset)
    now, pose = 0., config.session.initial_position_yx
    scans, events = [], []
    termination = "duration_limit"
    while now < config.duration_seconds - 1e-10:
        if source.end_time_seconds is not None and now >= source.end_time_seconds:
            termination = "source_exhausted"
            break
        ceiling = config.initial_budget_seconds if not scans else config.budget_seconds
        request = ScanRequestV1(min(ceiling, config.duration_seconds - now))
        plan = controller.plan_next(now, pose, request, wind=config.wind)
        events.append(call_type("plan_next", now, pose, request, config.wind, plan, (), None))
        if not plan.samples:
            termination = "no_legal_action"
            break
        prior = controller.belief_view()
        distribution = None
        if hasattr(config.session, "motion"):
            distribution = controller.predictive_distribution(plan.completed_at_seconds, config.capabilities.lattice_shape)
            mean, variance = distribution.mean, distribution.variance
        else:
            mean, variance = controller.predictive_moments(plan.completed_at_seconds, config.capabilities.lattice_shape)
        threshold = getattr(config.session, "allocation_threshold", config.model.background_threshold)
        raw_mass = (np.asarray(config.session.physical_quadrature_weights).reshape(mean.shape)
                    * (distribution.expected_excess(threshold) if distribution is not None else positive_expectation(mean, variance, threshold)))
        total_mass = float(raw_mass.sum())
        mass = raw_mass / total_mass if total_mass > config.session.policy.mass_floor and plan.predicted_mass_recall is not None else np.zeros(mean.shape)
        predicted = BeliefView(plan.completed_at_seconds, mean, np.sqrt(variance), mass)
        execution = scanner.execute(plan, delivery=config.delivery, delivery_delay=config.delivery_delay_seconds,
                                    completion_overrun=config.completion_overrun_seconds)
        batches = ((execution.readings, execution.receipt.received_at_seconds),) if config.delivery == "batch" else (
            ((reading,), reading.available_at_seconds) for reading in execution.readings)
        for batch, delivery_time in batches:
            controller.ingest_readings(batch, available_at_seconds=delivery_time)
            events.append(call_type("ingest_readings", delivery_time, None, None, None, None, batch, None))
        posterior = controller.finish_scan(execution.receipt)
        events.append(call_type("finish_scan", execution.receipt.received_at_seconds, None, None, None, None, (), execution.receipt))
        scans.append(SensingScanTrace(plan, execution, prior, predicted, posterior, controller.assimilation_events, config.wind,
                                     distribution, controller.motion_diagnostics()))
        now, pose = execution.receipt.received_at_seconds, execution.receipt.final_position_yx
        if execution.exhausted:
            termination = "source_exhausted"
            break
    return SensingRunV1(config, tuple(scans), tuple(events), termination, controller.startup_complete,
                        float(np.mean(controller.visit_times >= 0)))


class CausalReplayV1:
    """Replay public calls in delivery order; query predictions never alter state."""
    def __init__(self, session: SensingSessionV1, events: tuple[ControllerCallV1, ...]):
        if any(b.at_seconds < a.at_seconds for a, b in zip(events, events[1:])):
            raise ValueError("Trace events must be chronological.")
        self.controller = SensingController.from_session(session)
        self.events, self.index, self.time_seconds = events, 0, 0.

    def advance_to(self, time_seconds: float):
        if time_seconds < self.time_seconds - 1e-10:
            raise ValueError("Replay queries must be chronological.")
        while self.index < len(self.events) and self.events[self.index].at_seconds <= time_seconds + 1e-10:
            event = self.events[self.index]
            if event.operation == "plan_next":
                plan = self.controller.plan_next(event.at_seconds, event.position_yx, event.request, wind=event.wind)
                if plan != event.plan:
                    raise ValueError(f"Replayed plan differs from trace: {plan.scan_id}")
            elif event.operation == "ingest_readings":
                self.controller.ingest_readings(event.readings, available_at_seconds=event.at_seconds)
            else:
                self.controller.finish_scan(event.receipt)
            self.index += 1
        self.time_seconds = time_seconds
        return self.controller
