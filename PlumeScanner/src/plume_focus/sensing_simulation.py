"""Headless simulator adapter for the public controller; no file or video writes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Protocol

import numpy as np

from plume_focus.controller import AssimilationEvent, SensingController
from plume_focus.sensing_contracts import (
    CausalWind, ExecutionReceipt, Position, ScanConstraints, ScanPlan, ScannerCapabilities, SensorReading,
)
from plume_focus.signal_belief import BeliefView, SignalModelConfig, footprint_points
from plume_focus.video_sources import SyntheticMethaneLeakConfig, generate_synthetic_methane_values


class SensorSource(Protocol):
    end_time_seconds: float | None

    def signal_at(self, times: np.ndarray, positions_yx: np.ndarray) -> np.ndarray:
        """Noise-free instrument signal at physical times and positions."""


class SourceExhausted(ValueError):
    pass


@dataclass(frozen=True)
class CleanMethaneSource:
    config: SyntheticMethaneLeakConfig = field(default_factory=SyntheticMethaneLeakConfig)
    end_time_seconds: float | None = None

    def signal_at(self, times: np.ndarray, positions_yx: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        positions = np.asarray(positions_yx, dtype=float)
        if positions.shape != (*times.shape, 2):
            raise ValueError("Positions must match times with a final (y,x) axis.")
        if np.any(~np.isfinite(times)) or np.any(times < 0) or (self.end_time_seconds is not None and np.any(times > self.end_time_seconds + 1e-10)):
            raise SourceExhausted("Synthetic source has no sample at the requested time.")
        return generate_synthetic_methane_values(
            self.config, times * self.config.fps,
            positions[..., 0] * (self.config.shape[0] - 1), positions[..., 1] * (self.config.shape[1] - 1),
            clean_signal=True,
        )


def pose_at(plan: ScanPlan, time_seconds: float) -> Position:
    for segment in plan.segments:
        if time_seconds <= segment.completed_at_seconds:
            duration = segment.completed_at_seconds - segment.started_at_seconds
            f = np.clip((time_seconds - segment.started_at_seconds) / max(duration, 1e-15), 0, 1)
            return tuple((1 - f) * np.array(segment.start_yx) + f * np.array(segment.end_yx))
    return plan.final_position_yx


@dataclass(frozen=True)
class SimulatedExecution:
    readings: tuple[SensorReading, ...]
    receipt: ExecutionReceipt
    executed_plan: ScanPlan
    exhausted: bool


class FakeScanner:
    """Example hardware-shaped adapter, including exposure and finite input bounds."""
    def __init__(self, source: SensorSource, *, noise_std: float = .05, seed: int = 0,
                 gain: float = 1.0, offset: float = 0.0):
        if not np.isfinite(noise_std) or noise_std < 0 or type(seed) is not int or seed < 0:
            raise ValueError("Noise and seed must be nonnegative.")
        self.source, self.noise_std, self.seed = source, noise_std, seed
        if not np.isfinite(gain) or gain <= 0 or not np.isfinite(offset):
            raise ValueError("Sensor gain must be positive and offset finite.")
        self.gain, self.offset = gain, offset

    def execute(self, plan: ScanPlan, *, delivery: str = "stream", delivery_delay: float = 0.0,
                completion_overrun: float = 0.0) -> SimulatedExecution:
        if delivery not in ("stream", "batch") or delivery_delay < 0 or completion_overrun < 0:
            raise ValueError("Invalid simulated delivery settings.")
        end = plan.completed_at_seconds + completion_overrun
        exhausted = self.source.end_time_seconds is not None and end > self.source.end_time_seconds
        if exhausted:
            end = max(plan.started_at_seconds, self.source.end_time_seconds)
        readings, accepted_samples = [], []
        for index, sample in enumerate(plan.samples):
            exposure_end = sample.acquired_at_seconds + sample.integration_seconds / 2
            if exposure_end > end + 1e-10:
                break
            # Simpson exposure integration and four-point spatial quadrature.
            values = []
            for offset in (-.5, 0, .5):
                t = sample.acquired_at_seconds + offset * sample.integration_seconds
                points = footprint_points(pose_at(plan, t), sample.footprint)
                values.append(float(np.mean(self.source.signal_at(np.full(len(points), t), points))))
            signal = (values[0] + 4 * values[1] + values[2]) / 6
            # Same seed, acquisition time and position give the same noise even
            # when other policies choose a different number of previous samples.
            entropy = [self.seed, int(round(sample.acquired_at_seconds * 1e6)),
                       *(int(round(v * 1e6)) for v in sample.position_yx)]
            noise = np.random.default_rng(np.random.SeedSequence(entropy)).normal(0, self.noise_std)
            available = (end if delivery == "batch" else exposure_end) + delivery_delay
            readings.append(SensorReading(plan.scan_id, f"{plan.scan_id}:{index}", sample.position_yx,
                                           sample.acquired_at_seconds, available, self.offset + self.gain * signal + noise,
                                           sample.footprint, sample.integration_seconds))
            accepted_samples.append(sample)
        actual_pose = pose_at(plan, end)
        segments = []
        for segment in plan.segments:
            if segment.started_at_seconds >= end:
                break
            segments.append(replace(segment, completed_at_seconds=min(segment.completed_at_seconds, end),
                                    end_yx=pose_at(plan, min(segment.completed_at_seconds, end))))
        executed = replace(plan, completed_at_seconds=end, final_position_yx=actual_pose,
                           segments=tuple(segments), samples=tuple(accepted_samples))
        receipt = ExecutionReceipt(plan.scan_id, end, end + delivery_delay, actual_pose)
        return SimulatedExecution(tuple(readings), receipt, executed, exhausted)

    @staticmethod
    def deliver(controller: SensingController, execution: SimulatedExecution, *, delivery: str = "stream") -> BeliefView:
        if delivery == "batch":
            controller.ingest_readings(execution.readings, available_at_seconds=execution.receipt.received_at_seconds)
        else:
            for reading in execution.readings:
                controller.ingest_readings([reading], available_at_seconds=reading.available_at_seconds)
        return controller.finish_scan(execution.receipt)


@dataclass(frozen=True)
class SensingSimulationConfig:
    capabilities: ScannerCapabilities = field(default_factory=ScannerCapabilities)
    model: SignalModelConfig = field(default_factory=SignalModelConfig)
    budget_seconds: float = 12.0
    initial_budget_seconds: float = 36.0
    duration_seconds: float = 90.0
    max_scans: int = 8
    seed: int = 0
    source_free: bool = False
    start_full_scene: bool = True
    policy: str = "adaptive"
    wind_bias_yx: Position = (0.0, 0.0)
    wind_error_speed: float = .005
    delivery: str = "stream"

    def __post_init__(self) -> None:
        for value in (self.budget_seconds, self.initial_budget_seconds, self.duration_seconds):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Simulation durations must be positive.")
        if type(self.max_scans) is not int or self.max_scans < 1:
            raise ValueError("max_scans must be a positive integer.")
        if self.policy not in ("adaptive", "static_corridor", "rescan") or self.delivery not in ("stream", "batch"):
            raise ValueError("Unknown policy or delivery mode.")


@dataclass(frozen=True)
class SensingScanTrace:
    plan: ScanPlan
    execution: SimulatedExecution
    prior: BeliefView
    predicted: BeliefView
    posterior: BeliefView
    updates: tuple[AssimilationEvent, ...]
    wind: CausalWind
    predicted_distribution: object | None = None
    motion_diagnostics: dict | None = None


@dataclass(frozen=True)
class SensingRun:
    config: SensingSimulationConfig
    scans: tuple[SensingScanTrace, ...]
    termination_reason: str
    source_assumption: str
    declared_source_position: Position | None = None

    def operational_payload(self) -> dict:
        return {"profile": "recovery_scalar_v1", "model": self.config.model.to_dict(),
                "capabilities": self.config.capabilities.to_dict(), "policy": self.config.policy,
                "source_assumption": self.source_assumption, "termination_reason": self.termination_reason,
                "declared_source_position": self.declared_source_position,
                "scans": [{"plan": s.plan.to_dict(), "readings": [r.to_dict() for r in s.execution.readings],
                           "receipt": s.execution.receipt.to_dict(), "wind": s.wind.to_dict(),
                           "posterior_mean": s.posterior.mean.tolist(), "posterior_std": s.posterior.std.tolist()}
                          for s in self.scans]}

    @property
    def operational_hash(self) -> str:
        data = json.dumps(self.operational_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(data).hexdigest()


def run_sensing_simulation(config: SensingSimulationConfig = SensingSimulationConfig(), *,
                           source: CleanMethaneSource | None = None) -> SensingRun:
    source = source or CleanMethaneSource()
    source_position = tuple(np.array(source.config.source) / (np.array(source.config.shape) - 1))
    # Explicitly supplied synthetic wind proxy. The controller never gets the generator.
    velocity = np.array(source.config.wind_velocity) * source.config.fps / (np.array(source.config.shape) - 1)
    wind = CausalWind(velocity_yx=tuple(velocity + config.wind_bias_yx), error_speed=config.wind_error_speed)
    controller = SensingController(config.capabilities, config.model,
                                   source_position=None if config.source_free else source_position,
                                   start_full_scene=config.start_full_scene, policy=config.policy)
    scanner = FakeScanner(source, noise_std=config.model.sensor_noise_std, seed=config.seed,
                          gain=config.model.gain, offset=config.model.offset)
    now, pose, scans = 0., (0., 0.), []
    termination = "max_scans"
    for index in range(config.max_scans):
        if source.end_time_seconds is not None and now >= source.end_time_seconds:
            termination = "source_exhausted"
            break
        remaining = config.duration_seconds - now
        if remaining <= 1e-10:
            termination = "duration_limit"
            break
        budget = min(config.initial_budget_seconds if index == 0 else config.budget_seconds, remaining)
        plan = controller.plan_next(now, pose, ScanConstraints(budget), wind=wind)
        if not plan.samples:
            termination = "no_feasible_scan"
            break
        prior = controller.belief_view()
        predicted = controller.predictive_view(plan.completed_at_seconds)
        execution = scanner.execute(plan, delivery=config.delivery)
        posterior = scanner.deliver(controller, execution, delivery=config.delivery)
        scans.append(SensingScanTrace(plan, execution, prior, predicted, posterior, controller.assimilation_events, wind))
        now, pose = execution.receipt.received_at_seconds, execution.receipt.final_position_yx
        if execution.exhausted:
            termination = "source_exhausted"
            break
    return SensingRun(config, tuple(scans), termination,
                      "source_free_with_provided_wind" if config.source_free else "source_anchored_with_provided_wind",
                      None if config.source_free else source_position)
