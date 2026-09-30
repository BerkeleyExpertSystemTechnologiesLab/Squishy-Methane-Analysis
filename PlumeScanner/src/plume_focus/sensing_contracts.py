"""Version-one, transport-neutral messages for a single detector.

Coordinates are (y, x), each in [0, 1] across the declared field of view.
Seconds use a session origin, never video frame indices or wall-clock dates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import types
from typing import ClassVar, Literal, Union, get_args, get_origin, get_type_hints

Position = tuple[float, float]


def finite(name: str, value: float, minimum: float = 0.0, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value < minimum or (positive and value == minimum):
        raise ValueError(f"{name} must be finite and {'greater than' if positive else 'at least'} {minimum}.")


def position(value: Position) -> None:
    if len(value) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in value):
        raise ValueError("position must be (y, x) within the normalized field of view.")


def _decode(annotation: object, value: object) -> object:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        return _decode(next(a for a in args if a is not type(None)), value)
    if origin is Literal:
        if value not in args:
            raise ValueError(f"Expected one of {args}.")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError("Expected a sequence.")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], v) for v in value)
        if len(value) != len(args):
            raise ValueError("Sequence has the wrong length.")
        return tuple(_decode(a, v) for a, v in zip(args, value))
    if isinstance(annotation, type) and issubclass(annotation, Message):
        return annotation._from_fields(value)
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("Expected a finite number.")
        return float(value)
    if annotation in (str, int, bool) and type(value) is not annotation:
        raise ValueError(f"Expected {annotation.__name__}.")
    return value


class Message:
    schema_version: ClassVar[int] = 1

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "message_type": type(self).__name__, **asdict(self)}

    @classmethod
    def _from_fields(cls, value: object):
        if not isinstance(value, dict):
            raise ValueError("Message must be an object.")
        names = {f.name for f in fields(cls)}
        if set(value) - names:
            raise ValueError(f"Unknown fields for {cls.__name__}: {set(value) - names}")
        annotations = get_type_hints(cls)
        return cls(**{k: _decode(annotations[k], v) for k, v in value.items()})

    @classmethod
    def from_dict(cls, value: dict):
        payload = dict(value)
        version = payload.pop("schema_version", None)
        if type(version) is not int or version != cls.schema_version or payload.pop("message_type", None) != cls.__name__:
            raise ValueError("Unsupported message version or type.")
        return cls._from_fields(payload)


@dataclass(frozen=True)
class Footprint(Message):
    height: float = 0.05
    width: float = 1 / 32

    def __post_init__(self) -> None:
        for name in ("height", "width"):
            finite(name, getattr(self, name), positive=True)
            if getattr(self, name) > 1:
                raise ValueError("Footprint cannot exceed the field of view.")


@dataclass(frozen=True)
class ScannerCapabilities(Message):
    lattice_shape: tuple[int, int] = (20, 32)
    vertical_speed: float = 1.0
    horizontal_speed: float = 0.625
    readout_hz: float = 20.0
    integration_seconds: float = 0.025
    footprint: Footprint = Footprint()
    scan_mode: Literal["point", "column"] = "column"
    coordinate_frame: Literal["normalized_fov_yx"] = "normalized_fov_yx"

    def __post_init__(self) -> None:
        if len(self.lattice_shape) != 2 or any(type(n) is not int or n < 1 for n in self.lattice_shape):
            raise ValueError("lattice_shape must contain two positive integers.")
        for name in ("vertical_speed", "horizontal_speed", "readout_hz", "integration_seconds"):
            finite(name, getattr(self, name), positive=True)
        if self.integration_seconds > 1 / self.readout_hz + 1e-12:
            raise ValueError("Integration cannot exceed the declared readout period.")
        if self.scan_mode not in ("point", "column") or self.coordinate_frame != "normalized_fov_yx":
            raise ValueError("Unsupported scanner mode or coordinate frame.")


@dataclass(frozen=True)
class CausalWind(Message):
    measured_at_seconds: float = 0.0
    available_at_seconds: float = 0.0
    velocity_yx: Position = (0.0, 0.0)
    error_speed: float = 0.02

    def __post_init__(self) -> None:
        finite("wind measurement time", self.measured_at_seconds)
        finite("wind availability", self.available_at_seconds, self.measured_at_seconds)
        finite("wind error speed", self.error_speed)
        if len(self.velocity_yx) != 2 or any(not math.isfinite(v) for v in self.velocity_yx):
            raise ValueError("Wind velocity must contain two finite components.")


@dataclass(frozen=True)
class SensorReading(Message):
    scan_id: str
    reading_id: str
    position_yx: Position
    acquired_at_seconds: float
    available_at_seconds: float
    value: float
    footprint: Footprint = Footprint()
    integration_seconds: float = 0.025

    def __post_init__(self) -> None:
        if not self.scan_id or not self.reading_id:
            raise ValueError("Reading and scan identities are required.")
        position(self.position_yx)
        finite("integration", self.integration_seconds, positive=True)
        finite("acquisition midpoint", self.acquired_at_seconds, self.integration_seconds / 2)
        finite("reading availability", self.available_at_seconds,
               self.acquired_at_seconds + self.integration_seconds / 2 - 1e-12)
        if not math.isfinite(self.value):
            raise ValueError("Reading value must be finite.")


@dataclass(frozen=True)
class PathSegment(Message):
    kind: Literal["move", "sweep", "dwell"]
    start_yx: Position
    end_yx: Position
    started_at_seconds: float
    completed_at_seconds: float

    def __post_init__(self) -> None:
        position(self.start_yx)
        position(self.end_yx)
        finite("segment start", self.started_at_seconds)
        finite("segment completion", self.completed_at_seconds, self.started_at_seconds)
        if self.kind not in ("move", "sweep", "dwell"):
            raise ValueError("Unsupported segment kind.")


@dataclass(frozen=True)
class SampleTarget(Message):
    position_yx: Position
    acquired_at_seconds: float
    integration_seconds: float
    footprint: Footprint

    def __post_init__(self) -> None:
        position(self.position_yx)
        finite("integration", self.integration_seconds, positive=True)
        finite("acquisition", self.acquired_at_seconds, self.integration_seconds / 2)


@dataclass(frozen=True)
class ScanConstraints(Message):
    max_duration_seconds: float = 12.0
    target_mass: float = 0.9
    minimum_samples: int = 4
    max_planning_iterations: int = 5

    def __post_init__(self) -> None:
        finite("scan duration budget", self.max_duration_seconds, positive=True)
        if not 0 < self.target_mass <= 1:
            raise ValueError("target_mass must be in (0, 1].")
        if type(self.minimum_samples) is not int or self.minimum_samples < 1:
            raise ValueError("minimum_samples must be a positive integer.")
        if type(self.max_planning_iterations) is not int or self.max_planning_iterations < 1:
            raise ValueError("max_planning_iterations must be positive.")


def validate_physical_plan(plan):
    """Shared path/exposure validation, independent of allocation messages."""
    finite("plan start", plan.started_at_seconds)
    finite("plan completion", plan.completed_at_seconds, plan.started_at_seconds)
    position(plan.start_position_yx)
    position(plan.final_position_yx)
    if not plan.scan_id:
        raise ValueError("Invalid plan identity.")
    if len(plan.lattice_shape) != 2 or any(type(n) is not int or n < 1 for n in plan.lattice_shape):
        raise ValueError("Invalid plan lattice shape.")
    for cells in (plan.requested_cells, plan.swept_cells):
        if len(set(cells)) != len(cells) or any(type(i) is not int or not 0 <= i < math.prod(plan.lattice_shape) for i in cells):
            raise ValueError("Plan cells must be unique valid lattice indices.")
    clock, pose = plan.started_at_seconds, plan.start_position_yx
    for segment in plan.segments:
        if abs(segment.started_at_seconds - clock) > 1e-9 or any(abs(a - b) > 1e-9 for a, b in zip(segment.start_yx, pose)):
            raise ValueError("Path segments must join in time and position.")
        clock, pose = segment.completed_at_seconds, segment.end_yx
    if clock > plan.completed_at_seconds + 1e-9 or any(abs(a - b) > 1e-9 for a, b in zip(pose, plan.final_position_yx)):
        raise ValueError("Path does not match plan completion or final position.")
    prior_exposure_end = plan.started_at_seconds
    for sample in plan.samples:
        exposure_start = sample.acquired_at_seconds - sample.integration_seconds / 2
        exposure_end = sample.acquired_at_seconds + sample.integration_seconds / 2
        if exposure_start < prior_exposure_end - 1e-9 or exposure_end > plan.completed_at_seconds + 1e-9:
            raise ValueError("Sample exposures overlap or lie outside the plan.")
        prior_exposure_end = exposure_end


@dataclass(frozen=True)
class ScanPlan(Message):
    scan_id: str
    started_at_seconds: float
    completed_at_seconds: float
    start_position_yx: Position
    final_position_yx: Position
    segments: tuple[PathSegment, ...]
    samples: tuple[SampleTarget, ...]
    lattice_shape: tuple[int, int]
    requested_cells: tuple[int, ...]
    swept_cells: tuple[int, ...]
    predicted_mass_recall: float
    target_mass: float
    status: Literal["target_met", "coverage_target_unmet", "reacquisition", "infeasible"]
    converged: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.scan_id or not 0 <= self.predicted_mass_recall <= 1 + 1e-9:
            raise ValueError("Invalid plan identity or predicted mass recall.")
        if not 0 < self.target_mass <= 1:
            raise ValueError("Invalid target mass.")
        if self.status not in ("target_met", "coverage_target_unmet", "reacquisition", "infeasible"):
            raise ValueError("Invalid planning status.")
        validate_physical_plan(self)

    @property
    def duration_seconds(self) -> float:
        return self.completed_at_seconds - self.started_at_seconds


@dataclass(frozen=True)
class ExecutionReceipt(Message):
    scan_id: str
    completed_at_seconds: float
    received_at_seconds: float
    final_position_yx: Position

    def __post_init__(self) -> None:
        if not self.scan_id:
            raise ValueError("scan_id is required.")
        finite("actual completion", self.completed_at_seconds)
        finite("receipt availability", self.received_at_seconds, self.completed_at_seconds)
        position(self.final_position_yx)
