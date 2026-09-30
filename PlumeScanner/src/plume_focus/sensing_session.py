"""Explicit immutable instrument/session contracts; no file or source access."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from typing import ClassVar, Literal

from plume_focus.sensing_contracts import (
    Footprint, Message, Position, ScannerCapabilities, finite, position,
)
from plume_focus.signal_belief import SignalModelConfig


def canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def configuration_hash(value: dict) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def required_fields(cls, value: object) -> None:
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
        raise ValueError(f"{cls.__name__} requires exactly its declared fields.")


class RequiredMessage(Message):
    @classmethod
    def _from_fields(cls, value):
        required_fields(cls, value)
        return super()._from_fields(value)


def text_fields(obj, *names: str) -> None:
    for name in names:
        if not isinstance(getattr(obj, name), str) or not getattr(obj, name).strip():
            raise ValueError(f"{name} requires nonempty text.")


@dataclass(frozen=True)
class ProfileProvenance(RequiredMessage):
    kind: Literal["synthetic", "hardware_characterization"]
    note: str

    def __post_init__(self):
        text_fields(self, "note")
        if self.kind not in ("synthetic", "hardware_characterization"):
            raise ValueError("Unknown profile provenance.")


@dataclass(frozen=True)
class FovMapping(RequiredMessage):
    units: Literal["radian", "degree", "synthetic_fov"]
    origin_yx: Position
    span_yx: Position

    def __post_init__(self):
        object.__setattr__(self, "origin_yx", tuple(self.origin_yx))
        object.__setattr__(self, "span_yx", tuple(self.span_yx))
        if self.units not in ("radian", "degree", "synthetic_fov"):
            raise ValueError("Unsupported affine FOV mapping units.")
        if len(self.origin_yx) != 2 or any(not math.isfinite(v) for v in self.origin_yx):
            raise ValueError("FOV origin needs two finite components.")
        if len(self.span_yx) != 2:
            raise ValueError("FOV span needs two components.")
        for v in self.span_yx:
            finite("FOV span", v, positive=True)

    def to_fov(self, pointing: Position) -> Position:
        if len(pointing) != 2 or any(not math.isfinite(v) for v in pointing):
            raise ValueError("Pointing needs exactly two finite affine-map coordinates.")
        result = tuple((v - o) / s for v, o, s in zip(pointing, self.origin_yx, self.span_yx))
        position(result)
        return result


@dataclass(frozen=True)
class ProfileFootprint(RequiredMessage):
    height: float
    width: float
    edge_model: Literal["clipped_fov"]

    def __post_init__(self):
        Footprint(self.height, self.width)
        if self.edge_model != "clipped_fov":
            raise ValueError("Only the declared clipped-FOV approximation is supported.")


@dataclass(frozen=True)
class ProfileMechanics(RequiredMessage):
    sweep_vertical_fov_per_second: float
    move_vertical_fov_per_second: float
    move_horizontal_fov_per_second: float
    settling_seconds_per_visit: float

    def __post_init__(self):
        for name in ("sweep_vertical_fov_per_second", "move_vertical_fov_per_second", "move_horizontal_fov_per_second"):
            finite(name, getattr(self, name), positive=True)
        finite("settling time", self.settling_seconds_per_visit)


@dataclass(frozen=True)
class ProfileAcquisition(RequiredMessage):
    cadence_hz: float
    detector_capacity_hz: float
    integration_seconds: float

    def __post_init__(self):
        for name in ("cadence_hz", "detector_capacity_hz", "integration_seconds"):
            finite(name, getattr(self, name), positive=True)
        if self.cadence_hz > self.detector_capacity_hz:
            raise ValueError("Commanded cadence exceeds detector capacity.")
        if self.integration_seconds > 1 / self.cadence_hz + 1e-12:
            raise ValueError("Integration exceeds the commanded cadence slot.")


@dataclass(frozen=True)
class ProfileRetrieval(RequiredMessage):
    quantity: Literal["beam_integrated_concentration"]
    units: str
    path_convention: str
    calibration_id: str
    temporal_model: Literal["exposure_midpoint_approximation"]
    assumptions: str

    def __post_init__(self):
        text_fields(self, "units", "path_convention", "calibration_id", "assumptions")
        if self.quantity != "beam_integrated_concentration" or self.temporal_model != "exposure_midpoint_approximation":
            raise ValueError("Unsupported retrieved quantity or temporal likelihood.")


@dataclass(frozen=True)
class InstrumentProfileV1(RequiredMessage):
    profile_id: str
    provenance: ProfileProvenance
    coordinate_frame: Literal["normalized_fov_yx"]
    mapping: FovMapping
    lattice_shape: tuple[int, int]
    footprint: ProfileFootprint
    scan_mode: Literal["column", "point"]
    mechanics: ProfileMechanics
    acquisition: ProfileAcquisition
    retrieval: ProfileRetrieval

    def __post_init__(self):
        object.__setattr__(self, "lattice_shape", tuple(self.lattice_shape))
        text_fields(self, "profile_id")
        if self.coordinate_frame != "normalized_fov_yx" or self.scan_mode not in ("column", "point"):
            raise ValueError("Unsupported profile coordinates or scan mode.")
        if len(self.lattice_shape) != 2 or any(type(n) is not int or n < 1 for n in self.lattice_shape):
            raise ValueError("Physical lattice needs two positive integers.")
        required = self.mechanics.sweep_vertical_fov_per_second * self.lattice_shape[0]
        if self.scan_mode == "column" and not math.isclose(self.acquisition.cadence_hz, required, rel_tol=1e-10, abs_tol=0):
            raise ValueError(f"cadence_hz must equal sweep speed / row pitch ({required:g} Hz); speed is fixed.")

    def to_dict(self) -> dict:
        if hasattr(self, "_canonical_input"):
            return json.loads(self._canonical_input)
        result = super().to_dict()
        result.pop("message_type")
        return {**result, "artifact_type": "instrument_profile"}

    @classmethod
    def from_dict(cls, value: dict):
        payload = dict(value)
        version = payload.pop("schema_version", None)
        if type(version) is not int or version != 1 or payload.pop("artifact_type", None) != "instrument_profile":
            raise ValueError("Unsupported instrument profile version/type.")
        result = cls._from_fields(payload)
        # Preserve the declared canonical bytes, including JSON number spelling.
        object.__setattr__(result, "_canonical_input", canonical_json(value))
        return result

    @property
    def digest(self) -> str:
        return configuration_hash(self.to_dict())

    def capabilities(self) -> ScannerCapabilities:
        # Compatibility view only. The v1 compiler uses this full profile.
        # Legacy capabilities combine cadence/capacity; use commanded cadence here
        # so legal integrations need not fit the detector's *maximum* rate slot.
        return ScannerCapabilities(self.lattice_shape, self.mechanics.sweep_vertical_fov_per_second,
                                   self.mechanics.move_horizontal_fov_per_second, self.acquisition.cadence_hz,
                                   self.acquisition.integration_seconds,
                                   Footprint(self.footprint.height, self.footprint.width), self.scan_mode)


@dataclass(frozen=True)
class PolicyConfigV1(RequiredMessage):
    allowed_policy_ids: ClassVar[tuple[str, ...]] = ("mass90_column_v1", "static_corridor_with_reacquisition_v1", "age_rescan_v1")
    policy_id: Literal["mass90_column_v1", "static_corridor_with_reacquisition_v1", "age_rescan_v1"]
    gamma: float
    minimum_samples: int
    max_planning_iterations: int
    time_tolerance_seconds: float
    mass_tolerance: float
    uniform_density_tolerance: float
    mass_floor: float
    stale_after_seconds: float
    max_prediction_error_fov: float
    confidence_delta: float
    corridor_yxyx: tuple[float, float, float, float]

    def __post_init__(self):
        object.__setattr__(self, "corridor_yxyx", tuple(self.corridor_yxyx))
        if self.policy_id not in self.allowed_policy_ids:
            raise ValueError("Unsupported frozen-v1 policy.")
        if self.gamma != .90:
            raise ValueError("mass90 v1 fixes gamma=0.90.")
        for name in ("minimum_samples", "max_planning_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("time_tolerance_seconds", "mass_tolerance", "uniform_density_tolerance", "mass_floor"):
            finite(name, getattr(self, name))
        for name in ("stale_after_seconds", "max_prediction_error_fov"):
            finite(name, getattr(self, name), positive=True)
        if not 0 < self.confidence_delta < 1:
            raise ValueError("confidence_delta must be in (0, 1).")
        if len(self.corridor_yxyx) != 4:
            raise ValueError("Corridor needs two corners.")
        position(self.corridor_yxyx[:2])
        position(self.corridor_yxyx[2:])
        if self.corridor_yxyx[0] >= self.corridor_yxyx[2] or self.corridor_yxyx[1] >= self.corridor_yxyx[3]:
            raise ValueError("Corridor must have positive height and width.")


@dataclass(frozen=True)
class SensingSessionV1(RequiredMessage):
    formulation_name: ClassVar[str] = "beam_reconstruction_v1"
    policy_type: ClassVar[type[PolicyConfigV1]] = PolicyConfigV1
    formulation_id: Literal["beam_reconstruction_v1"]
    instrument: InstrumentProfileV1
    model: SignalModelConfig
    policy: PolicyConfigV1
    source_position: Position | None
    startup_mode: Literal["full_scene", "prior"]
    initial_position_yx: Position
    clock_origin: str
    wind_mapping: str
    physical_quadrature_weights: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "initial_position_yx", tuple(float(v) for v in self.initial_position_yx))
        object.__setattr__(self, "physical_quadrature_weights", tuple(float(w) for w in self.physical_quadrature_weights))
        if self.source_position is not None:
            object.__setattr__(self, "source_position", tuple(float(v) for v in self.source_position))
        # Canonical typed numbers also for direct Python construction, so a JSON
        # round trip cannot change a session's model/policy configuration hashes.
        object.__setattr__(self, "model", SignalModelConfig._from_fields(asdict(self.model)))
        object.__setattr__(self, "policy", self.policy_type._from_fields(asdict(self.policy)))
        if self.formulation_id != self.formulation_name or self.startup_mode not in ("full_scene", "prior"):
            raise ValueError("Unsupported formulation or startup mode.")
        if self.instrument.scan_mode != "column":
            raise ValueError("The frozen v1 policies require column mode.")
        if self.model.signal_units != self.instrument.retrieval.units:
            raise ValueError("Model and retrieved scalar units must agree.")
        position(self.initial_position_yx)
        if self.source_position is not None:
            position(self.source_position)
        text_fields(self, "clock_origin", "wind_mapping")
        weights = self.physical_quadrature_weights
        if len(weights) != math.prod(self.instrument.lattice_shape) or any(not math.isfinite(w) or w <= 0 for w in weights) or not math.isclose(sum(weights), 1, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Physical quadrature needs positive normalized weights for every lattice cell.")

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["instrument"] = self.instrument.to_dict()
        return result

    @classmethod
    def _from_fields(cls, value):
        required_fields(cls, value)
        payload = dict(value)
        required_fields(SignalModelConfig, payload["model"])
        profile = InstrumentProfileV1.from_dict(payload["instrument"])
        # Nested profile objects are already validated; decode the other fields.
        from plume_focus.sensing_contracts import _decode
        from typing import get_type_hints
        annotations = get_type_hints(cls)
        decoded = {k: _decode(annotations[k], v) for k, v in payload.items() if k != "instrument"}
        return cls(instrument=profile, **decoded)

    @property
    def hashes(self) -> dict:
        return {"session": configuration_hash(self.to_dict()), "instrument": self.instrument.digest,
                "model": configuration_hash(self.model.to_dict()), "policy": configuration_hash(self.policy.to_dict())}
