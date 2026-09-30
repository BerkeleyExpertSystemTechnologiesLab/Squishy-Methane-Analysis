"""Experimental acquisition binding; estimator and instrument mathematics unchanged."""

from dataclasses import asdict, dataclass, fields
from typing import ClassVar, Literal

from plume_focus.motion_config import PersistentMotionConfig
from plume_focus.sensing_contracts import finite, position
from plume_focus.sensing_session import PolicyConfigV1, configuration_hash
from plume_focus.sensing_session_v2 import PolicyConfigV2, SensingSessionV2


@dataclass(frozen=True)
class PolicyConfigV4(PolicyConfigV2):
    policy_id: Literal["reconstruction_risk_segment_v4_candidate"]
    risk_step_seconds: float = .1
    risk_spatial_shape: tuple[int, int] = (120, 160)
    history_count: int = 32
    random_seed: int = 0
    expected_delivery: Literal["stream", "batch"] = "stream"
    expected_delivery_delay_seconds: float = 0.
    expected_completion_overrun_seconds: float = 0.
    risk_absolute_tolerance: float = 1e-12
    risk_relative_tolerance: float = 1e-6

    def __post_init__(self):
        # Do not invoke the frozen mass90 validator: gamma is a diagnostic here.
        if self.policy_id != "reconstruction_risk_segment_v4_candidate":
            raise ValueError("Unsupported reconstruction-risk policy.")
        for name in ("minimum_samples", "max_planning_iterations", "history_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.minimum_samples < 4:
            raise ValueError("The v4 acquisition policy retains a minimum of four readings.")
        if type(self.random_seed) is not int or self.random_seed < 0 or self.history_count > 128:
            raise ValueError("Invalid bounded history sampling settings.")
        for name in ("risk_step_seconds", "stale_after_seconds", "max_prediction_error_fov",
                     "survey_height_fov", "survey_interval_seconds"):
            finite(name, getattr(self, name), positive=True)
        for name in ("time_tolerance_seconds", "mass_tolerance", "uniform_density_tolerance", "mass_floor",
                     "reportable_threshold", "expected_delivery_delay_seconds", "expected_completion_overrun_seconds",
                     "risk_absolute_tolerance", "risk_relative_tolerance"):
            finite(name, getattr(self, name))
        if not 0 < self.gamma <= 1 or not 0 < self.confidence_delta < 1 or self.survey_height_fov > 1:
            raise ValueError("Invalid coverage diagnostic/service configuration.")
        if self.expected_delivery not in ("stream", "batch") or not self.threshold_provenance.strip():
            raise ValueError("Delivery and threshold provenance are required.")
        object.__setattr__(self, "risk_spatial_shape", tuple(self.risk_spatial_shape))
        if len(self.risk_spatial_shape) != 2 or any(type(n) is not int or n < 1 for n in self.risk_spatial_shape):
            raise ValueError("Risk projection needs positive integer dimensions.")
        object.__setattr__(self, "corridor_yxyx", tuple(self.corridor_yxyx))
        if len(self.corridor_yxyx) != 4:
            raise ValueError("Corridor needs two corners.")
        position(self.corridor_yxyx[:2]); position(self.corridor_yxyx[2:])
        if any(a >= b for a, b in zip(self.corridor_yxyx[:2], self.corridor_yxyx[2:])):
            raise ValueError("Corridor must have positive area.")

    def legacy_policy(self):
        values = {f.name: getattr(self, f.name) for f in fields(PolicyConfigV2)}
        return PolicyConfigV2(**{**values, "policy_id": "threshold90_segment_v2", "gamma": .90})


@dataclass(frozen=True)
class SensingSessionV4(SensingSessionV2):
    formulation_name: ClassVar[str] = "beam_reconstruction_risk_v4_candidate"
    policy_type: ClassVar[type[PolicyConfigV1]] = PolicyConfigV4
    formulation_id: Literal["beam_reconstruction_risk_v4_candidate"]
    policy: PolicyConfigV4

    @property
    def estimator_id(self):
        return "beam_reconstruction_v2_candidate"

    def legacy_session(self):
        values = {f.name: getattr(self, f.name) for f in fields(SensingSessionV2)}
        return SensingSessionV2(**{**values, "formulation_id": "beam_reconstruction_v2_candidate",
                                  "policy": self.policy.legacy_policy()})


@dataclass(frozen=True)
class SensingMotionSessionV4(SensingSessionV4):
    formulation_name: ClassVar[str] = "beam_reconstruction_motion_risk_v4_candidate"
    formulation_id: Literal["beam_reconstruction_motion_risk_v4_candidate"]
    motion: PersistentMotionConfig

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.motion, PersistentMotionConfig):
            raise TypeError("A validated motion configuration is required.")
        object.__setattr__(self, "motion", PersistentMotionConfig._from_fields(asdict(self.motion)))

    @property
    def estimator_id(self):
        return "beam_reconstruction_motion_v3_candidate"

    @property
    def hashes(self):
        return {**super().hashes, "motion": configuration_hash(self.motion.to_dict())}
