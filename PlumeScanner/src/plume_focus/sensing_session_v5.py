"""Signal-gated threshold allocation; existing estimator/instrument settings."""

from dataclasses import dataclass, fields
from typing import ClassVar, Literal

from plume_focus.sensing_session_v4 import PolicyConfigV4, SensingSessionV4, SensingMotionSessionV4
from plume_focus.sensing_session import configuration_hash
from plume_focus.motion_config import PersistentMotionConfig
from plume_focus.sensing_session_v2 import SensingSessionV2


UPPER_POLICY = "upper_signal_threshold_segment_v5_candidate"
MEAN_POLICY = "mean_signal_threshold_segment_v5_ablation"


@dataclass(frozen=True)
class TransportBaselineSessionV2(SensingSessionV2):
    """Existing corridor/age rules with an explicitly identified transport."""
    formulation_name: ClassVar[str] = "beam_reconstruction_transport_baseline_v1"
    formulation_id: Literal["beam_reconstruction_transport_baseline_v1"]
    transport_id: Literal["zero_inflow_v1"]

    def __post_init__(self):
        super().__post_init__()
        if self.transport_id!="zero_inflow_v1" or self.policy.policy_id not in ("static_corridor_segment_v2","age_segment_v2"):
            raise ValueError("Expected a corrected-transport historical corridor/age control.")

    @property
    def hashes(self):
        return {**super().hashes,"transport":configuration_hash({"transport_id":self.transport_id}),
            "effective_model":configuration_hash({"model":self.model.to_dict(),"transport_id":self.transport_id})}


@dataclass(frozen=True)
class PolicyConfigV5(PolicyConfigV4):
    policy_id: Literal["upper_signal_threshold_segment_v5_candidate", "mean_signal_threshold_segment_v5_ablation"]
    upper_quantile: float = .95

    def __post_init__(self):
        if self.policy_id not in (UPPER_POLICY, MEAN_POLICY):
            raise ValueError("Unsupported threshold policy.")
        # Reuse physical/service/delivery validation, not the frozen mass gate.
        checked = PolicyConfigV4(**{**{f.name:getattr(self,f.name) for f in fields(PolicyConfigV4)},
                                   "policy_id":"reconstruction_risk_segment_v4_candidate"})
        for name in ("corridor_yxyx", "risk_spatial_shape"):
            object.__setattr__(self,name,getattr(checked,name))
        if isinstance(self.upper_quantile,bool) or not .5 < self.upper_quantile < 1:
            raise ValueError("An upper quantile must lie strictly between .5 and 1.")


@dataclass(frozen=True)
class SensingSessionV5(SensingSessionV4):
    formulation_name: ClassVar[str] = "beam_reconstruction_threshold_v5_candidate"
    policy_type: ClassVar[type] = PolicyConfigV5
    formulation_id: Literal["beam_reconstruction_threshold_v5_candidate"]
    policy: PolicyConfigV5
    transport_id: Literal["zero_inflow_v1"]

    def __post_init__(self):
        super().__post_init__()
        if self.transport_id != "zero_inflow_v1":
            raise ValueError("The v5 experiment requires explicit corrected transport.")

    @property
    def hashes(self):
        return {**super().hashes,"transport":configuration_hash({"transport_id":self.transport_id}),
                "effective_model":configuration_hash({"model":self.model.to_dict(),"transport_id":self.transport_id,
                    **({"motion":self.motion.to_dict()} if hasattr(self,"motion") else {})})}


@dataclass(frozen=True)
class SensingMotionSessionV5(SensingSessionV5):
    formulation_name: ClassVar[str] = "beam_reconstruction_motion_threshold_v5_candidate"
    formulation_id: Literal["beam_reconstruction_motion_threshold_v5_candidate"]
    motion: PersistentMotionConfig

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.motion,PersistentMotionConfig):
            raise TypeError("A validated persistent motion configuration is required.")

    @property
    def estimator_id(self):
        return "beam_reconstruction_motion_v3_candidate"

    @property
    def hashes(self):
        return {**super().hashes,"motion":configuration_hash(self.motion.to_dict())}
