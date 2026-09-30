"""Experimental mechanism-grounded policy configuration; frozen v1 stays distinct."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from plume_focus.sensing_contracts import finite
from plume_focus.sensing_session import PolicyConfigV1, SensingSessionV1


@dataclass(frozen=True)
class PolicyConfigV2(PolicyConfigV1):
    allowed_policy_ids: ClassVar[tuple[str, ...]] = (
        "threshold90_segment_v2", "mass90_segment_v2", "static_corridor_segment_v2", "age_segment_v2")
    policy_id: Literal["threshold90_segment_v2", "mass90_segment_v2", "static_corridor_segment_v2", "age_segment_v2"]
    reportable_threshold: float
    threshold_provenance: str
    survey_height_fov: float
    survey_interval_seconds: float

    def __post_init__(self):
        super().__post_init__()
        finite("reportable threshold", self.reportable_threshold)
        finite("survey height", self.survey_height_fov, positive=True)
        finite("survey interval", self.survey_interval_seconds, positive=True)
        if self.survey_height_fov > 1 or not self.threshold_provenance.strip():
            raise ValueError("Survey height must fit the FOV and threshold provenance is required.")


@dataclass(frozen=True)
class SensingSessionV2(SensingSessionV1):
    formulation_name: ClassVar[str] = "beam_reconstruction_v2_candidate"
    policy_type: ClassVar[type[PolicyConfigV1]] = PolicyConfigV2
    formulation_id: Literal["beam_reconstruction_v2_candidate"]
    policy: PolicyConfigV2

    def __post_init__(self):
        super().__post_init__()
        if self.policy.reportable_threshold < self.model.background_threshold:
            raise ValueError("Reportable signal cannot lie below the declared background threshold.")

    @property
    def allocation_threshold(self):
        return self.model.background_threshold if self.policy.policy_id == "mass90_segment_v2" else self.policy.reportable_threshold
