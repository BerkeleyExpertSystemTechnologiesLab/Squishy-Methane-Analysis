"""Serializable risk-policy experiment binding; public acquisition API unchanged."""

from dataclasses import dataclass, fields
from typing import ClassVar

from plume_focus.sensing_execution_v1 import ControllerCallV1
from plume_focus.sensing_execution_v2 import SensingRunConfigV2
from plume_focus.sensing_messages_v3 import ScanPlanV3
from plume_focus.sensing_session_v4 import PolicyConfigV4, SensingSessionV4, SensingMotionSessionV4


@dataclass(frozen=True)
class SensingRunConfigV4(SensingRunConfigV2):
    session: SensingSessionV4

    def __post_init__(self):
        super().__post_init__()
        policy = self.session.policy
        if (self.delivery != policy.expected_delivery or self.delivery_delay_seconds != policy.expected_delivery_delay_seconds
                or self.completion_overrun_seconds != policy.expected_completion_overrun_seconds):
            raise ValueError("Synthetic delivery/receipt settings must match the declared planning assumptions.")


@dataclass(frozen=True)
class SensingMotionRunConfigV4(SensingRunConfigV4):
    session: SensingMotionSessionV4


@dataclass(frozen=True)
class ControllerCallV2(ControllerCallV1):
    schema_version: ClassVar[int] = 2
    plan: ScanPlanV3 | None


def with_risk_policy(base, **settings):
    policy = {f.name: getattr(base.session.policy, f.name) for f in fields(base.session.policy)}
    policy.update(policy_id="reconstruction_risk_segment_v4_candidate", expected_delivery=base.delivery,
                  expected_delivery_delay_seconds=base.delivery_delay_seconds,
                  expected_completion_overrun_seconds=base.completion_overrun_seconds, **settings)
    policy = PolicyConfigV4(**policy)
    values = {f.name:getattr(base.session,f.name) for f in fields(base.session)}
    motion = hasattr(base.session,"motion")
    cls = SensingMotionSessionV4 if motion else SensingSessionV4
    session = cls(**{**values,"formulation_id":cls.formulation_name,"policy":policy})
    cls = SensingMotionRunConfigV4 if motion else SensingRunConfigV4
    return cls(**{**{f.name:getattr(base,f.name) for f in fields(base)},"session":session})
