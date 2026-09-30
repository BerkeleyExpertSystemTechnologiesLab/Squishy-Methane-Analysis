"""Versioned v5 simulation bindings and policy-only detached state substitution."""

from dataclasses import dataclass, fields, replace
from typing import ClassVar
from plume_focus.sensing_execution_v4 import SensingRunConfigV4, ControllerCallV2
from plume_focus.sensing_session_v4 import SensingSessionV4
from plume_focus.sensing_session_v5 import PolicyConfigV5, SensingSessionV5, SensingMotionSessionV5, UPPER_POLICY, TransportBaselineSessionV2
from plume_focus.sensing_execution_v2 import SensingRunConfigV2
from plume_focus.sensing_messages_v4 import ScanPlanV4
from plume_focus.sensing_snapshot import DecisionSnapshot


@dataclass(frozen=True)
class SensingRunConfigV5(SensingRunConfigV4):
    session: SensingSessionV5


@dataclass(frozen=True)
class SensingMotionRunConfigV5(SensingRunConfigV5):
    session: SensingMotionSessionV5


@dataclass(frozen=True)
class TransportBaselineRunConfigV2(SensingRunConfigV2):
    session: TransportBaselineSessionV2


def with_transport_baseline(base,policy_id):
    if hasattr(base.session,'motion'):
        raise ValueError("This bounded study uses Gaussian corridor/age controls only.")
    values={f.name:getattr(base.session,f.name) for f in fields(TransportBaselineSessionV2) if hasattr(base.session,f.name)}
    policy=replace(base.session.policy.legacy_policy(),policy_id=policy_id)
    session=TransportBaselineSessionV2(**{**values,'formulation_id':TransportBaselineSessionV2.formulation_name,
                                        'policy':policy,'transport_id':'zero_inflow_v1'})
    return TransportBaselineRunConfigV2(**{**{f.name:getattr(base,f.name) for f in fields(SensingRunConfigV2)},'session':session})


@dataclass(frozen=True)
class ControllerCallV3(ControllerCallV2):
    schema_version: ClassVar[int] = 3
    plan: ScanPlanV4 | None


def with_threshold_policy(base,policy_id=UPPER_POLICY,**settings):
    policy={f.name:getattr(base.session.policy,f.name) for f in fields(PolicyConfigV5)
            if hasattr(base.session.policy,f.name)}
    policy.update(policy_id=policy_id,expected_delivery=base.delivery,
        expected_delivery_delay_seconds=base.delivery_delay_seconds,
        expected_completion_overrun_seconds=base.completion_overrun_seconds,**settings)
    policy=PolicyConfigV5(**policy)
    cls=SensingMotionSessionV5 if hasattr(base.session,"motion") else SensingSessionV5
    values={f.name:getattr(base.session,f.name) for f in fields(cls) if hasattr(base.session,f.name)}
    session=cls(**{**values,"policy":policy,"formulation_id":cls.formulation_name,"transport_id":"zero_inflow_v1"})
    run_cls=SensingMotionRunConfigV5 if hasattr(session,"motion") else SensingRunConfigV5
    return run_cls(**{**{f.name:getattr(base,f.name) for f in fields(run_cls)},"session":session})


def substitute_snapshot_policy(snapshot,session):
    """Evaluator-side binding only: reject any change to physical/model state."""
    from plume_focus.controller import SensingController
    before=snapshot.fork_controller(SensingController)
    old=before._session
    for name in ("instrument","model","source_position","startup_mode","initial_position_yx","physical_quadrature_weights"):
        if getattr(old,name)!=getattr(session,name):raise ValueError(f"Policy substitution changes {name}.")
    if getattr(old,"motion",None)!=getattr(session,"motion",None) or getattr(old,"transport_id","legacy_clipped_v1")!=getattr(session,"transport_id","legacy_clipped_v1"):
        raise ValueError("Policy substitution changes estimator or transport.")
    payload=snapshot.to_dict();payload['session']=session.to_dict()
    result=DecisionSnapshot.from_dict(payload)
    result.fork_controller(SensingController)
    return result
