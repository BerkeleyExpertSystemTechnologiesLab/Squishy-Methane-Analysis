"""Opt-in transport diagnostic binding; historical configuration bytes are unchanged."""

from dataclasses import dataclass, fields
from typing import ClassVar, Literal

from plume_focus.sensing_session_v4 import SensingSessionV4, SensingMotionSessionV4
from plume_focus.sensing_execution_v4 import SensingRunConfigV4, SensingMotionRunConfigV4
from plume_focus.sensing_session import configuration_hash
from plume_focus.signal_belief import TRANSPORT_IDS


from plume_focus.sensing_session_diagnostic import TransportDiagnosticSession, MotionTransportDiagnosticSession


@dataclass(frozen=True)
class TransportDiagnosticRunConfig(SensingRunConfigV4):
    session: TransportDiagnosticSession


@dataclass(frozen=True)
class MotionTransportDiagnosticRunConfig(SensingMotionRunConfigV4):
    session: MotionTransportDiagnosticSession


def with_transport_diagnostic(config, transport_id):
    motion = hasattr(config.session, "motion")
    session_type = MotionTransportDiagnosticSession if motion else TransportDiagnosticSession
    values = {f.name: getattr(config.session, f.name) for f in fields(config.session)}
    session = session_type(**{**values, "formulation_id": session_type.formulation_name, "transport_id": transport_id})
    run_type = MotionTransportDiagnosticRunConfig if motion else TransportDiagnosticRunConfig
    return run_type(**{**{f.name: getattr(config, f.name) for f in fields(config)}, "session": session})
