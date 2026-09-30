"""Persistent wind candidate configuration for the existing simulator adapter."""

from dataclasses import dataclass, fields

from plume_focus.sensing_execution_v2 import SensingRunConfigV2
from plume_focus.sensing_session_v3 import SensingSessionV3
from plume_focus.motion_config import PersistentMotionConfig


@dataclass(frozen=True)
class SensingRunConfigV3(SensingRunConfigV2):
    session: SensingSessionV3


def with_persistent_motion(base, covariance, order=3):
    """Bind a declared bias prior without changing base instrument/model/policy."""
    values = {f.name: getattr(base.session, f.name) for f in fields(base.session) if f.name != "motion"}
    values.update(formulation_id="beam_reconstruction_motion_v3_candidate", motion=PersistentMotionConfig(
        covariance, "session_constant", order,
        "Predeclared synthetic session-constant bias prior, normalized FOV^2/s^2; independent of scheduler error_speed; no hardware calibration."))
    session = SensingSessionV3(**values)
    return SensingRunConfigV3(**{**{f.name: getattr(base, f.name) for f in fields(base)}, "session": session})
