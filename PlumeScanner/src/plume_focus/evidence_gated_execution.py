"""Headless execution binding for the evidence-gated candidate; I/O stays outside."""
from dataclasses import dataclass

from plume_focus.calibration_execution import CalibrationRunConfig
from plume_focus.evidence_gated_loop import EvidenceGatedScanner, EvidenceGatedSession
from plume_focus.sensing_session import required_fields


@dataclass(frozen=True)
class EvidenceGatedRunConfig(CalibrationRunConfig):
    """Same truth, mechanics and estimator as the calibrated candidate; gated mode decision."""
    session: EvidenceGatedSession

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.session, EvidenceGatedSession):
            raise TypeError("Evidence-gated session required.")

    @classmethod
    def _from_fields(cls, value):
        required_fields(cls, value)
        from plume_focus.sensing_contracts import _decode
        from typing import get_type_hints
        payload = dict(value)
        session = EvidenceGatedSession.from_dict(payload.pop("session"))
        annotations = get_type_hints(cls)
        return cls(session=session, **{k: _decode(annotations[k], v) for k, v in payload.items()})

    def make_controller(self):
        return EvidenceGatedScanner(self.session)
