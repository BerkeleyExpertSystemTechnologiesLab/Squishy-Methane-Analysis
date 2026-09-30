"""Pure, immutable transport diagnostic sessions; no execution or evaluator imports."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from plume_focus.sensing_session_v4 import SensingSessionV4, SensingMotionSessionV4
from plume_focus.sensing_session import configuration_hash
from plume_focus.signal_belief import TRANSPORT_IDS


@dataclass(frozen=True)
class TransportDiagnosticSession(SensingSessionV4):
    formulation_name: ClassVar[str] = "beam_reconstruction_transport_diagnostic_v1"
    formulation_id: Literal["beam_reconstruction_transport_diagnostic_v1"]
    transport_id: Literal["legacy_clipped_v1", "zero_inflow_v1"]

    def __post_init__(self):
        super().__post_init__()
        if self.transport_id not in TRANSPORT_IDS:
            raise ValueError("Unknown transport identifier.")

    @property
    def hashes(self):
        return {**super().hashes, "transport": configuration_hash({"transport_id": self.transport_id}),
                "effective_model": configuration_hash({"model": self.model.to_dict(), "transport_id": self.transport_id})}


@dataclass(frozen=True)
class MotionTransportDiagnosticSession(SensingMotionSessionV4):
    formulation_name: ClassVar[str] = "beam_reconstruction_motion_transport_diagnostic_v1"
    formulation_id: Literal["beam_reconstruction_motion_transport_diagnostic_v1"]
    transport_id: Literal["legacy_clipped_v1", "zero_inflow_v1"]

    def __post_init__(self):
        super().__post_init__()
        if self.transport_id not in TRANSPORT_IDS:
            raise ValueError("Unknown transport identifier.")

    @property
    def hashes(self):
        return {**super().hashes, "transport": configuration_hash({"transport_id": self.transport_id}),
                "effective_model": configuration_hash({"model": self.model.to_dict(), "motion": self.motion.to_dict(),
                                                       "transport_id": self.transport_id})}
