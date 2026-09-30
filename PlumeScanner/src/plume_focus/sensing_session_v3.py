"""Immutable persistent-motion candidate; policy v2 and instrument v1 retained."""

from dataclasses import asdict, dataclass
from typing import ClassVar, Literal

from plume_focus.sensing_session import configuration_hash
from plume_focus.sensing_session_v2 import SensingSessionV2
from plume_focus.motion_config import PersistentMotionConfig


@dataclass(frozen=True)
class SensingSessionV3(SensingSessionV2):
    formulation_name: ClassVar[str] = "beam_reconstruction_motion_v3_candidate"
    formulation_id: Literal["beam_reconstruction_motion_v3_candidate"]
    motion: PersistentMotionConfig

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.motion, PersistentMotionConfig):
            raise TypeError("A validated PersistentMotionConfig is required.")
        object.__setattr__(self, "motion", PersistentMotionConfig._from_fields(asdict(self.motion)))

    @property
    def hashes(self):
        return {**super().hashes, "motion": configuration_hash(self.motion.to_dict())}
