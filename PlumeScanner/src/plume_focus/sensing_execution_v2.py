"""V2 configuration for the shared public-call simulator/replay implementation."""

from dataclasses import asdict, dataclass

from plume_focus.sensing_contracts import finite
from plume_focus.sensing_execution_v1 import SensingRunConfigV1, SyntheticSourceV1
from plume_focus.sensing_session_v2 import SensingSessionV2


@dataclass(frozen=True)
class ScaledSyntheticSource:
    source: object
    scale: float

    @property
    def end_time_seconds(self):
        return self.source.end_time_seconds

    def signal_at(self, times, positions_yx):
        return self.scale * self.source.signal_at(times, positions_yx)


@dataclass(frozen=True)
class SyntheticSourceV2(SyntheticSourceV1):
    # Zero is a declared blank-scene diagnostic, not missing truth.
    signal_scale: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        finite("synthetic signal scale", self.signal_scale)

    def build(self):
        values = asdict(self)
        scale = values.pop("signal_scale")
        return ScaledSyntheticSource(SyntheticSourceV1(**values).build(), scale)


@dataclass(frozen=True)
class SensingRunConfigV2(SensingRunConfigV1):
    session: SensingSessionV2
    source: SyntheticSourceV2
    simulated_noise_std: float
    simulated_noise_provenance: str

    def __post_init__(self):
        super().__post_init__()
        finite("simulated retrieval noise", self.simulated_noise_std)
        if not self.simulated_noise_provenance.strip():
            raise ValueError("Independent simulator noise provenance is required.")
