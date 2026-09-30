"""Headless calibration adapter; truth and signal windows stay outside inference."""
from dataclasses import dataclass
import numpy as np
from plume_focus.full_fov_execution import FullFovRunConfig
from plume_focus.calibrated_scanner import CalibrationSession, CalibratedScanner
from plume_focus.sensing_session import required_fields


class WindowedSignal:
    def __init__(self,base,windows):
        self.base,self.windows=base,windows
        self.end_time_seconds=base.end_time_seconds

    def signal_at(self,times,positions_yx):
        values=self.base.signal_at(times,positions_yx)
        if self.windows is None:return values
        t=np.asarray(times);active=np.zeros(t.shape,bool)
        for start,end in self.windows:active |= (t>=start)&(t<end)
        return np.where(active,values,0.)


@dataclass(frozen=True)
class CalibrationRunConfig(FullFovRunConfig):
    session: CalibrationSession
    active_windows: tuple[tuple[float,float],...] | None = None

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.session,CalibrationSession):raise TypeError('Calibration session required')
        if self.active_windows is not None:
            end=-1.
            for a,b in self.active_windows:
                if not 0<=a<b<=self.duration_seconds or a<end:raise ValueError('Invalid signal windows')
                end=b

    @classmethod
    def _from_fields(cls,value):
        required_fields(cls,value)
        from plume_focus.sensing_contracts import _decode
        from typing import get_type_hints
        payload=dict(value);session=CalibrationSession.from_dict(payload.pop('session'))
        annotations=get_type_hints(cls)
        return cls(session=session,**{k:_decode(annotations[k],v) for k,v in payload.items()})

    def make_controller(self):return CalibratedScanner(self.session)

    def build_source(self):return WindowedSignal(super().build_source(),self.active_windows)
