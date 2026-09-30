"""Versioned margin and decay binding over the existing lean operating loop."""
from dataclasses import dataclass, fields, asdict
import math
import numpy as np

from plume_focus.full_fov_loop import FullFovSession, FullFovSettings, FullFovPlan, FullFovController
from plume_focus.sensing_session import required_fields, InstrumentProfileV1
from plume_focus.sensing_contracts import finite
from plume_focus.signal_belief import SignalModelConfig, GaussianSignalBelief


def decay_factors(rate, dt):
    if rate == 0: return 1., dt, dt
    return math.exp(-rate*dt), -math.expm1(-rate*dt)/rate, -math.expm1(-2*rate*dt)/(2*rate)


class DecayGaussianBelief(GaussianSignalBelief):
    def __init__(self, config, *, half_life_seconds=None, **kwargs):
        super().__init__(config, **kwargs)
        if half_life_seconds is not None: finite('half life',half_life_seconds,positive=True)
        self.half_life_seconds=half_life_seconds

    @property
    def decay_rate(self):
        return 0. if self.half_life_seconds is None else math.log(2)/self.half_life_seconds

    def clone(self):
        result=super().clone();result.half_life_seconds=self.half_life_seconds
        return result

    def _propagation_factors(self,dt):return decay_factors(self.decay_rate,dt)


@dataclass(frozen=True)
class CalibrationSettings(FullFovSettings):
    vertical_margin_fov: float = 0.
    full_scan_only: bool = False

    def __post_init__(self):
        super().__post_init__()
        finite('vertical margin',self.vertical_margin_fov)
        if self.vertical_margin_fov>1 or type(self.full_scan_only) is not bool:
            raise ValueError('Invalid calibration settings')


@dataclass(frozen=True)
class CalibrationSession(FullFovSession):
    settings: CalibrationSettings
    policy_id: str = 'full_fov_margin_loop_v2_candidate'
    half_life_seconds: float | None = None
    formulation_id: str = 'beam_reconstruction_decay_v1_candidate'

    def __post_init__(self):
        if not isinstance(self.settings,CalibrationSettings):raise TypeError('CalibrationSettings required')
        original=FullFovSession(self.instrument,self.model,
            FullFovSettings(**{f.name:getattr(self.settings,f.name) for f in fields(FullFovSettings)}),
            self.source_position,self.home_yx,self.transport_id,clock_origin=self.clock_origin)
        for name in ('instrument','model','source_position','home_yx'):
            object.__setattr__(self,name,getattr(original,name))
        if self.policy_id!='full_fov_margin_loop_v2_candidate' or self.formulation_id!='beam_reconstruction_decay_v1_candidate':
            raise ValueError('Unsupported calibration binding')
        if self.half_life_seconds is not None:finite('half life',self.half_life_seconds,positive=True)

    @classmethod
    def _from_fields(cls,value):
        required_fields(cls,value)
        return cls(instrument=InstrumentProfileV1.from_dict(value['instrument']),
            model=SignalModelConfig._from_fields(value['model']),
            settings=CalibrationSettings._from_fields(value['settings']),
            **{k:v for k,v in value.items() if k not in ('instrument','model','settings')})


@dataclass(frozen=True)
class CalibrationPlan(FullFovPlan):
    selected_cells: tuple[int,...] = ()
    padding_cells: tuple[int,...] = ()
    clipped_selected_cells: tuple[int,...] = ()
    vertical_margin_fov: float = 0.

    def __post_init__(self):
        super().__post_init__()
        finite('margin',self.vertical_margin_fov)
        if self.mode=='FOCUS':
            if set(self.selected_cells)|set(self.padding_cells)!=set(self.requested_cells):
                raise ValueError('Core and padding must partition requested cells')
            if set(self.selected_cells)&set(self.padding_cells):raise ValueError('Overlapping core/padding')


class CalibratedScanner(FullFovController):
    def __init__(self,session):
        if not isinstance(session,CalibrationSession):raise TypeError('CalibrationSession required')
        super().__init__(session)
        self._belief=DecayGaussianBelief(session.model,half_life_seconds=session.half_life_seconds,
                                       transport_id=session.transport_id)

    def _focus_supported(self,gate_mean):
        return not self.session.settings.full_scan_only and super()._focus_supported(gate_mean)

    def _focus_mask(self,mask):
        radius=math.ceil(self.session.settings.vertical_margin_fov*mask.shape[0]-1e-12)
        expanded=mask.copy()
        for shift in range(1,min(radius,mask.shape[0]-1)+1):
            expanded[shift:] |= mask[:-shift]
            expanded[:-shift] |= mask[shift:]
        return expanded

    def _decorate_plan(self,plan,core):
        radius=math.ceil(self.session.settings.vertical_margin_fov*plan.lattice_shape[0]-1e-12)
        rows,cols=plan.lattice_shape
        return CalibrationPlan(**{f.name:getattr(plan,f.name) for f in fields(FullFovPlan)},
            selected_cells=core,padding_cells=tuple(sorted(set(plan.requested_cells)-set(core))) if core else (),
            clipped_selected_cells=tuple(i for i in core if i//cols-radius<0 or i//cols+radius>=rows),
            vertical_margin_fov=self.session.settings.vertical_margin_fov)
