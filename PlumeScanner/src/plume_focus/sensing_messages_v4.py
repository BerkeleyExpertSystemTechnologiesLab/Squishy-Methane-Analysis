"""Version-four threshold decisions; coverage is descriptive, never a mass gate."""

from dataclasses import dataclass, fields
import math
from typing import ClassVar, Literal
from plume_focus.sensing_contracts import Message, finite
from plume_focus.sensing_messages_v3 import ScanPlanV3


@dataclass(frozen=True)
class SignalBoundSample(Message):
    cell: int
    at_seconds: float
    upper_signal: float
    mean_signal: float

    def __post_init__(self):
        if type(self.cell) is not int or self.cell < 0:
            raise ValueError("Bound cell must be a nonnegative integer.")
        finite("bound time",self.at_seconds)
        finite("upper signal",self.upper_signal,minimum=float('-inf'))
        finite("mean signal",self.mean_signal,minimum=float('-inf'))


@dataclass(frozen=True)
class ScanPlanV4(ScanPlanV3):
    schema_version: ClassVar[int] = 4
    status: Literal["threshold_selected", "threshold_partial", "startup_full_scene", "reacquisition", "no_legal_action"]
    signal_gate_enabled: bool = False
    signal_gate_max_mean: float = 0.
    signal_support: Literal["declared_prior", "delivered_posterior"] = "declared_prior"
    signal_threshold: float = .15
    upper_quantile: float = .95
    forecast_at_seconds: float = 0.
    eligible_cells: tuple[int,...] = ()
    selected_eligible_cells: tuple[int,...] = ()
    unvisited_eligible_cells: tuple[int,...] = ()
    fallback_reason: str = ""
    acquisition_bounds: tuple[SignalBoundSample,...] = ()
    completion_upper_signal: tuple[float,...] = ()
    selection_criterion: str = "signal_gate_then_upper_signal_prefix"
    numerical_status: str = "model_conditional_pointwise_bounds"

    def __post_init__(self):
        allowed=("threshold_selected","threshold_partial","startup_full_scene","reacquisition","no_legal_action")
        if self.status not in allowed:
            raise ValueError("Unsupported threshold plan status.")
        values={f.name:getattr(self,f.name) for f in fields(ScanPlanV3)}
        ScanPlanV3(**{**values,"status":"candidate" if self.status.startswith('threshold_') else self.status})
        if type(self.signal_gate_enabled) is not bool or self.signal_support not in ("declared_prior","delivered_posterior"):
            raise ValueError("Invalid signal gate provenance.")
        for name in ("signal_threshold","forecast_at_seconds"):
            finite(name,getattr(self,name))
        finite("gate mean",self.signal_gate_max_mean,minimum=float('-inf'))
        if not .5 < self.upper_quantile < 1:
            raise ValueError("Invalid upper quantile.")
        for name in ("eligible_cells","selected_eligible_cells","unvisited_eligible_cells"):
            values=tuple(getattr(self,name));object.__setattr__(self,name,values)
            if len(set(values))!=len(values) or any(type(i) is not int or i<0 for i in values):
                raise ValueError("Threshold cells must be unique nonnegative integers.")
        if not set(self.selected_eligible_cells).issubset(self.eligible_cells):
            raise ValueError("Selected threshold cells must qualify.")
        if any(i>=math.prod(self.lattice_shape) for i in self.eligible_cells):
            raise ValueError("Eligible cells lie outside the physical lattice.")
        if not set(self.selected_eligible_cells).issubset(self.requested_cells):
            raise ValueError("Explicitly selected cells must be requested.")
        if set(self.unvisited_eligible_cells)!=set(self.eligible_cells)-set(self.swept_cells):
            raise ValueError("Unvisited eligibility must reflect mechanical closure.")
        if self.status=='threshold_selected' and self.unvisited_eligible_cells:
            raise ValueError("Full threshold selection cannot hide unvisited eligible cells.")
        if self.status=='threshold_partial' and not self.unvisited_eligible_cells:
            raise ValueError("Partial threshold selection must name its uncovered region.")
        if self.signal_gate_enabled!=(self.signal_gate_max_mean>self.signal_threshold):
            raise ValueError("Signal gate must match the strict current-mean threshold.")
        if self.status.startswith('threshold_') and not self.signal_gate_enabled:
            raise ValueError("Focused action requires signal support.")
        if self.risk_candidates or self.expected_integrated_map_risk is not None:
            raise ValueError("Threshold plans are not risk-selected decisions.")
        object.__setattr__(self,"acquisition_bounds",tuple(self.acquisition_bounds))
        object.__setattr__(self,"completion_upper_signal",tuple(self.completion_upper_signal))
        if self.completion_upper_signal and len(self.completion_upper_signal)!=math.prod(self.lattice_shape):
            raise ValueError("Completion bounds must match the physical lattice.")
        for v in self.completion_upper_signal:
            finite("completion bound",v,minimum=float('-inf'))


def threshold_plan(plan, **changes):
    return ScanPlanV4(**{**{f.name:getattr(plan,f.name) for f in fields(ScanPlanV3) if hasattr(plan,f.name)},**changes})
