"""Full field first; supported signal focuses, absent signal returns home/rescans.

Gaussian estimator and acquisition lifecycle are shared with historical modes.
This module knows no simulation, truth, files, risk policy or media settings.
"""
from dataclasses import asdict, dataclass, fields
from typing import Literal
import numpy as np

from plume_focus.acquisition import AcquisitionController
from plume_focus.physical_scan import PhysicalPlan, compile_physical_scan
from plume_focus.sensing_contracts import CausalWind, Position, finite, position
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_session import InstrumentProfileV1, RequiredMessage, configuration_hash, required_fields
from plume_focus.signal_belief import GaussianSignalBelief, SignalModelConfig


@dataclass(frozen=True)
class FullFovSettings(RequiredMessage):
    threshold: float = .15
    upper_quantile: float = .95
    focus_seconds: float = 12.
    minimum_readings: int = 4
    delivery_delay_seconds: float = 0.
    completion_overrun_seconds: float = 0.

    def __post_init__(self):
        finite("threshold", self.threshold)
        finite("focus duration", self.focus_seconds, positive=True)
        finite("delivery delay", self.delivery_delay_seconds)
        finite("completion overrun", self.completion_overrun_seconds)
        if isinstance(self.upper_quantile, bool) or not .5 < self.upper_quantile < 1:
            raise ValueError("Expected an upper quantile between .5 and 1.")
        if type(self.minimum_readings) is not int or self.minimum_readings < 4:
            raise ValueError("At least four focused readings are required.")


@dataclass(frozen=True)
class FullFovSession(RequiredMessage):
    instrument: InstrumentProfileV1
    model: SignalModelConfig
    settings: FullFovSettings
    source_position: Position | None
    home_yx: Position = (0., 0.)
    transport_id: Literal["zero_inflow_v1"] = "zero_inflow_v1"
    policy_id: Literal["full_fov_threshold_loop_v1_candidate"] = "full_fov_threshold_loop_v1_candidate"
    clock_origin: str = "seconds from session start"

    def __post_init__(self):
        if not isinstance(self.instrument, InstrumentProfileV1) or not isinstance(self.model, SignalModelConfig):
            raise TypeError("Validated instrument and model required.")
        if not isinstance(self.settings, FullFovSettings):
            raise TypeError("Validated loop settings required.")
        object.__setattr__(self, "model", SignalModelConfig._from_fields(asdict(self.model)))
        object.__setattr__(self, "settings", FullFovSettings._from_fields(asdict(self.settings)))
        if self.instrument.scan_mode != "column" or self.model.signal_units != self.instrument.retrieval.units:
            raise ValueError("Column scanner and matching retrieved-scalar units required.")
        if self.transport_id != "zero_inflow_v1" or self.policy_id != "full_fov_threshold_loop_v1_candidate":
            raise ValueError("Unsupported loop binding.")
        position(self.home_yx)
        object.__setattr__(self, "home_yx", tuple(float(v) for v in self.home_yx))
        if self.source_position is not None:
            position(self.source_position)
            object.__setattr__(self, "source_position", tuple(float(v) for v in self.source_position))
        if not self.clock_origin:
            raise ValueError("Declare a session clock origin.")

    def to_dict(self):
        result = super().to_dict()
        result["instrument"] = self.instrument.to_dict()
        return result

    @classmethod
    def _from_fields(cls, value):
        required_fields(cls, value)
        required_fields(SignalModelConfig, value["model"])
        return cls(instrument=InstrumentProfileV1.from_dict(value["instrument"]),
                   model=SignalModelConfig._from_fields(value["model"]),
                   settings=FullFovSettings._from_fields(value["settings"]),
                   **{k:v for k,v in value.items() if k not in ("instrument", "model", "settings")})

    @property
    def digest(self):
        return configuration_hash(self.to_dict())


@dataclass(frozen=True)
class FullFovPlan(PhysicalPlan):
    mode: Literal["FULL", "FOCUS", "NONE"]
    reason: str
    gate_max_mean: float
    threshold: float
    eligible_cells: tuple[int, ...]
    omitted_cells: tuple[int, ...]
    full_epoch: int
    return_home_until_seconds: float
    required_duration_seconds: float

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in ("FULL", "FOCUS", "NONE") or not self.reason:
            raise ValueError("Explicit action mode and reason required.")
        finite("gate mean", self.gate_max_mean, minimum=-float('inf'))
        finite("threshold", self.threshold)
        finite("required duration", self.required_duration_seconds)
        finite("return-home clock", self.return_home_until_seconds)
        if type(self.full_epoch) is not int or self.full_epoch < 0:
            raise ValueError("Invalid full-scan epoch.")
        n = int(np.prod(self.lattice_shape))
        for cells in (self.eligible_cells, self.omitted_cells):
            if len(set(cells)) != len(cells) or any(type(i) is not int or not 0 <= i < n for i in cells):
                raise ValueError("Invalid eligibility cells.")
        if set(self.omitted_cells) != set(self.eligible_cells)-set(self.swept_cells):
            raise ValueError("Omissions must reflect the physical closure.")
        if self.mode == "NONE" and (self.samples or self.segments):
            raise ValueError("No-action result cannot execute motion or readings.")

    @property
    def status(self):
        return {"FULL":"full_scan", "FOCUS":"focused_scan", "NONE":"no_legal_action"}[self.mode]


class FullFovController(AcquisitionController):
    """Minimal operating state over the shared, causal acquisition engine."""
    def __setattr__(self, name, value):
        if name in {"session", "capabilities", "model", "source_position"} and getattr(self, "_bound", False):
            raise AttributeError("Instrument, model and loop settings are immutable within a session.")
        super().__setattr__(name, value)

    def __init__(self, session: FullFovSession):
        if not isinstance(session, FullFovSession):
            raise TypeError("FullFovSession required.")
        self.session = session
        self._initialize_acquisition(session.instrument.capabilities(), session.model,
                                     session.source_position, session.instrument)
        self._belief = GaussianSignalBelief(session.model, transport_id=session.transport_id)
        self._last_pose = session.home_yx
        self._full_missing = np.ones(self.capabilities.lattice_shape, bool)
        self._return_pending = True
        self._epoch = 1
        self.completed_full_scans = 0
        self._bound = True

    @property
    def startup_complete(self):
        return self.completed_full_scans > 0

    @property
    def full_scan_missing(self):
        return () if self._full_missing is None else tuple(np.flatnonzero(self._full_missing).tolist())

    def _focus_supported(self, gate_mean):
        return gate_mean > self.session.settings.threshold

    def _focus_mask(self, mask):
        return mask

    def _decorate_plan(self, plan, core):
        return plan

    def _on_plan_opened(self, plan):
        """Derived-policy hook; the shared lifecycle is unchanged when it is inert."""

    def _on_scan_finished(self, receipt):
        """Derived-policy hook after receipt validation, before pending evidence is cleared."""

    def plan_next(self, now_seconds, current_position, constraints: ScanRequestV1, *, wind: CausalWind | None = None):
        finite("decision time", now_seconds)
        position(current_position)
        if not isinstance(constraints, ScanRequestV1):
            raise TypeError("Pass remaining session duration in ScanRequestV1.")
        if self._active is not None:
            raise RuntimeError("Finish the current scan before choosing another path.")
        if now_seconds < max(self._last_available, self._belief.time_seconds)-1e-10:
            raise ValueError("Decision precedes available evidence.")
        if not np.allclose(current_position, self._last_pose, atol=1e-10, rtol=0):
            raise ValueError("Position must match the last execution receipt.")
        if wind is not None and not self._wind.available_at_seconds <= wind.available_at_seconds <= now_seconds:
            raise ValueError("Wind must be causal and chronological.")
        self._belief.advance(now_seconds, self._wind, self.source_position)
        if wind is not None:
            self._wind = wind
        self._last_available = now_seconds
        cfg, shape = self.session.settings, self.capabilities.lattice_shape
        remaining = max(0., constraints.max_duration_seconds)
        receipt_lag = cfg.delivery_delay_seconds + cfg.completion_overrun_seconds
        gate_mean = float(np.max(self.predictive_moments(now_seconds, shape)[0]))
        last = self._last_detector_reading

        def compile(mask, home=False):
            return compile_physical_scan(mask, self.session.instrument,
                scan_id=f"scan-{self._scan_count:06d}", started_at_seconds=now_seconds,
                current_position=tuple(current_position), return_home=self.session.home_yx if home else None,
                last_reading_at_seconds=last.acquired_at_seconds if last else None,
                last_integration_seconds=last.integration_seconds if last else None)

        def result(path, mode, reason, eligible=(), required=0., returning=False, core=()):
            # The compiler first visits home along horizontal then vertical axes.
            home_time = now_seconds
            if returning:
                mech = self.session.instrument.mechanics
                home_time += abs(current_position[1]-self.session.home_yx[1])/mech.move_horizontal_fov_per_second
                home_time += abs(current_position[0]-self.session.home_yx[0])/mech.move_vertical_fov_per_second
            plan = FullFovPlan(**{f.name:getattr(path,f.name) for f in fields(PhysicalPlan)},
                mode=mode, reason=reason, gate_max_mean=gate_mean, threshold=cfg.threshold,
                eligible_cells=eligible, omitted_cells=tuple(i for i in eligible if i not in set(path.swept_cells)),
                full_epoch=self._epoch, return_home_until_seconds=home_time,
                required_duration_seconds=required or path.duration_seconds+receipt_lag)
            plan = self._decorate_plan(plan, core)
            if plan.samples:
                self._open_scan(plan)
                self._on_plan_opened(plan)
            return plan

        reason = "startup" if not self.startup_complete else "resume_full_scan"
        eligible = ()
        if self._full_missing is None and self._focus_supported(gate_mean):
            ceiling = min(remaining, cfg.focus_seconds)
            distribution = self.predictive_distribution(now_seconds+ceiling/2, shape)
            upper = distribution.quantile(cfg.upper_quantile).ravel()
            ids = np.flatnonzero(upper >= cfg.threshold)
            eligible = tuple(ids.tolist())
            order = ids[np.lexsort(((ids-self._rotation) % np.prod(shape), self._last_sampled.ravel()[ids], -upper[ids]))]
            mask = np.zeros(shape, bool)
            best = None
            for cell in order:
                mask.ravel()[cell] = True
                expanded = self._focus_mask(mask)
                count = sum(np.ptp(np.flatnonzero(expanded[:,j]))+1 for j in np.flatnonzero(expanded.any(axis=0)))
                if count/self.session.instrument.acquisition.cadence_hz > ceiling+1e-9:
                    break
                candidate = compile(expanded)
                if len(candidate.samples) >= cfg.minimum_readings and candidate.duration_seconds+receipt_lag <= ceiling+1e-9:
                    best = candidate, tuple(np.flatnonzero(mask).tolist())
            if best is not None:
                return result(best[0], "FOCUS", "signal_supported", eligible, core=best[1])
            reason = "empty_focus_region" if not eligible else "focus_infeasible"
        elif self._full_missing is None:
            reason = "signal_not_supported"
        if self._full_missing is None:
            self._epoch += 1
            self._full_missing = np.ones(shape, bool)
            self._return_pending = True
        path = compile(self._full_missing, self._return_pending)
        if path.duration_seconds+receipt_lag > remaining+1e-9:
            return result(compile(np.zeros(shape, bool)), "NONE", "incomplete_full_scan_budget",
                          eligible, path.duration_seconds+receipt_lag)
        return result(path, "FULL", reason, eligible, returning=self._return_pending)

    def ingest_readings(self, readings, *, available_at_seconds):
        # Verify the declared instrument and requested physical cell before the
        # shared atomic duplicate/time validation mutates any evidence.
        for reading in readings:
            if reading.reading_id in self._seen or reading.reading_id in self._pending:
                continue
            cell = tuple(np.minimum((np.asarray(reading.position_yx)*self.capabilities.lattice_shape).astype(int),
                                    np.asarray(self.capabilities.lattice_shape)-1))
            index = np.ravel_multi_index(cell, self.capabilities.lattice_shape)
            if self._active is None or index not in self._active.swept_cells:
                raise ValueError("Reading does not cover a requested physical cell.")
            if reading.footprint != self.capabilities.footprint or not np.isclose(reading.integration_seconds, self.capabilities.integration_seconds):
                raise ValueError("Reading must respect the fixed instrument.")
        return super().ingest_readings(readings, available_at_seconds=available_at_seconds)

    def finish_scan(self, receipt):
        self._finish_evidence(receipt)
        if self._active.mode == "FULL":
            for reading in self._pending.values():
                cell = tuple(np.minimum((np.asarray(reading.position_yx)*self.capabilities.lattice_shape).astype(int),
                                        np.asarray(self.capabilities.lattice_shape)-1))
                self._full_missing[cell] = False
            self._return_pending = receipt.completed_at_seconds < self._active.return_home_until_seconds-1e-9
            if not self._full_missing.any():
                self.completed_full_scans += 1
                self._full_missing = None
                self._return_pending = False
        self._rotation = (self._rotation+1) % int(np.prod(self.capabilities.lattice_shape))
        self._on_scan_finished(receipt)
        self._close_scan()
        return self._belief.view()
