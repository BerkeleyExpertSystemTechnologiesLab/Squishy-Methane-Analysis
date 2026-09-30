"""Public callable API: sensor messages in, ordered physical plans out.

No timeline, image, truth, hardware driver, renderer, or filesystem dependency.
The chosen path remains fixed while a scan is open. Delayed evidence can be
replayed within that scan from its saved starting state, never across a closed
decision boundary.
"""

from __future__ import annotations

from plume_focus.acquisition import AcquisitionController, AssimilationEvent
import numpy as np

from plume_focus.sensing_contracts import (
    CausalWind, ExecutionReceipt, Footprint, Position, ScanConstraints, ScanPlan,
    ScannerCapabilities, SensorReading, finite, position,
)
from plume_focus.sensing_policy import plan_scan
from plume_focus.sensing_session import SensingSessionV1
from plume_focus.sensing_session_v2 import SensingSessionV2
from plume_focus.sensing_messages_v2 import ScanPlanV2, ScanRequestV1
from plume_focus.signal_belief import BeliefView, GaussianSignalBelief, SignalModelConfig
from plume_focus.signal_distribution import PredictiveDistribution
from plume_focus.motion_belief import PersistentWindBelief
from plume_focus.sensing_session_v3 import SensingSessionV3
from plume_focus.sensing_session_v4 import SensingSessionV4, SensingMotionSessionV4
from plume_focus.sensing_session_v5 import SensingSessionV5, SensingMotionSessionV5
from plume_focus.sensing_compat import HistoricalPolicy


class SensingController(AcquisitionController):
    def __setattr__(self, name, value):
        if name in {"capabilities", "model", "source_position", "start_full_scene", "policy"} and getattr(self, "_session", None) is not None:
            raise AttributeError("Frozen-v1 settings cannot change during a session; create a new explicit session.")
        super().__setattr__(name, value)

    def __init__(self, capabilities: ScannerCapabilities = ScannerCapabilities(),
                 model: SignalModelConfig = SignalModelConfig(), *, source_position: Position | None = None,
                 start_full_scene: bool = True, policy: str = "adaptive"):
        if source_position is not None:
            position(source_position)
        if policy not in ("adaptive", "static_corridor", "rescan"):
            raise ValueError("Unknown policy.")
        self._initialize_acquisition(capabilities, model, source_position)
        self.start_full_scene, self.policy = start_full_scene, policy
        self._session = None

    @classmethod
    def from_session(cls, session: SensingSessionV1):
        from plume_focus.evidence_gated_loop import EvidenceGatedScanner, EvidenceGatedSession
        if isinstance(session, EvidenceGatedSession):
            return EvidenceGatedScanner(session)
        from plume_focus.calibrated_scanner import CalibrationSession, CalibratedScanner
        if isinstance(session, CalibrationSession):
            return CalibratedScanner(session)
        from plume_focus.full_fov_loop import FullFovSession, FullFovController
        if isinstance(session, FullFovSession):
            return FullFovController(session)
        if not isinstance(session, SensingSessionV1):
            raise TypeError("A validated immutable SensingSessionV1 is required.")
        controller = cls(session.instrument.capabilities(), session.model,
                         source_position=session.source_position,
                         start_full_scene=session.startup_mode == "full_scene")
        controller.policy = session.policy.policy_id
        controller._session = session
        controller._instrument = session.instrument
        controller._policy_adapter = HistoricalPolicy(session)
        controller._last_pose = session.initial_position_yx
        transport_id = getattr(session, "transport_id", "legacy_clipped_v1")
        controller._belief = GaussianSignalBelief(session.model, transport_id=transport_id)
        if isinstance(session, (SensingSessionV3, SensingMotionSessionV4, SensingMotionSessionV5)):
            controller._belief = PersistentWindBelief(session.model, session.motion, transport_id=transport_id)
        return controller

    def decision_snapshot(self):
        """Detached READY-state diagnostic view; never opens or changes a scan."""
        from plume_focus.sensing_snapshot import DecisionSnapshot
        return DecisionSnapshot.capture(self)

    @classmethod
    def from_snapshot(cls, snapshot):
        """Restore an independent READY controller; the snapshot stays immutable."""
        return snapshot.fork_controller(cls)

    @property
    def startup_complete(self) -> bool:
        return bool(np.all(self._last_sampled >= 0)) if self.start_full_scene else True

    def plan_next(self, now_seconds: float, current_position: Position,
                  constraints: ScanConstraints | ScanRequestV1 = ScanConstraints(), *, wind: CausalWind | None = None) -> ScanPlan | ScanPlanV2:
        finite("decision time", now_seconds)
        position(current_position)
        if self._active is not None:
            raise RuntimeError("Finish the current scan before choosing another path.")
        if now_seconds < max(self._belief.time_seconds, self._last_available) - 1e-10:
            raise ValueError("Decision precedes available evidence or the prior completion.")
        if self._last_pose is not None and not np.allclose(current_position, self._last_pose, atol=1e-10, rtol=0):
            raise ValueError("Current position must match the last execution receipt; external moves need a receipt.")
        if wind is not None and wind.available_at_seconds > now_seconds + 1e-10:
            raise ValueError("Future wind is not available to this decision.")
        if wind is not None and wind.available_at_seconds < self._wind.available_at_seconds:
            raise ValueError("Wind updates must have nondecreasing availability times.")
        if self._session is not None and not isinstance(constraints, ScanRequestV1):
            raise TypeError("Frozen-v1 sessions require ScanRequestV1; policy parameters are bound to the session.")
        if self._session is None and not isinstance(constraints, ScanConstraints):
            raise TypeError("Legacy controllers require ScanConstraints.")
        self._belief.advance(now_seconds, self._wind, self.source_position)
        if wind is not None:
            self._wind = wind
        if self._session is None:
            plan = plan_scan(self._belief, self.capabilities, constraints, self._wind, self.source_position,
                             self._last_sampled, scan_id=f"scan-{self._scan_count:06d}", now=now_seconds,
                             pose=current_position, rotation=self._rotation,
                             full_scene=self.start_full_scene and self._scan_count == 0, policy=self.policy)
        else:
            plan = self._policy_adapter.plan(self, constraints, now_seconds, current_position)
        self._last_available = now_seconds
        if plan.samples:
            self._active = plan
            self._base = self._belief.clone()
            self._pending, self._received, self._events = {}, {}, []
            self._last_available = now_seconds
        return plan

    def finish_scan(self, receipt: ExecutionReceipt) -> BeliefView:
        self._finish_evidence(receipt)
        if self._session is not None:
            self._policy_adapter.finished(self, receipt)
        else:
            self._rotation = (self._rotation + 1) % self.capabilities.lattice_shape[1]
        self._close_scan()
        return self._belief.view()


__all__ = ["SensingController", "SignalModelConfig", "ScannerCapabilities", "ScanConstraints", "ScanPlan",
           "SensorReading", "CausalWind", "ExecutionReceipt", "Footprint", "AssimilationEvent",
           "SensingSessionV1", "SensingSessionV2", "ScanRequestV1", "ScanPlanV2"]
