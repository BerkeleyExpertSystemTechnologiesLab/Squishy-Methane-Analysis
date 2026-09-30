"""Immutable, serializable READY-state snapshots. No source, evaluator or I/O."""

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from plume_focus.sensing_contracts import CausalWind, SensorReading


@dataclass(frozen=True)
class DecisionSnapshot:
    _payload_json: str

    @classmethod
    def capture(cls, controller):
        if controller.active_plan is not None or controller._session is None:
            raise ValueError("A diagnostic snapshot requires an explicit READY session.")
        belief = controller._belief
        payload = {
            "schema_version": 1, "message_type": "DecisionSnapshot",
            "session": controller._session.to_dict(),
            "transport_id": belief.transport_id,
            "inference_time": belief.time_seconds,
            "components": [{"mean": b.mean.tolist(), "covariance": b.covariance.tolist()}
                           for b in getattr(belief, "components", (belief,))],
            "log_weights": belief.log_weights.tolist() if hasattr(belief, "log_weights") else None,
            "wind": controller._wind.to_dict(), "last_available": controller._last_available,
            "last_pose": controller._last_pose, "scan_count": controller._scan_count,
            "rotation": controller._rotation, "last_survey_completed": controller._last_survey_completed,
            "last_detector_reading": controller._last_detector_reading.to_dict() if controller._last_detector_reading else None,
            "visit_times": controller._last_sampled.tolist(), "startup_complete": controller.startup_complete,
            "seen": [r.to_dict() for _, r in sorted(controller._seen.items())],
        }
        return cls(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))

    def to_dict(self):
        return json.loads(self._payload_json)

    @property
    def digest(self):
        return hashlib.sha256(self._payload_json.encode()).hexdigest()

    @classmethod
    def from_dict(cls, payload):
        result = cls(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
        if payload.get("schema_version") != 1 or payload.get("message_type") != "DecisionSnapshot":
            raise ValueError("Unsupported snapshot.")
        return result

    def fork_controller(self, controller_type):
        from plume_focus.sensing_session_v4 import SensingSessionV4, SensingMotionSessionV4
        from plume_focus.sensing_session_diagnostic import TransportDiagnosticSession, MotionTransportDiagnosticSession
        from plume_focus.sensing_session import SensingSessionV1
        from plume_focus.sensing_session_v2 import SensingSessionV2
        from plume_focus.sensing_session_v3 import SensingSessionV3
        from plume_focus.sensing_session_v5 import SensingSessionV5, SensingMotionSessionV5, TransportBaselineSessionV2
        p = self.to_dict()
        if p.get("schema_version") != 1 or p.get("message_type") != "DecisionSnapshot":
            raise ValueError("Unsupported snapshot.")
        types = {c.__name__: c for c in (SensingSessionV1, SensingSessionV2, SensingSessionV3,
                 SensingSessionV4, SensingMotionSessionV4, TransportDiagnosticSession, MotionTransportDiagnosticSession,
                 SensingSessionV5, SensingMotionSessionV5, TransportBaselineSessionV2)}
        session = types[p["session"]["message_type"]].from_dict(p["session"])
        controller = controller_type.from_session(session)
        belief = controller._belief
        components = getattr(belief, "components", (belief,))
        if len(components) != len(p["components"]) or belief.transport_id != p["transport_id"]:
            raise ValueError("Snapshot estimator identity differs.")
        for state, saved in zip(components, p["components"]):
            mean, covariance = np.array(saved["mean"], float), np.array(saved["covariance"], float)
            if (mean.shape != state.mean.shape or covariance.shape != state.covariance.shape
                    or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance))):
                raise ValueError("Invalid snapshot moments.")
            state.mean, state.covariance, state.time_seconds = mean, covariance, float(p["inference_time"])
        if hasattr(belief, "log_weights"):
            logs = np.array(p["log_weights"], float)
            if logs.shape != belief.log_weights.shape or not np.all(np.isfinite(logs)):
                raise ValueError("Invalid snapshot weights.")
            belief.log_weights = logs
        controller._wind = CausalWind.from_dict(p["wind"])
        controller._last_available = float(p["last_available"])
        controller._last_pose = tuple(p["last_pose"]) if p["last_pose"] is not None else None
        controller._scan_count, controller._rotation = int(p["scan_count"]), int(p["rotation"])
        controller._last_survey_completed = float(p["last_survey_completed"])
        controller._last_detector_reading = SensorReading.from_dict(p["last_detector_reading"]) if p["last_detector_reading"] else None
        controller._last_sampled = np.asarray(p["visit_times"], float)
        if controller._last_sampled.shape != session.instrument.lattice_shape or not np.all(np.isfinite(controller._last_sampled)):
            raise ValueError("Invalid snapshot visits.")
        controller._seen = {r["reading_id"]: SensorReading.from_dict(r) for r in p["seen"]}
        if controller.startup_complete != p["startup_complete"]:
            raise ValueError("Snapshot startup state differs from actual visits.")
        return controller
