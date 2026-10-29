"""Version-two READY snapshots for the full-scan/calibrated/evidence lifecycles.

No source, truth, simulator, filesystem or forced-action injection. Unknown
subclasses are rejected: their extra state cannot safely be guessed or omitted.
"""
import numpy as np

from plume_focus.sensing_contracts import CausalWind, SensorReading, finite, position


def _bindings():
    from plume_focus.full_fov_loop import FullFovController, FullFovSession
    from plume_focus.calibrated_scanner import CalibratedScanner, CalibrationSession
    from plume_focus.evidence_gated_loop import EvidenceGatedScanner, EvidenceGatedSession
    return {c.__name__:(c,s) for c,s in ((FullFovController,FullFovSession),
        (CalibratedScanner,CalibrationSession),(EvidenceGatedScanner,EvidenceGatedSession))}


def capture_payload(controller):
    if type(controller).__name__ not in _bindings() or type(controller) is not _bindings()[type(controller).__name__][0]:
        raise TypeError("A custom full-FOV controller needs its own explicit snapshot state adapter.")
    if controller.active_plan is not None or controller._pending or getattr(controller,"_gate_open_action",None) is not None:
        raise ValueError("A diagnostic snapshot requires a completed-scan READY state.")
    b=controller._belief
    p={"schema_version":2,"message_type":"DecisionSnapshot","controller_type":type(controller).__name__,
       "session":controller.session.to_dict(),"transport_id":b.transport_id,"inference_time":b.time_seconds,
       "mean":b.mean.tolist(),"covariance":b.covariance.tolist(),"wind":controller._wind.to_dict(),
       "last_pose":controller._last_pose,"last_available":controller._last_available,
       "scan_count":controller._scan_count,"rotation":controller._rotation,
       "last_survey_completed":controller._last_survey_completed,
       "last_detector_reading":controller._last_detector_reading.to_dict() if controller._last_detector_reading else None,
       "visit_times":controller._last_sampled.tolist(),"seen":[r.to_dict() for _,r in sorted(controller._seen.items())],
       "full_missing":None if controller._full_missing is None else controller._full_missing.tolist(),
       "return_pending":controller._return_pending,"epoch":controller._epoch,
       "completed_full_scans":controller.completed_full_scans,"gate_state":None,
       "gate_decision_evidence":[],"gate_last_fully_executed":None}
    if hasattr(controller,"gate_state"):
        p.update(gate_state=controller.gate_state().to_dict(),
                 gate_decision_evidence=[e.to_dict() for e in controller._gate_decision_evidence],
                 gate_last_fully_executed=controller._gate_last_fully_executed)
    return p


def restore_controller(p, controller_type):
    bindings=_bindings()
    if p.get("controller_type") not in bindings:
        raise ValueError("Unsupported full-FOV snapshot controller.")
    expected,session_type=bindings[p["controller_type"]]
    session=session_type.from_dict(p["session"])
    controller=controller_type.from_session(session)
    if type(controller) is not expected:
        raise ValueError("Snapshot factory returned a different controller type.")
    # Exact key checking prevents silent loss of extension/recovery state.
    if set(p)!=set(capture_payload(controller)):
        raise ValueError("Full-FOV snapshot must contain exactly its declared state fields.")
    b=controller._belief
    mean=np.array(p["mean"],float);cov=np.array(p["covariance"],float)
    from plume_focus.sensing_snapshot import validate_moments
    validate_moments(mean,cov,b.mean.shape)
    if p["transport_id"]!=b.transport_id:
        raise ValueError("Snapshot estimator transport differs.")
    finite("snapshot inference time",p["inference_time"])
    finite("snapshot availability",p["last_available"],p["inference_time"])
    position(p["last_pose"])
    for key in ("scan_count","rotation","completed_full_scans","epoch"):
        if type(p[key]) is not int or p[key]<(1 if key=="epoch" else 0):
            raise ValueError("Invalid snapshot lifecycle count.")
    if p["completed_full_scans"]>p["scan_count"] or type(p["return_pending"]) is not bool:
        raise ValueError("Invalid full-scan lifecycle state.")
    visits=np.asarray(p["visit_times"],float)
    if (visits.shape!=session.instrument.lattice_shape or not np.all(np.isfinite(visits))
            or np.any((visits<0)&(visits!=-1)) or np.any(visits>p["inference_time"]+1e-10)):
        raise ValueError("Invalid snapshot visits.")
    missing=p["full_missing"]
    if missing is not None:
        raw=np.asarray(missing)
        if raw.shape!=visits.shape or raw.dtype!=np.dtype(bool) or not raw.any():
            raise ValueError("Invalid pending full-scan coverage.")
        missing=raw.copy()
    elif p["completed_full_scans"]==0 or p["return_pending"]:
        raise ValueError("Full coverage cannot be absent before startup or while returning home.")
    wind=CausalWind.from_dict(p["wind"])
    if wind.available_at_seconds>p["last_available"]:
        raise ValueError("Snapshot wind is not yet available.")
    seen=[SensorReading.from_dict(r) for r in p["seen"]]
    if len({r.reading_id for r in seen})!=len(seen) or any(r.available_at_seconds>p["last_available"]+1e-10 for r in seen):
        raise ValueError("Invalid snapshot evidence identities or availability.")
    last=SensorReading.from_dict(p["last_detector_reading"]) if p["last_detector_reading"] else None
    if last is not None and last not in seen:
        raise ValueError("Last detector reading must belong to the delivered evidence.")
    finite("last survey completion",p["last_survey_completed"])
    if p["last_survey_completed"]>p["last_available"]:
        raise ValueError("Survey completion is in the future.")
    b.mean,b.covariance,b.time_seconds=mean,cov,float(p["inference_time"])
    controller._last_pose=tuple(p["last_pose"]);controller._last_available=float(p["last_available"])
    controller._wind=wind;controller._last_sampled=visits.copy()
    controller._seen={r.reading_id:r for r in seen};controller._last_detector_reading=last
    controller._scan_count=p["scan_count"];controller._rotation=p["rotation"]
    controller._last_survey_completed=float(p["last_survey_completed"])
    controller._full_missing=missing;controller._return_pending=p["return_pending"]
    controller._epoch=p["epoch"];controller.completed_full_scans=p["completed_full_scans"]
    if hasattr(controller,"gate_state"):
        from plume_focus.evidence_gate import EvidenceGateState, GateActionEvidence
        state=EvidenceGateState.from_dict(p["gate_state"])
        if (state.epoch>controller._epoch or (state.last_decision is not None
                and state.focus_allowed!=state.last_decision.focus_allowed)):
            raise ValueError("Inconsistent snapshot gate state.")
        if p["gate_last_fully_executed"] is not None and type(p["gate_last_fully_executed"]) is not bool:
            raise ValueError("Invalid last execution flag.")
        controller._gate_epoch=state.epoch;controller._gate_action_index=state.action_index
        controller._gate_epoch_alpha=state.epoch_alpha_allocated;controller._gate_focus_allowed=state.focus_allowed
        controller._gate_contract_violations=state.contract_violations
        controller._gate_focus_evidence=state.focus_evidence;controller._gate_last_decision=state.last_decision
        controller._gate_full_evidence=list(state.full_epoch_evidence);controller._gate_transitions=list(state.transitions)
        controller._gate_decision_evidence=tuple(GateActionEvidence.from_dict(e) for e in p["gate_decision_evidence"])
        controller._gate_last_fully_executed=p["gate_last_fully_executed"]
    elif p["gate_state"] is not None or p["gate_decision_evidence"] or p["gate_last_fully_executed"] is not None:
        raise ValueError("Non-gated snapshot cannot contain evidence-gate state.")
    return controller
