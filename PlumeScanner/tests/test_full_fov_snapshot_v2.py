"""READY-state invariants at startup, partial FULL, entry and retention."""
import numpy as np
import pytest

from plume_focus.controller import SensingController
from plume_focus.sensing_snapshot import DecisionSnapshot
from plume_focus.sensing_counterfactual import run_counterfactual
from plume_focus.sensing_messages_v2 import ScanRequestV1
from test_full_fov_loop import controller as full_controller, plan, execute
from test_evidence_gate import controller as gated_controller


def detached(c):
    snapshot=c.decision_snapshot()
    copy=DecisionSnapshot.from_dict(snapshot.to_dict()).fork_controller(SensingController)
    assert copy.decision_snapshot().digest==snapshot.digest
    return snapshot,copy


@pytest.mark.parametrize("factory", [full_controller,gated_controller])
@pytest.mark.parametrize("stage", ["startup","partial_full","entry","renewal"])
def test_full_lifecycle_snapshot_preserves_next_plan_and_evidence(factory,stage):
    c=factory()
    if stage!="startup":
        p=plan(c);execute(c,p,.6,count=8 if stage=="partial_full" else None)
    if stage=="renewal":
        p=plan(c);assert p.mode=="FOCUS";execute(c,p,0.)
    snapshot,clone=detached(c)
    before=c._belief.mean.copy()
    clone._belief.mean+=.01
    np.testing.assert_array_equal(c._belief.mean,before)
    clone=snapshot.fork_controller(SensingController)
    p=plan(c);q=plan(clone)
    assert p==q
    execute(c,p,.6);execute(clone,q,.6)
    assert clone.decision_snapshot().digest==c.decision_snapshot().digest
    assert clone.full_scan_missing==c.full_scan_missing
    if hasattr(c,"gate_state"):
        assert c.gate_state()==clone.gate_state()


def test_calibrated_snapshot_preserves_decay_binding():
    from plume_focus.calibrated_scanner import CalibratedScanner, CalibrationSession, CalibrationSettings
    base=full_controller().session
    session=CalibrationSession(base.instrument,base.model,CalibrationSettings(),None,half_life_seconds=6.)
    c=CalibratedScanner(session);execute(c,plan(c),.6)
    _,clone=detached(c)
    assert clone._belief.half_life_seconds==6.
    now=c._last_available+1.
    np.testing.assert_array_equal(c.predictive_view(now).mean,clone.predictive_view(now).mean)


def test_active_and_unknown_extension_snapshots_are_rejected():
    c=gated_controller();plan(c)
    with pytest.raises(ValueError,match="READY"):c.decision_snapshot()
    class RecoveryExtension(type(gated_controller())):
        pass
    extension=RecoveryExtension(gated_controller().session)
    extension.recovery_budget=3.
    with pytest.raises(TypeError,match="own explicit snapshot"):extension.decision_snapshot()


@pytest.mark.parametrize("change", ["future_visit","missing_field","duplicate_reading"])
def test_corrupt_snapshot_cannot_restore(change):
    c=gated_controller();execute(c,plan(c),.6)
    p=c.decision_snapshot().to_dict()
    if change=="future_visit":p["visit_times"][0][0]=p["last_available"]+1.
    elif change=="missing_field":p.pop("epoch")
    else:p["seen"].append(p["seen"][0])
    with pytest.raises(ValueError):DecisionSnapshot.from_dict(p).fork_controller(SensingController)


def test_legacy_counterfactual_cannot_silently_force_a_gated_scan():
    c=gated_controller()
    with pytest.raises(ValueError,match="lifecycle adapter"):
        run_counterfactual(c.decision_snapshot(),None,None,(),continuation="idle")


@pytest.mark.parametrize("version", [1,2])
def test_snapshot_rejects_a_covariance_with_positive_diagonal_but_negative_eigenvalue(version):
    if version==1:
        from test_scan_diagnosis import config
        c=SensingController.from_session(config().session)
    else:c=gated_controller()
    p=c.decision_snapshot().to_dict()
    state=p["components"][0] if version==1 else p
    n=len(state["mean"]);cov=np.eye(n);cov[0,1]=cov[1,0]=2.
    state["covariance"]=cov.tolist()
    with pytest.raises(ValueError,match="positive semidefinite"):
        DecisionSnapshot.from_dict(p).fork_controller(SensingController)
