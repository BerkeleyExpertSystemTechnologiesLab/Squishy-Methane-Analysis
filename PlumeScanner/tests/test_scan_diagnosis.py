from dataclasses import replace
import json

import numpy as np
import pytest

from plume_focus.controller import SensingController
from plume_focus.signal_belief import GaussianSignalBelief, SignalModelConfig, grid_positions, advection_stencil
from plume_focus.sensing_contracts import CausalWind
from plume_focus.sensing_risk import linear_transition, score_pattern
from plume_focus.sensing_transport_diagnostic import with_transport_diagnostic
from plume_focus.sensing_cli_v1 import config_from_dict
from plume_focus.sensing_snapshot import DecisionSnapshot
from test_reconstruction_risk import config


@pytest.mark.parametrize("velocity",[(0,.04),(0,-.04),(.04,0),(-.04,0),(.03,-.04)])
def test_zero_inflow_positive_substochastic_and_no_edge_source(velocity):
    model=SignalModelConfig(grid_shape=(5,7),source_rate=0,diffusivity=0,process_std_per_sqrt_second=0)
    a,_,_=linear_transition(model,velocity,None,.05,"zero_inflow_v1")
    assert np.min(a.data)>=0
    assert np.max(np.asarray(a.sum(axis=0)))<=1+1e-13
    assert np.max(np.asarray(a.sum(axis=1)))<=1+1e-13
    state=GaussianSignalBelief(model,transport_id="zero_inflow_v1")
    state.mean[:]=1; initial=state.mean.sum()
    state.advance(100,CausalWind(velocity_yx=velocity))
    assert 0<=state.mean.sum()<initial*.01


@pytest.mark.parametrize("transport",["legacy_clipped_v1","zero_inflow_v1"])
def test_runtime_scorer_transport_matches(transport):
    model=SignalModelConfig(grid_shape=(4,5),diffusivity=.001)
    state=GaussianSignalBelief(model,transport_id=transport)
    state.mean[:]=np.random.default_rng(0).normal(size=20)
    mu,p=state.mean.copy(),state.covariance.copy()
    wind=CausalWind(velocity_yx=(.03,-.04));source=(.5,.2)
    a,b,q=linear_transition(model,wind.velocity_yx,source,1.3,transport)
    state.advance(1.3,wind,source)
    np.testing.assert_allclose(state.mean,a@mu+b,atol=1e-13)
    np.testing.assert_allclose(state.covariance,a@p@a.T+q,atol=1e-13)
    assert state.clone().transport_id==transport


@pytest.mark.parametrize("mixture",[False,True])
def test_diagnostic_roundtrip_snapshot_and_historical_identity(mixture):
    old=config(mixture);payload=old.to_dict(); hashes=old.session.hashes
    for transport in ("legacy_clipped_v1","zero_inflow_v1"):
        new=with_transport_diagnostic(old,transport)
        assert config_from_dict(json.loads(json.dumps(new.to_dict())))==new
        controller=SensingController.from_session(new.session)
        snapshot=controller.decision_snapshot(); digest=snapshot.digest
        detached=DecisionSnapshot.from_dict(snapshot.to_dict()).fork_controller(SensingController)
        detached._belief.advance(1,CausalWind(velocity_yx=(.03,.02)))
        assert snapshot.digest==digest and controller._belief.time_seconds==0
        assert detached._belief.transport_id==transport
        score_pattern(controller._belief,new.session,new.wind,(),1.)
    assert old.to_dict()==payload and old.session.hashes==hashes
    assert "transport_id" not in payload["session"]


def test_high_zero_reading_same_covariance_different_correlated_mean():
    model=SignalModelConfig(grid_shape=(2,2),correlation_length=.8)
    a=GaussianSignalBelief(model);b=a.clone();h=np.array([1.,0,0,0])
    a.condition(0.,h,.01);b.condition(1.,h,.01)
    np.testing.assert_array_equal(a.covariance,b.covariance)
    assert np.all(b.mean>a.mean)


def test_forced_execution_is_detached_and_checks_physical_paths():
    from plume_focus.sensing_counterfactual import run_counterfactual, begin_forced_scan, TruthQuadrature, evaluate_saved_branch
    from plume_focus.sensing_policy_v4 import CandidatePath, compile_for
    c=with_transport_diagnostic(config(),"zero_inflow_v1")
    controller=SensingController.from_session(c.session)
    snapshot=controller.decision_snapshot()
    mask=np.zeros(c.capabilities.lattice_shape,bool);mask[:,0]=True
    plan=compile_for(c.session,mask,"diagnostic",0.,c.session.initial_position_yx)
    candidate=CandidatePath("test",tuple(np.flatnonzero(mask)),plan)
    q=TruthQuadrature(c.source.build(),0.,8.,c.model.grid_shape,(8,8),.1)
    result=run_counterfactual(snapshot,c,candidate,(),horizon=8.,truth=q)
    assert result["events"][0]["operation"]=="forced_plan"
    assert result["spent_seconds"]<=8 and result["J"]>=0
    assert evaluate_saved_branch(snapshot,result,q)['J']==pytest.approx(result['J'],abs=1e-14)
    assert controller.active_plan is None and snapshot.digest==controller.decision_snapshot().digest
    with pytest.raises(ValueError,match="compiled"):
        begin_forced_scan(controller,replace(plan,completed_at_seconds=plan.completed_at_seconds+.01),c.wind,8.)


def test_delayed_forced_evidence_cannot_improve_earlier_online_loss():
    from plume_focus.sensing_counterfactual import run_counterfactual, TruthQuadrature
    from plume_focus.sensing_policy_v4 import CandidatePath, compile_for
    c=with_transport_diagnostic(config(),"zero_inflow_v1")
    c=replace(c,session=replace(c.session,policy=replace(c.session.policy,expected_delivery="batch",
        expected_delivery_delay_seconds=.4,expected_completion_overrun_seconds=.2)),delivery="batch",
        delivery_delay_seconds=.4,completion_overrun_seconds=.2)
    controller=SensingController.from_session(c.session);snapshot=controller.decision_snapshot()
    mask=np.zeros(c.capabilities.lattice_shape,bool);mask[:,0]=True
    plan=compile_for(c.session,mask,"delayed",0.,c.session.initial_position_yx)
    candidate=CandidatePath("delayed",tuple(np.flatnonzero(mask)),plan)
    q=TruthQuadrature(c.source.build(),0.,8.,c.model.grid_shape,(8,8),.1)
    measured=run_counterfactual(snapshot,c,candidate,(),horizon=8.,truth=q)
    idle=run_counterfactual(snapshot,c,None,(),horizon=8.,truth=q,continuation="idle")
    delivered=plan.completed_at_seconds+.6
    for a,b in zip(measured['quadrature_samples'],idle['quadrature_samples']):
        if a['time']<delivered-1e-9:assert a==b
    assert measured['scans'][0]['receipt']['completed_at_seconds']==pytest.approx(plan.completed_at_seconds+.2)


def test_corrected_zero_mixture_matches_gaussian_risk_and_mean():
    from plume_focus.motion_belief import PersistentWindBelief
    from plume_focus.motion_config import PersistentMotionConfig
    g=with_transport_diagnostic(config(),"zero_inflow_v1")
    m=with_transport_diagnostic(config(True,zero=True),"zero_inflow_v1")
    gc=SensingController.from_session(g.session);mc=SensingController.from_session(m.session)
    for state in (gc._belief,mc._belief):
        state.advance(2.,CausalWind(velocity_yx=(.04,-.03)))
    np.testing.assert_allclose(gc._belief.mean,mc._belief.mean,atol=1e-14)
    np.testing.assert_allclose(gc._belief.covariance,mc._belief.covariance,atol=1e-14)
    assert score_pattern(gc._belief,g.session,g.wind,(),1.)['risk']==pytest.approx(
        score_pattern(mc._belief,m.session,m.wind,(),1.)['risk'],abs=1e-14)


def test_gaussian_cache_exact_independent_of_mean_and_detached():
    from plume_focus.sensing_risk import _score_pattern
    c=with_transport_diagnostic(config(),"zero_inflow_v1")
    controller=SensingController.from_session(c.session)
    expected=_score_pattern(controller._belief,c.session,c.wind,(),3.,include_history_samples=True)
    result=score_pattern(controller._belief,c.session,c.wind,(),3.,include_history_samples=True)
    assert result==expected
    controller._belief.mean[:]=300.
    result['risk']=-1;result['between_history_integrals'].append(999)
    assert score_pattern(controller._belief,c.session,c.wind,(),3.,include_history_samples=True)==expected
