from dataclasses import replace
import json
import numpy as np
import pytest
from scipy.special import ndtr, ndtri
from scipy.optimize import brentq

from test_reconstruction_risk import config
from plume_focus.signal_distribution import PredictiveDistribution
from plume_focus.sensing_execution_v5 import with_threshold_policy, substitute_snapshot_policy, ControllerCallV3
from plume_focus.sensing_session_v5 import UPPER_POLICY, MEAN_POLICY
from plume_focus.sensing_messages_v4 import ScanPlanV4
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_cli_v1 import config_from_dict
from plume_focus.controller import SensingController
from plume_focus.sensing_simulation import FakeScanner
from plume_focus.sensing_execution_v1 import run_sensing_v1, CausalReplayV1
from plume_focus.sensing_transport_diagnostic import with_transport_diagnostic


@pytest.mark.parametrize('q',[.1,.5,.9,.95,.99])
def test_quantile_independent_cdf_and_sampling(q):
    weights=np.array([.2,.8]);means=np.array([[-1.,2.],[1.,-.5]]);var=np.array([[.3,.1],[.8,.4]])
    d=PredictiveDistribution(0.,weights,means,var);actual=d.quantile(q)
    for j in range(2):
        reference=brentq(lambda x:sum(weights[k]*ndtr((x-means[k,j])/np.sqrt(var[k,j])) for k in range(2))-q,-20,20,xtol=1e-13)
        assert actual[j]==pytest.approx(reference,abs=1e-10)
    np.testing.assert_allclose(d.cdf(actual),q,atol=1e-11)
    rng=np.random.default_rng(74);ids=rng.choice(2,size=100000,p=weights)
    sample=rng.normal(means[ids,0],np.sqrt(var[ids,0]))
    assert abs(np.mean(sample<=actual[0])-q)<.005


def test_quantile_atoms_singleton_and_zero_weight():
    d=PredictiveDistribution(0.,np.array([.2,.8,0.]),np.array([[0.],[2.],[1e6]]),np.zeros((3,1)))
    assert d.quantile(.1)[0]==pytest.approx(0,abs=1e-10)
    assert d.quantile(.95)[0]==2
    np.testing.assert_allclose(d.cdf(np.array([0.])),.2)
    singleton=PredictiveDistribution(0.,np.ones(1),np.array([[2.,-3.]]),np.array([[.4,0.]]))
    np.testing.assert_allclose(singleton.quantile(.95),[2+ndtri(.95)*np.sqrt(.4),-3])
    for q in (0,1,float('nan')):
        with pytest.raises(ValueError):d.quantile(q)


def controller(*,mixture=False,zero=False,policy=UPPER_POLICY,**settings):
    c=with_threshold_policy(config(mixture,zero),policy,**settings)
    return c,SensingController.from_session(c.session)


def set_field(s,mean,var=.0001):
    for b in getattr(s._belief,'components',(s._belief,)):
        b.mean[:]=mean;b.covariance[:]=np.eye(len(b.mean))*var


def decide(c,s,budget=8):
    return s.plan_next(s._last_available,s._last_pose,ScanRequestV1(budget),wind=c.wind)


def finish(c,s,plan,partial=False):
    execution=FakeScanner(c.source.build(),noise_std=c.simulated_noise_std,seed=0).execute(plan,
        delivery=c.delivery,delivery_delay=c.delivery_delay_seconds,completion_overrun=c.completion_overrun_seconds)
    readings=execution.readings[:1] if partial else execution.readings
    s.ingest_readings(readings,available_at_seconds=execution.receipt.received_at_seconds)
    s.finish_scan(execution.receipt)
    return execution


@pytest.mark.parametrize('mean,enabled',[(.149,False),(.15,False),(.151,True)])
def test_signal_gate_is_strict_mean_not_uncertainty(mean,enabled):
    c,s=controller();set_field(s,mean,var=4.)
    plan=decide(c,s)
    assert plan.signal_gate_enabled is enabled
    assert plan.signal_support=='declared_prior'
    if not enabled:assert plan.status=='reacquisition' and plan.fallback_reason=='mean_not_above_threshold'
    else:assert plan.fallback_reason!='mean_not_above_threshold'


@pytest.mark.parametrize('mixture',[False,True])
def test_binding_plan_call_snapshot_roundtrips(mixture):
    old=config(mixture);old_payload=old.to_dict();old_hashes=old.session.hashes
    c=with_threshold_policy(old);assert config_from_dict(json.loads(json.dumps(c.to_dict())))==c
    s=SensingController.from_session(c.session);set_field(s,[.5,.0,.5,.0])
    snap=s.decision_snapshot();fork=SensingController.from_snapshot(snap)
    plan=decide(c,s);other=decide(c,fork)
    assert plan==other and plan.status in ('threshold_partial','threshold_selected')
    assert ScanPlanV4.from_dict(json.loads(json.dumps(plan.to_dict())))==plan
    call=ControllerCallV3('plan_next',0.,(0.,0.),ScanRequestV1(8),c.wind,plan,(),None)
    assert ControllerCallV3.from_dict(json.loads(json.dumps(call.to_dict())))==call
    assert old.to_dict()==old_payload and old.session.hashes==old_hashes
    assert snap.digest==SensingController.from_snapshot(snap).decision_snapshot().digest


def test_zero_motion_equivalence_and_no_detector_noise_bonus():
    c,a=controller();d,b=controller(mixture=True,zero=True)
    set_field(a,[.6,0,.6,0]);set_field(b,[.6,0,.6,0])
    pa,pb=decide(c,a),decide(d,b)
    assert pa.eligible_cells==pb.eligible_cells and pa.samples==pb.samples
    noisy=replace(c,session=replace(c.session,model=replace(c.model,sensor_noise_std=10.)))
    s=SensingController.from_session(noisy.session);set_field(s,[.6,0,.6,0])
    p=decide(noisy,s);assert p.eligible_cells==pa.eligible_cells


def test_mean_ablation_and_upper_bound_add_uncertain_edges():
    c,a=controller();d,b=controller(policy=MEAN_POLICY)
    set_field(a,[.3,0,.3,0],var=.01);set_field(b,[.3,0,.3,0],var=.01)
    pa,pb=decide(c,a),decide(d,b)
    assert set(pb.eligible_cells)<set(pa.eligible_cells)
    c,a=controller();d,b=controller(policy=MEAN_POLICY)
    set_field(a,[.3,0,.3,0],var=0);set_field(b,[.3,0,.3,0],var=0)
    assert decide(c,a).samples==decide(d,b).samples


def test_regions_move_expand_and_retain_full_state():
    regions=[]
    for mean in ([.5,0,.5,0],[0,.5,0,.5],[.5,.5,.5,.5],[.5,0,.5,0]):
        c,s=controller();set_field(s,mean,var=0);regions.append(decide(c,s).eligible_cells)
    assert regions[0]!=regions[1] and set(regions[0])<set(regions[2]) and regions[0]==regions[3]


def test_partial_does_not_raise_threshold_and_closure_preserves_cells():
    c,s=controller();set_field(s,[.6,.3,.6,.3],var=0)
    plan=decide(c,s,budget=8.)
    assert plan.signal_threshold==.15 and plan.eligible_cells==tuple(range(16))
    assert plan.status=='threshold_partial' and plan.unvisited_eligible_cells
    assert set(plan.selected_eligible_cells)<=set(plan.requested_cells)<=set(plan.swept_cells)
    assert set(plan.unvisited_eligible_cells)==set(plan.eligible_cells)-set(plan.swept_cells)
    assert len(plan.samples)>=4 and plan.duration_seconds<=8.


def test_service_partial_execution_overrun_and_duplicate_atomicity():
    c,s=controller(survey_interval_seconds=.01);set_field(s,.4)
    s._last_available=.1;s._belief.advance(.1,c.wind,None)
    plan=decide(c,s)
    assert plan.service_status=='required' and plan.service_required_cells
    with pytest.raises(RuntimeError):decide(c,s)
    execution=FakeScanner(c.source.build(),noise_std=0,seed=0).execute(plan,completion_overrun=.2)
    sample=execution.readings[-1]
    s.ingest_readings([sample],available_at_seconds=sample.available_at_seconds)
    digest=s._belief.mean.copy()
    s.ingest_readings([sample],available_at_seconds=sample.available_at_seconds)
    np.testing.assert_array_equal(s._belief.mean,digest)
    with pytest.raises(ValueError,match='Conflicting'):
        s.ingest_readings([replace(sample,value=sample.value+1)],available_at_seconds=sample.available_at_seconds)
    s.finish_scan(execution.receipt)
    assert s._last_survey_completed==0 and s._last_available==execution.receipt.received_at_seconds
    next_plan=decide(c,s);assert next_plan.signal_support=='delivered_posterior'


def test_receipt_budget_and_no_action():
    c,s=controller();set_field(s,.5)
    c=replace(c,session=replace(c.session,policy=replace(c.session.policy,expected_delivery_delay_seconds=.4,
        expected_completion_overrun_seconds=.3)),delivery_delay_seconds=.4,completion_overrun_seconds=.3)
    s=SensingController.from_session(c.session);set_field(s,.5)
    p=decide(c,s,3);assert p.duration_seconds+.7<=3+1e-10
    c,s=controller();p=decide(c,s,.01);assert p.status=='no_legal_action' and not p.samples


def test_simulator_fake_scanner_replay_and_causal_views():
    c,_=controller();c=replace(c,duration_seconds=8.)
    run=run_sensing_v1(c);replay=CausalReplayV1(c.session,run.events)
    s=replay.advance_to(8.)
    assert s.startup_complete==run.startup_complete
    before=s.decision_snapshot().digest
    s.predictive_distribution(9.,(11,13));s.predictive_distribution(8.,(4,4))
    assert before==s.decision_snapshot().digest
    for e in run.events:assert ControllerCallV3.from_dict(e.to_dict())==e
    with pytest.raises(ValueError,match='precedes'):s.predictive_view(0.)


def test_policy_substitution_preserves_state_and_rejects_model_change():
    old=with_transport_diagnostic(config(),'zero_inflow_v1');s=SensingController.from_session(old.session)
    set_field(s,[.1,.2,.3,.4]);snapshot=s.decision_snapshot()
    c=with_threshold_policy(old);changed=substitute_snapshot_policy(snapshot,c.session)
    before,after=snapshot.to_dict(),changed.to_dict();before.pop('session');after.pop('session')
    assert before==after
    with pytest.raises(ValueError,match='model'):
        substitute_snapshot_policy(snapshot,replace(c.session,model=replace(c.model,source_rate=.3)))


def test_autonomous_adapter_reproduces_public_calls_and_late_delivery():
    from plume_focus.sensing_threshold_evaluation import run_policy,evaluate_execution
    from plume_focus.sensing_counterfactual import TruthQuadrature
    c,s=controller();c=replace(c,session=replace(c.session,policy=replace(c.session.policy,
        expected_delivery='batch',expected_delivery_delay_seconds=.3)),delivery='batch',delivery_delay_seconds=.3)
    s=SensingController.from_session(c.session);snapshot=s.decision_snapshot()
    q=TruthQuadrature(c.source.build(),0.,8.,c.model.grid_shape,(8,8),.1)
    record=run_policy(snapshot,c,horizon=8.,seed=3,truth=q)
    assert evaluate_execution(snapshot,record,q,verify_policy=True)['J']==pytest.approx(record['J'],abs=1e-14)
    first=next(e for e in record['events'] if e['operation']=='ingest_readings')
    assert all(r['available_at_seconds']==first['at_seconds'] for r in first['readings'])
    assert snapshot.digest==s.decision_snapshot().digest


def test_corrected_historical_baseline_decodes_without_changing_prior():
    from plume_focus.sensing_execution_v5 import with_transport_baseline
    original=with_transport_diagnostic(config(),'zero_inflow_v1')
    snapshot=SensingController.from_session(original.session).decision_snapshot()
    for policy in ('static_corridor_segment_v2','age_segment_v2'):
        baseline=with_transport_baseline(original,policy)
        assert config_from_dict(json.loads(json.dumps(baseline.to_dict())))==baseline
        child=substitute_snapshot_policy(snapshot,baseline.session)
        assert child.to_dict()['components']==snapshot.to_dict()['components']


def test_prefix_search_does_not_stop_at_an_expensive_route(monkeypatch):
    import plume_focus.sensing_policy_v5 as policy
    c,s=controller();set_field(s,[.6,.3,.6,.3],var=0)
    seen=[];original=policy.receipt_time
    def nonmonotone(plan,cfg):
        n=len(plan.requested_cells);seen.append(n)
        if n==4:return 100.
        if n==5:return 1.
        return original(plan,cfg)
    monkeypatch.setattr(policy,'receipt_time',nonmonotone)
    plan=decide(c,s,8.)
    assert 4 in seen and 5 in seen and len(plan.selected_eligible_cells)>=5


def test_partial_startup_requires_actual_scene_visits():
    c,s=controller();c=replace(c,session=replace(c.session,startup_mode='full_scene'))
    s=SensingController.from_session(c.session);plan=decide(c,s,8.)
    assert plan.fallback_reason=='startup_incomplete'
    finish(c,s,plan,partial=True)
    assert not s.startup_complete
    assert decide(c,s,8.).fallback_reason=='startup_incomplete'


@pytest.mark.parametrize('mixture',[False,True])
def test_detached_acquisition_bounds_match_runtime_projection(mixture):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('bound_observers_test',Path(__file__).parents[1]/'scripts/threshold_bound_observers.py')
    observers=importlib.util.module_from_spec(spec);spec.loader.exec_module(observers)
    c,s=controller(mixture=mixture);set_field(s,[.6,.2,.6,.2])
    p=decide(c,s);records=observers.point_forecasts(s,p)
    mean_before=s._belief.mean.copy()
    for point,sample in zip(records,p.samples):
        d=s.predictive_distribution(sample.acquired_at_seconds,p.lattice_shape)
        assert point['mean']==pytest.approx(d.mean.ravel()[point['cell']],abs=1e-12)
        assert point['upper']==pytest.approx(d.quantile(.95).ravel()[point['cell']],abs=1e-10)
    np.testing.assert_array_equal(s._belief.mean,mean_before)


def test_plan_rejects_misleading_gate_and_partial_metadata():
    c,s=controller();set_field(s,[.6,.3,.6,.3],var=0);p=decide(c,s)
    with pytest.raises(ValueError):replace(p,signal_gate_enabled=False)
    with pytest.raises(ValueError):replace(p,unvisited_eligible_cells=())
    with pytest.raises(ValueError):replace(p,eligible_cells=(*p.eligible_cells,999))
