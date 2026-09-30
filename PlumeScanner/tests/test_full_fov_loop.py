from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest

from test_sensing_v1 import profile
from plume_focus.controller import SensingController
from plume_focus.full_fov_loop import FullFovController, FullFovSession, FullFovSettings, FullFovPlan
from plume_focus.physical_scan import compile_physical_scan, compile_profile_scan
from plume_focus.sensing_contracts import CausalWind, ExecutionReceipt, SensorReading
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_simulation import FakeScanner, pose_at
from plume_focus.signal_belief import SignalModelConfig


class ConstantSource:
    end_time_seconds = None
    def __init__(self, value=0.): self.value = value
    def signal_at(self, times, positions_yx): return np.full(np.shape(times), self.value)


def controller(**settings):
    model = SignalModelConfig(grid_shape=(2,2), signal_units="synthetic_concentration_path",
                              source_rate=0, diffusivity=0, process_std_per_sqrt_second=0)
    session = FullFovSession(profile(), model, FullFovSettings(**settings), None)
    return FullFovController(session)


def plan(c, budget=100., wind=None):
    return c.plan_next(c._last_available, c._last_pose, ScanRequestV1(budget), wind=wind)


def execute(c, p, value=0., delivery='stream', delay=0., overrun=0., count=None):
    ex = FakeScanner(ConstantSource(value), noise_std=0).execute(p, delivery=delivery,
        delivery_delay=delay, completion_overrun=overrun)
    readings = ex.readings if count is None else ex.readings[:count]
    if delivery == 'batch': c.ingest_readings(readings, available_at_seconds=ex.receipt.received_at_seconds)
    else:
        for r in readings: c.ingest_readings([r], available_at_seconds=r.available_at_seconds)
    c.finish_scan(ex.receipt)
    return ex


def field(c, mean, variance=1e-5):
    c._belief.mean[:] = mean
    c._belief.covariance[:] = np.eye(len(c._belief.mean))*variance


def test_full_startup_even_with_positive_prior_then_focus():
    c=controller();field(c,.6)
    p=plan(c);assert p.mode=='FULL' and len(p.samples)==16 and not c.startup_complete
    with pytest.raises(RuntimeError):plan(c)
    execute(c,p,.6)
    assert c.startup_complete and c.completed_full_scans==1
    q=plan(c);assert q.mode=='FOCUS' and q.duration_seconds<=12
    assert not {'target_mass','risk_candidates','service_required_cells'} & q.to_dict().keys()


def test_blank_repeats_full_scan_and_charges_return_home():
    c=controller();p=plan(c);execute(c,p)
    q=plan(c)
    assert q.mode=='FULL' and q.reason=='signal_not_supported' and q.full_epoch==2
    assert len(q.samples)==16 and not np.allclose(q.start_position_yx,c.session.home_yx)
    assert pose_at(q,q.return_home_until_seconds)==c.session.home_yx
    assert q.return_home_until_seconds>q.started_at_seconds
    assert q.duration_seconds>p.duration_seconds
    assert len(c.full_scan_missing)==16
    execute(c,q);assert c.completed_full_scans==2


def test_focus_to_full_on_mean_gate_without_resetting_belief():
    c=controller();execute(c,plan(c),.6);assert plan(c).mode=='FOCUS'
    execute(c,c.active_plan,.0)
    field(c,.1);mu=c._belief.mean.copy();cov=c._belief.covariance.copy()
    q=plan(c);assert q.mode=='FULL'
    np.testing.assert_array_equal(c._belief.mean,mu)
    np.testing.assert_array_equal(c._belief.covariance,cov)


@pytest.mark.parametrize('value,mode',[(.149,'FULL'),(.15,'FULL'),(.151,'FOCUS')])
def test_strict_mean_gate_not_upper_bound(value,mode):
    c=controller();execute(c,plan(c));field(c,value,1.)
    assert plan(c).mode==mode


def test_each_full_sweep_needs_new_actual_readings():
    c=controller();execute(c,plan(c))
    p=plan(c);execute(c,p,count=1)
    assert c.completed_full_scans==1 and len(c.full_scan_missing)==15
    field(c,.9)
    q=plan(c);assert q.mode=='FULL' and q.reason=='resume_full_scan'
    assert q.return_home_until_seconds==q.started_at_seconds
    execute(c,q,.9);assert c.completed_full_scans==2
    assert plan(c).mode=='FOCUS'


def test_return_interruption_does_not_complete_full_obligation():
    c=controller();execute(c,plan(c));p=plan(c)
    at=(p.started_at_seconds+p.return_home_until_seconds)/2
    receipt=ExecutionReceipt(p.scan_id,at,at,pose_at(p,at));c.finish_scan(receipt)
    assert c.completed_full_scans==1 and len(c.full_scan_missing)==16
    q=plan(c);assert q.mode=='FULL' and q.return_home_until_seconds>q.started_at_seconds
    assert q.start_position_yx==receipt.final_position_yx


def test_horizon_cannot_replace_full_startup_with_small_scan():
    c=controller();p=plan(c,12)
    assert p.mode=='NONE' and p.reason=='incomplete_full_scan_budget' and p.required_duration_seconds>12
    assert c.active_plan is None and not c.startup_complete
    assert plan(c,100).mode=='FULL'


def test_missing_reading_and_wide_footprint_cannot_fabricate_coverage():
    c=controller();p=plan(c);ex=FakeScanner(ConstantSource(),noise_std=0).execute(p)
    r=ex.readings[0]
    with pytest.raises(ValueError,match='instrument'):
        c.ingest_readings([replace(r,footprint=replace(r.footprint,height=1.))],available_at_seconds=r.available_at_seconds)
    assert not c._pending
    c.ingest_readings([r],available_at_seconds=r.available_at_seconds);c.finish_scan(ex.receipt)
    assert not c.startup_complete and len(c.full_scan_missing)==15


def test_delayed_batch_duplicate_conflict_and_overrun():
    a=controller(delivery_delay_seconds=.4,completion_overrun_seconds=.2)
    b=controller(delivery_delay_seconds=.4,completion_overrun_seconds=.2)
    pa,pb=plan(a),plan(b)
    execute(a,pa,.6,delay=.4,overrun=.2)
    ex=FakeScanner(ConstantSource(.6),noise_std=0).execute(pb,delivery='batch',delivery_delay=.4,completion_overrun=.2)
    # Deliver the latest acquisition first, then replay earlier acquisitions.
    last=ex.readings[-1];b.ingest_readings([last],available_at_seconds=ex.receipt.received_at_seconds)
    before=b._belief.mean.copy()
    with pytest.raises(ValueError,match='Conflicting'):
        b.ingest_readings([ex.readings[0],replace(last,value=9.)],available_at_seconds=ex.receipt.received_at_seconds)
    np.testing.assert_array_equal(b._belief.mean,before)
    b.ingest_readings(ex.readings,available_at_seconds=ex.receipt.received_at_seconds)
    assert b.ingest_readings(ex.readings,available_at_seconds=ex.receipt.received_at_seconds)==()
    b.finish_scan(ex.receipt)
    np.testing.assert_allclose(a._belief.mean,b._belief.mean,atol=1e-12)
    np.testing.assert_allclose(a._belief.covariance,b._belief.covariance,atol=1e-12)
    with pytest.raises(ValueError,match='precedes'):
        b.predictive_view(ex.receipt.received_at_seconds-.1)
    q=plan(b);assert q.started_at_seconds==ex.receipt.received_at_seconds
    assert q.samples[0].acquired_at_seconds-last.acquired_at_seconds>=1/b.session.instrument.acquisition.detector_capacity_hz


def test_queries_and_transport_do_not_change_state_or_geometry():
    a=controller();b=controller()
    session=FullFovSession.from_dict(json.loads(json.dumps(a.session.to_dict())))
    assert session.digest==a.session.digest
    assert isinstance(SensingController.from_session(session),FullFovController)
    a.predictive_view(40.);a.predictive_distribution(40.,(80,120))
    p,q=plan(a),plan(b);assert p==q
    assert FullFovPlan.from_dict(json.loads(json.dumps(p.to_dict())))==p
    with pytest.raises(AttributeError):a.model=replace(a.model,initial_std=.5)
    with pytest.raises(ValueError):plan(controller(),wind=CausalWind(2.,2.))


def test_neutral_compiler_matches_historical_geometry_and_has_no_policy_fields():
    mask=np.zeros((4,4),bool);mask[[0,3],[0,2]]=True
    kw=dict(scan_id='a',started_at_seconds=0.,current_position=(.7,.8))
    neutral=compile_physical_scan(mask,profile(),**kw);old=compile_profile_scan(mask,profile(),**kw)
    assert neutral.samples==old.samples and neutral.segments==old.segments
    assert neutral.duration_seconds==old.duration_seconds
    assert 'target_mass' not in neutral.to_dict()


def test_focus_search_can_stay_expand_and_move_across_the_whole_field():
    c=controller();execute(c,plan(c),.6)
    regions=[]
    for values in ([.6,0,.6,0],[.6,0,.6,0],[.6,.6,.6,.6],[0,.6,0,.6]):
        field(c,values,1e-12)
        p=plan(c);assert p.mode=='FOCUS';regions.append(set(p.eligible_cells))
        execute(c,p,.6)
    assert regions[0]==regions[1] and regions[0]<regions[2]
    assert regions[3]!=regions[0] and any(cell%4==3 for cell in regions[3])


def test_headless_run_json_replay_and_noise_separation():
    from test_reconstruction_risk import config
    from plume_focus.full_fov_execution import FullFovRunConfig, run_full_fov, verify_replay
    from plume_focus.sensing_counterfactual import TruthQuadrature
    base=config();c=controller()
    run=FullFovRunConfig(c.session,replace(base.source,signal_scale=0.),CausalWind(),duration_seconds=40.,
                        evaluation_shape=(8,8),evaluation_step_seconds=.2)
    decoded=FullFovRunConfig.from_dict(json.loads(json.dumps(run.to_dict())))
    assert decoded.digest==run.digest
    truth=TruthQuadrature(run.build_source(),0.,40.,run.model.grid_shape,(8,8),.2)
    result=run_full_fov(run,truth=truth)
    restored=verify_replay(json.loads(json.dumps(result)),truth=truth)
    assert restored.completed_full_scans==2
    assert all(p['plan']['mode']=='FULL' for p in result['scans'])
    assert all(p['readings']==p['clean_readings'] for p in result['scans'])
    assert result['J']==0 and result['configuration']['session']['model']['sensor_noise_std']==.05


def test_lean_render_uses_causal_state_and_keeps_playback_out_of_acquisition():
    from types import SimpleNamespace
    from plume_focus.sensing_simulation import SensingScanTrace
    from plume_focus.sensing_rendering import compose_sensing_frame, RenderSettings
    c=controller();p=plan(c);prior=c.belief_view()
    ex=execute(c,p,.6);after=c.belief_view()
    trace=SensingScanTrace(p,ex,prior,after,after,c.assimilation_events,c._wind)
    run=SimpleNamespace(config=SimpleNamespace(model=c.model,capabilities=c.capabilities,
        policy=c.session.policy_id,session=c.session,simulated_noise_std=0.),source_assumption='source-free')
    payload=p.to_dict();mean=c._belief.mean.copy()
    at=ex.receipt.received_at_seconds
    a=compose_sensing_frame(trace,run,at,RenderSettings(),None,causal_belief=after)
    b=compose_sensing_frame(trace,run,at,RenderSettings(fps=2,playback_speed=3),None,
                            causal_belief=replace(after,mean=np.zeros_like(after.mean)))
    # Right-hand signal panel responds to the supplied causal state, while the
    # middle measured-scalar panel is independent of playback/estimate settings.
    assert not np.array_equal(a[120:320,640:960],b[120:320,640:960])
    np.testing.assert_array_equal(a[120:320,320:640],b[120:320,320:640])
    np.testing.assert_array_equal(c._belief.mean,mean)
    assert p.to_dict()==payload
