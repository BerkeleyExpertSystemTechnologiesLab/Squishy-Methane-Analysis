from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from plume_focus.controller import SensingController
from plume_focus.motion_belief import PersistentWindBelief
from plume_focus.motion_config import PersistentMotionConfig
from plume_focus.sensing_cli_v1 import load_config, config_from_dict
from plume_focus.sensing_contracts import CausalWind, ExecutionReceipt
from plume_focus.sensing_execution_v1 import CausalReplayV1, ReconstructionGridV1, run_sensing_v1
from plume_focus.sensing_execution_v3 import with_persistent_motion
from plume_focus.sensing_execution_v4 import with_risk_policy, ControllerCallV2
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_messages_v3 import ScanPlanV3
from plume_focus.sensing_policy_v4 import compile_for, candidate_paths, repeated_pattern, service_requirement
from plume_focus.sensing_risk import ForecastSweep, mean_transition, linear_transition, projection_gram, score_pattern
from plume_focus.signal_belief import GaussianSignalBelief, observation_weights, grid_positions, interpolation_stencil
from plume_focus.sensing_simulation import FakeScanner, pose_at
from plume_focus.sensing_reconstruction import reconstruction_loss, summarize_v1
from test_sensing_v1 import profile


def config(mixture=False, zero=False, **settings):
    c = load_config(Path(__file__).parents[1]/"configs/sensing-v2/coarse-full-startup.json")
    c = replace(c, session=replace(c.session, instrument=profile(), physical_quadrature_weights=(1/16,)*16,
        startup_mode="prior", source_position=None, model=replace(c.model, grid_shape=(2,2), diffusivity=0,
        process_std_per_sqrt_second=0, source_rate=0)), duration_seconds=16., initial_budget_seconds=8., budget_seconds=8.,
        wind=CausalWind(), evaluation=ReconstructionGridV1((8,8),.1,"uniform_cell_area","left_endpoint"))
    if mixture:
        c=with_persistent_motion(c, ((0.,0.),(0.,0.)) if zero else ((.0001,0.),(0.,.0001)))
    return with_risk_policy(c, risk_spatial_shape=(8,8), **settings)


def test_configuration_and_plan_transport_separate_diagnostic_gamma():
    for c in (config(),config(True)):
        assert config_from_dict(json.loads(json.dumps(c.to_dict()))) == c
        controller=SensingController.from_session(c.session)
        plan=controller.plan_next(0.,(0.,0.),ScanRequestV1(8.),wind=c.wind)
        assert plan.status=="value_selected" and len(plan.risk_candidates)<=14
        assert plan.predicted_mass_recall < .90
        assert ScanPlanV3.from_dict(json.loads(json.dumps(plan.to_dict())))==plan
        assert plan.expected_integrated_map_risk <= plan.idle_integrated_map_risk+1e-10
    with pytest.raises(ValueError,match="four readings"):
        replace(c.session.policy,minimum_samples=3)
    changed=replace(c.session.policy,gamma=.7)
    assert changed.gamma==.7
    with pytest.raises(ValueError,match="delivery"):
        replace(c,delivery="batch")


def test_projection_gram_matches_direct_bilinear_variance():
    c=config();b=GaussianSignalBelief(c.model)
    rng=np.random.default_rng(4);a=rng.normal(size=(4,4));b.covariance=a@a.T
    w=projection_gram((2,2),(13,17))
    assert np.sum(w*b.covariance)==pytest.approx(b.moments_on((13,17))[1].mean(),abs=1e-12)
    assert not np.isclose(np.trace(b.covariance)/4,np.sum(w*b.covariance))


@pytest.mark.parametrize("elapsed",[.001,.13,1.3,6.])
def test_linear_mean_operator_preserves_existing_propagation(elapsed):
    c=config();m=replace(c.model,grid_shape=(4,5),diffusivity=.001,source_rate=.15,process_std_per_sqrt_second=.013)
    b=GaussianSignalBelief(m);b.mean=np.random.default_rng(2).normal(size=20)
    initial=b.mean.copy();wind=CausalWind(velocity_yx=(.03,-.05));source=(.5,.2)
    a,offset=mean_transition(m,wind.velocity_yx,source,elapsed)
    operator,_,process=linear_transition(m,wind.velocity_yx,source,elapsed)
    covariance=(operator@(operator@b.covariance).T).T+process
    b.advance(elapsed,wind,source)
    np.testing.assert_allclose(a@initial+offset,b.mean,atol=1e-13,rtol=1e-12)
    np.testing.assert_allclose(covariance,b.covariance,atol=1e-13,rtol=1e-12)


def test_gaussian_risk_matches_independent_joint_conditioning_and_delay():
    c=config();b=GaussianSignalBelief(c.model);mask=np.zeros((4,4),bool);mask[:2,0]=True
    plan=compile_for(c.session,mask,"x",0.,(0.,.125))
    sweeps=(ForecastSweep(plan,plan.completed_at_seconds),)
    result=score_pattern(b,c.session,c.wind,sweeps,4.)
    h=np.stack([observation_weights(c.model.grid_shape,s.position_yx,s.footprint) for s in plan.samples])
    deliveries=[s.acquired_at_seconds+s.integration_seconds/2 for s in plan.samples]
    boundaries=[0.,*deliveries,4.];expected=0.;p=b.covariance;w=projection_gram((2,2),(8,8))
    for n,(lo,hi) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        H=h[:n]
        posterior=p-p@H.T@np.linalg.solve(H@p@H.T+c.model.sensor_noise_std**2*np.eye(n),H@p) if n else p
        expected+=(hi-lo)*np.sum(w*posterior)
    assert result['risk']==pytest.approx(expected,abs=1e-12)
    delayed=replace(c.session,policy=replace(c.session.policy,expected_delivery="batch"))
    assert score_pattern(b,delayed,c.wind,sweeps,4.)['risk']>result['risk']
    untouched=GaussianSignalBelief(c.model)
    np.testing.assert_array_equal(b.covariance,untouched.covariance)


def test_mixture_expected_risk_against_independent_scalar_integral():
    c=config(True,history_count=128)
    b=PersistentWindBelief(c.model,c.session.motion)
    b.components=(b.components[0].clone(),b.components[0].clone())
    b.offsets=np.zeros((2,2));b.log_weights=np.log([.5,.5])
    for state,mu in zip(b.components,(-.12,.12)):
        state.mean[:]=mu;state.covariance=np.eye(4)*.01
    mask=np.zeros((4,4),bool);mask[0,0]=True
    plan=compile_for(c.session,mask,"x",0.,(0.,.125))
    score=score_pattern(b,c.session,c.wind,(ForecastSweep(plan,plan.completed_at_seconds),),2.,seed=13)
    sample=plan.samples[0];h=observation_weights((2,2),sample.position_yx,sample.footprint)
    W=projection_gram((2,2),(8,8));P=b.components[0].covariance
    variance=float(h@P@h+c.model.sensor_noise_std**2);gain=P@h/variance
    conditional=P-np.outer(P@h,P@h)/variance
    def integrand(y):
        means=np.stack([state.mean+gain*(y-h@state.mean) for state in b.components])
        logs=np.array([-.5*((y-h@state.mean)**2/variance+np.log(2*np.pi*variance))+np.log(.5) for state in b.components])
        marginal=np.exp(logsumexp(logs));weights=np.exp(logs-logsumexp(logs))
        mean=weights@means
        between=sum(w*((mu-mean)@W@(mu-mean)) for w,mu in zip(weights,means))
        return marginal*between
    posterior_between=quad(integrand,-2,2,epsabs=1e-12)[0]
    delivery=sample.acquired_at_seconds+sample.integration_seconds/2
    expected=delivery*np.sum(W*b.covariance)+(2-delivery)*(np.sum(W*conditional)+posterior_between)
    assert abs(score['risk']-expected)<=max(4*score['standard_error'],1e-6)
    assert score['between']>0 and score['within']>0
    np.testing.assert_array_equal(b.log_weights,np.log([.5,.5]))


def test_zero_motion_risk_and_execution_match_gaussian():
    plain,mixed=config(),config(True,zero=True)
    a,b=run_sensing_v1(plain),run_sensing_v1(mixed)
    assert len(a.scans)==len(b.scans)>0
    for x,y in zip(a.scans,b.scans):
        assert x.plan.samples==y.plan.samples
        assert x.plan.expected_integrated_map_risk==pytest.approx(y.plan.expected_integrated_map_risk,abs=1e-14)
        assert y.plan.history_count==0
    events=tuple(ControllerCallV2.from_dict(json.loads(json.dumps(e.to_dict()))) for e in a.events)
    replay=CausalReplayV1(plain.session,events);replay.advance_to(plain.duration_seconds)
    np.testing.assert_allclose(replay.controller.belief_view().mean,a.scans[-1].posterior.mean)


def test_candidates_full_fov_budget_and_readonly_service():
    c=config();b=GaussianSignalBelief(c.model);b.advance(31.,c.wind)
    ages=np.zeros((4,4));ages[3,3]=-1
    candidates,required,status=candidate_paths(b,c.session,ScanRequestV1(7.),c.wind,ages,
        scan_id="x",now=31.,pose=(0.,0.),rotation=0,last_survey_completed=0.,startup_complete=True)
    assert status=="required" and required and candidates
    for candidate in candidates:
        assert set(required)<=set(candidate.plan.swept_cells)
        assert candidate.plan.duration_seconds<=7.+1e-8 and len(candidate.plan.samples)>=4
        pattern=repeated_pattern(candidate,c.session,ages,0,0.,required,horizon=7.)
        assert all(s.receipt_at_seconds<=38.+1e-8 for s in pattern)
    assert b.time_seconds==31 and ages[3,3]==-1
    cells,status=service_requirement(c.session,ages,0,0.,now=31.,pose=(0.,0.),budget=.01)
    assert not cells and status=="unmet_no_reachable_block"


def test_service_credit_requires_actual_complete_required_block():
    c=config();controller=SensingController.from_session(c.session)
    plan=controller.plan_next(31.,(0.,0.),ScanRequestV1(8.),wind=c.wind)
    assert plan.service_required_cells and plan.service_status=="required"
    controller.finish_scan(ExecutionReceipt(plan.scan_id,31.1,31.1,pose_at(plan,31.1)))
    assert controller._last_survey_completed==0
    plan=controller.plan_next(31.1,controller._last_pose,ScanRequestV1(8.),wind=c.wind)
    actual=FakeScanner(c.source.build(),noise_std=0).execute(plan)
    controller.ingest_readings(actual.readings,available_at_seconds=actual.receipt.received_at_seconds)
    controller.finish_scan(actual.receipt)
    assert controller._last_survey_completed==actual.receipt.received_at_seconds


def test_v4_delayed_actual_evidence_duplicate_atomicity_and_fixed_path():
    c=config(True);a,b=(SensingController.from_session(c.session) for _ in range(2))
    plan=a.plan_next(0.,(0.,0.),ScanRequestV1(8.),wind=c.wind)
    assert b.plan_next(0.,(0.,0.),ScanRequestV1(8.),wind=c.wind)==plan
    actual=FakeScanner(c.source.build(),noise_std=.05).execute(plan,completion_overrun=.3)
    for reading in actual.readings:
        a.ingest_readings([reading],available_at_seconds=reading.available_at_seconds)
        assert a.active_plan==plan
    with pytest.raises(RuntimeError):
        a.plan_next(1.,(0.,0.),ScanRequestV1(8.))
    before=a.belief_view();last=actual.readings[-1]
    assert a.ingest_readings([last],available_at_seconds=last.available_at_seconds)==()
    with pytest.raises(ValueError):
        a.ingest_readings([replace(last,value=last.value+1)],available_at_seconds=last.available_at_seconds)
    np.testing.assert_array_equal(before.mean,a.belief_view().mean)
    b.ingest_readings(actual.readings,available_at_seconds=actual.receipt.received_at_seconds)
    va=a.finish_scan(actual.receipt);vb=b.finish_scan(actual.receipt)
    np.testing.assert_allclose(va.mean,vb.mean,atol=1e-12)
    np.testing.assert_allclose(a._belief.covariance,b._belief.covariance,atol=1e-12)
    with pytest.raises(ValueError):
        a.ingest_readings([replace(last,reading_id="new-after-close")],available_at_seconds=actual.receipt.received_at_seconds)


def test_queries_and_reporting_do_not_mutate_execution_or_imply_mass_target():
    c=config();run=run_sensing_v1(c);digest=run.operational_hash
    loss=reconstruction_loss(run,c.source.build())
    report=summarize_v1(run,c.source.build(),loss=loss)
    assert report['value_selected_count']>0
    assert all(not row['mass_target_is_execution_requirement'] for row in report['scan_schedule'])
    assert all(row['expected_integrated_map_risk'] is not None for row in report['scan_schedule'])
    assert run.operational_hash==digest
    no_truth=reconstruction_loss(run,None)
    assert no_truth['J'] is None


def test_repeated_patterns_credit_earlier_new_measurements_and_charge_travel():
    c=config();b=GaussianSignalBelief(c.model)
    candidates,required,_=candidate_paths(b,c.session,ScanRequestV1(16.),c.wind,np.full((4,4),-1.),
        scan_id='x',now=0.,pose=(0.,0.),rotation=0,last_survey_completed=0.,startup_complete=True)
    candidate=min(candidates,key=lambda p:p.plan.duration_seconds)
    repeated=repeated_pattern(candidate,c.session,np.full((4,4),-1.),0,0.,required,horizon=16.)
    assert len(repeated)>1
    assert score_pattern(b,c.session,c.wind,repeated,16.)['risk'] < score_pattern(b,c.session,c.wind,repeated[:1],16.)['risk']
    for first,second in zip(repeated,repeated[1:]):
        assert second.plan.started_at_seconds==first.receipt_at_seconds
        assert second.plan.start_position_yx==first.plan.final_position_yx
        assert second.plan.samples[0].acquired_at_seconds-first.plan.samples[-1].acquired_at_seconds >= .5


def test_mean_amplitude_does_not_change_fixed_gaussian_schedule_risk():
    c=config();b=GaussianSignalBelief(c.model);mask=np.zeros((4,4),bool);mask[:,0]=True
    p=compile_for(c.session,mask,'x',0.,(0.,0.));sweeps=(ForecastSweep(p,p.completed_at_seconds),)
    expected=score_pattern(b,c.session,c.wind,sweeps,8.)
    b.mean[:]=np.arange(4)*10.
    assert score_pattern(b,c.session,c.wind,sweeps,8.)==expected


def test_full_field_proposals_follow_moving_and_expanding_signal_without_roi_input():
    c=config(); b=GaussianSignalBelief(c.model); ages=np.full((4,4),-1.)
    def proposed(mean):
        b.mean[:]=mean
        candidates,_,_=candidate_paths(b,c.session,ScanRequestV1(8.),c.wind,ages,
            scan_id='same',now=0.,pose=(0.,0.),rotation=0,last_survey_completed=0.,startup_complete=True)
        return tuple(p for p in candidates if p.name.startswith('expected_excess'))
    left=proposed([10.,0.,0.,0.]); right=proposed([0.,0.,0.,10.])
    assert left==proposed([10.,0.,0.,0.])
    assert left and right and left[0].core_cells!=right[0].core_cells
    assert any(15 in p.core_cells for p in right)
    broad=proposed([10.,10.,10.,10.])
    assert broad and broad!=left


def test_mixture_batched_projection_matches_component_propagation_without_readings():
    c=config(True,history_count=128,risk_step_seconds=.13)
    b=PersistentWindBelief(c.model,c.session.motion)
    for i,state in enumerate(b.components): state.mean[:]=np.arange(4)*(.1+i*.01)
    result=score_pattern(b,c.session,c.wind,(),1.3,seed=5)
    rng=np.random.default_rng(5); generating=np.searchsorted(np.cumsum(b.weights),rng.random(128))
    W=projection_gram((2,2),(8,8)); expected=0.
    times=sorted(set([0.,1.3]+list(np.arange(0,1.3,.13))))
    for lo,hi in zip(times[:-1],times[1:]):
        states=[]
        for state,offset in zip(b.components,b.offsets):
            state=state.clone();state.advance(lo,replace(c.wind,velocity_yx=tuple(offset)))
            states.append(state)
        mean=sum(w*s.mean for w,s in zip(b.weights,states))
        within=sum(w*np.sum(W*s.covariance) for w,s in zip(b.weights,states))
        between=sum(w*((s.mean-mean)@W@(s.mean-mean)) for w,s in zip(b.weights,states))
        expected+=(hi-lo)*(within+between)
    assert result['risk']==pytest.approx(expected,abs=1e-12)


def test_gaussian_risk_agrees_with_model_consistent_simulated_histories():
    c=config(); b=GaussianSignalBelief(c.model); mask=np.zeros((4,4),bool);mask[:2,0]=True
    plan=compile_for(c.session,mask,'simulation-check',0.,(0.,.125))
    score=score_pattern(b,c.session,c.wind,(ForecastSweep(plan,plan.completed_at_seconds),),4.)
    rng=np.random.default_rng(41); count=20000
    truth=rng.multivariate_normal(b.mean,b.covariance,size=count)
    means=np.repeat(b.mean[None],count,axis=0); P=b.covariance.copy(); W=projection_gram((2,2),(8,8))
    loss=np.zeros(count);at=0.
    for sample in (*plan.samples,None):
        delivered=4. if sample is None else sample.acquired_at_seconds+sample.integration_seconds/2
        error=truth-means; loss+=(delivered-at)*np.einsum('bi,ij,bj->b',error,W,error);at=delivered
        if sample is None:break
        h=observation_weights((2,2),sample.position_yx,sample.footprint)
        cross=P@h; variance=h@cross+c.model.sensor_noise_std**2
        y=truth@h+rng.normal(0,c.model.sensor_noise_std,count)
        means+=(y-means@h)[:,None]*(cross/variance)
        P-=np.outer(cross,cross)/variance
    assert abs(loss.mean()-score['risk'])<4*loss.std()/np.sqrt(count)


def test_pure_decision_cache_reuses_only_identical_evidence_and_lifecycle(monkeypatch):
    import plume_focus.sensing_policy_v4 as policy
    c=config(); b=GaussianSignalBelief(c.model); ages=np.full((4,4),-1.)
    policy._decision_cache.clear(); calls=[]; original=policy.score_pattern
    def counted(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(policy,'score_pattern',counted)
    kwargs=dict(scan_id='cache-check',now=0.,pose=[0.,0.],rotation=0,
                startup_complete=True,last_survey_completed=0.)
    first=policy.plan_scan_v4(b,c.session,ScanRequestV1(8.),c.wind,ages,**kwargs)
    count=len(calls)
    assert first==policy.plan_scan_v4(b,c.session,ScanRequestV1(8.),c.wind,ages,**kwargs)
    assert len(calls)==count
    b.mean[0]+=1.
    policy.plan_scan_v4(b,c.session,ScanRequestV1(8.),c.wind,ages,**kwargs)
    assert len(calls)>count

    count=len(calls);ages[0,0]=0.
    policy.plan_scan_v4(b,c.session,ScanRequestV1(8.),c.wind,ages,**kwargs)
    assert len(calls)>count


def test_v4_playback_settings_leave_execution_unchanged(tmp_path):
    import cv2
    from plume_focus.sensing_rendering import RenderSettings, render_sensing_video
    c=config();run=run_sensing_v1(c);digest=run.operational_hash
    records=[]
    for mode,fps,speed in (('physical',2.,4.),('physical',4.,8.),('summary',3.,1.)):
        path=tmp_path/f'{mode}-{fps}.mp4'
        result=render_sensing_video(run,path,source=c.source.build(),
            settings=RenderSettings(mode=mode,fps=fps,playback_speed=speed,
                                    panel_shape=(160,240) if fps==4. else (200,320)))
        capture=cv2.VideoCapture(str(path));count=0
        while capture.read()[0]:count+=1
        capture.release()
        assert count==result['encoded_frames']
        assert result['operational_hash']==digest==run.operational_hash
        records.append(result)
    assert records[0]['encoded_frames']==records[1]['encoded_frames']
    assert records[2]['encoded_frames']==len(run.scans)


@pytest.mark.parametrize('mixture',[False,True])
def test_existing_serialized_fake_scanner_example_matches_v4_runner(mixture):
    import runpy
    execute=runpy.run_path(str(Path(__file__).parents[1]/'examples/sensing_controller_v1.py'))['execute_example']
    c=config(mixture); run=run_sensing_v1(c); example=execute(c,count=2)
    for (plan,posterior),scan in zip(example,run.scans):
        assert plan==scan.plan
        np.testing.assert_array_equal(posterior.mean,scan.posterior.mean)
        np.testing.assert_array_equal(posterior.std,scan.posterior.std)


def test_noise_limits_and_redundancy_affect_covariance_risk():
    c=config();b=GaussianSignalBelief(c.model);mask=np.zeros((4,4),bool);mask[:,0]=True
    plan=compile_for(c.session,mask,'noise-check',0.,(0.,.125))
    pattern=(ForecastSweep(plan,plan.completed_at_seconds),)
    scores=[]
    for noise in (1e-7,.01,1e6):
        session=replace(c.session,model=replace(c.model,sensor_noise_std=noise))
        scores.append(score_pattern(b,session,c.wind,pattern,8.)['risk'])
    assert scores[0]<scores[1]<scores[2]
    assert scores[2]==pytest.approx(score_pattern(b,c.session,c.wind,(),8.)['risk'],abs=1e-10)


def test_unreachable_service_block_does_not_suppress_other_legal_measurements():
    from plume_focus.sensing_policy_v4 import plan_scan_v4
    # A deliberately taller block makes the distinction visible on a 4x4
    # arithmetic fixture. The study retains the frozen 0.2-FOV blocks.
    c=config(survey_height_fov=1.); b=GaussianSignalBelief(c.model);b.advance(31.,c.wind)
    ages=np.zeros((4,4));ages[2:, :2]=-1.
    plan=plan_scan_v4(b,c.session,ScanRequestV1(5.),c.wind,ages,
        scan_id='service-unreachable',now=31.,pose=(.5,.125),rotation=0,
        startup_complete=True,last_survey_completed=0.)
    assert plan.service_status=='unmet_no_reachable_block'
    assert plan.status=='value_selected' and len(plan.samples)>=4
    assert plan.duration_seconds<=5. and not plan.service_required_cells


def test_dynamic_gaussian_risk_matches_independent_joint_brownian_field():
    c=config(risk_step_seconds=.1)
    c=replace(c,session=replace(c.session,model=replace(c.model,process_std_per_sqrt_second=.1)))
    b=GaussianSignalBelief(c.model);mask=np.zeros((4,4),bool);mask[:2,0]=True
    plan=compile_for(c.session,mask,'joint-time',0.,(0.,.125));sweeps=(ForecastSweep(plan,plan.completed_at_seconds),)
    score=score_pattern(b,c.session,c.wind,sweeps,4.)
    Q=c.model.process_std_per_sqrt_second**2*b.kernel;P=b.covariance
    H=np.stack([observation_weights(c.model.grid_shape,s.position_yx,s.footprint) for s in plan.samples])
    acquired=np.array([s.acquired_at_seconds for s in plan.samples])
    delivered=acquired+np.array([s.integration_seconds/2 for s in plan.samples])
    times=sorted(set([0.,4.,plan.completed_at_seconds,*delivered,*np.arange(0,4.,.1)]))
    W=projection_gram((2,2),(8,8));expected=0.
    for t,end in zip(times[:-1],times[1:]):
        selected=np.flatnonzero(delivered<=t+1e-10);post=P+Q*t
        if len(selected):
            V=np.array([[H[i]@(P+Q*min(acquired[i],acquired[j]))@H[j]
                         +(c.model.sensor_noise_std**2 if i==j else 0.) for j in selected] for i in selected])
            cross=np.stack([(P+Q*acquired[i])@H[i] for i in selected],axis=1)
            post=post-cross@np.linalg.solve(V,cross.T)
        expected+=(end-t)*np.sum(W*post)
    assert score['risk']==pytest.approx(expected,abs=1e-12)


def test_v4_finite_source_exhaustion_preserves_partial_startup_and_unavailable_loss():
    c=config();c=replace(c,session=replace(c.session,startup_mode='full_scene'),
                        source=replace(c.source,end_time_seconds=2.2))
    run=run_sensing_v1(c)
    assert run.termination_reason=='source_exhausted' and not run.startup_complete
    assert 0<run.visited_fraction<1 and run.scans[-1].execution.exhausted
    loss=reconstruction_loss(run,c.source.build())
    assert loss['status']=='incomplete_truth_horizon' and loss['J'] is None


def test_persistent_mixture_histories_match_independent_joint_integration():
    c=config(True,history_count=128); b=PersistentWindBelief(c.model,c.session.motion)
    b.components=(b.components[0].clone(),b.components[0].clone())
    b.offsets=np.zeros((2,2));b.log_weights=np.log([.5,.5])
    P=.005*np.eye(4)+.005*np.ones((4,4));mus=np.repeat(np.array([-.12,.12])[:,None],4,axis=1)
    for state,mu in zip(b.components,mus):state.mean=mu.copy();state.covariance=P.copy()
    mask=np.zeros((4,4),bool);mask[:2,0]=True
    plan=compile_for(c.session,mask,'joint-mixture',0.,(0.,.125))
    score=score_pattern(b,c.session,c.wind,(ForecastSweep(plan,plan.completed_at_seconds),),4.,seed=13)
    H=np.stack([observation_weights(c.model.grid_shape,s.position_yx,s.footprint) for s in plan.samples])
    W=projection_gram((2,2),(8,8));nodes,weights=np.polynomial.hermite.hermgauss(32)
    def expected_posterior(n):
        h=H[:n];V=h@P@h.T+c.model.sensor_noise_std**2*np.eye(n)
        inverse=np.linalg.inv(V);gain=P@h.T@inverse;within=np.sum(W*(P-gain@h@P))
        z=np.stack(np.meshgrid(*([nodes]*n),indexing='ij'),axis=-1).reshape(-1,n)*np.sqrt(2)
        q=np.prod(np.stack(np.meshgrid(*([weights/np.sqrt(np.pi)]*n),indexing='ij'),axis=-1),axis=-1).ravel()
        centers=mus@h.T;result=0.
        for generating in range(2):
            y=centers[generating]+z@np.linalg.cholesky(V).T
            residual=y[None]-centers[:,None]
            logs=-.5*np.einsum('kbi,ij,kbj->kb',residual,inverse,residual)
            posterior_weights=np.exp(logs-logsumexp(logs,axis=0))
            means=mus[:,None]+residual@gain.T
            average=np.sum(posterior_weights[...,None]*means,axis=0)
            delta=means-average[None]
            between=np.sum(posterior_weights*np.einsum('kbi,ij,kbj->kb',delta,W,delta),axis=0)
            result+=.5*np.dot(q,within+between)
        return result
    deliveries=[s.acquired_at_seconds+s.integration_seconds/2 for s in plan.samples]
    expected=(deliveries[0]*np.sum(W*b.covariance)+(deliveries[1]-deliveries[0])*expected_posterior(1)
              +(4.-deliveries[1])*expected_posterior(2))
    assert abs(score['risk']-expected)<max(4*score['standard_error'],1e-6)
