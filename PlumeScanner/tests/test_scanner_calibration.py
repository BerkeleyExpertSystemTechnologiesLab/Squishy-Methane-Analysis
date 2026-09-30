from dataclasses import replace, fields
import json
import math
import numpy as np
import pytest
from plume_focus.calibrated_scanner import (CalibrationSession,CalibrationSettings,CalibrationPlan,
                                           CalibratedScanner,DecayGaussianBelief,decay_factors)
from plume_focus.calibration_evaluation import AffineBelief, mean_forecast, fit_rate
from plume_focus.full_fov_loop import FullFovPlan
from plume_focus.full_fov_execution import decode_config,decode_plan,run_full_fov,verify_replay
from plume_focus.calibration_execution import CalibrationRunConfig
from plume_focus.signal_belief import GaussianSignalBelief,SignalModelConfig
from plume_focus.sensing_contracts import CausalWind, Footprint, SensorReading
from plume_focus.signal_belief import observation_weights
from test_full_fov_loop import controller,plan,execute,field


def calibrated(old,margin=0,half=None):
    s=old.session
    return CalibratedScanner(CalibrationSession(s.instrument,s.model,CalibrationSettings(vertical_margin_fov=margin),
                                               s.source_position,half_life_seconds=half))


def test_zero_modes_preserve_paths_and_posteriors():
    old=controller();new=calibrated(old)
    for value in (.6,.2,0):
        a,b=plan(old),plan(new)
        assert all(getattr(a,f.name)==getattr(b,f.name) for f in fields(FullFovPlan))
        execute(old,a,value);execute(new,b,value)
        np.testing.assert_array_equal(old._belief.mean,new._belief.mean)
        np.testing.assert_array_equal(old._belief.covariance,new._belief.covariance)
    assert decode_plan(json.loads(json.dumps(b.to_dict())))==b
    assert CalibrationSession.from_dict(json.loads(json.dumps(new.session.to_dict())))==new.session


def test_margin_rounds_outwards_clips_and_never_expands_horizontally():
    c=calibrated(controller(),margin=.1)
    mask=np.zeros((20,32),bool);mask[0,3]=True;mask[10,7]=True
    result=c._focus_mask(mask)
    assert np.flatnonzero(result[:,3]).tolist()==[0,1,2]
    assert np.flatnonzero(result[:,7]).tolist()==[8,9,10,11,12]
    assert np.flatnonzero(result.any(axis=0)).tolist()==[3,7]


def test_margin_is_in_compiled_path_and_charged():
    old=controller();c=calibrated(old,margin=.25)
    p=plan(c);execute(c,p,.6)
    p=plan(c)
    assert p.mode=='FOCUS'
    assert set(p.selected_cells)|set(p.padding_cells)==set(p.requested_cells)
    assert set(p.requested_cells)<=set(p.swept_cells)
    assert len(p.samples)>=len(p.requested_cells)
    assert p.duration_seconds<=12+1e-9
    with pytest.raises(RuntimeError):plan(c)


@pytest.mark.parametrize('half',[None,3.,6.,12.])
def test_decay_matches_independent_stationary_solution_and_query(half):
    m=SignalModelConfig(grid_shape=(2,3),diffusivity=0,initial_mean=.7,max_transition_seconds=.5)
    b=DecayGaussianBelief(m,half_life_seconds=half,transport_id='zero_inflow_v1')
    initial=b.covariance.copy();source=(.5,.5);wind=CausalWind();t=2.
    emission=m.source_rate*np.exp(-np.sum((b.positions-source)**2,axis=1)/(2*m.source_sigma**2))
    rate=0 if half is None else math.log(2)/half
    a=math.exp(-rate*t);s=t if not rate else (1-a)/rate;q=t if not rate else (1-a*a)/(2*rate)
    prediction=mean_forecast(b,b.mean,t,wind,source)
    b.advance(t,wind,source)
    np.testing.assert_allclose(b.mean,.7*a+emission*s,atol=1e-13)
    np.testing.assert_allclose(b.covariance,initial*a*a+m.process_std_per_sqrt_second**2*q*b.kernel,atol=1e-13)
    np.testing.assert_allclose(prediction,b.mean,atol=1e-13)
    assert b.clone().half_life_seconds==half


def test_decay_query_matches_advected_runtime_without_mutation():
    b=DecayGaussianBelief(SignalModelConfig(grid_shape=(3,4)),half_life_seconds=3.,transport_id='zero_inflow_v1')
    b.mean[3]=1;wind=CausalWind(velocity_yx=(.04,-.02));before=b.mean.copy()
    for time in (.025,.1,.9,3.3):
        expected=b.clone();expected.advance(time,wind,(.4,.2))
        np.testing.assert_allclose(mean_forecast(b,b.mean,time,wind,(.4,.2)),expected.mean,atol=1e-12)
    np.testing.assert_array_equal(before,b.mean)


def test_zero_decay_vector_query_matches_composed_transition_at_long_horizons():
    from plume_focus.sensing_risk import linear_transition
    model=SignalModelConfig(grid_shape=(3,4),source_rate=.15)
    b=DecayGaussianBelief(model,transport_id='zero_inflow_v1')
    b.mean=np.linspace(-.2,.7,12);wind=CausalWind(velocity_yx=(.04,-.02));source=(.4,.2)
    for time in (.0125,.15,1.,12.,60.,100.):
        a,s,_=linear_transition(model,wind.velocity_yx,source,time,b.transport_id)
        np.testing.assert_allclose(mean_forecast(b,b.mean,time,wind,source),a@b.mean+s,rtol=1e-12,atol=1e-12)


def test_shared_covariance_affine_replay_equals_independent_rates():
    model=SignalModelConfig(grid_shape=(3,4),source_rate=.3)
    affine=AffineBelief(model,half_life_seconds=6.,transport_id='zero_inflow_v1')
    rates=[0.,.07,.3]
    references=[DecayGaussianBelief(replace(model,source_rate=r),half_life_seconds=6.,transport_id='zero_inflow_v1') for r in rates]
    wind=CausalWind(velocity_yx=(.01,.03));source=(.5,.3)
    for t,value in [(1.,.2),(2.7,0.),(3.,.7)]:
        h=observation_weights(model.grid_shape,(.5,.4),Footprint(.1,.1))
        affine.advance(t,wind,source);affine.condition(value,h,.0025)
        for rate,b in zip(rates,references):
            b.advance(t,wind,source);b.condition(value,h,.0025)
            expected=affine.mean_zero+(rate/.3)*(affine.mean-affine.mean_zero)
            np.testing.assert_allclose(b.mean,expected,atol=1e-12)
            np.testing.assert_allclose(b.covariance,affine.covariance,atol=1e-12)
    assert fit_rate([4,-1,5])['source_rate']==.075
    assert fit_rate([1,1,5])['source_rate']==0
    assert fit_rate([1,-2,5])['source_rate']==.3


def test_new_binding_delayed_batch_duplicate_and_overrun():
    c=calibrated(controller(),half=3);p=plan(c)
    ex=execute(c,p,.3,delivery='batch',delay=1,overrun=2)
    assert c._belief.time_seconds==ex.receipt.received_at_seconds
    mean=c._belief.mean.copy();c.ingest_readings(ex.readings,available_at_seconds=ex.receipt.received_at_seconds)
    np.testing.assert_array_equal(c._belief.mean,mean)
    with pytest.raises(ValueError):c.predictive_view(ex.receipt.received_at_seconds-.1)
    p=plan(c)
    with pytest.raises(RuntimeError):plan(c)


def test_calibration_run_roundtrip_and_causal_replay():
    from test_reconstruction_risk import config
    original=config()
    session=CalibrationSession(original.session.instrument,original.model,
        CalibrationSettings(vertical_margin_fov=.05),original.session.source_position,half_life_seconds=6.)
    run=CalibrationRunConfig(session,original.source,original.wind,duration_seconds=4.,
                             evaluation_shape=(8,8),evaluation_step_seconds=.1,active_windows=((0.,2.),))
    assert decode_config(json.loads(json.dumps(run.to_dict())))==run
    result=run_full_fov(run)
    from plume_focus.sensing_counterfactual import TruthQuadrature
    truth=TruthQuadrature(run.build_source(),0,4,run.model.grid_shape,(8,8),.1)
    verify_replay(result,truth=truth)


def test_regular_truth_grid_matches_sensor_source():
    import importlib.util
    from pathlib import Path
    from dataclasses import replace
    from plume_focus.signal_belief import grid_positions
    spec=importlib.util.spec_from_file_location('truth_grid',Path(__file__).parents[1]/'scripts/calibration_truth_grid.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    from test_reconstruction_risk import config
    original=config()
    session=CalibrationSession(original.session.instrument,original.model,
        CalibrationSettings(),original.session.source_position)
    run=CalibrationRunConfig(session,original.source,original.wind,duration_seconds=360.)
    for settings in ({},{'cessation_seconds':180.},{'active_windows':((0.,120.),(240.,360.))}):
        candidate=replace(run,**settings);source=candidate.build_source()
        for t in (0.,.25,33.575,120.,179.999,180.,240.,359.95):
            expected=source.signal_at(np.full(13*17,t),grid_positions((13,17))).reshape(13,17)
            np.testing.assert_allclose(helper.clean_grid(candidate,t,(13,17)),expected,rtol=1e-12,atol=1e-12)


def test_affine_causal_loss_matches_independent_fixed_evidence_replays():
    from test_reconstruction_risk import config
    from plume_focus.calibration_evaluation import AffineReplay,AffineLoss
    from plume_focus.focus_extent_diagnostics import EvidenceReplay
    from plume_focus.sensing_counterfactual import OnlineLoss,TruthQuadrature
    original=config()
    session=CalibrationSession(original.session.instrument,original.model,CalibrationSettings(),
                               original.session.source_position,half_life_seconds=6.)
    run=CalibrationRunConfig(session,original.source,original.wind,duration_seconds=4.,
                             evaluation_shape=(8,8),evaluation_step_seconds=.1)
    q=TruthQuadrature(run.build_source(),0,4,run.model.grid_shape,(8,8),.1)
    record=run_full_fov(run,truth=q)
    fitted=replace(session,model=replace(session.model,source_rate=.3))
    c=AffineReplay(fitted);loss=AffineLoss(q)
    for event in record['events']:
        loss.before(c,event['at_seconds']);c.event(event)
    loss.before(c,4.+1e-8);a,b,d=loss.coefficients
    for rate in (0.,.09,.3):
        s=replace(session,model=replace(session.model,source_rate=rate))
        c=EvidenceReplay(s);c._belief=DecayGaussianBelief(s.model,half_life_seconds=6.,transport_id=s.transport_id)
        check=OnlineLoss(q)
        for event in record['events']:
            check.before(c,event['at_seconds']);c.event(event)
        check.before(c,4.+1e-8);fraction=rate/.3
        np.testing.assert_allclose(check.result()['J'],a*fraction**2+2*b*fraction+d,rtol=1e-12,atol=1e-12)
