"""Independent analytic and refinement checks, not policy-performance evidence."""
from dataclasses import replace
import math
import numpy as np
import pytest
from scipy.integrate import quad

from plume_focus.sensor_quadrature import (SensorQuadratureV2, rectangle_gauss,
    exact_rectangle_observation_weights, exposure_average)
from plume_focus.sensing_contracts import Footprint, SampleTarget
from plume_focus.sensing_simulation import CleanMethaneSource, FakeScanner
from plume_focus.scheduled_puff_source import ScheduledPuffSource
from plume_focus.sensing_counterfactual import TruthQuadrature
from plume_focus.full_fov_execution import ControlledSignal
from plume_focus.evaluation_quadrature import time_quadrature
from plume_focus.signal_belief import grid_positions
from plume_focus.video_sources import SyntheticMethaneLeakConfig, generate_synthetic_methane_values
from test_full_fov_loop import controller, plan, ConstantSource


def test_narrow_gaussian_footprint_matches_closed_form():
    footprint = Footprint(.05, 1/32); center = (.5,.5)
    sy, sx = 1.8/219, 2.43/319
    exact = (math.sqrt(2*math.pi)*sy/footprint.height*math.erf(footprint.height/(2*math.sqrt(2)*sy))
             * math.sqrt(2*math.pi)*sx/footprint.width*math.erf(footprint.width/(2*math.sqrt(2)*sx)))
    estimates = []
    for order in (2,16,32):
        points, weights = rectangle_gauss(center, footprint, order)
        values = np.exp(-.5*((points[:,0]-.5)**2/sy**2+(points[:,1]-.5)**2/sx**2))
        estimates.append(float(weights@values))
    assert abs(estimates[1]-exact) < 1e-10
    assert abs(estimates[2]-estimates[1]) < 1e-10
    assert estimates[0] < .5*exact


@pytest.mark.parametrize("center", [(.44,.36), (.02,.98)])
def test_exact_latent_operator_matches_independent_piecewise_integration(center):
    shape=(3,4); footprint=Footprint(.7,.8)
    field=np.random.default_rng(17).normal(size=shape)
    ys=(np.arange(shape[0])+.5)/shape[0]; xs=(np.arange(shape[1])+.5)/shape[1]
    lo=np.maximum(0,np.array(center)-np.array((footprint.height,footprint.width))/2)
    hi=np.minimum(1,np.array(center)+np.array((footprint.height,footprint.width))/2)
    def at(y,x):
        return np.interp(y,ys,[np.interp(x,xs,row) for row in field])
    reference=quad(lambda y:quad(lambda x:at(y,x),lo[1],hi[1],points=xs[(xs>lo[1])&(xs<hi[1])],epsabs=1e-12)[0],
                   lo[0],hi[0],points=ys[(ys>lo[0])&(ys<hi[0])],epsabs=1e-12)[0]/np.prod(hi-lo)
    weights=exact_rectangle_observation_weights(shape,center,footprint)
    assert np.all(weights>=0) and weights.sum()==pytest.approx(1.)
    assert weights@field.ravel()==pytest.approx(reference,abs=1e-12)


def test_moving_exposure_matches_polynomial_integral():
    class Polynomial:
        def signal_at(self,times,positions_yx):
            return times**2+positions_yx[:,0]**2
    footprint=Footprint(.05,.03); sample=SampleTarget((.5,.5),1.,.1,footprint)
    value=exposure_average(Polynomial(),sample,lambda t:(.5+.2*(t-1),.5),SensorQuadratureV2())
    reference=1.+.1**2/12+.5**2+.2**2*.1**2/12+footprint.height**2/12
    assert value==pytest.approx(reference,abs=1e-12)


def test_exposure_splits_known_discontinuities():
    class Step:
        def temporal_breakpoints(self,start,end):return (1.,)
        def signal_at(self,times,positions_yx):return (times>=1.).astype(float)
    sample=SampleTarget((.5,.5),1.,.1,Footprint())
    profile=SensorQuadratureV2(spatial_order=1,temporal_order=1,max_panel_seconds=1.)
    assert exposure_average(Step(),sample,lambda t:(.5,.5),profile)==pytest.approx(.5,abs=1e-14)


def test_sensor_reference_preserves_receipts_and_constant_calibration():
    p=plan(controller())
    legacy=FakeScanner(ConstantSource(7.),noise_std=0,gain=2,offset=3).execute(p,delivery_delay=.2)
    refined=FakeScanner(ConstantSource(7.),noise_std=0,gain=2,offset=3,quadrature=SensorQuadratureV2()).execute(p,delivery_delay=.2)
    assert refined.receipt==legacy.receipt
    assert refined.executed_plan==legacy.executed_plan
    np.testing.assert_allclose([r.value for r in refined.readings],17.,atol=1e-12,rtol=0)
    assert [r.available_at_seconds for r in refined.readings]==[r.available_at_seconds for r in legacy.readings]


def test_shutoff_preserves_previously_emitted_puffs():
    cfg=SyntheticMethaneLeakConfig(); stop=1.
    scheduled=ScheduledPuffSource(config=cfg,emission_windows=((-cfg.memory_frames/cfg.fps,stop),))
    hard_clear=ControlledSignal(CleanMethaneSource(cfg),stop)
    points=grid_positions((30,40)); times=np.full(len(points),stop+.01)
    assert scheduled.signal_at(times,points).sum()>0.
    assert np.count_nonzero(hard_clear.signal_at(times,points))==0
    assert np.count_nonzero(scheduled.signal_at(np.full(len(points),stop+cfg.memory_frames/cfg.fps),points))==0


def test_half_open_puff_lifetime_removes_isolated_endpoint_spike():
    cfg=SyntheticMethaneLeakConfig(); points=grid_positions((30,40))
    yy,xx=points[:,0]*219,points[:,1]*319; t=np.full(len(points),120.)
    def value(times,mode):return generate_synthetic_methane_values(cfg,times,yy,xx,clean_signal=True,lifetime_convention=mode)
    assert np.max(np.abs(value(t,"closed_v1")-value(t+1e-7,"closed_v1")))>1e-4
    assert np.max(np.abs(value(t,"half_open_v2")-value(t+1e-7,"half_open_v2")))<1e-6


def test_phase_seed_controls_actual_source_configuration():
    cfg=SyntheticMethaneLeakConfig(); windows=((-6.,10.),)
    a=ScheduledPuffSource.from_phase_seed(cfg,emission_windows=windows,phase_seed=6101)
    b=ScheduledPuffSource.from_phase_seed(cfg,emission_windows=windows,phase_seed=6101)
    c=ScheduledPuffSource.from_phase_seed(cfg,emission_windows=windows,phase_seed=6102)
    assert a.configuration()==b.configuration() and a.configuration()!=c.configuration()
    points=grid_positions((20,24)); times=np.full(len(points),2.)
    np.testing.assert_array_equal(a.signal_at(times,points),b.signal_at(times,points))
    assert np.max(np.abs(a.signal_at(times,points)-c.signal_at(times,points)))>.001


def test_midpoint_truth_integral_converges_to_polynomial_reference():
    class Ramp:
        def signal_at(self,times,positions_yx):return times
    def integrate(step,rule):
        q=TruthQuadrature(Ramp(),0.,1.,(2,2),shape=(2,2),step=step,temporal_rule=rule)
        return np.dot(q.square,q.durations)
    old=integrate(.25,"left_endpoint_v1"); coarse=integrate(.25,"event_split_midpoint_v2")
    fine=integrate(1/64,"event_split_midpoint_v2")
    assert abs(fine-1/3)<3e-5
    assert abs(fine-1/3)<abs(coarse-1/3)<abs(old-1/3)


def test_time_quadrature_splits_events_and_covers_partial_terminal_cell():
    times,weights=time_quadrature(0.,1.,.4,rule="event_split_midpoint_v2",breakpoints=(.3,))
    assert weights.sum()==pytest.approx(1.) and np.all(weights>0)
    assert float(weights@(times>=.3))==pytest.approx(.7)
    expected=np.arange(4)*.3
    actual,durations=time_quadrature(0.,1.,.3)
    np.testing.assert_array_equal(actual,expected)
    assert durations[-1]==pytest.approx(.1)


@pytest.mark.parametrize("kwargs", [{"spatial_order": True},{"temporal_order":0},{"max_panel_seconds":np.nan}])
def test_invalid_numerical_profiles_rejected(kwargs):
    with pytest.raises(ValueError):SensorQuadratureV2(**kwargs)


def test_numerical_profile_transport_round_trip():
    profile=SensorQuadratureV2()
    assert SensorQuadratureV2.from_dict(profile.to_dict())==profile
