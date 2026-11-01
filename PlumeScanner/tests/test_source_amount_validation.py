import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from plume_focus.source_amount_validation import bounded_posterior, ClockDynamics, ClockAffineFilter, OnlineAmountBelief
from plume_focus.signal_belief import SignalModelConfig, grid_positions, observation_weights
from plume_focus.source_input import gaussian_source_input, CELL_INTEGRATED_V1
from plume_focus.sensing_contracts import CausalWind, SensorReading, Footprint


@pytest.mark.parametrize('L,K',[(np.array([0.,0.]),np.array([0.,0.])),
    (np.array([300.,-400.,50.]),np.array([100.,100.,100.])),
    (np.array([20000.,-20000.,0.]),np.array([10000.,10000.,10000.]))])
def test_bounded_amount_integration_matches_independent_quad(L,K):
    p=bounded_posterior(L,K,order=128)
    lo,hi=.25,1.75
    optimum=np.clip(np.divide(L,2*K,out=np.full_like(L,lo),where=K>0),lo,hi)
    scale=max(optimum*L-optimum**2*K)
    z=[];m=[];s=[]
    for a,b in zip(L,K):
        f=lambda x:np.exp(a*x-b*x*x-scale)/(hi-lo)
        z.append(quad(f,lo,hi,epsabs=1e-13,points=[float(np.clip(a/(2*b),lo,hi))] if b else None)[0])
        m.append(quad(lambda x:x*f(x),lo,hi,epsabs=1e-13)[0])
        s.append(quad(lambda x:x*x*f(x),lo,hi,epsabs=1e-13)[0])
    assert p['w'].sum()==pytest.approx(1.)
    assert p['w']@p['a1']==pytest.approx(sum(m)/sum(z),abs=2e-11)
    assert p['w']@p['a2']==pytest.approx(sum(s)/sum(z),abs=2e-11)
    assert p['omitted_bound']<1e-10


def test_fixed_clock_affine_filter_matches_direct_candidates_and_clone():
    model=SignalModelConfig(grid_shape=(3,4));d=ClockDynamics(model,(0.,.03))
    sources=np.array([[.3,.2],[.65,.7]])
    E=gaussian_source_input(model.grid_shape,sources,source_sigma=.08,total_rate=.002,formulation_id=CELL_INTEGRATED_V1)
    f=ClockAffineFilter(d,E);direct=np.zeros((2,12));P=d.P0.copy()
    for target,value in [(4,.1),(9,-.05),(15,.3)]:
        steps=target-f.tick;direct=d.means(direct,E,steps)
        for _ in range(steps): P=(d.A@(d.A@P).T).T+d.Q
        f.advance(target);h=observation_weights(model.grid_shape,(.4,.35),Footprint(.1,.1))
        cross=P@h;v=h@cross+model.sensor_noise_std**2
        direct+=(value-direct@h)[:,None]*(cross/v)[None]
        P-=np.outer(cross,cross)/v
        f.condition([value],h)
        np.testing.assert_allclose(f.m[0]+E@f.B.T,direct,atol=1e-13)
        np.testing.assert_allclose(f.P,P,atol=1e-13)
    belief=OnlineAmountBelief(d,sources,E,fixed=1.);belief.filter=f
    clone=belief.clone();old=f.P.copy();clone.advance(.25,CausalWind(velocity_yx=(0.,.03)))
    np.testing.assert_array_equal(f.P,old)
    assert belief.time_seconds==pytest.approx(15*.0125)
    distribution=belief.distribution_on((3,4))
    np.testing.assert_allclose(distribution.mean,belief.summary()[0].reshape(3,4),atol=1e-12)
    np.testing.assert_allclose(distribution.variance,belief.moments_on((3,4))[1],atol=1e-12)
    assert belief.view().std is None  # Cheap event view is explicitly mean-only.


def test_clock_rejects_off_grid_or_backwards():
    d=ClockDynamics(SignalModelConfig(grid_shape=(3,4)),(0.,0.))
    with pytest.raises(ValueError):d.tick_of(.013)
    f=ClockAffineFilter(d,np.zeros((2,12)));f.advance(3)
    with pytest.raises(ValueError):f.advance(2)


def test_joint_projection_cdf_agrees_with_independent_amount_integral():
    from scipy.special import ndtr
    model=SignalModelConfig(grid_shape=(3,4));d=ClockDynamics(model,(0.,.03))
    sources=np.array([[.3,.2],[.65,.7]])
    E=gaussian_source_input(model.grid_shape,sources,source_sigma=.08,total_rate=.02,formulation_id=CELL_INTEGRATED_V1)
    b=OnlineAmountBelief(d,sources,E,amount_order=128)
    f=b.filter;f.advance(32)
    h=observation_weights(model.grid_shape,(.4,.35),Footprint(.1,.1));f.condition([.1],h)
    distribution=b.distribution_on((3,4));target=.04;cell=5
    g=E@f.B[cell];base=f.m[0,cell];std=np.sqrt(f.P[cell,cell])
    scale=max(np.clip(f.L[0]/(2*f.K),.25,1.75)*f.L[0]-np.clip(f.L[0]/(2*f.K),.25,1.75)**2*f.K)
    numerator=0.;denominator=0.
    for i in range(2):
        likelihood=lambda a:np.exp(a*f.L[0,i]-a*a*f.K[i]-scale)
        denominator+=quad(likelihood,.25,1.75,epsabs=1e-12)[0]
        numerator+=quad(lambda a:likelihood(a)*ndtr((target-base-a*g[i])/std),.25,1.75,epsabs=1e-12)[0]
    assert distribution.cdf(target).ravel()[cell]==pytest.approx(numerator/denominator,abs=1e-10)
    evaluation=f.validation(0,b.posterior(),np.eye(12)[[cell]],np.array([target]))
    assert evaluation['pit'][0]==pytest.approx(numerator/denominator,abs=1e-10)


def test_weak_likelihood_shortcut_preserves_full_order_moments():
    rng=np.random.default_rng(20261101)
    K=np.exp(rng.uniform(-18,8,2000));L=rng.normal(size=2000)*np.sqrt(K)*3
    full=bounded_posterior(L,K,order=128)
    fast=bounded_posterior(L,K,order=128,components=False)
    np.testing.assert_array_equal(full['ids'],fast['ids'])
    np.testing.assert_allclose(full['w'],fast['w'],atol=1e-14,rtol=1e-11)
    np.testing.assert_allclose(full['a1'],fast['a1'],atol=1e-12)
    np.testing.assert_allclose(full['a2'],fast['a2'],atol=1e-12)
    assert fast['nodes'] is None


def test_batched_posterior_inputs_match_scalar_dense_and_sparse_queries():
    from plume_focus.source_amount_validation import posterior_inputs
    rng=np.random.default_rng(3);E=rng.normal(size=(100,12))
    for n in [4,90]:
        ps=[]
        for _ in range(6):
            ids=rng.choice(100,n,replace=False);w=rng.random(n);w/=w.sum()
            ps.append({'ids':ids,'w':w,'a1':rng.random(n)})
        expected=np.stack([(p['w']*p['a1'])@E[p['ids']] for p in ps])
        np.testing.assert_allclose(posterior_inputs(E,ps),expected,atol=1e-14)


def test_existing_policy_can_focus_with_joint_adapter_and_reject_future_evidence():
    from pathlib import Path
    from dataclasses import replace
    import json
    from plume_focus.full_fov_execution import decode_config
    from plume_focus.evidence_gated_loop import EvidenceGatedScanner
    from plume_focus.sensing_messages_v2 import ScanRequestV1
    from plume_focus.sensing_contracts import ExecutionReceipt
    config=json.loads((Path(__file__).parents[1]/'configs/evidence-gated-focus/coarse-demo.json').read_text())
    run=decode_config(config);model=replace(run.model,grid_shape=(3,4))
    controller=EvidenceGatedScanner(replace(run.session,model=model,source_position=None))
    d=ClockDynamics(model,run.wind.velocity_yx)
    sources=np.array([[.3,.2],[.65,.7]])
    E=gaussian_source_input(model.grid_shape,sources,source_sigma=.035,total_rate=.002,formulation_id=CELL_INTEGRATED_V1)
    controller._belief=OnlineAmountBelief(d,sources,E,amount_order=32)
    first=controller.plan_next(0.,(0.,0.),ScanRequestV1(60.),wind=run.wind)
    for i,s in enumerate(first.samples):
        r=SensorReading(first.scan_id,f'{first.scan_id}:{i}',s.position_yx,s.acquired_at_seconds,
            s.acquired_at_seconds+s.integration_seconds/2,.5,s.footprint,s.integration_seconds)
        with pytest.raises(ValueError):controller.ingest_readings([r],available_at_seconds=r.available_at_seconds-.001)
        controller.ingest_readings([r],available_at_seconds=r.available_at_seconds)
    receipt=ExecutionReceipt(first.scan_id,first.completed_at_seconds,first.completed_at_seconds,first.final_position_yx)
    controller.finish_scan(receipt)
    assert controller.startup_complete and controller.gate_focus_allowed
    second=controller.plan_next(receipt.received_at_seconds,receipt.final_position_yx,
                               ScanRequestV1(60.-receipt.received_at_seconds),wind=run.wind)
    assert second.mode=='FOCUS' and second.duration_seconds<=12.+1e-10
    assert second.instrument_hash==first.instrument_hash


def test_parallel_histories_preserve_identical_posteriors():
    from concurrent.futures import ThreadPoolExecutor
    rng=np.random.default_rng(44);K=rng.random(500)*10;L=rng.normal(size=(8,500))*4
    scalar=[bounded_posterior(x,K,components=False) for x in L]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel=list(pool.map(lambda x:bounded_posterior(x,K,components=False),L))
    for a,b in zip(scalar,parallel):
        for key in ['ids','w','a1','a2']:np.testing.assert_array_equal(a[key],b[key])


def test_emission_cache_is_shared_by_forecasts_and_invalidated_by_a_reading():
    model=SignalModelConfig(grid_shape=(3,4));wind=CausalWind(velocity_yx=(0.,.03))
    d=ClockDynamics(model,wind.velocity_yx);sources=np.array([[.3,.2],[.65,.7]])
    E=gaussian_source_input(model.grid_shape,sources,source_sigma=.08,total_rate=.02,formulation_id=CELL_INTEGRATED_V1)
    b=OnlineAmountBelief(d,sources,E,amount_order=64)
    _,emission=b.summary();forecast=b.clone();forecast.advance(.25,wind)
    assert forecast.summary()[1] is emission
    assert b.time_seconds==0.
    r=SensorReading('s','r',(.3,.2),.5,.5125,.8,Footprint(.1,.1),.025)
    b.assimilate(r,wind)
    mean,updated=b.summary()
    assert updated is not emission and np.max(np.abs(updated-emission))>1e-6
    direct=b.filter.summary(0,b.filter.posterior(0,None,64))
    np.testing.assert_allclose(mean,direct[0],atol=1e-14)
    np.testing.assert_allclose(updated,direct[1],atol=1e-14)


def test_analytic_amount_moments_match_full_quadrature_including_tails():
    rng=np.random.default_rng(20261102);used=0
    for _ in range(250):
        K=np.array([np.exp(rng.uniform(-8,10))]);L=2*K*rng.uniform(-5,5)
        exact=bounded_posterior(L,K,order=256)
        fast=bounded_posterior(L,K,order=128,components=False,analytic_moments=True)
        np.testing.assert_allclose(fast['a1'],exact['a1'],atol=2e-10,rtol=1e-10)
        np.testing.assert_allclose(fast['a2'],exact['a2'],atol=2e-10,rtol=1e-10)
        assert fast['log_evidence']==pytest.approx(exact['log_evidence'],abs=1e-8)
        used+=fast['analytic_count']
    assert used>25
    K=rng.uniform(1,100,2000);L=2*K*rng.uniform(-2,3,2000)
    exact=bounded_posterior(L,K,order=256)
    fast=bounded_posterior(L,K,order=128,components=False,analytic_moments=True)
    np.testing.assert_array_equal(fast['ids'],exact['ids'])
    np.testing.assert_allclose(fast['w'],exact['w'],atol=1e-12,rtol=1e-9)


def test_streamed_mixture_matches_dense_cdf_moments_and_quantiles():
    from plume_focus.source_amount_validation import StreamingMarginals
    from plume_focus.signal_distribution import PredictiveDistribution
    rng=np.random.default_rng(8);base=rng.normal(size=12);g=rng.normal(size=(11,12))
    ids=rng.integers(0,11,500);a=rng.uniform(.25,1.75,500);w=rng.random(500);w/=w.sum()
    variance=rng.uniform(.01,.05,12)
    dense=PredictiveDistribution(1.,w,(base+a[:,None]*g[ids]).reshape(500,3,4),np.broadcast_to(variance,(500,12)).reshape(500,3,4))
    streamed=StreamingMarginals(1.,base,g,ids,a,w,variance,(3,4))
    np.testing.assert_allclose(streamed.mean,dense.mean,atol=1e-13)
    np.testing.assert_allclose(streamed.variance,dense.variance,atol=1e-13)
    np.testing.assert_allclose(streamed.cdf(.15),dense.cdf(.15),atol=1e-13)
    np.testing.assert_allclose(streamed.quantile(.95),dense.quantile(.95),atol=2e-10)


def test_large_adapter_projection_streams_and_preserves_gaussian_startup():
    from plume_focus.source_amount_validation import StreamingMarginals
    from scipy.special import ndtri
    model=SignalModelConfig(grid_shape=(3,4));wind=CausalWind(velocity_yx=(0.,.03))
    d=ClockDynamics(model,wind.velocity_yx);sources=grid_positions((11,11))
    E=gaussian_source_input(model.grid_shape,sources,source_sigma=.035,total_rate=.002,formulation_id=CELL_INTEGRATED_V1)
    b=OnlineAmountBelief(d,sources,E,amount_order=128)
    distribution=b.distribution_on((3,4))
    assert isinstance(distribution,StreamingMarginals)
    np.testing.assert_allclose(distribution.quantile(.95),ndtri(.95)*np.sqrt(np.diag(d.P0)).reshape(3,4),atol=1e-11)
    b.advance(.5,wind);distribution=b.distribution_on((3,4))
    mean,variance=b.moments_on((3,4))
    np.testing.assert_allclose(distribution.mean,mean,atol=1e-12)
    np.testing.assert_allclose(distribution.variance,variance,atol=1e-12)


def test_negligible_amount_precision_avoids_unused_reciprocal_overflow():
    with np.errstate(over='raise',divide='raise',invalid='raise'):
        p=bounded_posterior(np.array([1e-160,0.]),np.array([1e-320,0.]),components=False,analytic_moments=True)
    np.testing.assert_allclose(p['a1'],1.,atol=1e-14)
    assert p['analytic_count']==0
