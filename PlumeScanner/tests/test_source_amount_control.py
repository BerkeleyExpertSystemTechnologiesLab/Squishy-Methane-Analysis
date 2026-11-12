import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp, ndtr
from plume_focus.source_amount_control import (InputMatchedGenerator,SourceAmountBank,
    amount_quadrature,source_amount_posterior,forecast_footprint_cdf)
from plume_focus.source_input import CELL_INTEGRATED_V1
from plume_focus.signal_belief import SignalModelConfig
from plume_focus.sensing_contracts import CausalWind
from plume_focus.source_likelihood_quadrature import AffineSourceLikelihood


def test_joint_quadrature_matches_independent_continuous_integrals():
    nodes,weights=amount_quadrature(65,.25,1.75)
    linear=np.array([1.,4.,9.]);quadratic=np.array([.5,2.,5.])
    p=source_amount_posterior(linear,quadratic,nodes,weights,amount_support=(.25,1.75))
    raw=np.array([[quad(lambda a:a**moment*np.exp(a*l-a*a*k),.25,1.75,
                         epsabs=1e-11)[0]/1.5 for l,k in zip(linear,quadratic)] for moment in range(3)])
    z=raw[0].sum()
    for moment in range(3):np.testing.assert_allclose(p[f'w{moment}'],raw[moment]/z,rtol=1e-12)
    np.testing.assert_allclose(p['amount_mean'],raw[1].sum()/z,rtol=1e-12)


def test_flat_likelihood_recovers_uniform_prior_and_exact_forecast_mixture_cdf():
    nodes,weights=amount_quadrature(65,.25,1.75);L=np.zeros(3);K=np.zeros(3)
    p=source_amount_posterior(L,K,nodes,weights)
    np.testing.assert_allclose(p['w0'],np.ones(3)/3,atol=1e-15)
    np.testing.assert_allclose(p['amount_mean'],1.,atol=1e-15)
    np.testing.assert_allclose(p['amount_variance'],1.5**2/12,atol=1e-14)
    g=np.array([0.,.1,.5]);base=.2;std=.3;truth=.7
    actual=forecast_footprint_cdf(p,L,K,nodes,weights,g,base,std,truth)
    expected=sum(quad(lambda a:ndtr((truth-base-response*a)/std),.25,1.75)[0]/1.5 for response in g)/3
    np.testing.assert_allclose(actual,expected,atol=1e-13)


def test_screening_bound_covers_actual_omitted_probability_and_fallback():
    nodes,weights=amount_quadrature(65,.25,1.75)
    L=np.array([1.,2.,3.,-100.]);K=np.array([1.,1.,1.,30.])
    full=source_amount_posterior(L,K,nodes,weights,gap=np.inf)
    screened=source_amount_posterior(L,K,nodes,weights,gap=10.,omitted_limit=.1)
    assert screened['retained_locations']==3
    omitted=full['w0'][screened['w0']==0].sum()
    assert omitted<=screened['omitted_mass_bound']
    assert np.abs(full['w0']-screened['w0']).sum()/2<=screened['omitted_mass_bound']+1e-15
    fallback=source_amount_posterior(L,K,nodes,weights,gap=0.,omitted_limit=0.)
    np.testing.assert_array_equal(fallback['w0'],full['w0'])


def test_direct_bank_matches_affine_likelihood_and_field_mixture_variance():
    model=SignalModelConfig(grid_shape=(4,5),diffusivity=.01,source_sigma=.1)
    sources=np.repeat([[.3,.2],[.6,.7]],3,axis=0);amounts=np.tile([.5,1.,1.5],2)
    bank=SourceAmountBank(model,sources,amounts,np.ones((1,6)),total_rate=.002,
        input_id=CELL_INTEGRATED_V1,half_life_seconds=6.,transport_id='zero_inflow_v1')
    a=AffineSourceLikelihood(model,half_life_seconds=6.,transport_id='zero_inflow_v1')
    h=np.zeros(20);h[7:9]=[.3,.7]
    for t,value in [(.1,.1),(.7,-.05),(1.2,.3),(2.1,.2)]:
        wind=CausalWind(velocity_yx=(.02,.05));bank.advance(t,wind);a.advance(t,wind)
        bank.condition(value,h,.0025);a.condition(value,h,.0025)
        ll=a.log_likelihood(bank.emissions);expected=np.exp(ll-logsumexp(ll))
        np.testing.assert_allclose(expected,bank.weights[0],atol=1e-12)
        components=a.means(bank.emissions)
        np.testing.assert_allclose(components,bank.means,atol=1e-12)
        mean=expected@components
        variance=np.diag(a.reference.covariance)+expected@(components**2)-mean**2
        explicit=np.diag(bank.reference.covariance)+np.sum(expected[:,None]*(bank.means-mean)**2,axis=0)
        np.testing.assert_allclose(variance,explicit,atol=1e-13)


def test_explicit_generator_mean_matches_known_input_filter_and_common_noise():
    model=SignalModelConfig(grid_shape=(4,5),source_sigma=.1)
    wind=CausalWind(velocity_yx=(.02,.05));trajectories={}
    for amount in [.5,1.,1.5]:
        for seed in [None,42]:
            gen=InputMatchedGenerator(model,[.4,.3],total_rate=.002*amount,input_id=CELL_INTEGRATED_V1,
                half_life_seconds=6.,transport_id='zero_inflow_v1',seed=seed)
            bank=SourceAmountBank(model,[[.4,.3]],[amount],[[1.]],total_rate=.002,
                input_id=CELL_INTEGRATED_V1,half_life_seconds=6.,transport_id='zero_inflow_v1')
            values=[]
            for t in [.1,.1125,.7,.7125,2.1]:
                gen.advance(t,wind);bank.advance(t,wind);values.append(gen.state.copy())
                if seed is None:np.testing.assert_allclose(gen.state,bank.means[0],atol=1e-13)
            trajectories[amount,seed]=np.array(values)
    ref=trajectories[.5,42]-trajectories[.5,None]
    for amount in [1.,1.5]:np.testing.assert_allclose(trajectories[amount,42]-trajectories[amount,None],ref,atol=1e-14)


@pytest.mark.parametrize('args',[(0,.25,1.75),(2,1.,1.),(2,-1.,1.)])
def test_invalid_amount_quadrature(args):
    with pytest.raises(ValueError):amount_quadrature(*args)


def test_invalid_posterior_statistics():
    nodes,weights=amount_quadrature(5,.25,1.75)
    with pytest.raises(ValueError):source_amount_posterior([1.],[-1.],nodes,weights)
    with pytest.raises(ValueError):source_amount_posterior([1.],[1.],nodes,weights,fixed_amount=-1.)
