import numpy as np
from plume_focus.source_likelihood_quadrature import AffineSourceLikelihood
from plume_focus.source_prior_belief import SourcePriorBank
from plume_focus.signal_belief import SignalModelConfig
from plume_focus.sensing_contracts import CausalWind


def test_affine_likelihood_matches_direct_filter_bank_at_every_event():
    model=SignalModelConfig(grid_shape=(4,5),diffusivity=.01,source_sigma=.12)
    bank=SourcePriorBank(model,[[.3,.2],[.4,.8],[.6,.4]],[[.2,.5,.3]],half_life_seconds=6.)
    affine=AffineSourceLikelihood(model,half_life_seconds=6.,transport_id='zero_inflow_v1')
    loglike=np.zeros(3);h=np.zeros(20);h[7:9]=[.3,.7];noise=.0025
    for t,v,w in [(.05,.1,(.02,.05)),(.1,-.1,(.02,.05)),(.7,.4,(-.03,.04)),(2.1,.3,(.01,-.03))]:
        wind=CausalWind(velocity_yx=w);affine.advance(t,wind);bank.advance(t,wind)
        var=float(h@bank.reference.covariance@h+noise)
        loglike+=-.5*(np.log(2*np.pi*var)+(v-bank.means@h)**2/var)
        affine.condition(v,h,noise);bank.condition(v,h,noise)
        np.testing.assert_allclose(affine.log_likelihood(bank.emissions),loglike,atol=1e-11)
        np.testing.assert_allclose(affine.means(bank.emissions),bank.means,atol=1e-12)
        np.testing.assert_allclose(affine.reference.covariance,bank.reference.covariance,atol=1e-12)
