import numpy as np
import pytest
from plume_focus.model_consistent_control import MatchedFieldGenerator
from plume_focus.calibrated_scanner import DecayGaussianBelief
from plume_focus.signal_belief import SignalModelConfig
from plume_focus.sensing_contracts import CausalWind


@pytest.mark.parametrize('transport', ['zero_inflow_v1','legacy_clipped_v1'])
def test_mean_generator_matches_original_unconditioned_dynamics(transport):
    model=SignalModelConfig(grid_shape=(4,5),diffusivity=.01,source_sigma=.12)
    gen=MatchedFieldGenerator(model,(.4,.3),half_life_seconds=6.,transport_id=transport)
    ref=DecayGaussianBelief(model,half_life_seconds=6.,transport_id=transport)
    for t,velocity in [(.15,(.02,.07)),(.8,(-.03,.05)),(2.1,(.1,-.1))]:
        wind=CausalWind(velocity_yx=velocity)
        gen.advance(t,wind);ref.advance(t,wind,(.4,.3))
        np.testing.assert_allclose(gen.state,ref.mean,atol=2e-14)
    with pytest.raises(ValueError): gen.advance(0.,wind)


def test_sampled_moments_match_original_multi_step_covariance():
    model=SignalModelConfig(grid_shape=(2,3),diffusivity=.01,
                           process_std_per_sqrt_second=.15,correlation_length=.2)
    wind=CausalWind(velocity_yx=(.08,.12)); source=(.4,.3)
    ref=DecayGaussianBelief(model,half_life_seconds=6.,transport_id='zero_inflow_v1')
    ref.advance(.7,wind,source);ref.advance(1.3,wind,source)
    samples=[]
    for seed in range(6000):
        gen=MatchedFieldGenerator(model,source,half_life_seconds=6.,transport_id='zero_inflow_v1',seed=seed)
        gen.advance(.7,wind);gen.advance(1.3,wind);samples.append(gen.state)
    samples=np.array(samples)
    # Sampling tolerance expressed in standard errors, not implementation equality.
    np.testing.assert_array_less(np.abs(samples.mean(0)-ref.mean),5*np.sqrt(np.diag(ref.covariance)/len(samples)))
    error=np.abs(np.cov(samples,rowvar=False)-ref.covariance)
    se=np.sqrt((ref.covariance**2+np.outer(np.diag(ref.covariance),np.diag(ref.covariance)))/(len(samples)-1))
    np.testing.assert_array_less(error,5*se)


def test_seed_reproducibility_and_independence():
    model=SignalModelConfig(grid_shape=(2,3));wind=CausalWind(velocity_yx=(.02,.1))
    def draw(seed):
        gen=MatchedFieldGenerator(model,(.5,.4),half_life_seconds=6.,transport_id='zero_inflow_v1',seed=seed)
        gen.advance(.9,wind);return gen.state
    np.testing.assert_array_equal(draw(42),draw(42))
    assert not np.array_equal(draw(42),draw(43))


def test_observation_noise_and_acquisition_clock():
    from types import SimpleNamespace
    from plume_focus.sensing_contracts import Footprint
    from plume_focus.signal_belief import observation_weights
    model=SignalModelConfig(grid_shape=(2,3),gain=2.,offset=.3,sensor_noise_std=.05)
    gen=MatchedFieldGenerator(model,(.5,.4),half_life_seconds=6.,transport_id='zero_inflow_v1',seed=19)
    gen.advance(.05,CausalWind())
    reading=SimpleNamespace(acquired_at_seconds=.05,available_at_seconds=.0625,
                            position_yx=(.4,.3),footprint=Footprint(height=.2,width=.2))
    h=observation_weights(model.grid_shape,reading.position_yx,reading.footprint)
    expected=model.gain*float(h@gen.state)+model.offset
    samples=np.array([gen.observe(reading) for _ in range(6000)])
    assert abs(samples.mean()-expected)<5*model.sensor_noise_std/np.sqrt(len(samples))
    assert abs(samples.std(ddof=1)/model.sensor_noise_std-1)<.05
    reading.acquired_at_seconds=.0625
    with pytest.raises(ValueError):gen.observe(reading)
