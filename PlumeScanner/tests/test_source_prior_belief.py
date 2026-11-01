import numpy as np
import pytest
from scipy.stats import multivariate_normal
from plume_focus.source_prior_belief import SourcePriorBank
from plume_focus.calibrated_scanner import DecayGaussianBelief
from plume_focus.signal_belief import SignalModelConfig
from plume_focus.sensing_contracts import CausalWind


def test_bank_matches_independent_original_filters():
    model = SignalModelConfig(grid_shape=(4, 5), source_sigma=.12)
    sources = np.array([[.4, .3], [.5, .7]])
    bank = SourcePriorBank(model, sources, np.eye(2))
    originals = [DecayGaussianBelief(model, half_life_seconds=6., transport_id='zero_inflow_v1') for _ in sources]
    wind = CausalWind(velocity_yx=(.02, .07))
    h = np.zeros(20); h[7:9] = [.4, .6]
    for at, value in [(.15, .3), (.8, -.1), (2.1, .7)]:
        bank.advance(at, wind)
        for original, source in zip(originals, sources):
            original.advance(at, wind, source)
        bank.condition(value, h, .03)
        for i, original in enumerate(originals):
            original.condition(value, h, .03)
            np.testing.assert_allclose(bank.means[i], original.mean, atol=1e-12)
            np.testing.assert_allclose(bank.reference.covariance, original.covariance, atol=1e-12)
        future = bank.forecast_aggregate(at+.7, wind)
        for i, original in enumerate(originals):
            copy = original.clone(); copy.advance(at+.7, wind, sources[i])
            np.testing.assert_allclose(future[i], copy.mean, atol=1e-12)


def test_sequential_weights_match_joint_gaussian_likelihood():
    model = SignalModelConfig(grid_shape=(2, 3))
    bank = SourcePriorBank(model, [[.3,.3],[.7,.7]], [[.4,.6]])
    bank.means[0] = np.linspace(0, 1, 6)
    bank.means[1] = np.linspace(1, 0, 6)
    means, covariance = bank.means.copy(), bank.reference.covariance.copy()
    h = np.eye(6)[[1,4]]; readings = np.array([.15,.5]); noise = .04
    expected = np.array([multivariate_normal.pdf(readings, h @ mean,
        h @ covariance @ h.T + noise*np.eye(2)) for mean in means])*[.4,.6]
    for value, row in zip(readings,h): bank.condition(value,row,noise)
    np.testing.assert_allclose(bank.weights[0],expected/expected.sum(),atol=1e-12)


def test_between_source_variance_and_duplicate_hypotheses():
    model = SignalModelConfig(grid_shape=(2, 3))
    bank = SourcePriorBank(model, [[.3,.3],[.7,.7]], [[.5,.5]])
    bank.means[0] = 0.; bank.means[1] = 2.
    view = bank.projected_snapshot(0., CausalWind(), (2,3))
    np.testing.assert_allclose(view['mean'],1.)
    np.testing.assert_allclose(view['between_variance'],1.)
    np.testing.assert_allclose(view['std']**2,view['within_variance'][None]+1.)
    bank.means[1] = 0.
    np.testing.assert_allclose(bank.projected_snapshot(0.,CausalWind(),(2,3))['between_variance'],0.)


def test_forecast_is_detached_and_rejects_time_reversal():
    bank = SourcePriorBank(SignalModelConfig(grid_shape=(2,3)), [[.5,.5]], [[1.]])
    bank.advance(1.,CausalWind())
    old = bank.means.copy(); old_cov = bank.reference.covariance.copy()
    bank.forecast_aggregate(2.,CausalWind())
    np.testing.assert_array_equal(bank.means,old)
    np.testing.assert_array_equal(bank.reference.covariance,old_cov)
    with pytest.raises(ValueError): bank.advance(.5,CausalWind())


@pytest.mark.parametrize('priors', [[[0.,0.]], [[-1.,2.]], [[float('nan'),1.]]])
def test_invalid_prior_rejected(priors):
    with pytest.raises(ValueError):
        SourcePriorBank(SignalModelConfig(grid_shape=(2,3)), [[.3,.3],[.7,.7]],priors)
