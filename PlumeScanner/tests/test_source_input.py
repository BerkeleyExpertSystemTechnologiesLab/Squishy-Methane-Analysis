"""Conservation, analytic-cell and refinement checks for opt-in source inputs."""
import numpy as np
import pytest
from scipy.integrate import quad
from plume_focus.source_input import (
    gaussian_source_input, POINT_NORMALIZED_V1, CELL_INTEGRATED_V1, SOURCE_INPUT_IDS,
)
from plume_focus.conserved_source_control import injection
from plume_focus.signal_belief import SignalModelConfig


@pytest.mark.parametrize('mode', SOURCE_INPUT_IDS)
@pytest.mark.parametrize('shape', [(12, 16), (24, 32)])
@pytest.mark.parametrize('sigma', [.0175, .035, .07])
def test_same_total_for_centers_gaps_boundaries_and_candidate_rates(mode, shape, sigma):
    sources = np.array([[.547945205479452, .15047021943573669], [.5, .5],
                        [6.5/12, 2.5/16], [.02, .97], [0., 0.], [1., 1.]])
    rates = np.array([.002, .003, .004, .005, .006, 0.])
    e = gaussian_source_input(shape, sources, source_sigma=sigma,
                              total_rate=rates, formulation_id=mode)
    assert np.all(e >= 0)
    np.testing.assert_allclose(e.mean(axis=1), rates, rtol=5e-14, atol=0.)


@pytest.mark.parametrize('source', [(.547945205479452, .15047021943573669), (.02, .97)])
def test_point_normalization_preserves_legacy_shape(source):
    model = SignalModelConfig(source_rate=.2829738193616142, source_sigma=.035)
    old = injection(model, source)
    new = gaussian_source_input(model.grid_shape, source, source_sigma=model.source_sigma,
                                total_rate=.002, formulation_id=POINT_NORMALIZED_V1)
    # Log-domain normalization can alter relative roundoff in vanishing tails.
    # Keep a strict absolute check on shape relative to its unit peak.
    np.testing.assert_allclose(new/new.max(), old/old.max(), rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize('source', [(.547945205479452, .15047021943573669), (.02, .97), (0., 0.)])
def test_integrated_cells_agree_with_independent_numerical_quadrature(source):
    shape = (12, 16); sigma = .035; amount = .002
    e = gaussian_source_input(shape, source, source_sigma=sigma,
                              total_rate=amount, formulation_id=CELL_INTEGRATED_V1).reshape(shape)
    f = lambda x, mu: np.exp(-.5*((x-mu)/sigma)**2)
    normalizers = [quad(f, 0., 1., args=(mu,), epsabs=1e-13)[0] for mu in source]
    # Check cells near the source and in both tails, including positive-tail CDFs.
    center = np.minimum((np.array(source)*shape).astype(int), np.array(shape)-1)
    for dy, dx in [(0, 0), (1, 1), (-1, -1), (2, 2)]:
        cell = np.clip(center+[dy, dx], 0, np.array(shape)-1)
        mass = 1.
        for k, mu in enumerate(source):
            lo, hi = cell[k]/shape[k], (cell[k]+1)/shape[k]
            mass *= quad(f, lo, hi, args=(mu,), epsabs=1e-14)[0]/normalizers[k]
        np.testing.assert_allclose(e[tuple(cell)]/np.prod(shape), amount*mass,
                                   rtol=2e-10, atol=1e-17)


@pytest.mark.parametrize('source', [(.547945205479452, .15047021943573669), (.5, .5), (0., 0.), (.02, .97)])
def test_integrated_input_coarse_fine_cell_masses_are_additive(source):
    coarse = gaussian_source_input((12, 16), source, source_sigma=.0175,
                                    total_rate=.002, formulation_id=CELL_INTEGRATED_V1).reshape(12, 16)
    fine = gaussian_source_input((24, 32), source, source_sigma=.0175,
                                  total_rate=.002, formulation_id=CELL_INTEGRATED_V1).reshape(24, 32)
    aggregated = fine.reshape(12, 2, 16, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(aggregated, coarse, rtol=2e-13, atol=1e-15)


@pytest.mark.parametrize('mode', SOURCE_INPUT_IDS)
def test_scalar_vector_consistency_and_amount_linearity(mode):
    kw = dict(source_sigma=.035, formulation_id=mode)
    one = gaussian_source_input((12, 16), [.4, .3], total_rate=.002, **kw)
    many = gaussian_source_input((12, 16), [[.4, .3], [.4, .3]], total_rate=[.002, .006], **kw)
    np.testing.assert_array_equal(many[0], one)
    np.testing.assert_allclose(many[1], 3*one, rtol=1e-14)


@pytest.mark.parametrize('updates', [dict(grid_shape=(1, 16)), dict(grid_shape=(12., 16)),
    dict(sources_yx=[1.1, .2]), dict(sources_yx=[]), dict(source_sigma=0.),
    dict(source_sigma=float('nan')), dict(total_rate=-1.), dict(total_rate=[1., 2.]),
    dict(formulation_id='unknown')])
def test_invalid_inputs_are_rejected(updates):
    args = dict(grid_shape=(12, 16), sources_yx=[.5, .4], source_sigma=.035,
                total_rate=.002, formulation_id=CELL_INTEGRATED_V1)
    args.update(updates)
    with pytest.raises(ValueError): gaussian_source_input(**args)
