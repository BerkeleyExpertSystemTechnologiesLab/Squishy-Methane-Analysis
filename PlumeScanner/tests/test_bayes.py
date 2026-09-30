import numpy as np

from plume_focus.bayes import (
    bayesian_roi_step,
    credible_region_mask,
    credible_region_roi,
    posterior_from_likelihood,
    scan_likelihood,
    source_downwind_prior,
)
from plume_focus.motion import MotionEstimate
from plume_focus.predictor import BoundingBox


def test_posterior_moves_mass_toward_likelihood() -> None:
    prior = np.full((20, 20), 1.0 / 400.0)
    likelihood = np.ones((20, 20))
    likelihood[8:12, 14:18] = 10.0

    posterior = posterior_from_likelihood(prior, likelihood)

    assert posterior[10, 16] > prior[10, 16]
    assert np.isclose(posterior.sum(), 1.0)


def test_scan_likelihood_is_neutral_outside_unsampled_region() -> None:
    scan_values = np.zeros((10, 12), dtype=np.uint8)
    sampled_mask = np.zeros((10, 12), dtype=bool)
    sampled_mask[4:7, 5:9] = True
    scan_values[5, 7] = 255

    likelihood = scan_likelihood(scan_values, sampled_mask, blur_sigma=0.0, neutral_likelihood=0.4)

    assert likelihood[0, 0] == 0.4
    assert likelihood[5, 7] > likelihood[4, 5]


def test_credible_region_roi_contains_requested_mass() -> None:
    distribution = np.zeros((30, 40))
    distribution[12:16, 20:24] = 1.0

    roi = credible_region_roi(distribution, mass=0.9, margin=2)
    contained_mass = distribution[roi.to_slices()].sum() / distribution.sum()

    assert contained_mass >= 0.9


def test_credible_region_mask_is_non_rectangular() -> None:
    distribution = np.zeros((40, 50))
    rows, cols = np.indices(distribution.shape)
    distribution = np.exp(-(((rows - 20) ** 2) / 50.0 + ((cols - 25) ** 2) / 80.0))

    mask = credible_region_mask(distribution, mass=0.6, blur_sigma=1.5)
    box = credible_region_roi(distribution, mass=0.6, margin=0)
    box_mask = np.zeros(mask.shape, dtype=bool)
    box_mask[box.to_slices()] = True

    assert mask.sum() < box_mask.sum()
    assert mask.sum() > 0


def test_credible_region_mask_respects_minimum_area_fraction() -> None:
    distribution = np.zeros((40, 50))
    distribution[20, 25] = 1.0

    mask = credible_region_mask(distribution, mass=0.6, blur_sigma=0.0, min_area_fraction=0.08)

    assert mask.sum() >= 0.08 * distribution.size


def test_minimum_area_expands_locally_instead_of_selecting_arbitrary_zeros() -> None:
    distribution = np.zeros((40, 50))
    distribution[20, 25] = 1.0

    mask = credible_region_mask(distribution, mass=0.6, blur_sigma=0.0, min_area_fraction=0.08)
    rows, cols = np.where(mask)

    assert mask.sum() == 160
    assert rows.min() >= 13
    assert rows.max() <= 27
    assert cols.min() >= 18
    assert cols.max() <= 32


def test_bayesian_roi_step_can_shrink_from_broad_roi() -> None:
    shape = (40, 60)
    source = (20.0, 8.0)
    motion = MotionEstimate(dy_per_t=0.0, dx_per_t=1.0, method="test")
    broad_roi = BoundingBox(5, 5, 35, 55)
    prior = source_downwind_prior(shape, source=source, wind_motion=motion, roi=broad_roi)
    scan_values = np.zeros(shape, dtype=np.uint8)
    sampled_mask = np.zeros(shape, dtype=bool)
    sampled_mask[broad_roi.to_slices()] = True
    scan_values[18:22, 20:32] = 230

    belief = bayesian_roi_step(
        prior,
        scan_values,
        sampled_mask,
        motion,
        current_roi=broad_roi,
        credible_mass=0.65,
        diffusion_sigma=2.0,
        source=None,
    )

    assert belief.next_roi.area < broad_roi.area
    assert np.isclose(belief.posterior.sum(), 1.0)
    assert np.isclose(belief.predictive.sum(), 1.0)


def test_bayesian_roi_step_advects_scan_evidence_without_source() -> None:
    shape = (40, 60)
    motion = MotionEstimate(dy_per_t=0.0, dx_per_t=4.0, method="known_wind_proxy")
    full_roi = BoundingBox(0, 0, shape[0] - 1, shape[1] - 1)
    prior = np.full(shape, 1.0 / np.prod(shape))
    scan_values = np.zeros(shape, dtype=np.uint8)
    sampled_mask = np.ones(shape, dtype=bool)
    scan_values[17:23, 18:26] = 240

    belief = bayesian_roi_step(
        prior,
        scan_values,
        sampled_mask,
        motion,
        current_roi=full_roi,
        credible_mass=0.7,
        diffusion_sigma=1.0,
        source=None,
        smooth_roi=True,
        roi_blur_sigma=1.0,
    )

    _rows, cols = np.indices(shape)
    posterior_col = float((belief.posterior * cols).sum())
    predictive_col = float((belief.predictive * cols).sum())

    assert predictive_col > posterior_col + 2.0
    assert belief.next_roi.max_col > 26
    assert np.isclose(belief.posterior.sum(), 1.0)
    assert np.isclose(belief.predictive.sum(), 1.0)


def test_predictive_motion_respects_elapsed_time() -> None:
    shape = (20, 30)
    motion = MotionEstimate(dy_per_t=0.0, dx_per_t=2.0, method="px_per_second")
    posterior = np.zeros(shape, dtype=float)
    posterior[10, 5] = 1.0

    belief = bayesian_roi_step(
        posterior,
        np.zeros(shape, dtype=np.uint8),
        np.zeros(shape, dtype=bool),
        motion,
        current_roi=BoundingBox(0, 0, shape[0] - 1, shape[1] - 1),
        credible_mass=0.8,
        diffusion_sigma=0.0,
        source=None,
        prediction_delta_t=3.0,
    )

    _rows, cols = np.indices(shape)
    predictive_col = float((belief.predictive * cols).sum())
    assert predictive_col > 10.5
