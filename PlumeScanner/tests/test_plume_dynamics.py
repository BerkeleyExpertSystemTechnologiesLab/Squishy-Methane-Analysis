import cv2
import numpy as np
import pytest

from plume_focus.experiment_metrics import (
    foreground_mask,
    plume_truth_distribution,
)
from plume_focus.motion import MotionEstimate
from plume_focus.plume_dynamics import (
    PlumeDynamicsConfig,
    propagate_plume_belief,
    source_oriented_prior,
)
from plume_focus.predictor import BoundConfig
from plume_focus.video_sources import (
    SyntheticMethaneLeakConfig,
    generate_synthetic_methane_frame,
)


def test_dynamics_config_rejects_invalid_physical_parameters() -> None:
    with pytest.raises(ValueError):
        PlumeDynamicsConfig(source_emission_rate_per_second=-0.1)
    with pytest.raises(ValueError):
        PlumeDynamicsConfig(max_emission_step_px=0.0)
    with pytest.raises(ValueError):
        PlumeDynamicsConfig(source_kernel_sigma_px=-1.0)
    with pytest.raises(ValueError):
        PlumeDynamicsConfig(prior_source_kernel_weight=-0.1)
    with pytest.raises(ValueError, match="sum to 1"):
        PlumeDynamicsConfig(
            prior_roi_background_weight=0.5,
            prior_roi_focus_weight=0.8,
        )


def test_zero_elapsed_time_preserves_normalized_posterior() -> None:
    posterior = np.zeros((20, 30), dtype=float)
    posterior[8:12, 10:14] = 3.0

    propagated = propagate_plume_belief(
        posterior,
        MotionEstimate(4.0, 7.0, "test"),
        diffusion_sigma=3.0,
        delta_t=0.0,
        source=(10.0, 10.0),
    )

    assert np.allclose(propagated, posterior / posterior.sum())


def test_source_free_propagation_remains_pure_advection() -> None:
    posterior = np.zeros((30, 50), dtype=float)
    posterior[15, 8] = 1.0

    propagated = propagate_plume_belief(
        posterior,
        MotionEstimate(0.0, 4.0, "test"),
        diffusion_sigma=0.0,
        delta_t=2.0,
        source=None,
    )

    peak_row, peak_col = np.unravel_index(
        int(np.argmax(propagated)),
        propagated.shape,
    )
    assert (peak_row, peak_col) == (15, 16)
    assert propagated[15, 8] == 0.0


@pytest.mark.parametrize(
    ("motion", "expected_row_sign", "expected_col_sign"),
    [
        (MotionEstimate(0.0, 4.0, "right"), 0, 1),
        (MotionEstimate(0.0, -4.0, "left"), 0, -1),
        (MotionEstimate(-3.0, 4.0, "tilted"), -1, 1),
    ],
)
def test_source_prior_uses_complete_wind_vector(
    motion: MotionEstimate,
    expected_row_sign: int,
    expected_col_sign: int,
) -> None:
    shape = (80, 100)
    source = (40.0, 50.0)
    prior = source_oriented_prior(
        shape,
        source=source,
        wind_motion=motion,
    )
    rows, cols = np.indices(shape)
    centroid_row = float((prior * rows).sum())
    centroid_col = float((prior * cols).sum())

    assert prior[40, 50] > 0.0
    if expected_row_sign:
        assert np.sign(centroid_row - source[0]) == expected_row_sign
    else:
        assert abs(centroid_row - source[0]) < 1.0
    assert np.sign(centroid_col - source[1]) == expected_col_sign


def test_zero_wind_source_prior_is_symmetric() -> None:
    source = (30.0, 40.0)
    prior = source_oriented_prior(
        (61, 81),
        source=source,
        wind_motion=MotionEstimate(0.0, 0.0, "still"),
    )
    rows, cols = np.indices(prior.shape)

    assert abs(float((prior * rows).sum()) - source[0]) < 1e-6
    assert abs(float((prior * cols).sum()) - source[1]) < 1e-6


def test_source_prior_width_comes_from_dynamics_config() -> None:
    shape = (80, 100)
    source = (40.0, 20.0)
    motion = MotionEstimate(0.0, 5.0, "right")
    narrow = source_oriented_prior(
        shape,
        source=source,
        wind_motion=motion,
        config=PlumeDynamicsConfig(
            prior_crosswind_sigma_min_px=2.0,
            prior_crosswind_sigma_frame_fraction=0.01,
        ),
    )
    wide = source_oriented_prior(
        shape,
        source=source,
        wind_motion=motion,
        config=PlumeDynamicsConfig(
            prior_crosswind_sigma_min_px=20.0,
            prior_crosswind_sigma_frame_fraction=0.3,
        ),
    )
    far_crosswind = np.abs(np.indices(shape)[0] - source[0]) >= 15

    assert float(wide[far_crosswind].sum()) > float(
        narrow[far_crosswind].sum()
    )


@pytest.mark.parametrize("horizon_seconds", [0.25, 0.5, 1.0])
def test_continuous_emission_prevents_advection_gap(
    horizon_seconds: float,
) -> None:
    shape = (60, 100)
    source = (30.0, 12.0)
    posterior = np.zeros(shape, dtype=float)
    posterior[27:34, 12:43] = 1.0
    posterior /= posterior.sum()

    propagated = propagate_plume_belief(
        posterior,
        MotionEstimate(0.0, 12.0, "wind_px_per_second"),
        diffusion_sigma=2.0,
        delta_t=horizon_seconds,
        source=source,
    )
    attention = propagated / propagated.max()
    mask = attention >= 0.22
    _count, labels = cv2.connectedComponents(
        mask.astype(np.uint8),
        connectivity=8,
    )
    source_label = int(labels[30, 12])

    assert source_label > 0
    assert np.any(labels[:, 20:] == source_label)


def test_emission_transition_is_approximately_time_step_consistent() -> None:
    shape = (40, 80)
    source = (20.0, 10.0)
    posterior = np.zeros(shape, dtype=float)
    posterior[18:23, 10:31] = 1.0
    posterior /= posterior.sum()
    motion = MotionEstimate(0.0, 12.0, "wind_px_per_second")
    config = PlumeDynamicsConfig(source_kernel_sigma_px=0.0)

    one_step = propagate_plume_belief(
        posterior,
        motion,
        diffusion_sigma=0.0,
        delta_t=1.0,
        source=source,
        config=config,
    )
    repeated = posterior
    for _index in range(4):
        repeated = propagate_plume_belief(
            repeated,
            motion,
            diffusion_sigma=0.0,
            delta_t=0.25,
            source=source,
            config=config,
        )

    assert np.abs(one_step - repeated).sum() < 0.03


@pytest.mark.parametrize("horizon_seconds", [0.25, 0.5, 1.0])
def test_canonical_methane_forecast_remains_source_connected(
    horizon_seconds: float,
) -> None:
    config = SyntheticMethaneLeakConfig()
    source_row, source_col = map(round, config.source)
    current = generate_synthetic_methane_frame(config, 20.0)
    support = foreground_mask(current, threshold_ratio=0.22)
    posterior = plume_truth_distribution(current, support)
    motion = MotionEstimate(
        config.wind_velocity[0] * config.fps,
        config.wind_velocity[1] * config.fps,
        "exact_wind_px_per_second",
    )
    bounds = BoundConfig(
        scan_age=horizon_seconds,
        wind_error_speed=0.25 * config.fps,
        model_error_radius=2.0,
        plume_spread_speed=0.45 * config.fps,
    )
    radius = bounds.uncertainty_radius(
        motion,
        delta_t=horizon_seconds,
    )

    predictive = propagate_plume_belief(
        posterior,
        motion,
        diffusion_sigma=max(0.5, radius / 3.0),
        delta_t=horizon_seconds,
        source=config.source,
    )
    attention = predictive / predictive.max()
    mask = attention >= 0.22
    _count, labels = cv2.connectedComponents(
        mask.astype(np.uint8),
        connectivity=8,
    )
    source_label = int(labels[source_row, source_col])

    assert source_label > 0
    assert np.any(labels[:, source_col + 8 :] == source_label)
