import numpy as np

from plume_focus.experiment_metrics import (
    estimate_opencv_motion,
    foreground_mask,
    js_divergence,
    kernel_plume_attention_field,
    kl_divergence,
    normalize_distribution,
)
from plume_focus.motion import MotionEstimate
from plume_focus.video_sources import (
    SyntheticPlumeVideoConfig,
    SyntheticVideoConfig,
    generate_synthetic_plume_frames,
    generate_synthetic_video_frames,
)


def test_opencv_motion_estimates_synthetic_object_direction() -> None:
    config = SyntheticVideoConfig(frames=4, velocity=(1.2, 2.0))
    frames = generate_synthetic_video_frames(config)

    mask = foreground_mask(frames[1])
    motion = estimate_opencv_motion(frames[0], frames[1], current_mask=mask)

    assert motion.dy_per_t > 0.5
    assert motion.dx_per_t > 1.0


def test_opencv_motion_estimates_plume_advection_direction() -> None:
    config = SyntheticPlumeVideoConfig(frames=8)
    frames = generate_synthetic_plume_frames(config)

    mask = foreground_mask(frames[3], threshold_ratio=0.24)
    motion = estimate_opencv_motion(frames[2], frames[3], current_mask=mask)

    assert motion.dx_per_t > 0.4


def test_kernel_plume_attention_advects_many_positive_pixels() -> None:
    frame = np.zeros((30, 40), dtype=np.uint8)
    frame[12:15, 8:12] = 120
    frame[16:19, 18:23] = 220
    mask = frame > 0
    motion = MotionEstimate(dy_per_t=0.0, dx_per_t=5.0, method="test")

    attention = kernel_plume_attention_field(frame, mask, motion, uncertainty_radius=6.0)
    original_mass = attention[:, :23].sum()
    advected_mass = attention[:, 23:].sum()

    assert attention.shape == frame.shape
    assert np.isclose(attention.max(), 1.0)
    assert advected_mass > original_mass


def test_distribution_metrics_reward_better_prediction() -> None:
    truth = normalize_distribution(np.array([[0.0, 1.0], [0.0, 0.0]]))
    good = normalize_distribution(np.array([[0.0, 0.9], [0.1, 0.0]]))
    bad = normalize_distribution(np.array([[1.0, 0.0], [0.0, 0.0]]))

    assert kl_divergence(truth, good) < kl_divergence(truth, bad)
    assert js_divergence(truth, good) < js_divergence(truth, bad)
