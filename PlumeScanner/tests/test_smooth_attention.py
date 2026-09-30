import numpy as np

from plume_focus.motion import MotionEstimate
from plume_focus.predictor import BoundConfig, predict_smooth_attention
from plume_focus.synthetic import gaussian_plume


def test_smooth_attention_peak_moves_with_motion() -> None:
    frame = gaussian_plume((50, 60), center=(20.0, 25.0), sigma=(4.0, 4.0))
    motion = MotionEstimate(dy_per_t=3.0, dx_per_t=5.0, method="test")

    prediction = predict_smooth_attention(
        frame,
        motion,
        bounds=BoundConfig(wind_error_speed=0.0, model_error_radius=1.0, plume_spread_speed=0.0),
    )
    peak_row, peak_col = np.unravel_index(int(np.argmax(prediction.field)), prediction.field.shape)

    assert abs(peak_row - 23) <= 1
    assert abs(peak_col - 30) <= 1


def test_smooth_attention_threshold_box_grows_with_uncertainty() -> None:
    frame = gaussian_plume((50, 60), center=(20.0, 25.0), sigma=(4.0, 4.0))
    motion = MotionEstimate(dy_per_t=1.0, dx_per_t=1.0, method="test")

    tight = predict_smooth_attention(frame, motion, bounds=BoundConfig(wind_error_speed=0.0, model_error_radius=0.5))
    wide = predict_smooth_attention(frame, motion, bounds=BoundConfig(wind_error_speed=3.0, model_error_radius=3.0))

    assert wide.threshold_box(0.25).area > tight.threshold_box(0.25).area
