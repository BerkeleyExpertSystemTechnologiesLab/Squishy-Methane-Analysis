from plume_focus.motion import MotionEstimate, estimate_centroid_motion
from plume_focus.predictor import BoundConfig, bounding_box_from_mask, predict_attention_area
from plume_focus.synthetic import gaussian_plume


def test_attention_area_contains_immediate_next_plume() -> None:
    shape = (80, 90)
    frame_0 = gaussian_plume(shape, center=(30.0, 35.0))
    frame_1 = gaussian_plume(shape, center=(32.0, 38.0))
    frame_2 = gaussian_plume(shape, center=(34.0, 41.0))

    motion = estimate_centroid_motion(frame_0, frame_1)
    prediction = predict_attention_area(
        frame_1,
        motion,
        bounds=BoundConfig(wind_error_speed=0.5, model_error_radius=1.0),
    )
    true_next_box = bounding_box_from_mask(frame_2 >= frame_2.max() * 0.25)

    assert prediction.attention_box.contains(true_next_box)


def test_scan_age_increases_attention_bound() -> None:
    frame = gaussian_plume((50, 50), center=(25.0, 25.0))
    motion = MotionEstimate(dy_per_t=2.0, dx_per_t=0.0, method="test")

    fresh = predict_attention_area(frame, motion, bounds=BoundConfig(scan_age=0.0))
    stale = predict_attention_area(frame, motion, bounds=BoundConfig(scan_age=3.0))

    assert stale.uncertainty_radius > fresh.uncertainty_radius
    assert stale.attention_box.area > fresh.attention_box.area
