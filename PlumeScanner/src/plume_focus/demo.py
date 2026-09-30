"""Command-line demonstration of the plume focus workflow."""

from __future__ import annotations

import json

from plume_focus.motion import MotionEstimate, estimate_centroid_motion
from plume_focus.predictor import BoundConfig, bounding_box_from_mask, predict_attention_area
from plume_focus.reconstruction import grid_sample_points, reconstruct_idw, sample_frame
from plume_focus.scanner import SinglePixelScanner
from plume_focus.synthetic import moving_gaussian_frame


def run_demo() -> dict[str, object]:
    """Run a compact end-to-end synthetic plume scenario."""

    shape = (64, 80)
    start_center = (25.0, 25.0)
    true_motion = MotionEstimate(dy_per_t=1.25, dx_per_t=2.0, method="truth")

    frame_0 = moving_gaussian_frame(shape, start_center, true_motion, time=0.0)
    frame_1 = moving_gaussian_frame(shape, start_center, true_motion, time=1.0)
    frame_2 = moving_gaussian_frame(shape, start_center, true_motion, time=2.0)

    measured_motion = estimate_centroid_motion(frame_0, frame_1)

    # This approximates a wind/proxy vector: useful, but less local than the
    # frame-to-frame motion estimate, so we give it a wider error bound.
    average_wind_motion = MotionEstimate(dy_per_t=1.0, dx_per_t=1.7, method="average_wind_proxy", confidence=0.6)

    scanner = SinglePixelScanner(shape, dwell_time=0.00005, stabilization_time=0.00001)
    scan = scanner.scan(lambda t: moving_gaussian_frame(shape, start_center, true_motion, time=1.0 + t))

    motion_bounds = BoundConfig(wind_error_speed=0.75, model_error_radius=1.0, scan_age=scan.max_age)
    wind_bounds = BoundConfig(wind_error_speed=1.5, model_error_radius=2.0, scan_age=scan.max_age)

    motion_prediction = predict_attention_area(frame_1, measured_motion, bounds=motion_bounds)
    wind_prediction = predict_attention_area(frame_1, average_wind_motion, bounds=wind_bounds)

    true_next_box = bounding_box_from_mask(frame_2 >= frame_2.max() * 0.25)

    points = grid_sample_points(shape, spacing=8)
    sampled_values = sample_frame(frame_1, points)
    reconstructed = reconstruct_idw(points, sampled_values, shape)

    return {
        "true_motion_px_per_t": {
            "dy": true_motion.dy_per_t,
            "dx": true_motion.dx_per_t,
        },
        "measured_motion_px_per_t": {
            "dy": round(measured_motion.dy_per_t, 4),
            "dx": round(measured_motion.dx_per_t, 4),
            "method": measured_motion.method,
        },
        "single_pixel_scan": {
            "duration": round(scan.scan_duration, 4),
            "oldest_pixel_age": round(scan.max_age, 4),
        },
        "motion_vector_prediction": {
            "attention_box": motion_prediction.attention_box.__dict__,
            "uncertainty_radius": round(motion_prediction.uncertainty_radius, 4),
            "contains_true_next_box": motion_prediction.attention_box.contains(true_next_box),
        },
        "average_wind_prediction": {
            "attention_box": wind_prediction.attention_box.__dict__,
            "uncertainty_radius": round(wind_prediction.uncertainty_radius, 4),
            "contains_true_next_box": wind_prediction.attention_box.contains(true_next_box),
        },
        "sparse_reconstruction": {
            "sample_count": int(points.shape[0]),
            "mean_reconstructed_value": round(float(reconstructed.mean()), 6),
        },
    }


def main() -> None:
    """Print the demo report as JSON."""

    print(json.dumps(run_demo(), indent=2))


if __name__ == "__main__":
    main()
