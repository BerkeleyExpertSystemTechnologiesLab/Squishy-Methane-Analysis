import numpy as np

from plume_focus.motion import estimate_centroid_motion, estimate_phase_motion, translate_frame
from plume_focus.synthetic import gaussian_plume


def test_centroid_motion_recovers_smooth_translation() -> None:
    frame_0 = gaussian_plume((60, 70), center=(25.0, 30.0))
    frame_1 = gaussian_plume((60, 70), center=(27.5, 26.5))

    motion = estimate_centroid_motion(frame_0, frame_1)

    assert np.isclose(motion.dy_per_t, 2.5, atol=0.1)
    assert np.isclose(motion.dx_per_t, -3.5, atol=0.1)


def test_phase_motion_recovers_integer_shift() -> None:
    frame_0 = gaussian_plume((40, 50), center=(18.0, 22.0), sigma=(3.0, 4.0))
    frame_1 = translate_frame(frame_0, dy=3, dx=-4)

    motion = estimate_phase_motion(frame_0, frame_1)

    assert motion.dy_per_t == 3
    assert motion.dx_per_t == -4


def test_translate_frame_does_not_wrap() -> None:
    frame = np.zeros((5, 5))
    frame[0, 0] = 1.0

    shifted = translate_frame(frame, dy=-1, dx=0)

    assert shifted.sum() == 0.0
