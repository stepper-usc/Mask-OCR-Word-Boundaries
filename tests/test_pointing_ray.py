import unittest

import numpy as np

from src.models import DetectedHand, HandLandmark
from src.pointing_ray import (
    estimate_index_pointing_ray,
    is_index_finger_extended,
)


def make_hand(
    pip_xy: tuple[float, float],
    dip_xy: tuple[float, float],
    tip_xy: tuple[float, float],
) -> DetectedHand:
    landmarks = [
        HandLandmark(
            landmark_id=landmark_id,
            x=0.0,
            y=0.0,
            z=None,
            pixel_x=0.0,
            pixel_y=0.0,
        )
        for landmark_id in range(21)
    ]
    for landmark_id, xy in ((6, pip_xy), (7, dip_xy), (8, tip_xy)):
        landmarks[landmark_id] = HandLandmark(
            landmark_id=landmark_id,
            x=0.0,
            y=0.0,
            z=None,
            pixel_x=xy[0],
            pixel_y=xy[1],
        )

    return DetectedHand(
        hand_id=0,
        handedness="Right",
        handedness_score=0.99,
        landmarks=landmarks,
        image_width=640,
        image_height=480,
    )


class PointingRayTests(unittest.TestCase):
    def test_straight_upward_finger_points_up(self) -> None:
        hand = make_hand(
            pip_xy=(100.0, 140.0),
            dip_xy=(100.0, 120.0),
            tip_xy=(100.0, 100.0),
        )

        ray = estimate_index_pointing_ray(hand)

        np.testing.assert_allclose(ray.direction, np.array([0.0, -1.0]), atol=1e-6)
        np.testing.assert_allclose(ray.origin, np.array([100.0, 100.0]), atol=1e-6)

    def test_diagonal_finger_direction_has_unit_length(self) -> None:
        hand = make_hand(
            pip_xy=(100.0, 140.0),
            dip_xy=(120.0, 120.0),
            tip_xy=(140.0, 100.0),
        )

        ray = estimate_index_pointing_ray(hand)

        self.assertAlmostEqual(float(np.linalg.norm(ray.direction)), 1.0, places=6)

    def test_duplicate_tip_and_dip_raises_value_error(self) -> None:
        hand = make_hand(
            pip_xy=(100.0, 140.0),
            dip_xy=(100.0, 120.0),
            tip_xy=(100.0, 120.0),
        )

        with self.assertRaises(ValueError):
            estimate_index_pointing_ray(hand)

    def test_index_finger_extended_uses_tip_to_pip_distance(self) -> None:
        extended_hand = make_hand(
            pip_xy=(100.0, 150.0),
            dip_xy=(100.0, 120.0),
            tip_xy=(100.0, 90.0),
        )
        collapsed_hand = make_hand(
            pip_xy=(100.0, 150.0),
            dip_xy=(100.0, 145.0),
            tip_xy=(100.0, 140.0),
        )

        self.assertTrue(is_index_finger_extended(extended_hand))
        self.assertFalse(is_index_finger_extended(collapsed_hand))


if __name__ == "__main__":
    unittest.main()
