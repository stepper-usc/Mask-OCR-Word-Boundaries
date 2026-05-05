import unittest

from src.hand_detector import _score_detected_hands
from src.models import DetectedHand, HandLandmark


def make_hand(tip_x: float, tip_y: float) -> DetectedHand:
    landmarks = [
        HandLandmark(
            landmark_id=landmark_id,
            x=0.5,
            y=0.5,
            z=None,
            pixel_x=50.0,
            pixel_y=50.0,
        )
        for landmark_id in range(21)
    ]
    landmarks[8] = HandLandmark(
        landmark_id=8,
        x=tip_x / 100.0,
        y=tip_y / 100.0,
        z=None,
        pixel_x=tip_x,
        pixel_y=tip_y,
    )
    return DetectedHand(
        hand_id=0,
        handedness="Right",
        handedness_score=0.9,
        landmarks=landmarks,
        image_width=100,
        image_height=100,
    )


class HandDetectorTests(unittest.TestCase):
    def test_scores_in_frame_fingertip_higher_than_out_of_frame_fingertip(self) -> None:
        in_frame_score = _score_detected_hands([make_hand(80.0, 80.0)])
        out_of_frame_score = _score_detected_hands([make_hand(-20.0, 80.0)])

        self.assertGreater(in_frame_score, 0.0)
        self.assertEqual(out_of_frame_score, 0.0)


if __name__ == "__main__":
    unittest.main()
