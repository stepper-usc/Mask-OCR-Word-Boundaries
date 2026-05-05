import numpy as np

from .models import DetectedHand, HandLandmark, PointingRay


INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8


def normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    magnitude = float(np.linalg.norm(v))
    if magnitude < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / magnitude


def estimate_index_pointing_ray(
    hand: DetectedHand,
    method: str = "weighted_index",
) -> PointingRay:
    if method != "weighted_index":
        raise ValueError("Only method='weighted_index' is currently supported.")

    pip = _get_landmark(hand, INDEX_PIP)
    dip = _get_landmark(hand, INDEX_DIP)
    tip = _get_landmark(hand, INDEX_TIP)

    pip_xy = _landmark_xy(pip)
    dip_xy = _landmark_xy(dip)
    tip_xy = _landmark_xy(tip)

    v1 = tip_xy - dip_xy
    v2 = dip_xy - pip_xy
    v_a = normalize_vector(v2)
    v_b = normalize_vector(v1)
    direction = normalize_vector(0.7 * v1 + 0.3 * v2)
    cos_angle = float(np.clip(np.dot(v_a, v_b), -1.0, 1.0))
    confidence = float(np.clip((cos_angle + 1.0) / 2.0, 0.0, 1.0))

    return PointingRay(
        origin=tip_xy,
        direction=direction,
        confidence=confidence,
        source_hand_id=hand.hand_id,
        fingertip_landmark_id=INDEX_TIP,
        metadata={
            "method": method,
            "pip": [pip.pixel_x, pip.pixel_y],
            "dip": [dip.pixel_x, dip.pixel_y],
            "tip": [tip.pixel_x, tip.pixel_y],
        },
    )


def is_index_finger_extended(
    hand: DetectedHand,
    min_tip_to_pip_distance: float = 25.0,
) -> bool:
    pip = _get_landmark(hand, INDEX_PIP)
    tip = _get_landmark(hand, INDEX_TIP)
    return float(np.linalg.norm(_landmark_xy(tip) - _landmark_xy(pip))) >= min_tip_to_pip_distance


def _get_landmark(hand: DetectedHand, landmark_id: int) -> HandLandmark:
    for landmark in hand.landmarks:
        if landmark.landmark_id == landmark_id:
            return landmark
    raise ValueError(f"Hand {hand.hand_id} is missing landmark {landmark_id}.")


def _landmark_xy(landmark: HandLandmark) -> np.ndarray:
    return np.array([landmark.pixel_x, landmark.pixel_y], dtype=float)
