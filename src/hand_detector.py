from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .models import DetectedHand, HandLandmark


PADDING_FALLBACKS = (
    ("replicate", 0.25),
    ("replicate", 0.5),
    ("replicate", 1.0),
    ("white", 0.5),
    ("white", 1.0),
    ("black", 0.25),
    ("black", 0.5),
    ("black", 1.0),
    ("reflect", 0.25),
    ("reflect", 0.5),
    ("reflect", 1.0),
)


def create_hand_landmarker(
    model_path: str,
    num_hands: int = 1,
    min_hand_detection_confidence: float = 0.5,
    min_hand_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"MediaPipe hand landmarker model not found: {model_path}"
        )

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands_in_image(
    image_bgr: np.ndarray,
    landmarker,
    max_detection_image_size: int | None = 1024,
    padding_fallback: bool = True,
) -> list[DetectedHand]:
    image_height, image_width = image_bgr.shape[:2]
    hands = _detect_hands_once(
        image_bgr=image_bgr,
        landmarker=landmarker,
        max_detection_image_size=max_detection_image_size,
        original_width=image_width,
        original_height=image_height,
        offset_x=0.0,
        offset_y=0.0,
    )
    if hands or not padding_fallback:
        return hands

    best_hands: list[DetectedHand] = []
    best_score = 0.0
    for padding_mode, padding_fraction in PADDING_FALLBACKS:
        padded_image, offset_x, offset_y = _pad_for_detection(
            image_bgr,
            padding_mode=padding_mode,
            padding_fraction=padding_fraction,
        )
        hands = _detect_hands_once(
            image_bgr=padded_image,
            landmarker=landmarker,
            max_detection_image_size=max_detection_image_size,
            original_width=image_width,
            original_height=image_height,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        score = _score_detected_hands(hands)
        if score > best_score:
            best_hands = hands
            best_score = score

    return best_hands


def _detect_hands_once(
    *,
    image_bgr: np.ndarray,
    landmarker,
    max_detection_image_size: int | None,
    original_width: int,
    original_height: int,
    offset_x: float,
    offset_y: float,
) -> list[DetectedHand]:
    source_height, source_width = image_bgr.shape[:2]
    detection_image = _resize_for_detection(image_bgr, max_detection_image_size)
    image_rgb = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    result = landmarker.detect(mp_image)
    hand_landmarks = result.hand_landmarks or []
    handedness_results = result.handedness or []

    detected_hands: list[DetectedHand] = []
    for hand_id, landmarks in enumerate(hand_landmarks):
        handedness = _extract_handedness(handedness_results, hand_id)
        hand_landmark_models = [
            _to_hand_landmark(
                landmark=landmark,
                landmark_id=landmark_id,
                source_width=source_width,
                source_height=source_height,
                original_width=original_width,
                original_height=original_height,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            for landmark_id, landmark in enumerate(landmarks)
        ]
        detected_hands.append(
            DetectedHand(
                hand_id=hand_id,
                handedness=handedness[0],
                handedness_score=handedness[1],
                landmarks=hand_landmark_models,
                image_width=original_width,
                image_height=original_height,
            )
        )

    return detected_hands


def _to_hand_landmark(
    *,
    landmark,
    landmark_id: int,
    source_width: int,
    source_height: int,
    original_width: int,
    original_height: int,
    offset_x: float,
    offset_y: float,
) -> HandLandmark:
    pixel_x = float(landmark.x * source_width - offset_x)
    pixel_y = float(landmark.y * source_height - offset_y)
    return HandLandmark(
        landmark_id=landmark_id,
        x=pixel_x / original_width,
        y=pixel_y / original_height,
        z=float(landmark.z) if getattr(landmark, "z", None) is not None else None,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
    )


def _pad_for_detection(
    image_bgr: np.ndarray,
    *,
    padding_mode: str,
    padding_fraction: float,
) -> tuple[np.ndarray, float, float]:
    image_height, image_width = image_bgr.shape[:2]
    padding = int(round(max(image_width, image_height) * padding_fraction))
    if padding <= 0:
        return image_bgr, 0.0, 0.0

    if padding_mode == "replicate":
        border_type = cv2.BORDER_REPLICATE
        border_value = None
    elif padding_mode == "reflect":
        border_type = cv2.BORDER_REFLECT_101
        border_value = None
    elif padding_mode == "white":
        border_type = cv2.BORDER_CONSTANT
        border_value = (255, 255, 255)
    elif padding_mode == "black":
        border_type = cv2.BORDER_CONSTANT
        border_value = (0, 0, 0)
    else:
        raise ValueError(f"Unsupported padding mode: {padding_mode}")

    if border_value is None:
        padded = cv2.copyMakeBorder(
            image_bgr,
            padding,
            padding,
            padding,
            padding,
            border_type,
        )
    else:
        padded = cv2.copyMakeBorder(
            image_bgr,
            padding,
            padding,
            padding,
            padding,
            border_type,
            value=border_value,
        )

    return padded, float(padding), float(padding)


def _score_detected_hands(hands: list[DetectedHand]) -> float:
    if not hands:
        return 0.0

    return sum(_score_detected_hand(hand) for hand in hands)


def _score_detected_hand(hand: DetectedHand) -> float:
    fingertip = _landmark_by_id(hand, 8)
    if fingertip is None or not _is_landmark_inside_image(fingertip, hand):
        return 0.0

    index_landmarks = [
        landmark
        for landmark_id in (6, 7, 8)
        if (landmark := _landmark_by_id(hand, landmark_id)) is not None
    ]
    inside_count = sum(
        1 for landmark in index_landmarks if _is_landmark_inside_image(landmark, hand)
    )
    inside_ratio = inside_count / 3.0

    return 1.0 + (hand.handedness_score or 0.0) + inside_ratio


def _landmark_by_id(hand: DetectedHand, landmark_id: int) -> HandLandmark | None:
    if 0 <= landmark_id < len(hand.landmarks):
        landmark = hand.landmarks[landmark_id]
        if landmark.landmark_id == landmark_id:
            return landmark

    for landmark in hand.landmarks:
        if landmark.landmark_id == landmark_id:
            return landmark

    return None


def _is_landmark_inside_image(landmark: HandLandmark, hand: DetectedHand) -> bool:
    return (
        0.0 <= landmark.pixel_x <= hand.image_width
        and 0.0 <= landmark.pixel_y <= hand.image_height
    )


def _resize_for_detection(
    image_bgr: np.ndarray,
    max_detection_image_size: int | None,
) -> np.ndarray:
    if max_detection_image_size is None:
        return image_bgr

    image_height, image_width = image_bgr.shape[:2]
    max_side = max(image_width, image_height)
    if max_side <= max_detection_image_size:
        return image_bgr

    scale = max_detection_image_size / max_side
    detection_width = int(round(image_width * scale))
    detection_height = int(round(image_height * scale))
    return cv2.resize(image_bgr, (detection_width, detection_height), interpolation=cv2.INTER_AREA)


def _extract_handedness(
    handedness_results,
    hand_id: int,
) -> tuple[str | None, float | None]:
    if hand_id >= len(handedness_results) or not handedness_results[hand_id]:
        return None, None

    category = handedness_results[hand_id][0]
    label = getattr(category, "category_name", None)
    score = getattr(category, "score", None)
    return label, float(score) if score is not None else None
