from pathlib import Path

import cv2
import numpy as np

from .models import DetectedHand, OCRPage, PointingRay
from .selection_models import WordCandidate, WordSelectionResult


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_character_boxes(
    image_path: str,
    page: OCRPage,
    output_path: str,
) -> None:
    image = _read_image(image_path)

    for char in page.characters:
        points = _box_to_points(char.box)
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        label_point = tuple(points[0, 0].tolist())
        cv2.putText(
            image,
            char.id,
            label_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )

    _write_image(output_path, image)


def draw_segmented_word_boxes(
    image_path: str,
    page: OCRPage,
    output_path: str,
) -> None:
    image = _read_image(image_path)
    labeled_word_instances: set[str] = set()

    for char in page.characters:
        word_instance_id = char.word_instance_id or char.id
        color = _color_for_word(word_instance_id)
        points = _box_to_points(char.box)
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

        if word_instance_id not in labeled_word_instances:
            labeled_word_instances.add(word_instance_id)
            label_point = tuple(points[0, 0].tolist())
            cv2.putText(
                image,
                word_instance_id,
                label_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    _write_image(output_path, image)


def draw_hand_landmarks_and_ray(
    image_bgr: np.ndarray,
    hands: list[DetectedHand],
    rays: list[PointingRay],
    output_path: str | None = None,
    ray_length: int = 800,
) -> np.ndarray:
    debug_image = image_bgr.copy()

    if not hands:
        cv2.putText(
            debug_image,
            "No hands detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for hand in hands:
        landmarks_by_id = {landmark.landmark_id: landmark for landmark in hand.landmarks}

        for start_id, end_id in HAND_CONNECTIONS:
            start = landmarks_by_id.get(start_id)
            end = landmarks_by_id.get(end_id)
            if start is None or end is None:
                continue
            cv2.line(
                debug_image,
                _point_tuple(start.pixel_x, start.pixel_y),
                _point_tuple(end.pixel_x, end.pixel_y),
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )

        for landmark in hand.landmarks:
            cv2.circle(
                debug_image,
                _point_tuple(landmark.pixel_x, landmark.pixel_y),
                4,
                (0, 255, 255),
                -1,
                cv2.LINE_AA,
            )

        for landmark_id in (5, 6, 7, 8):
            landmark = landmarks_by_id.get(landmark_id)
            if landmark is None:
                continue
            point = _point_tuple(landmark.pixel_x, landmark.pixel_y)
            cv2.putText(
                debug_image,
                str(landmark_id),
                (point[0] + 6, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        fingertip = landmarks_by_id.get(8)
        if fingertip is not None:
            cv2.circle(
                debug_image,
                _point_tuple(fingertip.pixel_x, fingertip.pixel_y),
                8,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )

    for ray in rays:
        origin = ray.origin.astype(float)
        end = origin + ray.direction.astype(float) * float(ray_length)
        cv2.line(
            debug_image,
            _point_tuple(origin[0], origin[1]),
            _point_tuple(end[0], end[1]),
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_image,
            f"ray conf={ray.confidence:.2f}",
            _point_tuple(origin[0] + 10, origin[1] + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if output_path is not None:
        _write_image(output_path, debug_image)

    return debug_image


def draw_word_selection_debug(
    image_bgr: np.ndarray,
    segmented_page: OCRPage,
    ray: PointingRay,
    selection_result: WordSelectionResult,
    output_path: str | None = None,
    ray_length: int = 800,
) -> np.ndarray:
    debug_image = image_bgr.copy()

    _draw_ray_with_threshold_guides(
        debug_image,
        selection_result.ray_origin,
        selection_result.ray_direction,
        selection_result.threshold_px,
        ray_length,
    )

    selected_word_id = (
        selection_result.selected.word_id
        if selection_result.selected is not None
        else None
    )

    for candidate in selection_result.rejected:
        _draw_word_candidate(debug_image, candidate, (150, 150, 150), thickness=1)

    for candidate in selection_result.candidates:
        if candidate.word_id == selected_word_id:
            continue
        _draw_word_candidate(debug_image, candidate, (0, 255, 255), thickness=2)

    if selection_result.selected is not None:
        selected = selection_result.selected
        _draw_word_candidate(debug_image, selected, (0, 255, 0), thickness=4, center_radius=6)
        label_point = _point_tuple(selected.center[0] + 8, selected.center[1] - 8)
        cv2.putText(
            debug_image,
            f"{selected.word_id} score={selected.heuristic_score:.1f}",
            label_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            debug_image,
            "No candidate selected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        debug_image,
        f"words={len(selection_result.candidates) + len(selection_result.rejected)} "
        f"accepted={len(selection_result.candidates)}",
        (20, max(75, min(120, debug_image.shape[0] - 20))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if output_path is not None:
        _write_image(output_path, debug_image)

    return debug_image


def _draw_ray_with_threshold_guides(
    image: np.ndarray,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    threshold_px: float,
    ray_length: int,
) -> None:
    origin = np.asarray(ray_origin, dtype=float)
    direction = _unit_vector(np.asarray(ray_direction, dtype=float))
    end = origin + direction * float(ray_length)
    normal = np.array([-direction[1], direction[0]], dtype=float)

    for offset in (-threshold_px, threshold_px):
        guide_origin = origin + normal * offset
        guide_end = end + normal * offset
        cv2.line(
            image,
            _point_tuple(guide_origin[0], guide_origin[1]),
            _point_tuple(guide_end[0], guide_end[1]),
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.line(
        image,
        _point_tuple(origin[0], origin[1]),
        _point_tuple(end[0], end[1]),
        (255, 0, 0),
        3,
        cv2.LINE_AA,
    )


def _draw_word_candidate(
    image: np.ndarray,
    candidate: WordCandidate,
    color: tuple[int, int, int],
    thickness: int,
    center_radius: int = 3,
) -> None:
    cv2.polylines(
        image,
        [_box_to_points(candidate.box)],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(
        image,
        _point_tuple(candidate.center[0], candidate.center[1]),
        center_radius,
        color,
        -1,
        cv2.LINE_AA,
    )


def _read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def _write_image(output_path: str, image: np.ndarray) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(output_path, image):
        raise RuntimeError(f"Could not write debug image: {output_path}")


def _box_to_points(box: list[list[float]]) -> np.ndarray:
    return np.array(box, dtype=np.int32).reshape((-1, 1, 2))


def _point_tuple(x: float, y: float) -> tuple[int, int]:
    return int(round(x)), int(round(y))


def _color_for_word(word_instance_id: str) -> tuple[int, int, int]:
    seed = sum((index + 1) * ord(char) for index, char in enumerate(word_instance_id))
    return (
        80 + seed % 176,
        80 + (seed * 37) % 176,
        80 + (seed * 73) % 176,
    )


def _unit_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        raise ValueError("Cannot draw ray with a zero-length direction.")

    return v / norm
