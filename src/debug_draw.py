from pathlib import Path

import cv2
import numpy as np

from .models import OCRPage


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


def _color_for_word(word_instance_id: str) -> tuple[int, int, int]:
    seed = sum((index + 1) * ord(char) for index, char in enumerate(word_instance_id))
    return (
        80 + seed % 176,
        80 + (seed * 37) % 176,
        80 + (seed * 73) % 176,
    )
