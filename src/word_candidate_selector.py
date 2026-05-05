from dataclasses import replace

import numpy as np

from .models import CharacterBox, OCRPage, PointingRay
from .selection_models import WordCandidate, WordSelectionResult


def build_word_box(
    segmented_page: OCRPage,
    word_id: str,
) -> list[list[float]]:
    word_chars = _characters_for_word(segmented_page, word_id)
    if not word_chars:
        raise ValueError(f"No segmented characters found for word_id={word_id!r}.")

    points = [
        point
        for char in word_chars
        for point in char.box
    ]
    min_x = min(float(point[0]) for point in points)
    min_y = min(float(point[1]) for point in points)
    max_x = max(float(point[0]) for point in points)
    max_y = max(float(point[1]) for point in points)

    return [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
    ]


def box_center(box: list[list[float]]) -> np.ndarray:
    points = np.asarray(box, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Invalid box points: {box!r}")

    return points[:, :2].mean(axis=0)


def build_word_candidates(
    segmented_page: OCRPage,
) -> list[WordCandidate]:
    word_groups = _group_characters_by_word(segmented_page)
    candidates: list[WordCandidate] = []

    for word_id, chars in word_groups:
        box = build_word_box(segmented_page, word_id)
        candidates.append(
            WordCandidate(
                word_id=word_id,
                word=chars[0].word or "",
                line_id=chars[0].line_id,
                char_ids=[char.id for char in chars],
                char_indices=[char.char_index for char in chars],
                box=box,
                center=box_center(box),
                distance_along_ray=float("inf"),
                perpendicular_distance=float("inf"),
                heuristic_score=float("inf"),
                metadata={
                    "global_char_indices": [char.global_char_index for char in chars],
                },
            )
        )

    return candidates


def compute_ray_metrics(
    point: np.ndarray,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
) -> tuple[float, float]:
    origin = np.asarray(ray_origin, dtype=float)
    direction = _normalize_vector(np.asarray(ray_direction, dtype=float))
    point = np.asarray(point, dtype=float)

    relative = point - origin
    distance_along_ray = float(np.dot(relative, direction))
    projection = origin + distance_along_ray * direction
    perpendicular_distance = float(np.linalg.norm(point - projection))

    return distance_along_ray, perpendicular_distance


def score_word_candidate(
    candidate: WordCandidate,
    ray: PointingRay,
) -> WordCandidate:
    distance_along_ray, perpendicular_distance = compute_ray_metrics(
        candidate.center,
        ray.origin,
        ray.direction,
    )
    heuristic_score = distance_along_ray + perpendicular_distance

    return replace(
        candidate,
        distance_along_ray=distance_along_ray,
        perpendicular_distance=perpendicular_distance,
        heuristic_score=heuristic_score,
    )


def select_best_word_candidate(
    segmented_page: OCRPage,
    ray: PointingRay,
    image_width: int,
    threshold_ratio: float = 0.05,
) -> WordSelectionResult:
    if image_width <= 0:
        raise ValueError("image_width must be positive.")
    if threshold_ratio < 0:
        raise ValueError("threshold_ratio must be non-negative.")

    threshold_px = float(image_width) * threshold_ratio
    all_candidates = build_word_candidates(segmented_page)

    accepted: list[WordCandidate] = []
    rejected: list[WordCandidate] = []

    for candidate in all_candidates:
        scored_candidate = score_word_candidate(candidate, ray)
        if (
            scored_candidate.distance_along_ray < 0
            or scored_candidate.perpendicular_distance > threshold_px
        ):
            rejected.append(scored_candidate)
        else:
            accepted.append(scored_candidate)

    accepted.sort(key=lambda candidate: candidate.heuristic_score)
    selected = accepted[0] if accepted else None

    return WordSelectionResult(
        selected=selected,
        candidates=accepted,
        rejected=rejected,
        threshold_px=threshold_px,
        ray_origin=np.asarray(ray.origin, dtype=float),
        ray_direction=_normalize_vector(np.asarray(ray.direction, dtype=float)),
        metadata={
            "threshold_ratio": threshold_ratio,
            "total_words": len(all_candidates),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "heuristic": "distance_along_ray + perpendicular_distance",
        },
    )


def _group_characters_by_word(
    segmented_page: OCRPage,
) -> list[tuple[str, list[CharacterBox]]]:
    word_groups: dict[str, list[CharacterBox]] = {}
    for char in segmented_page.characters:
        if char.word_instance_id is None or char.word is None:
            continue
        word_groups.setdefault(char.word_instance_id, []).append(char)

    return sorted(
        word_groups.items(),
        key=lambda item: min(char.global_char_index for char in item[1]),
    )


def _characters_for_word(
    segmented_page: OCRPage,
    word_id: str,
) -> list[CharacterBox]:
    return [
        char
        for char in segmented_page.characters
        if char.word_instance_id == word_id
    ]


def _normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        raise ValueError("Ray direction must have non-zero length.")

    return v / norm
