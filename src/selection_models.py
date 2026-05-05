from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class WordCandidate:
    word_id: str
    word: str
    line_id: int
    char_ids: list[str]
    char_indices: list[int]
    box: list[list[float]]
    center: np.ndarray
    distance_along_ray: float
    perpendicular_distance: float
    heuristic_score: float
    metadata: dict[str, Any] | None = None


@dataclass
class WordSelectionResult:
    selected: WordCandidate | None
    candidates: list[WordCandidate]
    rejected: list[WordCandidate]
    threshold_px: float
    ray_origin: np.ndarray
    ray_direction: np.ndarray
    metadata: dict[str, Any] | None = None
