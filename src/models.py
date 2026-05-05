from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CharacterBox:
    id: str
    char: str

    box: list[list[float]]
    line_id: int
    char_index: int
    global_char_index: int

    word: str | None = None
    word_instance_id: str | None = None

    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRPage:
    full_text: str
    character_lines: list[list[CharacterBox]]
    characters: list[CharacterBox] = field(init=False)

    def __post_init__(self) -> None:
        self.characters = [char for line_chars in self.character_lines for char in line_chars]


@dataclass
class HandLandmark:
    landmark_id: int
    x: float
    y: float
    z: float | None
    pixel_x: float
    pixel_y: float


@dataclass
class DetectedHand:
    hand_id: int
    handedness: str | None
    handedness_score: float | None
    landmarks: list[HandLandmark]
    image_width: int
    image_height: int


@dataclass
class PointingRay:
    origin: np.ndarray
    direction: np.ndarray
    confidence: float
    source_hand_id: int
    fingertip_landmark_id: int = 8
    metadata: dict[str, Any] | None = None
