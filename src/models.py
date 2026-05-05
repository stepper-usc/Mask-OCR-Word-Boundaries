from dataclasses import dataclass, field
from typing import Any


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
    characters: list[CharacterBox]
