import logging
from collections.abc import Sequence
from typing import Any

from paddleocr import PaddleOCR

from .models import CharacterBox, OCRPage


logger = logging.getLogger(__name__)


class OCRExtractor:
    def __init__(self, ocr: PaddleOCR | None = None) -> None:
        self.ocr = ocr if ocr is not None else create_ocr_engine()

    def extract_page(self, image_path: str) -> OCRPage:
        results = self._predict(image_path)
        payload = _extract_result_payload(results)
        return _payload_to_page(payload)

    def _predict(self, image_path: str) -> Any:
        if hasattr(self.ocr, "predict") and callable(self.ocr.predict):
            return self.ocr.predict(image_path)
        return self.ocr.ocr(image_path)


def create_ocr_engine() -> PaddleOCR:
    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
    )


def extract_ocr_page(image_path: str) -> OCRPage:
    return OCRExtractor().extract_page(image_path)


def _payload_to_page(payload: dict) -> OCRPage:
    character_lines = _extract_lines_from_payload(payload)
    characters = [char for line_chars in character_lines for char in line_chars]
    full_text = "\n".join("".join(char.char for char in line_chars) for line_chars in character_lines)

    return OCRPage(full_text=full_text, characters=characters)


def _result_to_dict(result: Any) -> dict:
    if isinstance(result, dict):
        return _to_plain(result)

    json_payload = getattr(result, "json", None)
    if json_payload is not None:
        if callable(json_payload):
            json_payload = json_payload()
        if isinstance(json_payload, dict):
            return _to_plain(json_payload)

    for method_name in ("to_dict", "dict"):
        method = getattr(result, method_name, None)
        if callable(method):
            value = method()
            if isinstance(value, dict):
                return _to_plain(value)

    result_dict = getattr(result, "__dict__", None)
    if isinstance(result_dict, dict) and result_dict:
        return _to_plain(result_dict)

    raise RuntimeError(f"Could not convert PaddleOCR result to a dictionary: {type(result)!r}")


def _extract_result_payload(results: Any) -> dict:
    if results is None:
        raise RuntimeError("PaddleOCR returned an empty OCR result.")

    if isinstance(results, dict):
        candidates = [results]
    elif isinstance(results, Sequence) and not isinstance(results, (str, bytes, bytearray)):
        if len(results) == 0:
            raise RuntimeError("PaddleOCR returned an empty OCR result.")
        candidates = list(results)
    else:
        candidates = [results]

    for candidate in candidates:
        payload = _unwrap_payload(_result_to_dict(candidate))
        if _contains_ocr_payload(payload):
            return payload

    available = [_sorted_keys(_unwrap_payload(_result_to_dict(candidate))) for candidate in candidates]
    raise RuntimeError(f"PaddleOCR result did not contain an OCR payload. Available keys: {available}")


def _extract_lines_from_payload(payload: dict) -> list[list[CharacterBox]]:
    payload = _to_plain(payload)
    rec_texts = _as_list(_first_present(payload, ("rec_texts", "texts", "text")))
    text_words_raw = _first_present(payload, ("text_word", "text_words", "chars", "characters"))

    if not rec_texts and text_words_raw is None:
        logger.warning("PaddleOCR payload keys: %s", _sorted_keys(payload))
        raise RuntimeError(
            "PaddleOCR result did not contain recognized text or boxed character text. "
            f"Available keys: {_sorted_keys(payload)}"
        )

    raw_char_boxes = _first_present(payload, ("text_word_boxes", "char_boxes", "character_boxes"))
    if raw_char_boxes is None:
        logger.warning("PaddleOCR payload keys: %s", _sorted_keys(payload))
        raise RuntimeError(
            "PaddleOCR did not return character boxes. Confirm return_word_box=True and inspect result keys."
        )

    line_count = len(rec_texts) if rec_texts else _infer_line_count(raw_char_boxes, text_words_raw)
    rec_scores = _as_list(_first_present(payload, ("rec_scores", "scores", "rec_confidences")))
    line_boxes = _as_list(_first_present(payload, ("rec_polys", "dt_polys", "rec_boxes", "boxes")))
    char_boxes_by_line = _normalize_line_items(raw_char_boxes, line_count, item_kind="boxes")
    text_words_by_line = (
        _normalize_line_items(text_words_raw, line_count, item_kind="text")
        if text_words_raw is not None
        else []
    )

    character_lines: list[list[CharacterBox]] = []
    global_cursor = 0

    for line_id in range(line_count):
        raw_line_char_boxes = _sequence_get(char_boxes_by_line, line_id)
        if raw_line_char_boxes is None:
            raise RuntimeError(
                "PaddleOCR did not return character boxes. Confirm return_word_box=True and inspect result keys."
            )

        line_char_boxes = _as_list(raw_line_char_boxes)
        text_word_items = _sequence_get(text_words_by_line, line_id)
        rec_line_text = _optional_str(_sequence_get(rec_texts, line_id))

        shared_box_metadata: list[dict[str, Any]] = []
        if text_word_items is not None:
            line_chars, line_char_boxes, shared_box_metadata = _expand_text_words_and_boxes(
                text_word_items,
                line_char_boxes,
                line_id,
            )
        elif rec_line_text is not None:
            line_chars = list(rec_line_text)
        else:
            raise RuntimeError(
                f"Missing recognized character text for line {line_id}. "
                "Expected text_word or rec_texts in the PaddleOCR result."
            )

        if len(line_chars) != len(line_char_boxes):
            raise RuntimeError(
                f"Character/box alignment mismatch for line {line_id}: recognized boxed text has "
                f"{len(line_chars)} characters but PaddleOCR returned {len(line_char_boxes)} boxes. "
                f"boxed_text={''.join(line_chars)!r}"
            )

        boxed_line_text = "".join(line_chars)
        if rec_line_text is not None and _candidate_text(rec_line_text) != _candidate_text(boxed_line_text):
            raise RuntimeError(
                f"PaddleOCR recognized text and boxed character text disagree for line {line_id}: "
                f"rec_text={rec_line_text!r}, boxed_text={boxed_line_text!r}."
            )

        confidence = _optional_float(_sequence_get(rec_scores, line_id))
        line_box = _optional_box(_sequence_get(line_boxes, line_id))
        line_characters: list[CharacterBox] = []

        for raw_char_index, (char, raw_box) in enumerate(zip(line_chars, line_char_boxes, strict=True)):
            if not _is_candidate_character(char):
                continue

            char_index = len(line_characters)
            global_char_index = global_cursor + char_index
            char_id = f"l{line_id}_c{char_index}_g{global_char_index}"
            metadata: dict[str, Any] = {"raw_char_index": raw_char_index}
            if rec_line_text is not None:
                metadata["rec_line_text"] = rec_line_text
            if line_box is not None:
                metadata["line_box"] = line_box
            metadata.update(shared_box_metadata[raw_char_index] if shared_box_metadata else {})

            line_characters.append(
                CharacterBox(
                    id=char_id,
                    char=char,
                    box=_normalize_box(raw_box),
                    line_id=line_id,
                    char_index=char_index,
                    global_char_index=global_char_index,
                    confidence=confidence,
                    metadata=metadata,
                )
            )

        if line_characters:
            character_lines.append(line_characters)
            global_cursor += len(line_characters) + 1

    return character_lines


def _unwrap_payload(result_dict: dict) -> dict:
    current = result_dict
    for key in ("res", "result", "data"):
        nested = current.get(key)
        if isinstance(nested, dict):
            current = nested
    return current


def _contains_ocr_payload(payload: dict) -> bool:
    return any(key in payload for key in ("rec_texts", "texts", "text", "text_word", "text_word_boxes"))


def _first_present(payload: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    value = _to_plain(value)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_line_items(value: Any, line_count: int, item_kind: str) -> list[Any]:
    items = _as_list(value)
    if line_count == 1 and items:
        first = items[0]
        if item_kind == "text" and all(isinstance(item, str) for item in items):
            return [items]
        if item_kind == "boxes" and _is_box_like(first):
            return [items]
    return items


def _infer_line_count(raw_char_boxes: Any, text_words_raw: Any) -> int:
    text_words = _as_list(text_words_raw)
    if text_words and isinstance(text_words[0], list):
        return len(text_words)

    char_boxes = _as_list(raw_char_boxes)
    if char_boxes and _is_box_like(char_boxes[0]):
        return 1
    return len(char_boxes)


def _expand_text_words_and_boxes(
    text_word_items: Any,
    line_char_boxes: list[Any],
    line_id: int,
) -> tuple[list[str], list[Any], list[dict[str, Any]]]:
    if isinstance(text_word_items, str):
        text_items = list(text_word_items)
    else:
        text_items = [str(item) for item in _as_list(text_word_items)]

    characters: list[str] = []
    expanded_boxes: list[Any] = []
    metadata: list[dict[str, Any]] = []

    if len(text_items) != len(line_char_boxes):
        raise RuntimeError(
            f"PaddleOCR text_word/box alignment mismatch for line {line_id}: "
            f"text_word has {len(text_items)} items but PaddleOCR returned {len(line_char_boxes)} boxes."
        )

    for source_index, (text, box) in enumerate(zip(text_items, line_char_boxes, strict=True)):
        if text == "":
            continue

        for source_char_offset, char in enumerate(text):
            characters.append(char)
            expanded_boxes.append(box)
            metadata.append(
                {
                    "source_text_word": text,
                    "source_text_word_index": source_index,
                    "source_text_word_char_offset": source_char_offset,
                    "shares_box_with_text_word": len(text) > 1,
                }
            )

    return characters, expanded_boxes, metadata


def _candidate_text(text: str) -> str:
    return "".join(char for char in text if _is_candidate_character(char))


def _is_candidate_character(char: str) -> bool:
    if len(char) != 1:
        return False

    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x2F800 <= codepoint <= 0x2FA1F
    )


def _optional_box(raw_box: Any) -> list[list[float]] | None:
    if raw_box is None:
        return None
    return _normalize_box(raw_box)


def _normalize_box(raw_box: Any) -> list[list[float]]:
    box = _to_plain(raw_box)
    if _is_rect_like(box):
        x1, y1, x2, y2 = [float(value) for value in box]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    if not isinstance(box, list) or not box:
        raise RuntimeError(f"Invalid OCR box format: {raw_box!r}")

    normalized: list[list[float]] = []
    for point in box:
        if not isinstance(point, list) or len(point) < 2:
            raise RuntimeError(f"Invalid OCR polygon point: {point!r}")
        normalized.append([float(point[0]), float(point[1])])

    return normalized


def _is_rect_like(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
    )


def _is_point_like(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    )


def _is_box_like(value: Any) -> bool:
    return _is_rect_like(value) or (
        isinstance(value, list)
        and len(value) >= 4
        and all(_is_point_like(point) for point in value)
    )


def _sequence_get(values: Sequence[Any] | list[Any], index: int) -> Any:
    if values is None or index >= len(values):
        return None
    return values[index]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _to_plain(value: Any) -> Any:
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def _sorted_keys(payload: dict) -> list[str]:
    return sorted(str(key) for key in payload.keys())
