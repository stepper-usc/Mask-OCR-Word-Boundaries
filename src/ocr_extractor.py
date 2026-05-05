from collections.abc import Sequence
from typing import Any

from paddleocr import PaddleOCR

from .models import CharacterBox, OCRPage


REQUIRED_PAYLOAD_KEYS = (
    "rec_texts",
    "rec_scores",
    "rec_polys",
    "text_word",
    "text_word_boxes",
)


class OCRExtractor:
    def __init__(self, ocr: PaddleOCR | None = None) -> None:
        self.ocr = ocr if ocr is not None else create_ocr_engine()

    def extract_page(self, image_path: str) -> OCRPage:
        results = self._predict(image_path)
        payload = _extract_result_payload(results)
        return _payload_to_page(payload)

    def _predict(self, image_path: str) -> Any:
        if not hasattr(self.ocr, "predict") or not callable(self.ocr.predict):
            raise RuntimeError(
                "PaddleOCR engine does not expose predict(...). "
                "This project expects PaddleOCR 3.5.0."
            )

        return self.ocr.predict(image_path)


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


def _extract_result_payload(results: Any) -> dict:
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes, bytearray)):
        raise RuntimeError("Expected PaddleOCR 3.5 predict(...) to return a non-empty sequence.")
    if len(results) == 0:
        raise RuntimeError("PaddleOCR returned an empty OCR result.")

    result = results[0]
    json_payload = getattr(result, "json", None)
    if not isinstance(json_payload, dict):
        raise RuntimeError("Expected PaddleOCR 3.5 result shape: results[0].json['res'].")

    payload = json_payload.get("res")
    if not isinstance(payload, dict):
        raise RuntimeError("Expected PaddleOCR 3.5 result shape: results[0].json['res'].")

    return payload


def _payload_to_page(payload: dict) -> OCRPage:
    character_lines = _extract_lines_from_payload(payload)
    full_text = "\n".join("".join(char.char for char in line_chars) for line_chars in character_lines)

    return OCRPage(full_text=full_text, character_lines=character_lines)


def _extract_lines_from_payload(payload: dict) -> list[list[CharacterBox]]:
    _validate_required_payload_keys(payload)

    rec_texts = _to_plain_py_list(payload["rec_texts"])
    rec_scores = _to_plain_py_list(payload["rec_scores"])
    rec_polys = _to_plain_py_list(payload["rec_polys"])
    text_words_by_line = _to_plain_py_list(payload["text_word"])
    char_boxes_by_line = _to_plain_py_list(payload["text_word_boxes"])

    _validate_line_lists(
        rec_texts=rec_texts,
        rec_scores=rec_scores,
        rec_polys=rec_polys,
        text_words_by_line=text_words_by_line,
        char_boxes_by_line=char_boxes_by_line,
    )

    character_lines: list[list[CharacterBox]] = []
    global_cursor = 0

    for line_id, rec_line_text in enumerate(rec_texts):
        rec_line_text = str(rec_line_text)
        text_word_items = text_words_by_line[line_id]
        line_char_boxes = char_boxes_by_line[line_id]

        if not isinstance(text_word_items, list):
            raise RuntimeError(f"Expected payload['text_word'][{line_id}] to be a list.")
        if not isinstance(line_char_boxes, list):
            raise RuntimeError(f"Expected payload['text_word_boxes'][{line_id}] to be a list.")
        if len(text_word_items) != len(line_char_boxes):
            raise RuntimeError(
                f"PaddleOCR text_word/box alignment mismatch for line {line_id}: "
                f"text_word has {len(text_word_items)} items but text_word_boxes has "
                f"{len(line_char_boxes)} boxes."
            )

        boxed_line_text = "".join(str(item) for item in text_word_items)
        rec_candidate_text = _candidate_text(rec_line_text)
        boxed_candidate_text = _candidate_text(boxed_line_text)
        if rec_candidate_text != boxed_candidate_text:
            raise RuntimeError(
                f"PaddleOCR recognized text and boxed character text disagree for line {line_id}: "
                f"cleaned rec_text={rec_candidate_text!r}, cleaned boxed_text={boxed_candidate_text!r}."
            )

        confidence = _required_float(rec_scores[line_id], f"rec_scores[{line_id}]")
        line_box = _normalize_box(rec_polys[line_id])
        line_characters: list[CharacterBox] = []

        for raw_char_index, (text_word, raw_box) in enumerate(
            zip(text_word_items, line_char_boxes, strict=True)
        ):
            candidate_chars = _candidate_text(str(text_word))
            if candidate_chars == "":
                continue
            if len(candidate_chars) > 1:
                raise RuntimeError(
                    f"PaddleOCR returned multiple Han characters for one box on line {line_id}: "
                    f"text_word[{raw_char_index}]={text_word!r}. "
                    "This pipeline expects one selectable Han character per box."
                )

            char = candidate_chars
            char_index = len(line_characters)
            global_char_index = global_cursor + char_index
            char_id = f"l{line_id}_c{char_index}_g{global_char_index}"

            line_characters.append(
                CharacterBox(
                    id=char_id,
                    char=char,
                    box=_normalize_box(raw_box),
                    line_id=line_id,
                    char_index=char_index,
                    global_char_index=global_char_index,
                    confidence=confidence,
                    metadata={
                        "raw_char_index": raw_char_index,
                        "rec_line_text": rec_line_text,
                        "line_box": line_box,
                        "source_text_word": str(text_word),
                    },
                )
            )

        if line_characters:
            character_lines.append(line_characters)
            global_cursor += len(line_characters) + 1

    if not character_lines:
        raise RuntimeError("PaddleOCR returned no selectable Han character boxes.")

    return character_lines


def _validate_required_payload_keys(payload: dict) -> None:
    missing = [key for key in REQUIRED_PAYLOAD_KEYS if key not in payload]
    if missing:
        raise RuntimeError(
            "Expected PaddleOCR 3.5 payload keys are missing: "
            f"{missing}. Available keys: {_sorted_keys(payload)}"
        )


def _validate_line_lists(
    *,
    rec_texts: Any,
    rec_scores: Any,
    rec_polys: Any,
    text_words_by_line: Any,
    char_boxes_by_line: Any,
) -> None:
    line_lists = {
        "rec_texts": rec_texts,
        "rec_scores": rec_scores,
        "rec_polys": rec_polys,
        "text_word": text_words_by_line,
        "text_word_boxes": char_boxes_by_line,
    }
    for key, value in line_lists.items():
        if not isinstance(value, list):
            raise RuntimeError(f"Expected payload[{key!r}] to be a list.")

    line_count = len(rec_texts)
    if line_count == 0:
        raise RuntimeError("PaddleOCR returned no recognized text lines.")

    for key, value in line_lists.items():
        if len(value) != line_count:
            raise RuntimeError(
                f"Expected payload[{key!r}] to contain {line_count} lines, found {len(value)}."
            )


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


def _normalize_box(raw_box: Any) -> list[list[float]]:
    box = _to_plain_py_list(raw_box)
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


def _to_plain_py_list(value: Any) -> Any:
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if isinstance(value, tuple):
        return [_to_plain_py_list(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_py_list(item) for item in value]
    return value


def _required_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Expected PaddleOCR payload field {field_name} to be numeric.") from exc


def _sorted_keys(payload: dict) -> list[str]:
    return sorted(str(key) for key in payload.keys())
