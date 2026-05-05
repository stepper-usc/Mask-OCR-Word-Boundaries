from collections import defaultdict

import jieba

from .models import CharacterBox, OCRPage


def add_custom_words(words: list[str]) -> None:
    for word in words:
        if word:
            jieba.add_word(word)


def segment_ocr_page(page: OCRPage, mode: str = "line") -> OCRPage:
    if mode != "line":
        raise ValueError("Only mode='line' is currently supported. Segmenting by line preserves OCR line breaks.")

    _validate_page_text_alignment(page)
    for char in page.characters:
        char.word = None
        char.word_instance_id = None

    for line_id, line_chars in _characters_by_line(page.characters):
        segment_line_with_indices(line_chars, line_id=line_id)

    unassigned = [char.id for char in page.characters if char.word is None or char.word_instance_id is None]
    if unassigned:
        raise RuntimeError(f"Not every character was assigned to a word: {unassigned}")

    return page


def segment_line_with_indices(line_chars: list[CharacterBox], line_id: int | None = None) -> None:
    if not line_chars:
        return

    resolved_line_id = line_chars[0].line_id if line_id is None else line_id
    _validate_line_characters(line_chars, resolved_line_id)
    line_text = "".join(char.char for char in line_chars)

    index = 0
    word_number = 0

    for word in jieba.cut(line_text):
        if word == "":
            continue

        start = index
        end = index + len(word)
        if end > len(line_chars):
            raise RuntimeError(
                f"Jieba token {word!r} on line {resolved_line_id} exceeds available OCR characters."
            )
        if line_text[start:end] != word:
            raise RuntimeError(
                f"Jieba token coverage mismatch on line {resolved_line_id}: "
                f"expected token {word!r} at character range [{start}, {end}), "
                f"found {line_text[start:end]!r}."
            )

        word_instance_id = f"l{resolved_line_id}_w{word_number}"
        for char in line_chars[start:end]:
            char.word = word
            char.word_instance_id = word_instance_id

        index = end
        word_number += 1

    if index != len(line_text):
        raise RuntimeError(
            f"Jieba token lengths did not cover full line {resolved_line_id}: "
            f"covered {index} of {len(line_text)} characters."
        )


def find_character(page: OCRPage, line_id: int, char_index: int) -> CharacterBox | None:
    for char in page.characters:
        if char.line_id == line_id and char.char_index == char_index:
            return char
    return None


def find_word_for_character(page: OCRPage, line_id: int, char_index: int) -> str | None:
    char = find_character(page, line_id, char_index)
    return None if char is None else char.word


def get_characters_for_word_instance(
    page: OCRPage,
    word_instance_id: str,
) -> list[CharacterBox]:
    return [char for char in page.characters if char.word_instance_id == word_instance_id]


def get_segmented_characters_for_word(
    page: OCRPage,
    word_instance_id: str,
) -> list[CharacterBox]:
    return get_characters_for_word_instance(page, word_instance_id)


def iter_word_instances(page: OCRPage) -> list[tuple[str, str, list[CharacterBox]]]:
    word_instances: dict[str, list[CharacterBox]] = {}
    for char in page.characters:
        if char.word_instance_id is None or char.word is None:
            continue
        word_instances.setdefault(char.word_instance_id, []).append(char)

    return [
        (word_instance_id, chars[0].word or "", chars)
        for word_instance_id, chars in sorted(
            word_instances.items(),
            key=lambda item: min(char.global_char_index for char in item[1]),
        )
    ]


def _characters_by_line(characters: list[CharacterBox]) -> list[tuple[int, list[CharacterBox]]]:
    grouped: dict[int, list[CharacterBox]] = defaultdict(list)
    for char in characters:
        grouped[char.line_id].append(char)

    return [
        (line_id, sorted(chars, key=lambda char: char.char_index))
        for line_id, chars in sorted(grouped.items())
    ]


def _validate_page_text_alignment(page: OCRPage) -> None:
    derived_full_text = "\n".join(
        "".join(char.char for char in line_chars)
        for _, line_chars in _characters_by_line(page.characters)
    )
    if page.full_text != derived_full_text:
        raise RuntimeError(
            "OCRPage.full_text does not match the text derived from CharacterBox instances. "
            f"full_text={page.full_text!r}, derived_full_text={derived_full_text!r}"
        )


def _validate_line_characters(line_chars: list[CharacterBox], line_id: int) -> None:
    for expected_index, char in enumerate(line_chars):
        if char.line_id != line_id:
            raise RuntimeError(
                f"Line grouping mismatch: expected line_id {line_id}, found {char.line_id} for {char.id}."
            )
        if char.char_index != expected_index:
            raise RuntimeError(
                f"Line {line_id} character indices must be contiguous and in reading order. "
                f"Expected char_index {expected_index}, found {char.char_index} for {char.id}."
            )
