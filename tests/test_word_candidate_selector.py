import unittest

import numpy as np

from src.models import CharacterBox, OCRPage, PointingRay
from src.word_candidate_selector import (
    build_word_candidates,
    select_best_word_candidate,
)


def make_char(
    *,
    char_id: str,
    char: str,
    word_id: str,
    word: str,
    line_id: int,
    char_index: int,
    global_char_index: int,
    center: tuple[float, float],
    size: float = 10.0,
) -> CharacterBox:
    half = size / 2.0
    x, y = center
    return CharacterBox(
        id=char_id,
        char=char,
        box=[
            [x - half, y - half],
            [x + half, y - half],
            [x + half, y + half],
            [x - half, y + half],
        ],
        line_id=line_id,
        char_index=char_index,
        global_char_index=global_char_index,
        word=word,
        word_instance_id=word_id,
    )


def make_page(chars: list[CharacterBox]) -> OCRPage:
    lines: dict[int, list[CharacterBox]] = {}
    for char in chars:
        lines.setdefault(char.line_id, []).append(char)

    character_lines = [
        sorted(line_chars, key=lambda char: char.char_index)
        for _, line_chars in sorted(lines.items())
    ]
    return OCRPage(
        full_text="\n".join("".join(char.char for char in line) for line in character_lines),
        character_lines=character_lines,
    )


def make_ray(
    origin: tuple[float, float] = (0.0, 0.0),
    direction: tuple[float, float] = (1.0, 0.0),
) -> PointingRay:
    return PointingRay(
        origin=np.array(origin, dtype=float),
        direction=np.array(direction, dtype=float),
        confidence=1.0,
        source_hand_id=0,
    )


class WordCandidateSelectorTests(unittest.TestCase):
    def test_selects_word_directly_along_ray(self) -> None:
        page = make_page([
            make_char(
                char_id="l0_c0_g0",
                char="中",
                word_id="l0_w0",
                word="中",
                line_id=0,
                char_index=0,
                global_char_index=0,
                center=(100.0, 0.0),
            )
        ])

        result = select_best_word_candidate(page, make_ray(), image_width=1000)

        self.assertIsNotNone(result.selected)
        self.assertEqual(result.selected.word_id, "l0_w0")
        self.assertAlmostEqual(result.selected.perpendicular_distance, 0.0)

    def test_rejects_word_behind_ray_origin(self) -> None:
        page = make_page([
            make_char(
                char_id="l0_c0_g0",
                char="后",
                word_id="l0_w0",
                word="后",
                line_id=0,
                char_index=0,
                global_char_index=0,
                center=(-30.0, 0.0),
            )
        ])

        result = select_best_word_candidate(page, make_ray(), image_width=1000)

        self.assertIsNone(result.selected)
        self.assertEqual(len(result.candidates), 0)
        self.assertEqual(len(result.rejected), 1)
        self.assertLess(result.rejected[0].distance_along_ray, 0.0)

    def test_rejects_word_farther_than_threshold_from_ray(self) -> None:
        page = make_page([
            make_char(
                char_id="l0_c0_g0",
                char="远",
                word_id="l0_w0",
                word="远",
                line_id=0,
                char_index=0,
                global_char_index=0,
                center=(100.0, 60.0),
            )
        ])

        result = select_best_word_candidate(page, make_ray(), image_width=1000)

        self.assertIsNone(result.selected)
        self.assertEqual(result.threshold_px, 50.0)
        self.assertGreater(result.rejected[0].perpendicular_distance, result.threshold_px)

    def test_lowest_distance_plus_perpendicular_score_wins(self) -> None:
        page = make_page([
            make_char(
                char_id="l0_c0_g0",
                char="近",
                word_id="l0_w0",
                word="近",
                line_id=0,
                char_index=0,
                global_char_index=0,
                center=(50.0, 20.0),
            ),
            make_char(
                char_id="l0_c1_g1",
                char="远",
                word_id="l0_w1",
                word="远",
                line_id=0,
                char_index=1,
                global_char_index=1,
                center=(80.0, 0.0),
            ),
        ])

        result = select_best_word_candidate(page, make_ray(), image_width=1000)

        self.assertIsNotNone(result.selected)
        self.assertEqual(result.selected.word_id, "l0_w0")
        self.assertAlmostEqual(result.selected.heuristic_score, 70.0)

    def test_repeated_same_word_string_remains_distinct_candidates(self) -> None:
        page = make_page([
            make_char(
                char_id="l0_c0_g0",
                char="书",
                word_id="l0_w0",
                word="书",
                line_id=0,
                char_index=0,
                global_char_index=0,
                center=(100.0, 0.0),
            ),
            make_char(
                char_id="l0_c1_g1",
                char="书",
                word_id="l0_w1",
                word="书",
                line_id=0,
                char_index=1,
                global_char_index=1,
                center=(200.0, 0.0),
            ),
        ])

        candidates = build_word_candidates(page)

        self.assertEqual([candidate.word for candidate in candidates], ["书", "书"])
        self.assertEqual([candidate.word_id for candidate in candidates], ["l0_w0", "l0_w1"])
        self.assertNotEqual(candidates[0].char_ids, candidates[1].char_ids)

    def test_no_accepted_candidates_selects_none(self) -> None:
        page = make_page([
            make_char(
                char_id="l0_c0_g0",
                char="偏",
                word_id="l0_w0",
                word="偏",
                line_id=0,
                char_index=0,
                global_char_index=0,
                center=(100.0, 80.0),
            )
        ])

        result = select_best_word_candidate(page, make_ray(), image_width=1000)

        self.assertIsNone(result.selected)
        self.assertEqual(result.candidates, [])


if __name__ == "__main__":
    unittest.main()
