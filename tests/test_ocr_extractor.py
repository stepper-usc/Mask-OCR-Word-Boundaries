import unittest

from src.ocr_extractor import OCRExtractor, _normalize_character_box


class FakePaddleResult:
    def __init__(self, payload: dict) -> None:
        self.json = {"res": payload}


def make_payload(text: str) -> dict:
    return {
        "rec_texts": [text],
        "rec_scores": [0.99],
        "rec_polys": [[[0, 0], [20, 0], [20, 10], [0, 10]]],
        "text_word": [list(text)],
        "text_word_boxes": [
            [[float(index), 0.0, float(index + 1), 10.0] for index, _ in enumerate(text)]
        ],
    }


class FakeOCR:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, image_path: str) -> list[FakePaddleResult]:
        self.calls.append(image_path)
        return [FakePaddleResult(make_payload("学生"))]


class OCRExtractorTests(unittest.TestCase):
    def test_reuses_injected_ocr_engine_across_extract_calls(self) -> None:
        fake_ocr = FakeOCR()
        extractor = OCRExtractor(ocr=fake_ocr)

        first_page = extractor.extract_page("first.jpg")
        second_page = extractor.extract_page("second.jpg")

        self.assertEqual(fake_ocr.calls, ["first.jpg", "second.jpg"])
        self.assertEqual(first_page.full_text, "学生")
        self.assertEqual(second_page.full_text, "学生")
        self.assertEqual(len(first_page.character_lines), 1)
        self.assertEqual("".join(char.char for char in first_page.characters), "学生")

    def test_rect_character_box_is_projected_onto_angled_line_polygon(self) -> None:
        line_box = [
            [0.0, 0.0],
            [100.0, 10.0],
            [100.0, 30.0],
            [0.0, 20.0],
        ]

        box = _normalize_character_box([20.0, 0.0, 40.0, 20.0], line_box)

        self.assertEqual(
            box,
            [
                [20.0, 2.0],
                [40.0, 4.0],
                [40.0, 24.0],
                [20.0, 22.0],
            ],
        )

    def test_polygon_character_box_is_preserved(self) -> None:
        box = _normalize_character_box(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        )

        self.assertEqual(
            box,
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        )


if __name__ == "__main__":
    unittest.main()
