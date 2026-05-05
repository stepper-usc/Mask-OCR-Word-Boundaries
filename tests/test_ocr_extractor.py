import unittest

from src.ocr_extractor import OCRExtractor


def make_payload(text: str) -> dict:
    return {
        "res": {
            "rec_texts": [text],
            "rec_scores": [0.99],
            "rec_polys": [[[0, 0], [20, 0], [20, 10], [0, 10]]],
            "text_word": [list(text)],
            "text_word_boxes": [
                [
                    [
                        [float(index), 0.0],
                        [float(index + 1), 0.0],
                        [float(index + 1), 1.0],
                        [float(index), 1.0],
                    ]
                    for index, _ in enumerate(text)
                ]
            ],
        }
    }


class FakeOCR:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, image_path: str) -> list[dict]:
        self.calls.append(image_path)
        return [make_payload("学生")]


class OCRExtractorTests(unittest.TestCase):
    def test_reuses_injected_ocr_engine_across_extract_calls(self) -> None:
        fake_ocr = FakeOCR()
        extractor = OCRExtractor(ocr=fake_ocr)

        first_page = extractor.extract_page("first.jpg")
        second_page = extractor.extract_page("second.jpg")

        self.assertEqual(fake_ocr.calls, ["first.jpg", "second.jpg"])
        self.assertEqual(first_page.full_text, "学生")
        self.assertEqual(second_page.full_text, "学生")


if __name__ == "__main__":
    unittest.main()
