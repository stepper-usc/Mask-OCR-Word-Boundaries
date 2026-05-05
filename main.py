import argparse

from src.debug_draw import draw_segmented_word_boxes
from src.ocr_extractor import OCRExtractor
from src.segmenter import add_custom_words, iter_word_instances, segment_ocr_page


DEFAULT_CUSTOM_WORDS = [
    "学生证",
    "刘海",
    "学生会",
    "名字",
    "照照片",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR + Jieba segmentation and draw colored word boxes."
    )
    parser.add_argument("image", help="Input image path.")
    parser.add_argument("output", help="Output debug image path.")
    args = parser.parse_args()

    extractor = OCRExtractor()
    page = extractor.extract_page(args.image)

    add_custom_words(DEFAULT_CUSTOM_WORDS)
    segment_ocr_page(page)
    draw_segmented_word_boxes(args.image, page, args.output)

    print("FULL TEXT:")
    print(page.full_text)
    print()
    print(f"Characters: {len(page.characters)}")
    print(f"Words: {len(iter_word_instances(page))}")
    print(f"Saved colored word boxes: {args.output}")


if __name__ == "__main__":
    main()
