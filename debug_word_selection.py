import argparse

import cv2

from src.debug_draw import draw_hand_landmarks_and_ray, draw_word_selection_debug
from src.hand_detector import create_hand_landmarker, detect_hands_in_image
from src.ocr_extractor import OCRExtractor
from src.pointing_ray import estimate_index_pointing_ray, is_index_finger_extended
from src.segmenter import add_custom_words, segment_ocr_page
from src.word_candidate_selector import select_best_word_candidate


DEFAULT_CUSTOM_WORDS = [
    "学生证",
    "刘海",
    "学生会",
    "名字",
    "照照片",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OCR, segment words, estimate hand ray, and select the best word candidate."
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--hand-model", required=True, help="MediaPipe hand landmarker .task path.")
    parser.add_argument("--output", required=True, help="Output debug image path.")
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.05,
        help="Candidate max perpendicular distance as a ratio of image width.",
    )
    parser.add_argument(
        "--max-detection-image-size",
        type=int,
        default=1024,
        help="Resize the longest image side to this size for MediaPipe detection; use 0 to disable.",
    )
    parser.add_argument("--num-hands", type=int, default=2, help="Maximum number of hands to detect.")
    parser.add_argument(
        "--disable-padding-fallback",
        action="store_true",
        help="Disable padded retry attempts for partially cropped hands.",
    )
    parser.add_argument(
        "--min-hand-detection-confidence",
        type=float,
        default=0.1,
        help="MediaPipe minimum hand detection confidence.",
    )
    parser.add_argument(
        "--min-hand-presence-confidence",
        type=float,
        default=0.1,
        help="MediaPipe minimum hand presence confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.1,
        help="MediaPipe minimum tracking confidence.",
    )
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    page = None
    ocr_warning = None
    try:
        page = OCRExtractor().extract_page(args.image)
        add_custom_words(DEFAULT_CUSTOM_WORDS)
        segment_ocr_page(page)
    except RuntimeError as exc:
        ocr_warning = str(exc)

    landmarker = create_hand_landmarker(
        args.hand_model,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    max_detection_image_size = (
        None if args.max_detection_image_size <= 0 else args.max_detection_image_size
    )
    hands = detect_hands_in_image(
        image_bgr,
        landmarker,
        max_detection_image_size=max_detection_image_size,
        padding_fallback=not args.disable_padding_fallback,
    )

    if not hands:
        draw_hand_landmarks_and_ray(image_bgr, hands, [], output_path=args.output)
        print("No hands detected.")
        print(f"Saved debug image: {args.output}")
        return

    print(f"Detected hands: {len(hands)}")
    if len(hands) > 1:
        print("Multiple hands detected; using hand 0 for word selection.")

    hand = hands[0]
    if len(hand.landmarks) <= 8:
        draw_hand_landmarks_and_ray(image_bgr, hands, [], output_path=args.output)
        print("Warning: hand 0 has incomplete landmarks; no selection made.")
        print(f"Saved debug image: {args.output}")
        return
    if not is_index_finger_extended(hand):
        draw_hand_landmarks_and_ray(image_bgr, hands, [], output_path=args.output)
        print("Warning: hand 0 index finger is not extended; no selection made.")
        print(f"Saved debug image: {args.output}")
        return

    try:
        ray = estimate_index_pointing_ray(hand)
    except ValueError as exc:
        draw_hand_landmarks_and_ray(image_bgr, hands, [], output_path=args.output)
        print(f"Warning: could not estimate ray for hand 0: {exc}")
        print(f"Saved debug image: {args.output}")
        return

    if page is None:
        draw_hand_landmarks_and_ray(image_bgr, hands, [ray], output_path=args.output)
        print("No OCR words found.")
        if ocr_warning:
            print(f"OCR warning: {ocr_warning}")
        print(f"Ray origin: {ray.origin.tolist()}")
        print(f"Ray direction: {ray.direction.tolist()}")
        print("Selected: None")
        print(f"Saved debug image: {args.output}")
        return

    selection_result = select_best_word_candidate(
        segmented_page=page,
        ray=ray,
        image_width=image_bgr.shape[1],
        threshold_ratio=args.threshold_ratio,
    )
    draw_word_selection_debug(
        image_bgr,
        page,
        ray,
        selection_result,
        output_path=args.output,
    )

    print(f"Ray origin: {ray.origin.tolist()}")
    print(f"Ray direction: {ray.direction.tolist()}")
    print(f"Threshold px: {selection_result.threshold_px:.1f}")
    print()
    print(f"Accepted candidates: {len(selection_result.candidates)}")
    print(f"Rejected candidates: {len(selection_result.rejected)}")

    if selection_result.metadata and selection_result.metadata.get("total_words") == 0:
        print("No OCR words found.")

    if selection_result.selected is None:
        print()
        print("Selected: None")
        print("Reason: no word centers were within threshold or in front of the ray.")
    else:
        selected = selection_result.selected
        print()
        print("Selected:")
        print(f"  word_id: {selected.word_id}")
        print(f"  word: {selected.word}")
        print(f"  line_id: {selected.line_id}")
        print(f"  score: {selected.heuristic_score:.1f}")
        print(f"  distance_along_ray: {selected.distance_along_ray:.1f}")
        print(f"  perpendicular_distance: {selected.perpendicular_distance:.1f}")

    print(f"Saved debug image: {args.output}")


if __name__ == "__main__":
    main()
