import argparse

import cv2

from src.debug_draw import draw_hand_landmarks_and_ray
from src.hand_detector import create_hand_landmarker, detect_hands_in_image
from src.pointing_ray import estimate_index_pointing_ray, is_index_finger_extended


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect a hand and draw the estimated index-finger pointing ray."
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--model", required=True, help="MediaPipe hand landmarker .task model path.")
    parser.add_argument("--output", required=True, help="Output debug image path.")
    parser.add_argument(
        "--max-detection-image-size",
        type=int,
        default=1024,
        help="Resize the longest image side to this size for MediaPipe detection; use 0 to disable.",
    )
    parser.add_argument("--num-hands", type=int, default=1, help="Maximum number of hands to detect.")
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

    image_path = args.image
    model_path = args.model
    output_path = args.output

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    landmarker = create_hand_landmarker(
        model_path,
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
        draw_hand_landmarks_and_ray(image_bgr, hands, [], output_path=output_path)
        print("No hands detected.")
        print(f"Saved debug image: {output_path}")
        return

    rays = []
    print(f"Detected hands: {len(hands)}")
    for hand in hands:
        print(
            f"Hand {hand.hand_id}: handedness={hand.handedness} "
            f"score={hand.handedness_score}"
        )
        if len(hand.landmarks) <= 8:
            print(f"Warning: hand {hand.hand_id} has incomplete landmarks; skipping ray.")
            continue
        if not is_index_finger_extended(hand):
            print(f"Warning: hand {hand.hand_id} index finger is not extended; skipping ray.")
            continue

        try:
            ray = estimate_index_pointing_ray(hand)
        except ValueError as exc:
            print(f"Warning: could not estimate ray for hand {hand.hand_id}: {exc}")
            continue

        rays.append(ray)
        print(f"Index fingertip: {ray.origin.tolist()}")
        print(f"Ray direction: {ray.direction.tolist()}")
        print(f"Ray confidence: {ray.confidence:.2f}")

    draw_hand_landmarks_and_ray(image_bgr, hands, rays, output_path=output_path)
    print(f"Saved debug image: {output_path}")


if __name__ == "__main__":
    main()
