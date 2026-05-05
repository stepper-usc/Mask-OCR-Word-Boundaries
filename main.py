from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    return_word_box=True,
)

results = ocr.predict("./test-images/t1.jpg")

for res in results:
    data = res.json["res"]

    print(data.keys())

    rec_texts = data.get("rec_texts", [])
    text_words = data.get("text_word", [])
    text_word_boxes = data.get("text_word_boxes", [])

    for line_text, chars, char_boxes in zip(rec_texts, text_words, text_word_boxes):
        print("LINE:", line_text)

        for ch, box in zip(chars, char_boxes):
            print(ch, box)
