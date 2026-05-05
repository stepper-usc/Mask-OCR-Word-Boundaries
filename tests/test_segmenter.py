import unittest

from src.models import CharacterBox, OCRPage
from src.segmenter import (
    add_custom_words,
    find_character,
    find_word_for_character,
    get_characters_for_word_instance,
    iter_word_instances,
    segment_ocr_page,
)


def make_page(line_texts: list[str]) -> OCRPage:
    characters: list[CharacterBox] = []
    global_start = 0

    for line_id, text in enumerate(line_texts):
        for char_index, char in enumerate(text):
            global_char_index = global_start + char_index
            char_id = f"l{line_id}_c{char_index}_g{global_char_index}"
            box = [
                [float(char_index), float(line_id)],
                [float(char_index + 1), float(line_id)],
                [float(char_index + 1), float(line_id + 1)],
                [float(char_index), float(line_id + 1)],
            ]
            characters.append(
                CharacterBox(
                    id=char_id,
                    char=char,
                    box=box,
                    line_id=line_id,
                    char_index=char_index,
                    global_char_index=global_char_index,
                )
            )

        global_start += len(text) + 1

    return OCRPage(
        full_text="\n".join(line_texts),
        characters=characters,
    )


class SegmenterTests(unittest.TestCase):
    def test_custom_word_assigns_every_character_to_word(self) -> None:
        add_custom_words(["学生证"])
        page = make_page(["今天要照学生证用的照片"])

        segment_ocr_page(page)

        self.assertEqual(len(page.characters), len(page.full_text))
        self.assertTrue(all(char.word for char in page.characters))
        self.assertTrue(all(char.word_instance_id for char in page.characters))

        student_id_chars = [find_character(page, 0, char_index) for char_index in (4, 5, 6)]
        word_instance_ids = {char.word_instance_id for char in student_id_chars if char is not None}

        self.assertEqual(len(word_instance_ids), 1)
        self.assertEqual(student_id_chars[0].word, "学生证")

        word_chars = get_characters_for_word_instance(page, student_id_chars[0].word_instance_id or "")
        self.assertEqual("".join(char.char for char in word_chars), "学生证")

    def test_repeated_characters_remain_distinct_instances(self) -> None:
        page = make_page(["我的书是他的书"])

        first_de = find_character(page, 0, 1)
        second_de = find_character(page, 0, 5)
        segment_ocr_page(page)

        self.assertIsNotNone(first_de)
        self.assertIsNotNone(second_de)
        self.assertEqual(first_de.char, "的")
        self.assertEqual(second_de.char, "的")
        self.assertNotEqual(first_de.id, second_de.id)
        self.assertEqual(find_word_for_character(page, 0, 1), "的")
        self.assertEqual(find_word_for_character(page, 0, 5), "的")
        self.assertNotEqual(first_de.word_instance_id, second_de.word_instance_id)

        first_word_chars = get_characters_for_word_instance(page, first_de.word_instance_id or "")
        second_word_chars = get_characters_for_word_instance(page, second_de.word_instance_id or "")

        self.assertIn(first_de, first_word_chars)
        self.assertNotIn(second_de, first_word_chars)
        self.assertIn(second_de, second_word_chars)
        self.assertNotIn(first_de, second_word_chars)

    def test_multiline_global_indices_include_newline_separators(self) -> None:
        add_custom_words(["学生证用"])
        page = make_page(["学生", "证用"])

        segment_ocr_page(page)

        self.assertEqual(page.full_text, "学生\n证用")
        self.assertEqual(find_character(page, 0, 0).global_char_index, 0)
        self.assertEqual(find_character(page, 0, 1).global_char_index, 1)
        self.assertEqual(find_character(page, 1, 0).global_char_index, 3)
        self.assertEqual(find_character(page, 1, 1).global_char_index, 4)

        words = [word for _, word, _ in iter_word_instances(page)]
        self.assertNotIn("学生证用", words)
        self.assertTrue(all("\n" not in word for word in words))


if __name__ == "__main__":
    unittest.main()
