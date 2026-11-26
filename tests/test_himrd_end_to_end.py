import unittest

from survey_exp.himrd_end_to_end import _parse_split, _render_text


class HimrdEndToEndTests(unittest.TestCase):
    def test_parse_split_accepts_source_shape(self):
        value = _parse_split("make a ( )\ncake\nanalysis")
        self.assertEqual(value, ("make a ( )", "cake"))

    def test_parse_split_rejects_missing_placeholder(self):
        self.assertIsNone(_parse_split("make a cake\ncake"))

    def test_text_renderer_has_locked_dimensions(self):
        image = _render_text("a phrase long enough to wrap onto another line")
        self.assertEqual(image.size, (512, 100))


if __name__ == "__main__":
    unittest.main()
