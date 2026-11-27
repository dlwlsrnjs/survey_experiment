from __future__ import annotations

import unittest

from survey_exp.util import parse_json_object


class ParseJsonObjectTests(unittest.TestCase):
    def test_parses_final_json_after_visible_thinking(self) -> None:
        text = '<think>Compare {"example": true} first.</think>\n{"refusal": false}'
        self.assertEqual(parse_json_object(text), {"refusal": False})

    def test_rejects_truncated_thinking_without_final_answer(self) -> None:
        with self.assertRaises(ValueError):
            parse_json_object('<think>Still reasoning about {"example": true}')


if __name__ == "__main__":
    unittest.main()
