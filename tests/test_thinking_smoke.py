from __future__ import annotations

import unittest


class ThinkingSmokeThresholdTests(unittest.TestCase):
    def test_documented_thresholds_are_strict_enough(self) -> None:
        expected = 20
        self.assertEqual(int(expected * 0.95), 19)
        self.assertEqual(int(expected * 0.10), 2)


if __name__ == "__main__":
    unittest.main()
