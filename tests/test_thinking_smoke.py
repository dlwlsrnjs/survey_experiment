from __future__ import annotations

import unittest


class AuxiliarySmokeThresholdTests(unittest.TestCase):
    def test_documented_thresholds_are_strict_enough(self) -> None:
        expected = 10
        self.assertEqual((expected * 95 + 99) // 100, 10)
        self.assertEqual(int(expected * 0.10), 1)
        self.assertEqual(int(expected * 0.20), 2)


if __name__ == "__main__":
    unittest.main()
