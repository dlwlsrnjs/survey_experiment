import tempfile
import unittest
from pathlib import Path

from survey_exp.vlm_benchmark_eval import build_cases


ROOT = Path(__file__).resolve().parents[1]


class VlmBenchmarkEvalTests(unittest.TestCase):
    def test_figstep_uses_exactly_two_hundred_source_cases(self):
        cases = build_cases(
            "FigStep", ROOT / "data/manifests/figstep_safebench_200.jsonl", 200
        )
        self.assertEqual(len(cases), 200)
        self.assertEqual(len({case["goal_id"] for case in cases}), 200)

    def test_hades_uses_one_fixed_step_for_two_hundred_goals(self):
        cases = build_cases(
            "HADES", ROOT / "data/manifests/hades_released_200.jsonl", 200, hades_step=5
        )
        self.assertEqual(len(cases), 200)
        self.assertEqual(len({case["goal_id"] for case in cases}), 200)
        self.assertEqual({case["hades_step"] for case in cases}, {5})

    def test_himrd_rejects_missing_state(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                build_cases(
                    "HIMRD", ROOT / "data/manifests/himrd_safebench_full.jsonl", 1,
                    himrd_state_dir=Path(directory),
                )


if __name__ == "__main__":
    unittest.main()
