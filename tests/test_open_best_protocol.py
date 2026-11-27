import json
import unittest
from pathlib import Path

import yaml

from scripts.build_open_best_matrix import build
from scripts.download_open_best_asset import assets


ROOT = Path(__file__).resolve().parents[1]


class OpenBestProtocolTests(unittest.TestCase):
    def test_candidate_families_have_at_least_five_models(self):
        candidates = yaml.safe_load((ROOT / "configs/open_best_candidates.yaml").read_text())
        self.assertGreaterEqual(len(candidates["vlm_candidates"]), 5)
        self.assertGreaterEqual(len(candidates["llm_candidates"]), 5)
        self.assertGreaterEqual(len(candidates["image_generator_candidates"]), 5)

    def test_factorial_has_at_least_one_hundred_combinations(self):
        candidates = yaml.safe_load((ROOT / "configs/open_best_candidates.yaml").read_text())
        factorial = candidates["combination_search"]["primary_factorial"]
        calculated = factorial["vlm_count"] * factorial["llm_count"] * factorial["image_generator_count"]
        self.assertEqual(factorial["combinations"], calculated)
        self.assertGreaterEqual(calculated, 100)
        payload, rows = build(ROOT / "configs/open_best_candidates.yaml")
        self.assertEqual(len(rows), calculated)
        self.assertEqual(
            payload,
            (ROOT / "configs/open_best_combinations.jsonl").read_bytes(),
        )

    def test_every_combination_has_two_hundred_cases_per_method(self):
        protocol = yaml.safe_load((ROOT / "configs/open_best_protocol.yaml").read_text())
        self.assertEqual(protocol["benchmark"]["groups_per_method"], 200)
        self.assertEqual(protocol["benchmark"]["outer_folds"], 5)
        search = protocol["combination_search"]
        self.assertEqual(search["groups_per_method_per_combination"], 200)
        self.assertEqual(search["planned_case_evaluations"], 125000)
        self.assertEqual(search["ranking_scope"], "fixed_benchmark_comparison_not_external_test")

    def test_open_best_revisions_are_asset_locked(self):
        candidates = yaml.safe_load((ROOT / "configs/open_best_candidates.yaml").read_text())
        lock = json.loads((ROOT / "configs/assets.lock.json").read_text())
        locked = {(item["repo_id"], item["revision"]) for item in lock["models"]}
        for family in ("vlm_candidates", "llm_candidates", "image_generator_candidates"):
            for model in candidates[family]:
                self.assertIn((model["repo_id"], model["revision"]), locked)

    def test_download_array_covers_every_open_best_asset(self):
        selected = assets(ROOT / "configs/assets.lock.json")
        self.assertEqual(len(selected), 16)
        self.assertEqual(len({(item["repo_id"], item["revision"]) for item in selected}), 16)


if __name__ == "__main__":
    unittest.main()

