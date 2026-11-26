import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from survey_exp.dataset import build_manifest
from survey_exp.util import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_manifest_counts_and_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.jsonl"
            summary = build_manifest(ROOT, output, 20261125)
            self.assertEqual(summary["counts"], {
                "HarmBench-Standard:dev": 20,
                "HarmBench-Standard:eval": 50,
                "MM-SafetyBench:dev": 20,
                "MM-SafetyBench:eval": 50,
            })
            self.assertEqual(summary["missing_images"], 0)
            rows = read_jsonl(output)
            by_goal = defaultdict(list)
            for row in rows:
                by_goal[row["goal_id"]].append(row)
            for goal_rows in by_goal.values():
                self.assertEqual(len({row["split"] for row in goal_rows}), 1)
                if goal_rows[0]["dataset"] == "MM-SafetyBench":
                    self.assertEqual({row["image_variant"] for row in goal_rows}, {"SD", "SD_TYPO"})
            trace_rows = by_goal["mm-safetybench:06-Fraud:6"]
            self.assertTrue(all(row["pilot"] and row["split"] == "dev" for row in trace_rows))

    def test_no_goal_leakage_between_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.jsonl"
            build_manifest(ROOT, output, 20261125)
            rows = read_jsonl(output)
            dev = {row["goal_id"] for row in rows if row["split"] == "dev"}
            evaluation = {row["goal_id"] for row in rows if row["split"] == "eval"}
            self.assertTrue(dev.isdisjoint(evaluation))

