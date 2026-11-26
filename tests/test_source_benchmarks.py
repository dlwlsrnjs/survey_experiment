import json
import tempfile
import unittest
from pathlib import Path

from survey_exp.source_benchmarks import build_himrd_safebench


ROOT = Path(__file__).resolve().parents[1]


class SourceBenchmarkTests(unittest.TestCase):
    def test_himrd_manifest_uses_all_source_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "himrd.jsonl"
            summary = build_himrd_safebench(ROOT, output)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            source_targets = [
                line.strip()
                for line in (ROOT / "third_party/HIMRD-jailbreak/data/SafeBench/SafeBench.txt")
                .read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(summary["rows"], 350)
            self.assertEqual(len(rows), len(source_targets))
            self.assertEqual(len({row["goal_id"] for row in rows}), 350)
            self.assertEqual([row["goal"] for row in rows], source_targets)
            self.assertTrue(output.with_suffix(".jsonl.sha256").is_file())


if __name__ == "__main__":
    unittest.main()
