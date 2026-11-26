import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_figstep_manifest import FIGSTEP_TEXT_PROMPT, build


class FigStepManifestTests(unittest.TestCase):
    def test_prompt_is_source_prompt(self):
        self.assertIn("list numbered 1, 2, and 3", FIGSTEP_TEXT_PROMPT)
        self.assertIn("approximately 100 words", FIGSTEP_TEXT_PROMPT)

    def test_build_requires_balanced_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "safe.csv"
            csv_path.write_text("dataset,category_id,task_id,category_name,question,instruction\nForbidQI,1,1,Test,Q,I\n", encoding="utf-8")
            image_root = root / "images"
            image_root.mkdir()
            Image.new("RGB", (10, 10)).save(image_root / "query_ForbidQI_1_1_6.png")
            summary = build(csv_path, image_root, root / "manifest.jsonl", per_category=1)
            self.assertEqual(summary["rows"], 1)


if __name__ == "__main__":
    unittest.main()
