from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from survey_exp.image_generation import select_profile


ROOT = Path(__file__).resolve().parents[1]


class ImageGenerationProfileTests(unittest.TestCase):
    def test_all_profiles_are_revision_locked_models(self) -> None:
        lock = json.loads((ROOT / "configs/assets.lock.json").read_text(encoding="utf-8"))
        locked = {(item["repo_id"], item["revision"]) for item in lock["models"]}
        for selector in ("controlled_default", "source_fidelity.hades", "source_fidelity.himrd"):
            profile = select_profile(ROOT / "configs/image_generators.yaml", selector)
            self.assertIn((profile["model"], profile["revision"]), locked)

    def test_unknown_profile_fails_closed(self) -> None:
        with self.assertRaises(KeyError):
            select_profile(ROOT / "configs/image_generators.yaml", "source_fidelity.unknown")

    def test_incomplete_profile_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.yaml"
            path.write_text("broken:\n  pipeline: Example\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                select_profile(path, "broken")


if __name__ == "__main__":
    unittest.main()
