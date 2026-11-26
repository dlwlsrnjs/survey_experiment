import tempfile
import unittest
from pathlib import Path

from survey_exp.paths import portable_path, resolve_project_path


class PathTests(unittest.TestCase):
    def test_relative_path_uses_project_root(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            self.assertEqual(resolve_project_path("data/example.json", root), root / "data/example.json")

    def test_missing_legacy_absolute_path_is_remapped(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            old = "/home/old/project/data/raw/example.jpg"
            self.assertEqual(resolve_project_path(old, root), root / "data/raw/example.jpg")

    def test_portable_path_is_relative_inside_project(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            self.assertEqual(portable_path(root / "data/file.json", root), "data/file.json")


if __name__ == "__main__":
    unittest.main()
