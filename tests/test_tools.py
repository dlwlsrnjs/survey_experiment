import tempfile
import unittest
from pathlib import Path

from survey_exp.tools import render_typography


class ToolTests(unittest.TestCase):
    def test_renderer_writes_checksumed_png(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "render.png"
            result = render_typography("A short test statement", output)
            self.assertEqual(result["status"], "success")
            self.assertTrue(output.is_file())
            self.assertEqual(len(result["sha256"]), 64)

