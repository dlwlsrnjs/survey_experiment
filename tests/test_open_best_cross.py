import json
import tempfile
import unittest
from pathlib import Path

from survey_exp.open_best_cross_judge_eval import _parse
from survey_exp.open_best_cross_image_eval import load_prompts
from survey_exp.open_best_cross_vlm_eval import load_cases


class OpenBestCrossTest(unittest.TestCase):
    def test_judge_parsers_keep_success_and_refusal_separate(self):
        self.assertTrue(_parse("Yes", "harmbench")["attack_success"])
        parsed = _parse("Safety: Safe\nCategories: None\nRefusal: Yes", "qwen_guard")
        self.assertEqual(parsed["safety_label"], "Safe")
        self.assertEqual(parsed["refusal_label"], "Yes")
        self.assertTrue(parsed["parse_valid"])

    def test_cross_vlm_loader_retains_upstream_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            row = {
                "cross_id": "llm:HADES:item-1", "method": "HADES",
                "condition_id": "item-1", "goal_id": "goal-1", "goal": "goal",
                "status": "generation_error", "artifact_path": None,
            }
            (run_dir / "outputs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            cases = load_cases([f"image={run_dir}"], per_image_model=1)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["status"], "generation_error")
            self.assertEqual(cases[0]["combination_case_id"], "image:llm:HADES:item-1")

    def test_missing_hades_diffusion_prompt_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            rows = [
                {"method": "HADES", "role": "caption_generation", "condition_id": "h1",
                 "goal_id": "g1", "goal": "goal", "parsed_output": None, "refusal_detected": True},
                {"method": "HIMRD", "role": "scene_generation", "condition_id": "m1",
                 "goal_id": "g2", "goal": "goal", "parsed_output": "scene", "refusal_detected": False},
                {"method": "HIMRD", "role": "prefix_rewrite", "condition_id": "m1",
                 "parsed_output": "prefix", "refusal_detected": False},
                {"method": "HIMRD", "role": "suffix_rewrite", "condition_id": "m1",
                 "parsed_output": "suffix", "refusal_detected": False},
            ]
            (run_dir / "outputs.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            prompts = load_prompts([f"llm={run_dir}"], per_method=1)
            hades = next(row for row in prompts if row["method"] == "HADES")
            self.assertFalse(hades["upstream_usable"])
            self.assertTrue(hades["upstream_refusal"])


if __name__ == "__main__":
    unittest.main()
