from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class AssetLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = json.loads((ROOT / "configs/assets.lock.json").read_text(encoding="utf-8"))

    def test_git_and_model_revisions_are_full_hashes(self) -> None:
        for item in [*self.lock["repositories"], *self.lock["models"]]:
            revision = item["revision"]
            self.assertEqual(len(revision), 40)
            int(revision, 16)

    def test_core_contains_32b_and_controlled_generator(self) -> None:
        core = {item["repo_id"] for item in self.lock["models"] if "core" in item["profiles"]}
        self.assertIn("Qwen/Qwen2.5-VL-32B-Instruct", core)
        self.assertIn("Qwen/Qwen3-VL-32B-Thinking", core)
        self.assertIn("Qwen/Qwen2.5-VL-7B-Instruct", core)
        self.assertIn("stabilityai/stable-diffusion-xl-base-1.0", core)

    def test_common_target_matches_asset_lock(self) -> None:
        config = yaml.safe_load((ROOT / "configs/common_target.yaml").read_text(encoding="utf-8"))
        model = next(item for item in self.lock["models"] if item["repo_id"] == config["model"]["id"])
        self.assertEqual(config["model"]["revision"], model["revision"])
        self.assertEqual(config["model"]["role"], "target_vlm")
        self.assertTrue(config["evaluation"]["judge_is_separate_from_target"])

    def test_sdxl_download_is_limited_to_fp16_diffusers_files(self) -> None:
        sdxl = next(item for item in self.lock["models"] if item["repo_id"] == "stabilityai/stable-diffusion-xl-base-1.0")
        patterns = sdxl["allow_patterns"]
        self.assertIn("unet/diffusion_pytorch_model.fp16.safetensors", patterns)
        self.assertNotIn("*", patterns)


if __name__ == "__main__":
    unittest.main()
