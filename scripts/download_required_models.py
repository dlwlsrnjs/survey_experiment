from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = [
    # HADES: exact diffusion generator and image captioner used by black_box/optimize_harm.py.
    ("PixArt-alpha/PixArt-XL-2-1024-MS", "b89adadeccd9ead2adcb9fa2825d3fabec48d404", "HADES:image_generator"),
    ("liuhaotian/llava-v1.5-7b", "4481d270cc22fd5c4d1bb5df129622006ccd9234", "HADES:image_captioner_and_target"),
    ("PKU-Alignment/beaver-dam-7b", "3a5b70888869f04336d0d8344185ed74268ef896", "HADES:final_safety_evaluator"),
    # HIMRD: exact image generator. This repository is gated and may require an accepted license.
    ("stabilityai/stable-diffusion-3-medium-diffusers", "ea42f8cef0f178587cf766dc8129abd379c90671", "HIMRD:image_generator"),
    ("llava-hf/llava-v1.6-mistral-7b-hf", "2424fdd47412fccc66d91719126b420e9fbd7065", "HIMRD:victim_vlm"),
    ("THUDM/glm-4v-9b", "3376fea6e54db68587a89bf1ac27a6889bafb867", "HIMRD:victim_vlm"),
    ("Qwen/Qwen-VL-Chat", "f57cfbd358cb56b710d963669ad1bcfb44cdcdd8", "HIMRD:victim_vlm"),
    ("deepseek-ai/deepseek-vl-7b-chat", "6f16f00805f45b5249f709ce21820122eeb43556", "HIMRD:victim_vlm"),
    ("01-ai/Yi-VL-34B", "2bd12c3b988c443b34b3c8b0355a01548aa2e33f", "HIMRD:victim_vlm"),
    # MiniGPT-4's HF repository contains code/config only; its LLM/vision weights require a separate upstream acquisition.
    ("Vision-CAIR/MiniGPT-4", "e427c1915f6e23c054279dc42d3a743907e6ec47", "HIMRD:minigpt4_code_and_config"),
    # MM-Plan original base planner plus the fixed local target used in controlled comparisons.
    ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17", "MM-Plan:original_base_planner"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5", "controlled_comparison:fixed_target_vlm"),
    # MemJack retrieval embedding model.
    ("Qwen/Qwen3-VL-Embedding-8B", "2c4565515e0f265c6511776e7193b22c0968ddc7", "MemJack:embedding"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    args.status.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for repo_id, revision, role in MODELS:
        started = time.time()
        item = {"repo_id": repo_id, "revision": revision, "role": role, "started_at_epoch": started}
        try:
            path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=str(args.cache_dir) if args.cache_dir else None)
            item.update(status="complete", snapshot_path=path, elapsed_seconds=time.time() - started)
        except Exception as exc:
            item.update(status="error", error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.time() - started)
        results.append(item)
        args.status.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if all(x["status"] == "complete" for x in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
