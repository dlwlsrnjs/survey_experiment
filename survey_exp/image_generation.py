from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .util import sha256_file


def load_profiles(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Image generator config must be a mapping: {path}")
    return value


def select_profile(path: Path, selector: str) -> dict[str, Any]:
    value: Any = load_profiles(path)
    for part in selector.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Unknown image generation profile: {selector}")
        value = value[part]
    if not isinstance(value, dict) or "model" not in value:
        raise ValueError(f"Incomplete image generation profile: {selector}")
    return value


@lru_cache(maxsize=3)
def _pipeline(config_json: str):
    import diffusers
    import torch

    config = json.loads(config_json)
    pipeline_class = getattr(diffusers, config["pipeline"])
    dtype = getattr(torch, config.get("torch_dtype", "float16"))
    pipe = pipeline_class.from_pretrained(
        config["model"], revision=config.get("revision"), torch_dtype=dtype,
        use_safetensors=True, **(config.get("load_kwargs") or {}),
    )
    if config.get("enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
        return pipe
    return pipe.to("cuda")


def generate_image(prompt: str, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for configured image generation")
    stable_config = json.dumps(config, ensure_ascii=False, sort_keys=True)
    pipe = _pipeline(stable_config)
    seed = int(config.get("seed", 20261125))
    kwargs: dict[str, Any] = {"prompt": prompt, "generator": torch.Generator(device="cuda").manual_seed(seed)}
    if not config.get("use_pipeline_defaults"):
        for key in ("num_inference_steps", "guidance_scale", "height", "width", "negative_prompt"):
            if config.get(key) is not None:
                kwargs[key] = config[key]
    image = pipe(**kwargs).images[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return {
        "status": "ok", "tool": "revision_pinned_text_to_image_v1",
        "model": config["model"], "revision": config.get("revision"),
        "pipeline": config["pipeline"], "seed": seed,
        "settings": {key: config.get(key) for key in ("num_inference_steps", "guidance_scale", "height", "width")},
        "artifact": str(output.resolve()), "sha256": sha256_file(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one image with a locked experiment profile")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", required=True, help="controlled_default, source_fidelity.hades, or source_fidelity.himrd")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate_image(args.prompt, args.output, select_profile(args.config, args.profile)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
