from __future__ import annotations

import argparse
import inspect
import json
import re
import time
from pathlib import Path
from typing import Any

from .util import append_jsonl, atomic_write_text, read_jsonl, sha256_file, utc_now


ROLE_BY_METHOD = {
    "HADES": "caption_generation",
    "HIMRD": "scene_generation",
}


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--llm-run must be TAG=PATH")
    tag, path = value.split("=", 1)
    return tag, Path(path).resolve()


def load_prompts(values: list[str], per_method: int) -> list[dict[str, Any]]:
    prompts = []
    for value in values:
        llm_tag, run_dir = _parse_run(value)
        output_path = run_dir / "outputs.jsonl"
        if not output_path.is_file():
            raise FileNotFoundError(f"Missing LLM outputs: {output_path}")
        records = read_jsonl(output_path)
        by_key = {
            (row.get("method"), row.get("role"), row.get("condition_id")): row
            for row in records
        }
        for method, role in ROLE_BY_METHOD.items():
            selected = [row for row in records if row.get("method") == method and row.get("role") == role]
            if len(selected) != per_method:
                raise ValueError(f"{llm_tag} {method}:{role} has {len(selected)} records; expected {per_method}")
            for row in selected:
                prompt_row = row
                if method == "HADES":
                    prompt_row = by_key.get(("HADES", "diffusion_prompt_generation", row["condition_id"]))
                prompt = prompt_row.get("parsed_output") if prompt_row else None
                prefix = None
                suffix = None
                target_prompt_usable = True
                if method == "HIMRD":
                    prefix_row = by_key.get(("HIMRD", "prefix_rewrite", row["condition_id"]))
                    suffix_row = by_key.get(("HIMRD", "suffix_rewrite", row["condition_id"]))
                    if prefix_row is None or suffix_row is None:
                        raise ValueError(f"{llm_tag} HIMRD target rewrites missing for {row['condition_id']}")
                    prefix = prefix_row.get("parsed_output")
                    suffix = suffix_row.get("parsed_output")
                    target_prompt_usable = bool(prefix and suffix)
                prompts.append({
                    "cross_id": f"{llm_tag}:{method}:{row['condition_id']}",
                    "llm_tag": llm_tag,
                    "llm_run": str(run_dir),
                    "llm_outputs_sha256": sha256_file(output_path),
                    "method": method,
                    "role": "diffusion_prompt_generation" if method == "HADES" else role,
                    "condition_id": row["condition_id"],
                    "goal_id": row["goal_id"],
                    "category": row.get("category"),
                    "goal": row.get("goal"),
                    "prompt": str(prompt) if prompt else None,
                    "upstream_usable": bool(prompt),
                    "upstream_refusal": bool(
                        row.get("refusal_detected")
                        or (prompt_row and prompt_row.get("refusal_detected"))
                    ),
                    "caption": row.get("parsed_output") if method == "HADES" else None,
                    "target_prefix": str(prefix) if prefix else None,
                    "target_suffix": str(suffix) if suffix else None,
                    "target_prompt_usable": target_prompt_usable,
                })
    return prompts


def run(args: argparse.Namespace) -> Path:
    import torch
    from diffusers import DiffusionPipeline

    output_dir = args.output_dir.resolve()
    artifact_dir = output_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    outputs = output_dir / "outputs.jsonl"
    prompts = load_prompts(args.llm_run, args.per_method)
    completed = {row["cross_id"] for row in read_jsonl(outputs)} if outputs.exists() else set()
    provenance = {
        "protocol": "open_best_llm_image_cross_v1",
        "started_at": utc_now(),
        "image_model": {"repo_id": args.model, "revision": args.revision},
        "llm_runs": args.llm_run,
        "planned": len(prompts),
        "settings": {"width": args.width, "height": args.height, "steps": args.steps, "guidance": args.guidance, "seed": args.seed},
    }
    provenance_path = output_dir / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        for key in ("image_model", "llm_runs", "planned", "settings"):
            if previous.get(key) != provenance.get(key):
                raise ValueError(f"Resume provenance mismatch: {key}")
    else:
        atomic_write_text(provenance_path, json.dumps(provenance, indent=2) + "\n")
    pipe = DiffusionPipeline.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")
    parameters = inspect.signature(pipe.__call__).parameters

    for index, item in enumerate(prompts):
        if item["cross_id"] in completed:
            continue
        artifact = artifact_dir / item["method"].lower() / item["llm_tag"] / f"{_safe(item['condition_id'])}.png"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        seed = args.seed + index
        error = None
        latency = 0.0
        if item["upstream_usable"]:
            call: dict[str, Any] = {
                "prompt": item["prompt"], "num_inference_steps": args.steps,
                "generator": torch.Generator(device="cuda").manual_seed(seed),
            }
            for key, value in (("width", args.width), ("height", args.height), ("guidance_scale", args.guidance), ("true_cfg_scale", args.guidance)):
                if key in parameters:
                    call[key] = value
            started = time.monotonic()
            for attempt in range(1, 3):
                try:
                    pipe(**call).images[0].save(artifact)
                    error = None
                    break
                except Exception as exc:
                    error = {"type": type(exc).__name__, "message": str(exc), "attempt": attempt}
            latency = time.monotonic() - started
        record = {
            **item,
            "timestamp": utc_now(), "image_model": args.model, "image_revision": args.revision,
            "seed": seed, "latency_seconds": latency,
            "artifact_path": str(artifact) if artifact.is_file() else None,
            "artifact_sha256": sha256_file(artifact) if artifact.is_file() else None,
            "error": error,
            "status": "generated" if artifact.is_file() else ("upstream_unusable" if not item["upstream_usable"] else "generation_error"),
        }
        append_jsonl(outputs, record)

    records = read_jsonl(outputs)
    summary = {
        "completed_at": utc_now(), "records": len(records), "planned": len(prompts),
        "generated": sum(row.get("status") == "generated" for row in records),
        "upstream_unusable": sum(row.get("status") == "upstream_unusable" for row in records),
        "generation_errors": sum(row.get("status") == "generation_error" for row in records),
        "complete": len(records) == len(prompts),
    }
    atomic_write_text(output_dir / "completed.json", json.dumps(summary, indent=2) + "\n")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross all fixed LLM role outputs with one image generator")
    parser.add_argument("--llm-run", action="append", required=True, help="TAG=RUN_DIR")
    parser.add_argument("--per-method", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20261127)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
