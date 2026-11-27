from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

from .util import append_jsonl, atomic_write_text, read_jsonl, sha256_file, utc_now


def run(args: argparse.Namespace) -> Path:
    import torch
    from diffusers import DiffusionPipeline

    output_dir = args.output_dir.resolve()
    artifact_dir = output_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    outputs = output_dir / "outputs.jsonl"
    rows = read_jsonl(args.manifest.resolve())[: args.limit]
    if len(rows) != args.limit:
        raise ValueError(f"expected {args.limit} prompts, found {len(rows)}")
    completed = {row["item_id"] for row in read_jsonl(outputs)} if outputs.exists() else set()
    provenance = {
        "protocol": "open_best_image_generator_comparison_v1",
        "scope": "fixed_component_comparison_not_source_fidelity_or_global_winner",
        "started_at": utc_now(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "model": {"repo_id": args.model, "revision": args.revision},
        "settings": {
            "width": args.width, "height": args.height,
            "num_inference_steps": args.steps, "guidance_scale": args.guidance,
            "base_seed": args.seed, "dtype": "bfloat16",
        },
    }
    atomic_write_text(output_dir / "provenance.json", json.dumps(provenance, indent=2) + "\n")
    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    parameters = inspect.signature(pipe.__call__).parameters

    for index, row in enumerate(rows):
        if row["item_id"] in completed:
            continue
        seed = args.seed + index
        artifact = artifact_dir / f"{row['item_id'].replace(':', '__')}.png"
        call = {
            "prompt": row["prompt"],
            "num_inference_steps": args.steps,
            "generator": torch.Generator(device="cuda").manual_seed(seed),
        }
        for key, value in (
            ("width", args.width), ("height", args.height),
            ("guidance_scale", args.guidance), ("true_cfg_scale", args.guidance),
        ):
            if key in parameters:
                call[key] = value
        started = time.monotonic()
        error = None
        try:
            image = pipe(**call).images[0]
            image.save(artifact)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        record = {
            **row,
            "timestamp": utc_now(),
            "model": args.model,
            "revision": args.revision,
            "seed": seed,
            "settings": {key: value for key, value in call.items() if key not in {"prompt", "generator"}},
            "latency_seconds": time.monotonic() - started,
            "artifact_path": str(artifact) if artifact.is_file() else None,
            "artifact_sha256": sha256_file(artifact) if artifact.is_file() else None,
            "error": error,
            "human_review_status": "pending",
        }
        append_jsonl(outputs, record)

    records = read_jsonl(outputs)
    summary = {
        "completed_at": utc_now(),
        "records": len(records),
        "planned": len(rows),
        "generated": sum(row.get("artifact_path") is not None for row in records),
        "errors": sum(row.get("error") is not None for row in records),
        "human_review_complete": False,
        "complete": len(records) == len(rows),
    }
    atomic_write_text(output_dir / "completed.json", json.dumps(summary, indent=2) + "\n")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a fixed 200-prompt image-model comparison")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20261127)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
