from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .hades_released_eval import LLAVA_ROOT, ROOT, _llava_response
from .util import append_jsonl, atomic_write_text, read_jsonl, sha256_file, utc_now


def run(args: argparse.Namespace) -> None:
    if str(LLAVA_ROOT) not in sys.path:
        sys.path.insert(0, str(LLAVA_ROOT))
    from llava.model.builder import load_pretrained_model

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = output_dir / "outputs.jsonl"
    events = output_dir / "events.jsonl"
    rows = read_jsonl(args.manifest.resolve())[: args.limit]
    done = {row["condition_id"] for row in read_jsonl(outputs)} if outputs.exists() else set()
    provenance = {
        "protocol": "figstep_source_eval_v1", "started_at": utc_now(),
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256_file(args.manifest.resolve()),
        "rows": len(rows),
        "target": {"id": "liuhaotian/llava-v1.5-7b", "revision": args.llava_revision},
        "vision_tower": {"id": "openai/clip-vit-large-patch14-336", "revision": args.clip_revision},
        "evaluation": "Paper-primary manual review; no automatic ASR is reported by this runner.",
        "generation_settings": {"do_sample": True, "temperature": 0.2, "max_new_tokens": 1024},
    }
    atomic_write_text(output_dir / "provenance.json", json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(args.llava_path.resolve()), None, "llava-v1.5-7b", device_map="auto",
    )
    model.eval()
    for index, row in enumerate(rows, 1):
        if row["condition_id"] in done:
            continue
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        response = _llava_response(tokenizer, model, image_processor, image_path, row["text_prompt"])
        record = {
            **row, "timestamp": utc_now(), "response": response,
            "target_model": "liuhaotian/llava-v1.5-7b", "manual_unsafe_label": None,
            "human_review_status": "pending",
        }
        append_jsonl(outputs, record)
        append_jsonl(events, {"timestamp": utc_now(), "event": "target_response", "condition_id": row["condition_id"], "index": index, "total": len(rows)})

    records = read_jsonl(outputs)
    annotation_path = output_dir / "manual_review.csv"
    if not annotation_path.exists():
        with annotation_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("condition_id", "category", "question", "response", "unsafe_label", "notes"))
            writer.writeheader()
            for record in records:
                writer.writerow({
                    "condition_id": record["condition_id"], "category": record["category"],
                    "question": record["question"], "response": record["response"],
                    "unsafe_label": "", "notes": "",
                })
    summary = {
        "completed_at": utc_now(), "records": len(records), "planned": len(rows),
        "responses_complete": len(records) == len(rows), "manual_review_complete": False,
        "asr": None,
    }
    atomic_write_text(output_dir / "completed.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run source FigStep images and prompt against original LLaVA target")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--llava-path", type=Path, required=True)
    parser.add_argument("--llava-revision", default="4481d270cc22fd5c4d1bb5df129622006ccd9234")
    parser.add_argument("--clip-revision", default="ce19dc912ca5cd21c8a653c79e251e808ccabcd1")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
