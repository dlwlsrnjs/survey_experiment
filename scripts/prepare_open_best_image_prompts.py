from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze 200 HIMRD scene prompts for image-model comparison")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/open_best_image_prompts_200.jsonl"))
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    selected = []
    with args.source.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("method") == "HIMRD" and row.get("role") == "scene_generation":
                selected.append({
                    "item_id": row["condition_id"],
                    "group_id": row["goal_id"],
                    "method": "HIMRD",
                    "category": row.get("category"),
                    "goal": row.get("goal"),
                    "prompt": row["parsed_output"],
                    "source_model": row.get("model_settings", {}).get("model") or "gpt-4o-mini-2024-07-18",
                    "source_outputs_sha256": sha256(args.source),
                })
    selected = selected[: args.limit]
    if len(selected) != args.limit or len({row["group_id"] for row in selected}) != args.limit:
        raise ValueError(f"expected {args.limit} unique prompts, found {len(selected)}")
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(selected), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
