from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from .util import sha256_file, stable_hash, utc_now, write_jsonl


def _ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_experience_bank(dataset_root: Path, output: Path, split: str = "train") -> dict[str, Any]:
    allowed = _ids(dataset_root / f"splits/{split}_ids.txt")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped_not_split = 0
    for record in _records(dataset_root / "rounds.jsonl"):
        image_name = str(record.get("image_name", ""))
        if image_name not in allowed:
            skipped_not_split += 1
            continue
        # The public corpus can mark an early round goal_achieved even when the final
        # trajectory is safe. Preserve round-level labels rather than recomputing them.
        status = "success" if record.get("goal_achieved") else "observed_failure"
        identity = stable_hash({
            "trajectory_id": record.get("trajectory_id"), "round": record.get("round"),
            "prompt": record.get("prompt"),
        })[:20]
        if identity in seen:
            continue
        seen.add(identity)
        rows.append({
            "experience_id": f"memjack:{identity}",
            "source": "MemJack-Bench",
            "source_split": split,
            "trajectory_id": record.get("trajectory_id"),
            "sample_id": record.get("sample_id"),
            "image_name": image_name,
            "source_corpus": record.get("source_corpus"),
            "victim_model": record.get("victim_model"),
            "round": record.get("round"),
            "goal": record.get("goal"),
            "anchor": record.get("anchor"),
            "output": record.get("prompt"),
            "strategy": record.get("strategy"),
            "attack_angle": record.get("attack_angle"),
            "defense_pattern": record.get("defense_pattern"),
            "status": status,
            "safety_label": record.get("safety_label"),
            "risk_score": record.get("risk_score"),
            "query_count_cum": record.get("query_count_cum"),
        })
    write_jsonl(output, rows)
    return {
        "created_at": utc_now(), "source": str(dataset_root.resolve()),
        "source_rounds_sha256": sha256_file(dataset_root / "rounds.jsonl"),
        "source_split_sha256": sha256_file(dataset_root / f"splits/{split}_ids.txt"),
        "split": split, "records": len(rows), "skipped_not_split": skipped_not_split,
        "output": str(output.resolve()), "output_sha256": sha256_file(output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a split-safe experience bank from MemJack-Bench")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/memjack"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/memjack_train_experience.jsonl"))
    parser.add_argument("--split", default="train", choices=("train", "val", "test", "ood"))
    args = parser.parse_args(argv)
    result = build_experience_bank(args.dataset_root, args.output, args.split)
    args.output.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

