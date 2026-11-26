from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .util import atomic_write_text, read_jsonl, stable_hash, utc_now


def build_bundle(run_dir: Path, adjudicated_csv: Path, output: Path) -> dict[str, Any]:
    with adjudicated_csv.open(encoding="utf-8-sig") as handle:
        labels = list(csv.DictReader(handle))
    by_blind = {row["blind_id"]: row for row in labels if row.get("error_type")}
    key = json.loads((adjudicated_csv.parent / "blind_key.json").read_text(encoding="utf-8"))
    blind_for_request = {item["request_id"]: item["blind_id"] for item in key["mapping"]}
    outputs = read_jsonl(run_dir / "outputs.jsonl")
    labeled = []
    for record in outputs:
        blind_id = blind_for_request.get(record["request_id"])
        if blind_id in by_blind:
            labeled.append((record, by_blind[blind_id]))
    frequencies = Counter(label["error_type"] for _, label in labeled)
    by_role: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for item in labeled:
        by_role[item[0]["role"]].append(item)

    selected = []
    for role in ("generation", "observation_planning", "evaluation_revision", "experience_use"):
        candidates = sorted(
            by_role.get(role, []),
            key=lambda item: (
                frequencies[item[1]["error_type"]],
                item[1].get("quality_0_1_2", ""),
                stable_hash(item[0]["request_id"]),
            ),
        )
        used_errors: set[str] = set()
        chosen = []
        for item in candidates:
            error = item[1]["error_type"]
            if error not in used_errors or not chosen:
                chosen.append(item)
                used_errors.add(error)
            if len(chosen) == 2:
                break
        selected.extend(chosen)

    bundle = {
        "created_at": utc_now(),
        "selection_rule": "two per role after full labeling; prefer distinct predeclared error types, then rarer types; never rank by success",
        "error_type_frequencies": dict(frequencies),
        "cases": [
            {
                "case_id": f"case-{index + 1:02d}",
                "role": record["role"], "goal_id": record["goal_id"],
                "condition_id": record["condition_id"], "image_path": record.get("image_path"),
                "source_id": record.get("source_id"), "goal": record.get("goal"),
                "question": record.get("question"), "model_request": record.get("request_payload"),
                "model_output": record.get("parsed"), "automatic_validation": record["automatic_validation"],
                "human_label": label,
            }
            for index, (record, label) in enumerate(selected)
        ],
    }
    atomic_write_text(output, json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    return {"cases": len(selected), "output": str(output.resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select eight contrastive qualitative cases after labeling")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_bundle(args.run_dir, args.adjudicated, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

