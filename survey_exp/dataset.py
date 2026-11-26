from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .util import sha256_file, stable_hash, utc_now, write_jsonl


MM_CATEGORIES = (
    "01-Illegal_Activitiy",
    "03-Malware_Generation",
    "04-Physical_Harm",
    "06-Fraud",
    "09-Privacy_Violence",
)


def _rank(rows: list[dict[str, Any]], seed: int, key: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: stable_hash({"seed": seed, "value": row[key]}))


def _balanced_take(groups: dict[str, list[dict[str, Any]]], count: int, seed: int) -> list[dict[str, Any]]:
    ranked = {name: _rank(rows, seed, "goal_id") for name, rows in sorted(groups.items())}
    positions = {name: 0 for name in ranked}
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for name in sorted(ranked):
            pos = positions[name]
            if pos < len(ranked[name]):
                selected.append(ranked[name][pos])
                positions[name] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"Only {len(selected)} rows available for requested count={count}")
    return selected


def _harmbench_rows(root: Path, seed: int) -> list[dict[str, Any]]:
    source = root / "third_party/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    with source.open(encoding="utf-8") as handle:
        raw = [row for row in csv.DictReader(handle) if row["FunctionalCategory"] == "standard"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        groups[row["SemanticCategory"]].append({
            "goal_id": f"harmbench:{row['BehaviorID']}",
            "dataset": "HarmBench-Standard",
            "source_id": row["BehaviorID"],
            "category": row["SemanticCategory"],
            "goal": row["Behavior"],
            "question": row["Behavior"],
            "image_variant": None,
            "image_path": None,
            "source_file": str(source.resolve()),
        })
    chosen = _balanced_take(groups, 70, seed)
    for index, row in enumerate(chosen):
        row["split"] = "dev" if index < 20 else "eval"
        row["pilot"] = index < 10
    return chosen


def _find_mm_image_root(root: Path) -> Path:
    candidates = [
        root / "data/raw/mm_safetybench/MM-SafetyBench(imgs)",
        root / "data/raw/mm_safetybench/imgs",
        root / "third_party/MM-SafetyBench/data/imgs",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    for candidate in (root / "data/raw/mm_safetybench").glob("**/imgs"):
        if (candidate / MM_CATEGORIES[0]).is_dir():
            return candidate
    return candidates[0]


def _mm_rows(root: Path, seed: int) -> list[dict[str, Any]]:
    questions = root / "third_party/MM-SafetyBench/data/processed_questions"
    image_root = _find_mm_image_root(root)
    goals: list[dict[str, Any]] = []
    for category in MM_CATEGORIES:
        source = questions / f"{category}.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        category_goals = []
        for source_id, item in payload.items():
            category_goals.append({
                "goal_id": f"mm-safetybench:{category}:{source_id}",
                "dataset": "MM-SafetyBench",
                "source_id": source_id,
                "category": category,
                "goal": item["Question"],
                "key_phrase": item.get("Key Phrase"),
                "phrase_type": item.get("Phrase Type"),
                "source_file": str(source.resolve()),
                "questions": {
                    "SD": item["Rephrased Question(SD)"],
                    "SD_TYPO": item["Rephrased Question"],
                },
            })
        ranked_all = _rank(category_goals, seed, "goal_id")
        if category == "06-Fraud":
            # The experiment plan names public sample 06-Fraud/6 as the trace example.
            pinned = next(goal for goal in category_goals if goal["source_id"] == "6")
            ranked = [pinned] + [goal for goal in ranked_all if goal["source_id"] != "6"][:13]
        else:
            ranked = ranked_all[:14]
        for index, goal in enumerate(ranked):
            goal["split"] = "dev" if index < 4 else "eval"
            goal["pilot"] = index < 2
            goals.append(goal)

    rows: list[dict[str, Any]] = []
    for goal in goals:
        for variant in ("SD", "SD_TYPO"):
            image = image_root / goal["category"] / variant / f"{goal['source_id']}.jpg"
            row = {key: value for key, value in goal.items() if key != "questions"}
            row.update({
                "condition_id": f"{goal['goal_id']}:{variant}",
                "question": goal["questions"][variant],
                "image_variant": variant,
                "image_path": str(image.resolve()),
                "image_exists": image.is_file(),
            })
            rows.append(row)
    return rows


def build_manifest(root: Path, output: Path, seed: int) -> dict[str, Any]:
    rows = _harmbench_rows(root, seed) + _mm_rows(root, seed)
    for row in rows:
        row.setdefault("condition_id", row["goal_id"])
        row["manifest_version"] = "1.0"
        row["selection_seed"] = seed
    rows.sort(key=lambda row: (row["dataset"], row["goal_id"], str(row["image_variant"])))
    unique_goals = {row["goal_id"]: row for row in rows}
    dev_goals = [row for row in unique_goals.values() if row["split"] == "dev"]
    eval_goals = [row for row in unique_goals.values() if row["split"] == "eval"]
    overlap_review = []
    for left in dev_goals:
        left_tokens = set(re.findall(r"[a-z0-9]+", left["goal"].lower()))
        for right in eval_goals:
            if left["dataset"] != right["dataset"]:
                continue
            right_tokens = set(re.findall(r"[a-z0-9]+", right["goal"].lower()))
            union = left_tokens | right_tokens
            similarity = len(left_tokens & right_tokens) / len(union) if union else 0.0
            if similarity >= 0.90:
                overlap_review.append({
                    "dev_goal_id": left["goal_id"], "eval_goal_id": right["goal_id"],
                    "token_jaccard": similarity, "dev_goal": left["goal"], "eval_goal": right["goal"],
                })
    write_jsonl(output, rows)
    digest = sha256_file(output)
    (output.with_suffix(output.suffix + ".sha256")).write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    summary = {
        "created_at": utc_now(),
        "manifest": str(output.resolve()),
        "sha256": digest,
        "seed": seed,
        "rows": len(rows),
        "goal_units": len({row["goal_id"] for row in rows}),
        "missing_images": sum(1 for row in rows if row.get("image_path") and not row.get("image_exists")),
        "potential_semantic_overlaps_at_jaccard_0_90": len(overlap_review),
        "counts": {
            f"{dataset}:{split}": len({row["goal_id"] for row in rows if row["dataset"] == dataset and row["split"] == split})
            for dataset in ("HarmBench-Standard", "MM-SafetyBench")
            for split in ("dev", "eval")
        },
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".semantic_overlap_review.json").write_text(
        json.dumps(overlap_review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build leakage-safe deterministic experiment manifest")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/manifests/fixed_inputs.jsonl"))
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    print(json.dumps(build_manifest(root, output, args.seed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

