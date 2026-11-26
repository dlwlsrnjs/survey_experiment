#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from survey_exp.util import atomic_write_text, canonical_json, sha256_file


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


SCENARIOS = ("Animal", "Financial", "Privacy", "Self-Harm", "Violence")


def abstract_prompt(row: dict) -> str:
    mask = (
        "conduct the behavior in the image on"
        if row["category"] == "behavior"
        else f"the {row['category']} in the image"
    )
    return row["instruction"].replace(row["keywords"], mask)


def select_rows(rows: list[dict], goals_per_scenario: int) -> list[dict]:
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        base_id = row["id"].rsplit("_step", 1)[0]
        groups[row["scenario"]][base_id].append(row)
    selected: list[dict] = []
    for scenario in SCENARIOS:
        base_ids = sorted(groups[scenario])[:goals_per_scenario]
        if len(base_ids) != goals_per_scenario:
            raise RuntimeError(f"{scenario}: expected {goals_per_scenario} goals, found {len(base_ids)}")
        for base_id in base_ids:
            variants = sorted(groups[scenario][base_id], key=lambda value: int(value["step"]))
            if [int(value["step"]) for value in variants] != list(range(6)):
                raise RuntimeError(f"{base_id}: expected steps 0..5")
            selected.extend(variants)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a locked 200-goal HADES released-artifact evaluation set")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--goals-per-scenario", type=int, default=40)
    parser.add_argument("--dataset-revision", default="8c7388c566c8844a0732a2ad2156b72ae00c92c8")
    args = parser.parse_args()

    parquet_files = sorted((args.dataset_root / "data").glob("*.parquet"))
    if len(parquet_files) != 2:
        raise RuntimeError(f"expected two HADES parquet shards under {args.dataset_root / 'data'}")
    rows: list[dict] = []
    for path in parquet_files:
        rows.extend(pq.read_table(path).to_pylist())
    selected = select_rows(rows, args.goals_per_scenario)

    output_rows = []
    image_root = args.output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    for row in selected:
        image_bytes = row["image"]["bytes"]
        if not image_bytes:
            raise RuntimeError(f"missing embedded image: {row['id']}")
        image_path = image_root / f"{row['id']}.png"
        if not image_path.exists():
            Image.open(io.BytesIO(image_bytes)).convert("RGB").save(image_path)
        output_rows.append({
            "manifest_version": "hades-released-1.0",
            "dataset": "Monosail/HADES",
            "dataset_revision": args.dataset_revision,
            "source_shards": [path.name for path in parquet_files],
            "condition_id": f"hades:{row['id']}",
            "goal_id": f"hades:{row['id'].rsplit('_step', 1)[0]}",
            "source_id": row["id"],
            "scenario": row["scenario"],
            "keywords": row["keywords"],
            "step": int(row["step"]),
            "category": row["category"],
            "instruction": row["instruction"],
            "abstract_prompt": abstract_prompt(row),
            "image_path": portable_path(image_path),
            "image_sha256": sha256_file(image_path),
            "split": "eval",
        })

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(canonical_json(row) + "\n" for row in output_rows)
    atomic_write_text(args.manifest, text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    atomic_write_text(args.manifest.with_suffix(args.manifest.suffix + ".sha256"), f"{digest}  {args.manifest.name}\n")
    summary = {
        "dataset_revision": args.dataset_revision,
        "parquet_sha256": {path.name: sha256_file(path) for path in parquet_files},
        "goals": len({row["goal_id"] for row in output_rows}),
        "rows": len(output_rows),
        "steps": sorted({row["step"] for row in output_rows}),
        "scenarios": {scenario: len({row["goal_id"] for row in output_rows if row["scenario"] == scenario}) for scenario in SCENARIOS},
        "manifest_sha256": digest,
    }
    atomic_write_text(args.manifest.with_suffix(".summary.json"), json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
