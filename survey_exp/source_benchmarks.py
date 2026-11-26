from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .paths import portable_path
from .util import sha256_file, utc_now, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def build_himrd_safebench(root: Path, output: Path) -> dict[str, Any]:
    targets_path = root / "third_party/HIMRD-jailbreak/data/SafeBench/SafeBench.txt"
    figstep_path = root / "third_party/FigStep/data/question/safebench.csv"
    targets = [line.strip() for line in targets_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with figstep_path.open(encoding="utf-8") as handle:
        categories = list(csv.DictReader(handle))
    if len(targets) < 200:
        raise ValueError(f"HIMRD SafeBench has fewer than 200 targets: {len(targets)}")
    if len(categories) < len(targets):
        raise ValueError("FigStep SafeBench category metadata is shorter than HIMRD SafeBench")

    rows = []
    for index, target in enumerate(targets):
        meta = categories[index]
        source_id = str(index + 1)
        rows.append({
            "manifest_version": "source-fidelity-1.0",
            "dataset": "HIMRD-SafeBench",
            "split": "eval",
            "goal_id": f"himrd-safebench:{source_id}",
            "condition_id": f"himrd-safebench:{source_id}",
            "source_id": source_id,
            "category": meta["category_name"],
            "goal": target,
            "question": target,
            "image_variant": None,
            "image_path": None,
            "source_file": portable_path(targets_path, root),
            "category_source_file": portable_path(figstep_path, root),
        })
    write_jsonl(output, rows)
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    summary = {
        "created_at": utc_now(), "dataset": "HIMRD-SafeBench", "rows": len(rows),
        "goal_units": len(rows), "sha256": digest,
        "source": portable_path(targets_path, root),
        "category_source": portable_path(figstep_path, root),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build manifests from the benchmark files shipped by source implementations")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/himrd_safebench_full.jsonl")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build_himrd_safebench(root, output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
