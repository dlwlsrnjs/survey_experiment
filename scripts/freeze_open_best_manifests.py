from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _stable_group_order(method: str, groups: list[str], seed: int) -> list[str]:
    return sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}:{method}:{value}".encode()).hexdigest(),
    )


def freeze(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent
    output_root = (config_dir / config["output_root"]).resolve()
    required = set(config["required_fields"])
    benchmark_groups = int(config["benchmark"]["groups_per_method"])
    outer_folds = int(config["benchmark"]["outer_folds"])
    if outer_folds < 2:
        raise ValueError("outer_folds must be at least 2")
    seed = int(config["freeze_seed"])

    all_split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summary: dict[str, Any] = {
        "protocol_version": config["protocol_version"],
        "freeze_seed": seed,
        "config_sha256": _sha256(config_path),
        "benchmark": {
            "groups_per_method": benchmark_groups,
            "outer_folds": outer_folds,
            "fold_assignment": config["benchmark"]["fold_assignment"],
        },
        "methods": {},
    }

    for method, method_config in config["methods"].items():
        pool_path = (config_dir / method_config["pool"]).resolve()
        if not pool_path.is_file():
            raise FileNotFoundError(f"Required frozen pool is missing for {method}: {pool_path}")
        rows = _read_jsonl(pool_path)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        item_ids: set[str] = set()
        for index, row in enumerate(rows, 1):
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(f"{pool_path}:{index}: missing fields {missing}")
            if row["method"] != method:
                raise ValueError(f"{pool_path}:{index}: method must be {method!r}")
            item_id = str(row["item_id"])
            if item_id in item_ids:
                raise ValueError(f"{pool_path}:{index}: duplicate item_id {item_id!r}")
            item_ids.add(item_id)
            grouped[str(row[config["group_key"]])].append(row)
        if len(grouped) < benchmark_groups:
            raise ValueError(
                f"{method} has {len(grouped)} unique groups; {benchmark_groups} are required "
                "for the fixed benchmark"
            )

        ordered = _stable_group_order(method, list(grouped), seed)[:benchmark_groups]
        method_summary = {
            "pool": str(pool_path),
            "pool_sha256": _sha256(pool_path),
            "pool_rows": len(rows),
            "pool_groups": len(grouped),
            "selected_groups": len(ordered),
            "fold_groups": {str(fold): 0 for fold in range(outer_folds)},
        }
        selected_rows = []
        for position, group_id in enumerate(ordered):
            fold = position % outer_folds
            method_summary["fold_groups"][str(fold)] += 1
            for row in grouped[group_id]:
                selected_rows.append({**row, "outer_fold": fold})
        all_split_rows["benchmark"].extend(selected_rows)
        method_summary["selected_rows"] = len(selected_rows)
        summary["methods"][method] = method_summary

    output_root.mkdir(parents=True, exist_ok=True)
    for split_name, rows in all_split_rows.items():
        output = output_root / f"{split_name}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary.setdefault("outputs", {})[split_name] = {
            "path": str(output),
            "rows": len(rows),
            "sha256": _sha256(output),
        }

    summary_path = output_root / "freeze_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the 200-group open-best benchmark and its fixed folds")
    parser.add_argument("--config", type=Path, default=Path("configs/open_best_protocol.yaml"))
    args = parser.parse_args()
    print(json.dumps(freeze(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

