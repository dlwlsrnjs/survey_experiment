from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import yaml


FAMILIES = {
    "vlm": ("vlm_candidates", "vlms"),
    "llm": ("llm_candidates", "llms"),
    "image_generator": ("image_generator_candidates", "image_generators"),
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build(config_path: Path, factorial_key: str = "primary_factorial") -> tuple[bytes, list[dict[str, Any]]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    factorial = config["combination_search"][factorial_key]
    selected: dict[str, list[dict[str, Any]]] = {}
    for role, (candidate_key, selection_key) in FAMILIES.items():
        registry = {item["repo_id"]: item for item in config[candidate_key]}
        names = factorial[selection_key]
        missing = [name for name in names if name not in registry]
        if missing:
            raise ValueError(f"Unknown {role} candidates: {missing}")
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate {role} candidates in primary factorial")
        expected = int(factorial[f"{role}_count"])
        if len(names) != expected:
            raise ValueError(f"{role} count is {len(names)}, expected {expected}")
        selected[role] = [registry[name] for name in names]

    rows: list[dict[str, Any]] = []
    for vlm, llm, image_generator in product(
        selected["vlm"], selected["llm"], selected["image_generator"]
    ):
        identity = {
            "vlm": {"repo_id": vlm["repo_id"], "revision": vlm["revision"]},
            "llm": {"repo_id": llm["repo_id"], "revision": llm["revision"]},
            "image_generator": {
                "repo_id": image_generator["repo_id"],
                "revision": image_generator["revision"],
            },
        }
        combination_id = "ob-" + _sha256_bytes(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        )[:12]
        row = {"combination_id": combination_id, **identity}
        if factorial_key == "alignment_ablation_factorial":
            row["alignment_status"] = {
                "vlm": "abliterated" if vlm.get("alignment_status") == "abliterated" else "aligned",
                "llm": "abliterated" if llm.get("alignment_status") == "abliterated" else "aligned",
            }
            row["paired_base_family"] = {
                "vlm": vlm.get("base_model", vlm["repo_id"]),
                "llm": llm.get("base_model", llm["repo_id"]),
            }
        rows.append(row)
    if len(rows) != int(factorial["combinations"]):
        raise ValueError(f"Generated {len(rows)} combinations, expected {factorial['combinations']}")
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in rows
    )
    return payload, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the revision-pinned open-best factorial matrix")
    parser.add_argument("--config", type=Path, default=Path("configs/open_best_candidates.yaml"))
    parser.add_argument("--output", type=Path, default=Path("configs/open_best_combinations.jsonl"))
    parser.add_argument("--factorial-key", default="primary_factorial")
    args = parser.parse_args()
    payload, rows = build(args.config.resolve(), args.factorial_key)
    args.output.write_bytes(payload)
    digest = _sha256_bytes(payload)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"combinations": len(rows), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
