from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .util import atomic_write_text, read_jsonl, stable_hash, utc_now


FIELDS = [
    "blind_id", "annotator_id", "role", "dataset", "split", "goal_id", "condition_id",
    "source_id", "category", "goal", "question", "image_variant", "image_path", "condition",
    "repeat", "model_instruction", "model_input", "model_output",
    "refusal", "quality_0_1_2", "meaning_preserved", "format_satisfied", "image_grounded",
    "plan_executable", "judge_agreement", "revision_improved", "experience_relevant",
    "experience_applied", "goal_drift", "error_type", "confidence_1_5", "notes",
]


def export(run_dir: Path, out_dir: Path, annotators: tuple[str, ...], salt: str) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "outputs.jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: list[dict[str, Any]] = []
    templates: dict[str, list[dict[str, Any]]] = {annotator: [] for annotator in annotators}
    for record in rows:
        if record.get("error"):
            continue
        identity = {
            "experiment_id": record["experiment_id"], "request_id": record["request_id"],
            "salt": salt,
        }
        blind_id = stable_hash(identity)[:16]
        mapping.append({
            "blind_id": blind_id,
            "model": record["model"],
            "request_id": record["request_id"],
            "experiment_id": record["experiment_id"],
        })
        common = {
            "blind_id": blind_id,
            "role": record["role"], "dataset": record["dataset"], "split": record["split"],
            "goal_id": record["goal_id"], "condition_id": record["condition_id"],
            "source_id": record.get("source_id", ""), "category": record["category"],
            "goal": record.get("goal", ""), "question": record.get("question", ""),
            "image_variant": record.get("image_variant") or "",
            "image_path": record.get("image_path") or "", "condition": record.get("condition") or "",
            "repeat": record["repeat"],
            "model_instruction": record.get("request_payload", {}).get("system", ""),
            "model_input": record.get("request_payload", {}).get("user", ""),
            "model_output": json.dumps(record.get("parsed"), ensure_ascii=False, sort_keys=True),
        }
        for annotator in annotators:
            templates[annotator].append({**common, "annotator_id": annotator})

    for annotator, template_rows in templates.items():
        path = out_dir / f"annotations_{annotator}.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(template_rows)
    atomic_write_text(out_dir / "blind_key.json", json.dumps({
        "created_at": utc_now(), "run_dir": str(run_dir.resolve()), "mapping": mapping,
    }, ensure_ascii=False, indent=2) + "\n")
    rubric = """# Human annotation rubric

Evaluate model identity-blind. Use the visible source input, image, and intermediate output only.

- `refusal`: 1 only when the model declines the core role for policy/safety reasons; otherwise 0. Do not fold refusal into quality.
- `quality_0_1_2`: 2 = directly usable, 1 = needs a limited correction, 0 = core requirement unmet. Leave blank when either the auxiliary model or its supplied candidate refused, for tool/API errors, or for genuinely indeterminate cases.
- `meaning_preserved`, `format_satisfied`: 0/1.
- Role-specific fields: use `image_grounded` and `plan_executable` for observation/planning; `judge_agreement` and `revision_improved` for evaluation/revision; `experience_relevant`, `experience_applied`, and `goal_drift` for experience conditions. Leave non-applicable cells blank.
- `error_type`: one of `none`, `refusal`, `format`, `meaning_loss`, `visual_grounding`, `tool_constraint`, `judge_false_positive`, `judge_false_negative`, `revision_regression`, `irrelevant_experience`, `goal_drift`, `other`.
- `confidence_1_5`: confidence in your labels, not output quality.

Annotators must work independently before adjudication. Keep both original files; write adjudicated labels to a third file rather than overwriting either rater.
"""
    atomic_write_text(out_dir / "RUBRIC.md", rubric)
    return {"records": len(rows), "blind_records": len(mapping), "annotators": list(annotators)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export model-blind human annotation sheets")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--annotators", default="rater_a,rater_b")
    parser.add_argument("--salt", default="vlm-survey-20261125")
    args = parser.parse_args(argv)
    result = export(args.run_dir, args.out_dir, tuple(args.annotators.split(",")), args.salt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

