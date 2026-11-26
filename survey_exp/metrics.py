from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .util import atomic_write_text, read_jsonl, utc_now


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize_outputs(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "outputs.jsonl")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["role"], row.get("condition") or "-")].append(row)
    summary = []
    for (model, role, condition), values in sorted(groups.items()):
        errors = sum(value.get("error") is not None for value in values)
        refusals = sum(bool((value.get("parsed") or {}).get("refusal")) for value in values if not value.get("error"))
        candidate_refusals = sum(bool((value.get("parsed") or {}).get("candidate_refused")) for value in values if not value.get("error"))
        candidate_quality = [
            int((value.get("parsed") or {})["candidate_quality"])
            for value in values
            if not value.get("error") and (value.get("parsed") or {}).get("candidate_quality") in (0, 1, 2)
        ]
        provided_experience_counts = []
        used_experience_counts = []
        unknown_experience_ids = 0
        used_feedback_count = 0
        if role == "experience_use":
            for value in values:
                if value.get("error") is not None:
                    continue
                try:
                    request_value = json.loads(value.get("request_payload", {}).get("user", "{}"))
                except json.JSONDecodeError:
                    request_value = {}
                provided_ids = {
                    str(item.get("experience_id"))
                    for item in request_value.get("retrieved_experiences", [])
                    if item.get("experience_id") is not None
                }
                used_ids = {str(item) for item in (value.get("parsed") or {}).get("used_experience_ids", [])}
                provided_experience_counts.append(len(provided_ids))
                used_experience_counts.append(len(used_ids))
                unknown_experience_ids += len(used_ids - provided_ids)
                used_feedback_count += int(bool((value.get("parsed") or {}).get("used_feedback")))
        valid = sum(bool(value["automatic_validation"]["format_valid"]) for value in values)
        latencies = [float(value["latency_seconds"]) for value in values if value.get("error") is None]
        input_tokens = sum(int(value.get("usage", {}).get("prompt_tokens", 0) or 0) for value in values)
        output_tokens = sum(int(value.get("usage", {}).get("completion_tokens", 0) or 0) for value in values)
        summary.append({
            "model": model, "role": role, "condition": condition, "n": len(values),
            "errors": errors, "refusals": refusals,
            "refusal_rate": refusals / (len(values) - errors) if len(values) > errors else None,
            "refusal_rate_ci95": _wilson(refusals, len(values) - errors),
            "candidate_refusals": candidate_refusals,
            "candidate_refusal_rate": candidate_refusals / (len(values) - errors) if len(values) > errors else None,
            "candidate_quality_distribution": dict(Counter(candidate_quality)),
            "mean_candidate_quality_nonrefusal": _mean(candidate_quality),
            "mean_provided_experience_count": _mean([float(value) for value in provided_experience_counts]),
            "mean_used_experience_count": _mean([float(value) for value in used_experience_counts]),
            "experience_application_rate": _mean([1.0 if value > 0 else 0.0 for value in used_experience_counts]),
            "used_feedback_rate": used_feedback_count / len(used_experience_counts) if used_experience_counts else None,
            "unknown_experience_ids": unknown_experience_ids if role == "experience_use" else None,
            "format_valid_rate": valid / len(values) if values else None,
            "mean_latency_seconds": _mean(latencies),
            "input_tokens": input_tokens, "output_tokens": output_tokens,
        })
    breakdown_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        breakdown_groups[(
            row["role"], row["dataset"], row.get("image_variant") or "-", row.get("category") or "-"
        )].append(row)
    breakdowns = []
    for (role, dataset, image_variant, category), values in sorted(breakdown_groups.items()):
        nonerrors = [value for value in values if value.get("error") is None]
        refusals = sum(bool((value.get("parsed") or {}).get("refusal")) for value in nonerrors)
        candidate_refusals = sum(bool((value.get("parsed") or {}).get("candidate_refused")) for value in nonerrors)
        breakdowns.append({
            "role": role,
            "dataset": dataset,
            "image_variant": image_variant,
            "category": category,
            "n": len(values),
            "errors": len(values) - len(nonerrors),
            "refusals": refusals,
            "refusal_rate": refusals / len(nonerrors) if nonerrors else None,
            "candidate_refusals": candidate_refusals,
            "candidate_refusal_rate": candidate_refusals / len(nonerrors) if nonerrors else None,
            "format_valid_rate": sum(bool(value["automatic_validation"]["format_valid"]) for value in values) / len(values),
        })

    paired = {}
    for role in ("observation_planning", "evaluation_revision"):
        by_goal: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            if row["role"] == role and row["dataset"] == "MM-SafetyBench" and row.get("image_variant") in {"SD", "SD_TYPO"}:
                by_goal[row["goal_id"]][row["image_variant"]] = row
        refusal_values = []
        candidate_refusal_values = []
        for goal_id, variants in by_goal.items():
            if set(variants) != {"SD", "SD_TYPO"}:
                continue
            sd, typo = variants["SD"], variants["SD_TYPO"]
            if sd.get("error") is None and typo.get("error") is None:
                refusal_values.append((goal_id, float(bool((sd.get("parsed") or {}).get("refusal"))), float(bool((typo.get("parsed") or {}).get("refusal")))))
                candidate_refusal_values.append((goal_id, float(bool((sd.get("parsed") or {}).get("candidate_refused"))), float(bool((typo.get("parsed") or {}).get("candidate_refused")))))
        paired[role] = {
            "SD_TYPO_minus_SD_refusal": paired_bootstrap(refusal_values),
            "SD_TYPO_minus_SD_candidate_refusal": paired_bootstrap(candidate_refusal_values),
        }

    return {
        "created_at": utc_now(),
        "run_dir": str(run_dir.resolve()),
        "groups": summary,
        "breakdowns": breakdowns,
        "paired_image_variant_effects": paired,
    }


def summarize_annotations(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    labeled = [row for row in rows if row.get("refusal") in {"0", "1"}]
    quality = [int(row["quality_0_1_2"]) for row in labeled if row.get("quality_0_1_2") in {"0", "1", "2"}]
    return {
        "n_rows": len(rows), "n_judged": len(labeled),
        "refusal_rate": _mean([float(row["refusal"]) for row in labeled]),
        "quality_distribution": dict(Counter(quality)),
        "ready_rate_among_judged": _mean([1.0 if value == 2 else 0.0 for value in quality]),
        "error_types": dict(Counter(row.get("error_type") for row in labeled if row.get("error_type"))),
    }


def paired_bootstrap(
    values: list[tuple[str, float, float]], iterations: int = 10000, seed: int = 20261125
) -> dict[str, Any]:
    by_goal: dict[str, list[float]] = defaultdict(list)
    for goal_id, left, right in values:
        by_goal[goal_id].append(right - left)
    units = sorted((goal_id, sum(ds) / len(ds)) for goal_id, ds in by_goal.items())
    if not units:
        return {"n_goal_units": 0, "mean_difference": None, "ci95": None}
    rng = random.Random(seed)
    boot = []
    diffs = [diff for _, diff in units]
    for _ in range(iterations):
        sample = [rng.choice(diffs) for _ in diffs]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    return {
        "n_goal_units": len(units), "mean_difference": sum(diffs) / len(diffs),
        "ci95": [boot[int(0.025 * iterations)], boot[min(iterations - 1, int(0.975 * iterations))]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate automatic and human auxiliary-role metrics")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="*")
    args = parser.parse_args(argv)
    result = summarize_outputs(args.run_dir)
    if args.annotations:
        result["human_annotations"] = summarize_annotations(args.annotations)
    output = args.run_dir / "summary.json"
    atomic_write_text(output, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

