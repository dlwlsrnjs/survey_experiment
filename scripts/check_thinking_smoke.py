from __future__ import annotations

import argparse
import json
from pathlib import Path

from survey_exp.util import atomic_write_text, read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate an auxiliary-model smoke run before the full evaluation")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=20)
    parser.add_argument("--min-format-rate", type=float, default=0.95)
    parser.add_argument("--max-length-rate", type=float, default=0.10)
    parser.add_argument("--min-nonrefusal-rate", type=float, default=0.20)
    args = parser.parse_args()

    rows = read_jsonl(args.run_dir / "outputs.jsonl")
    errors = sum(row.get("error") is not None for row in rows)
    valid = sum(bool((row.get("automatic_validation") or {}).get("format_valid")) for row in rows)
    length_stops = sum(row.get("finish_reason") == "length" for row in rows)
    nonrefusals = sum(
        bool((row.get("automatic_validation") or {}).get("format_valid"))
        and row.get("parsed") is not None
        and not bool((row.get("parsed") or {}).get("refusal"))
        for row in rows
    )
    count = len(rows)
    format_rate = valid / count if count else 0.0
    length_rate = length_stops / count if count else 1.0
    nonrefusal_rate = nonrefusals / count if count else 0.0
    passed = (
        count == args.expected
        and errors == 0
        and format_rate >= args.min_format_rate
        and length_rate <= args.max_length_rate
        and nonrefusal_rate >= args.min_nonrefusal_rate
    )
    report = {
        "passed": passed,
        "expected": args.expected,
        "count": count,
        "errors": errors,
        "format_valid": valid,
        "format_rate": format_rate,
        "length_stops": length_stops,
        "length_rate": length_rate,
        "nonrefusals": nonrefusals,
        "nonrefusal_rate": nonrefusal_rate,
        "thresholds": {
            "min_format_rate": args.min_format_rate,
            "max_length_rate": args.max_length_rate,
            "min_nonrefusal_rate": args.min_nonrefusal_rate,
        },
    }
    atomic_write_text(args.run_dir / "smoke_gate.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
