from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(description="Download revision-pinned experiment models")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--profile", choices=("core", "full", "open_best"), default="core")
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    args.status.parent.mkdir(parents=True, exist_ok=True)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    models = [item for item in lock["models"] if args.profile in item["profiles"]]
    results = []
    for item in models:
        started = time.time()
        result = {**item, "profile": args.profile, "started_at_epoch": started}
        try:
            path = snapshot_download(
                repo_id=item["repo_id"], revision=item["revision"],
                cache_dir=str(args.cache_dir) if args.cache_dir else None,
                allow_patterns=item.get("allow_patterns"),
            )
            result.update(status="complete", snapshot_path=path, elapsed_seconds=time.time() - started)
        except Exception as exc:
            result.update(status="error", error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.time() - started)
        results.append(result)
        args.status.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if all(item["status"] == "complete" for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
