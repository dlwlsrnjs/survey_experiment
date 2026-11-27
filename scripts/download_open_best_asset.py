from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def assets(lock_path: Path) -> list[dict[str, object]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selected = [model for model in lock["models"] if "open_best" in model.get("profiles", [])]
    seen: set[tuple[str, str]] = set()
    unique = []
    for model in selected:
        key = (model["repo_id"], model["revision"])
        if key not in seen:
            seen.add(key)
            unique.append(model)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Download one indexed open-best asset at its locked revision")
    parser.add_argument("--lock", type=Path, default=Path("configs/assets.lock.json"))
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--status-dir", type=Path, default=Path("runs/open_best_download_status"))
    args = parser.parse_args()
    selected = assets(args.lock.resolve())
    if not 0 <= args.index < len(selected):
        raise ValueError(f"index {args.index} is outside 0..{len(selected) - 1}")
    model = selected[args.index]
    args.status_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.status_dir / f"task-{args.index:02d}.json"
    status = {
        "index": args.index,
        "total": len(selected),
        "repo_id": model["repo_id"],
        "revision": model["revision"],
        "role": model["role"],
        "status": "started",
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    try:
        path = snapshot_download(
            repo_id=str(model["repo_id"]),
            revision=str(model["revision"]),
            allow_patterns=model.get("allow_patterns"),
        )
    except Exception as exc:
        status.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise
    status.update({"status": "complete", "snapshot_path": path})
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
