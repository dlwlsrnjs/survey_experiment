from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def prepare(root: Path, lock_path: Path) -> list[dict[str, str]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    third_party = root / "third_party"
    third_party.mkdir(parents=True, exist_ok=True)
    results = []
    for item in lock["repositories"]:
        destination = third_party / item["name"]
        if not (destination / ".git").is_dir():
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"Refusing to replace non-Git directory: {destination}")
            subprocess.run(["git", "clone", "--no-checkout", item["url"], str(destination)], check=True)
        origin = _run("git", "remote", "get-url", "origin", cwd=destination)
        if origin.rstrip("/").removesuffix(".git") != item["url"].rstrip("/").removesuffix(".git"):
            raise RuntimeError(f"Unexpected origin for {item['name']}: {origin}")
        status = _run("git", "status", "--porcelain", cwd=destination)
        if status:
            raise RuntimeError(f"Third-party repository has local changes: {destination}")
        try:
            _run("git", "cat-file", "-e", f"{item['revision']}^{{commit}}", cwd=destination)
        except subprocess.CalledProcessError:
            subprocess.run(["git", "fetch", "origin", item["revision"]], cwd=destination, check=True)
        subprocess.run(["git", "checkout", "--detach", item["revision"]], cwd=destination, check=True)
        actual = _run("git", "rev-parse", "HEAD", cwd=destination)
        if actual != item["revision"]:
            raise RuntimeError(f"Revision mismatch for {item['name']}: {actual}")
        results.append({"name": item["name"], "revision": actual, "path": str(destination)})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone and pin all third-party repositories")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root.resolve(), args.lock.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
