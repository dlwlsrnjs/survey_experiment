from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch: {path}\nexpected={expected}\nactual={actual}")


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as value:
        for member in value.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe zip member: {member.filename}")
        value.extractall(destination)


def prepare_mm(root: Path, spec: dict) -> dict:
    import gdown

    destination = root / "data/raw/mm_safetybench"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / spec["filename"]
    if not archive.is_file() or sha256(archive) != spec["sha256"]:
        gdown.download(id=spec["file_id"], output=str(archive), quiet=False)
    require_hash(archive, spec["sha256"])
    extracted = destination / spec["extract_root"]
    if not extracted.is_dir():
        safe_extract(archive, destination)
    image_count = sum(1 for _ in extracted.rglob("*.jpg"))
    if image_count < 1:
        raise RuntimeError(f"No MM-SafetyBench images extracted under {extracted}")
    return {"archive": str(archive), "sha256": spec["sha256"], "images": image_count}


def prepare_memjack(root: Path, spec: dict) -> dict:
    from huggingface_hub import snapshot_download

    destination = root / "data/raw/memjack"
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type=spec["repo_type"],
        revision=spec["revision"],
        local_dir=str(destination),
        allow_patterns=list(spec["files"]),
    )
    for relative, expected in spec["files"].items():
        require_hash(destination / relative, expected)
    return {"repo_id": spec["repo_id"], "revision": spec["revision"], "path": str(destination)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify experiment datasets")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    results = {
        "mm_safetybench": prepare_mm(root, lock["datasets"]["mm_safetybench_images"]),
        "memjack": prepare_memjack(root, lock["datasets"]["memjack"]),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
