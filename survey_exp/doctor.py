from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from .image_generation import select_profile
from .paths import resolve_project_path
from .util import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(profile: str) -> dict[str, Any]:
    lock = json.loads((ROOT / "configs/assets.lock.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any, severity: str = "error") -> None:
        checks.append({"name": name, "ok": ok, "severity": severity, "detail": detail})

    add("python", tuple(map(int, platform.python_version_tuple()[:2])) >= (3, 10), platform.python_version())
    add("project_root", (ROOT / "pyproject.toml").is_file(), str(ROOT))

    for item in lock["repositories"]:
        path = ROOT / "third_party" / item["name"]
        try:
            revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
            add(f"repo:{item['name']}", revision == item["revision"], {"expected": item["revision"], "actual": revision})
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            add(f"repo:{item['name']}", False, str(exc))

    mm = lock["datasets"]["mm_safetybench_images"]
    mm_archive = ROOT / "data/raw/mm_safetybench" / mm["filename"]
    add("dataset:mm_archive", mm_archive.is_file() and _sha256(mm_archive) == mm["sha256"], str(mm_archive))
    mm_root = ROOT / "data/raw/mm_safetybench" / mm["extract_root"]
    mm_images = sum(1 for _ in mm_root.rglob("*.jpg")) if mm_root.is_dir() else 0
    add("dataset:mm_images", mm_images > 0, {"root": str(mm_root), "jpg_count": mm_images})

    memjack = lock["datasets"]["memjack"]
    for relative, expected in memjack["files"].items():
        path = ROOT / "data/raw/memjack" / relative
        add(f"dataset:memjack:{relative}", path.is_file() and _sha256(path) == expected, str(path))

    manifest = ROOT / "data/manifests/fixed_inputs.jsonl"
    manifest_sidecar = ROOT / "data/manifests/fixed_inputs.jsonl.sha256"
    if manifest.is_file() and manifest_sidecar.is_file():
        manifest_sha = manifest_sidecar.read_text(encoding="utf-8").split()[0]
        add("manifest:sha256", _sha256(manifest) == manifest_sha, {"path": str(manifest), "expected": manifest_sha})
        manifest_rows = read_jsonl(manifest)
        image_rows = [row for row in manifest_rows if row.get("image_path")]
        missing = [row["condition_id"] for row in image_rows if not resolve_project_path(row["image_path"], ROOT).is_file()]
        add("manifest:images", not missing, {"rows": len(image_rows), "missing": missing[:10]})
    else:
        add("manifest:sha256", False, {"manifest": str(manifest), "sidecar": str(manifest_sidecar)})
        add("manifest:images", False, "manifest unavailable")

    bank = ROOT / "data/manifests/memjack_train_experience.jsonl"
    bank_summary_path = ROOT / "data/manifests/memjack_train_experience.summary.json"
    if bank.is_file() and bank_summary_path.is_file():
        bank_summary = json.loads(bank_summary_path.read_text(encoding="utf-8"))
        add("experience_bank", _sha256(bank) == bank_summary["output_sha256"], {"path": str(bank), "records": bank_summary["records"]})
    else:
        add("experience_bank", False, {"bank": str(bank), "summary": str(bank_summary_path)})

    image_config = ROOT / "configs/image_generators.yaml"
    for selector in ("controlled_default", "source_fidelity.hades", "source_fidelity.himrd"):
        try:
            selected = select_profile(image_config, selector)
            locked = any(model["repo_id"] == selected["model"] and model["revision"] == selected["revision"] for model in lock["models"])
            add(f"image_profile:{selector}", locked, {"model": selected["model"], "revision": selected["revision"]})
        except Exception as exc:
            add(f"image_profile:{selector}", False, str(exc))

    try:
        from huggingface_hub import snapshot_download

        cache_dir = str(Path(os.environ.get("HF_HOME", ROOT / ".cache/huggingface")) / "hub")
        for model in lock["models"]:
            if profile not in model["profiles"]:
                continue
            try:
                snapshot_download(
                    model["repo_id"], revision=model["revision"], cache_dir=cache_dir,
                    allow_patterns=model.get("allow_patterns"), local_files_only=True,
                )
                add(f"model:{model['repo_id']}", True, model["revision"])
            except Exception as exc:
                add(f"model:{model['repo_id']}", False, str(exc))
    except ImportError as exc:
        add("models", False, str(exc))

    try:
        import torch

        add("cuda", torch.cuda.is_available(), {"available": torch.cuda.is_available(), "count": torch.cuda.device_count()}, severity="warning")
    except ImportError as exc:
        add("cuda", False, str(exc), severity="warning")

    return {
        "profile": profile,
        "project_root": str(ROOT),
        "ok": not any(not check["ok"] and check["severity"] == "error" for check in checks),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a bootstrapped experiment environment")
    parser.add_argument("--profile", choices=("core", "full", "open_best"), default="core")
    parser.add_argument("--strict", action="store_true", help="return nonzero when required assets are missing")
    args = parser.parse_args()
    result = audit(args.profile)
    for check in result["checks"]:
        marker = "OK" if check["ok"] else ("WARN" if check["severity"] == "warning" else "FAIL")
        print(f"[{marker}] {check['name']}: {check['detail']}")
    print(f"overall: {'ready' if result['ok'] else 'incomplete'} ({args.profile})")
    return 0 if result["ok"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
