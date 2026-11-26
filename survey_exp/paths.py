from __future__ import annotations

from pathlib import Path


def project_root_from_config(config_path: Path | None = None) -> Path:
    if config_path is not None:
        return config_path.resolve().parent.parent
    return Path(__file__).resolve().parents[1]


def resolve_project_path(value: str | Path, root: Path) -> Path:
    """Resolve portable paths while preserving legacy frozen manifests.

    New manifests use repository-relative paths. The 2026-11-25 manifest contains
    absolute paths and has an active run tied to its SHA-256, so a missing legacy
    absolute path is remapped from its data/ or third_party/ suffix without changing
    the frozen file.
    """
    path = Path(value)
    if not path.is_absolute():
        return (root / path).resolve()
    if path.exists():
        return path.resolve()
    parts = path.parts
    for marker in ("data", "third_party"):
        if marker in parts:
            index = parts.index(marker)
            return (root / Path(*parts[index:])).resolve()
    return path


def portable_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)
