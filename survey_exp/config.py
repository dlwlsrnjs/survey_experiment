from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str
    revision: str | None = None
    api_key_env: str = "LOCAL_MODEL_API_KEY"
    temperature: float = 0.0
    seed: int = 20261125
    max_tokens: int = 1200
    timeout_seconds: int = 300
    backend: str = "openai_compatible"
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    manifest: Path
    output_root: Path
    model: ModelConfig
    roles: tuple[str, ...]
    split: str = "pilot"
    repeats: int = 1
    target_query_cap: int = 3
    allowed_image_ops: tuple[str, ...] = ("crop", "mask", "blur")
    source: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML configs; use JSON or install pyyaml")
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    raw = _load_mapping(source)
    model = ModelConfig(**raw.pop("model"))
    known = {
        "experiment_id", "manifest", "output_root", "roles", "split", "repeats",
        "target_query_cap", "allowed_image_ops",
    }
    extra = {key: raw.pop(key) for key in list(raw) if key not in known}
    return ExperimentConfig(
        experiment_id=raw["experiment_id"],
        manifest=(source.parent / raw["manifest"]).resolve(),
        output_root=(source.parent / raw["output_root"]).resolve(),
        model=model,
        roles=tuple(raw["roles"]),
        split=raw.get("split", "pilot"),
        repeats=int(raw.get("repeats", 1)),
        target_query_cap=int(raw.get("target_query_cap", 3)),
        allowed_image_ops=tuple(raw.get("allowed_image_ops", ["crop", "mask", "blur"])),
        source=source,
        extra=extra,
    )

