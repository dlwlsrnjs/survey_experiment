from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any

from .client import MockClient, OpenAICompatibleClient
from .config import ExperimentConfig, load_config
from .image_generation import generate_image, select_profile
from .paths import project_root_from_config, resolve_project_path
from .retrieval import build_index, matched_random, retrieve_indexed
from .roles import (
    auto_validate,
    evaluation_revision_prompt,
    experience_use_prompt,
    generation_prompt,
    observation_planning_prompt,
)
from .util import append_jsonl, atomic_write_text, read_jsonl, sha256_file, stable_hash, utc_now, write_jsonl


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _provenance(config: ExperimentConfig, client: OpenAICompatibleClient) -> dict[str, Any]:
    root = Path.cwd()
    third_party = {}
    for path in sorted((root / "third_party").glob("*")):
        revision = _git_revision(path)
        if revision:
            third_party[path.name] = revision
    slurm_keys = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NODELIST", "CUDA_VISIBLE_DEVICES")
    gpu_info = None
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return {
        "experiment_id": config.experiment_id,
        "started_at": utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "node_gpus_reported_by_nvidia_smi": gpu_info,
        "config_path": str(config.source),
        "config_sha256": sha256_file(config.source) if config.source else None,
        "manifest_path": str(config.manifest),
        "manifest_sha256": sha256_file(config.manifest),
        "model": client.fingerprint(),
        "slurm": {key: os.environ.get(key) for key in slurm_keys},
        "third_party_revisions": third_party,
    }


def _selected_rows(config: ExperimentConfig, limit: int | None) -> list[dict[str, Any]]:
    rows = read_jsonl(config.manifest)
    if config.split == "pilot":
        rows = [row for row in rows if row.get("pilot")]
    else:
        rows = [row for row in rows if row["split"] == config.split]
    if limit is not None:
        goals: list[str] = []
        for row in rows:
            if row["goal_id"] not in goals:
                goals.append(row["goal_id"])
        allowed = set(goals[:limit])
        rows = [row for row in rows if row["goal_id"] in allowed]
    return rows


def _load_optional_map(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    rows = read_jsonl(path)
    return {row["condition_id"]: row for row in rows}


def run(
    config: ExperimentConfig,
    limit: int | None = None,
    mock: bool = False,
    target_responses_path: Path | None = None,
) -> Path:
    run_dir = config.output_root / config.experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    client = MockClient(config.model) if mock else OpenAICompatibleClient(config.model)
    project_root = project_root_from_config(config.source)
    image_generation_settings = config.extra.get("image_generation") or {}
    image_generator = None
    if image_generation_settings.get("enabled"):
        generator_config = Path(image_generation_settings["config"])
        if not generator_config.is_absolute() and config.source:
            generator_config = (config.source.parent / generator_config).resolve()
        image_generator = select_profile(generator_config, image_generation_settings["profile"])
    provenance = _provenance(config, client)
    if image_generator is not None:
        provenance["image_generation"] = {
            "profile": image_generation_settings["profile"],
            "config_path": str(generator_config),
            "config_sha256": sha256_file(generator_config),
            "resolved_profile": image_generator,
        }
    atomic_write_text(run_dir / "provenance.json", json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    events_path = run_dir / "events.jsonl"
    outputs_path = run_dir / "outputs.jsonl"
    target_responses = _load_optional_map(target_responses_path)
    rows = _selected_rows(config, limit)
    upstream_candidate_cache: dict[str, dict[str, Any]] = {}
    evaluation_cache: dict[str, dict[str, Any]] = {}
    experience_bank: list[dict[str, Any]] = []
    retrieval_plan: dict[tuple[str, int], tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    completed_calls: set[tuple[str, str, int, str]] = set()
    if outputs_path.exists():
        for prior in read_jsonl(outputs_path):
            key = (
                prior["role"], prior.get("condition") or "-", int(prior["repeat"]),
                prior["condition_id"],
            )
            completed_calls.add(key)
            if prior.get("parsed") is not None and prior.get("error") is None:
                if prior["role"] in {"generation", "observation_planning"}:
                    upstream_candidate_cache[prior["condition_id"]] = prior["parsed"]
                elif prior["role"] == "evaluation_revision":
                    evaluation_cache[prior["condition_id"]] = prior["parsed"]
    bank_value = config.extra.get("experience_bank")
    if bank_value:
        bank_path = Path(bank_value)
        if not bank_path.is_absolute() and config.source:
            bank_path = (config.source.parent / bank_path).resolve()
        experience_bank = read_jsonl(bank_path)
        if any(item.get("source_split") in {"test", "eval"} for item in experience_bank):
            raise ValueError("Experience bank contains test/eval records; only dev/train records are allowed")

    role_repeats = config.extra.get("role_repeats", {})
    if experience_bank and "experience_use" in config.roles:
        experience_repeats = int(role_repeats.get("experience_use", config.repeats))
        index = build_index(experience_bank)
        plan_rows = []
        unique_goals = {row["goal_id"]: row for row in rows if row["split"] == "eval"}
        for goal_id, row in sorted(unique_goals.items()):
            relevant = retrieve_indexed(row["goal"], index, k=3)
            for repeat in range(experience_repeats):
                random_seed = int(stable_hash({"goal": goal_id, "repeat": repeat})[:12], 16)
                random_exp = matched_random(relevant, experience_bank, random_seed)
                retrieval_plan[(goal_id, repeat)] = (relevant, random_exp)
                plan_rows.append({
                    "goal_id": goal_id,
                    "repeat": repeat,
                    "query_sha256": stable_hash(row["goal"]),
                    "relevant": [
                        {"experience_id": item.get("experience_id"), "score": item.get("retrieval_score")}
                        for item in relevant
                    ],
                    "matched_random": [
                        {"experience_id": item.get("experience_id"), "matched_length_to": item.get("matched_length_to")}
                        for item in random_exp
                    ],
                })
        retrieval_plan_path = run_dir / "retrieval_plan.jsonl"
        write_jsonl(retrieval_plan_path, plan_rows)
        provenance["retrieval_plan_path"] = str(retrieval_plan_path.resolve())
        provenance["retrieval_plan_sha256"] = sha256_file(retrieval_plan_path)
        atomic_write_text(run_dir / "provenance.json", json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")

    for role in config.roles:
        repeats = int(role_repeats.get(role, config.repeats))
        if repeats < 1:
            raise ValueError(f"role repeat count must be positive: {role}={repeats}")
        for repeat in range(repeats):
            for row in rows:
                condition_id = row["condition_id"]
                image = resolve_project_path(row["image_path"], project_root) if row.get("image_path") else None
                skip_reason = None
                variants: list[tuple[str | None, str, str, Path | None]] = []
                if role == "generation":
                    if row["dataset"] != "HarmBench-Standard":
                        skip_reason = "role_applies_to_harmbench"
                    else:
                        system, user = generation_prompt(row)
                        variants.append((None, system, user, None))
                elif role == "observation_planning":
                    if row["dataset"] != "MM-SafetyBench":
                        skip_reason = "role_applies_to_mm_safetybench"
                    elif image is None or not image.is_file():
                        skip_reason = "image_missing"
                    else:
                        system, user = observation_planning_prompt(row, config.allowed_image_ops)
                        variants.append((None, system, user, image))
                elif role == "evaluation_revision":
                    candidate = upstream_candidate_cache.get(condition_id)
                    if candidate is None:
                        skip_reason = "no_upstream_candidate"
                    else:
                        target = target_responses.get(condition_id, {}).get("target_response")
                        system, user = evaluation_revision_prompt(row, candidate, target)
                        variants.append((None, system, user, image if image and image.is_file() else None))
                elif role == "experience_use":
                    if row["split"] != "eval":
                        skip_reason = "experience_conditions_only_on_eval"
                    elif not experience_bank:
                        skip_reason = "empty_dev_experience_bank"
                    else:
                        feedback = evaluation_cache.get(condition_id)
                        relevant, random_exp = retrieval_plan[(row["goal_id"], repeat)]
                        for condition, experiences in (("A", []), ("B", []), ("C", relevant), ("C_random", random_exp)):
                            system, user = experience_use_prompt(row, condition, feedback, experiences)
                            variants.append((condition, system, user, image if image and image.is_file() else None))
                else:
                    skip_reason = "unknown_role"

                if skip_reason:
                    append_jsonl(events_path, {
                        "event": "skipped", "timestamp": utc_now(), "role": role,
                        "condition_id": condition_id, "goal_id": row["goal_id"],
                        "repeat": repeat, "reason": skip_reason,
                    })
                    continue

                for ablation, system, user, call_image in variants:
                    call_key = (role, ablation or "-", repeat, condition_id)
                    if call_key in completed_calls:
                        append_jsonl(events_path, {
                            "event": "resume_skip", "timestamp": utc_now(), "role": role,
                            "condition": ablation, "condition_id": condition_id,
                            "goal_id": row["goal_id"], "repeat": repeat,
                        })
                        continue
                    prompt_hash = stable_hash({"system": system, "user": user, "image": str(call_image) if call_image else None})
                    append_jsonl(events_path, {
                        "event": "request_started", "timestamp": utc_now(), "role": role,
                        "condition": ablation, "condition_id": condition_id, "goal_id": row["goal_id"],
                        "repeat": repeat, "prompt_hash": prompt_hash,
                    })
                    result = client.complete(system, user, call_image)
                    validation = auto_validate(role, result.parsed, config.allowed_image_ops)
                    next_consumer = {
                        "generation": "fixed_renderer_or_fixed_t2i",
                        "observation_planning": "target_executor",
                        "evaluation_revision": "rewriter_or_stopping_controller",
                        "experience_use": "candidate_generator",
                    }.get(role, "unknown")
                    tool_result: dict[str, Any] = {
                        "typography": {"status": "not_run"},
                        "image_generation": {"status": "not_run"},
                    }
                    if role == "generation" and result.parsed and not result.parsed.get("refusal"):
                        statement = result.parsed.get("render_statement")
                        if isinstance(statement, str) and statement.strip():
                            from .tools import render_typography
                            artifact = run_dir / "artifacts/rendered" / f"{condition_id.replace(':', '__')}_r{repeat}.png"
                            font = project_root / "third_party/MM-SafetyBench/assets/font/ARIAL.TTF"
                            try:
                                tool_result["typography"] = render_typography(statement, artifact, font)
                            except Exception as exc:
                                tool_result["typography"] = {"status": "error", "tool": "deterministic_typography_renderer_v1", "error": type(exc).__name__, "message": str(exc)}
                        scene = result.parsed.get("scene_description")
                        if image_generator and isinstance(scene, str) and scene.strip():
                            generated = run_dir / "artifacts/generated" / f"{condition_id.replace(':', '__')}_r{repeat}.png"
                            try:
                                tool_result["image_generation"] = generate_image(scene, generated, image_generator)
                            except Exception as exc:
                                tool_result["image_generation"] = {
                                    "status": "error", "tool": "revision_pinned_text_to_image_v1",
                                    "error": type(exc).__name__, "message": str(exc),
                                    "model": image_generator.get("model"), "revision": image_generator.get("revision"),
                                }
                    output = {
                        "record_version": "1.0",
                        "timestamp": utc_now(),
                        "experiment_id": config.experiment_id,
                        "model": config.model.name,
                        "role": role,
                        "condition": ablation,
                        "repeat": repeat,
                        "dataset": row["dataset"],
                        "split": row["split"],
                        "goal_id": row["goal_id"],
                        "condition_id": condition_id,
                        "category": row["category"],
                        "source_id": row["source_id"],
                        "goal": row["goal"],
                        "question": row["question"],
                        "image_variant": row.get("image_variant"),
                        "image_path": str(call_image) if call_image else None,
                        "prompt_hash": prompt_hash,
                        "request_payload": {
                            "model": config.model.name,
                            "system": system,
                            "user": user,
                            "temperature": config.model.temperature,
                            "seed": config.model.seed,
                            "max_tokens": config.model.max_tokens,
                            "image_path": str(call_image.resolve()) if call_image else None,
                            "image_sha256": sha256_file(call_image) if call_image else None,
                        },
                        "request_id": result.request_id,
                        "response_id": result.response_id,
                        "latency_seconds": result.latency_seconds,
                        "usage": result.usage,
                        "error": result.error,
                        "raw_text": result.text,
                        "parsed": result.parsed,
                        "automatic_validation": validation,
                        "next_consumer": next_consumer,
                        "tool_result": tool_result,
                    }
                    append_jsonl(outputs_path, output)
                    completed_calls.add(call_key)
                    append_jsonl(events_path, {
                        "event": "request_finished", "timestamp": utc_now(), "role": role,
                        "condition": ablation, "condition_id": condition_id, "goal_id": row["goal_id"],
                        "repeat": repeat, "request_id": result.request_id,
                        "error": result.error, "format_valid": validation["format_valid"],
                    })
                    if result.parsed is not None and result.error is None:
                        if role in {"generation", "observation_planning"}:
                            upstream_candidate_cache[condition_id] = result.parsed
                        elif role == "evaluation_revision":
                            evaluation_cache[condition_id] = result.parsed
                        if row["split"] == "dev" and role in {"generation", "observation_planning"}:
                            experience_bank.append({
                                "experience_id": f"{role}:{condition_id}:r{repeat}",
                                "goal_id": row["goal_id"], "goal": row["goal"],
                                "role": role, "output": result.parsed,
                            })

    if experience_bank and not bank_value:
        write_jsonl(run_dir / "experience_bank.jsonl", experience_bank)
    atomic_write_text(run_dir / "completed.json", json.dumps({
        "completed_at": utc_now(), "rows_selected": len(rows), "roles": list(config.roles),
        "experience_records": len(experience_bank),
    }, indent=2) + "\n")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run auxiliary-role experiments with full JSONL traces")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--target-responses", type=Path)
    args = parser.parse_args(argv)
    print(run(load_config(args.config), args.limit, args.mock, args.target_responses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

