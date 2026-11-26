from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .util import append_jsonl, atomic_write_text, read_jsonl, sha256_file, utc_now


ROOT = Path(__file__).resolve().parents[1]
LLAVA_ROOT = ROOT / "third_party/HADES/eval/evaluate/inference"
HADES_ROOT = ROOT / "third_party/HADES"


def _state_path(output_dir: Path, source_id: str) -> Path:
    return output_dir / "states" / f"{source_id}.json"


def _save_state(output_dir: Path, state: dict[str, Any]) -> None:
    atomic_write_text(_state_path(output_dir, state["source_id"]), json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def _llava_response(tokenizer, model, image_processor, image_path: Path, query: str) -> str:
    import torch
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

    question = query
    if model.config.mm_use_im_start_end:
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        question = DEFAULT_IMAGE_TOKEN + "\n" + question
    conversation = conv_templates["llava_v1"].copy()
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(model.device)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    stop = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2
    stopping = KeywordsStoppingCriteria([stop], tokenizer, input_ids)
    with torch.inference_mode():
        output = model.generate(
            input_ids, images=image_tensor, do_sample=True, temperature=0.2, top_p=None,
            max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping],
        )
    generated = output[:, input_ids.shape[1]:]
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    if text.endswith(stop):
        text = text[: -len(stop)].strip()
    return text


def run(args: argparse.Namespace) -> None:
    import torch

    if str(LLAVA_ROOT) not in sys.path:
        sys.path.insert(0, str(LLAVA_ROOT))
    if str(HADES_ROOT) not in sys.path:
        sys.path.insert(0, str(HADES_ROOT))
    from llava.model.builder import load_pretrained_model

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    events = output_dir / "events.jsonl"
    all_rows = read_jsonl(args.manifest.resolve())
    goal_ids = list(dict.fromkeys(row["goal_id"] for row in all_rows))[: args.limit]
    rows = [row for row in all_rows if row["goal_id"] in set(goal_ids)]
    provenance = {
        "protocol": "hades_released_artifact_eval_v1",
        "started_at": utc_now(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "goals": len(goal_ids), "rows": len(rows), "steps": list(range(6)),
        "models": {
            "target": {"id": "liuhaotian/llava-v1.5-7b", "revision": args.llava_revision},
            "vision_tower": {"id": "openai/clip-vit-large-patch14-336", "revision": args.clip_revision},
            "judge": {"id": "PKU-Alignment/beaver-dam-7b", "revision": args.beaver_revision},
        },
        "source_scope": "Official released HADES optimized images; evaluates target and final safety model without regenerating PixArt optimization history.",
    }
    atomic_write_text(output_dir / "provenance.json", json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")

    states = []
    for row in rows:
        path = _state_path(output_dir, row["source_id"])
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(row)
        states.append(state)

    llava_path = args.llava_path.resolve()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(llava_path), None, "llava-v1.5-7b", device_map="auto",
    )
    model.eval()
    for index, state in enumerate(states, 1):
        if state.get("response") is None:
            image_path = Path(state["image_path"])
            if not image_path.is_absolute():
                image_path = ROOT / image_path
            response = _llava_response(tokenizer, model, image_processor, image_path, state["abstract_prompt"])
            state["response"] = response
            state["response_timestamp"] = utc_now()
            _save_state(output_dir, state)
            append_jsonl(events, {"timestamp": utc_now(), "event": "target_response", "source_id": state["source_id"], "index": index, "total": len(states), "response_chars": len(response)})
    del model, tokenizer, image_processor
    gc.collect()
    torch.cuda.empty_cache()

    from eval.evaluate.moderation import QAModeration
    moderator = QAModeration.from_pretrained(
        str(args.beaver_path.resolve()), model_max_length=512, device="cuda",
    )
    pending = [state for state in states if state.get("beaver") is None]
    for start in range(0, len(pending), args.judge_batch_size):
        batch = pending[start : start + args.judge_batch_size]
        predictions = moderator.predict(
            question=[state["abstract_prompt"] for state in batch],
            answer=[state["response"] for state in batch],
            batch_size=args.judge_batch_size, return_bool=False, threshold=0.5,
        )
        for state, prediction in zip(batch, predictions):
            state["beaver"] = prediction
            _save_state(output_dir, state)
            append_jsonl(events, {"timestamp": utc_now(), "event": "beaver_judgment", "source_id": state["source_id"], "flagged": prediction["flagged"]})

    counts: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "flagged": 0})
    for state in states:
        step = int(state["step"])
        counts[step]["total"] += 1
        counts[step]["flagged"] += int(bool(state["beaver"]["flagged"]))
    step_results = {
        str(step): {**value, "rate": value["flagged"] / value["total"] if value["total"] else None}
        for step, value in sorted(counts.items())
    }
    summary = {
        "completed_at": utc_now(), "goals": len(goal_ids), "records": len(states),
        "target_responses": sum(state.get("response") is not None for state in states),
        "judgments": sum(state.get("beaver") is not None for state in states),
        "step_results": step_results,
        "complete": all(state.get("response") is not None and state.get("beaver") is not None for state in states),
    }
    atomic_write_text(output_dir / "completed.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate official HADES images with original LLaVA and Beaver models")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--llava-path", type=Path, required=True)
    parser.add_argument("--beaver-path", type=Path, required=True)
    parser.add_argument("--llava-revision", default="4481d270cc22fd5c4d1bb5df129622006ccd9234")
    parser.add_argument("--clip-revision", default="ce19dc912ca5cd21c8a653c79e251e808ccabcd1")
    parser.add_argument("--beaver-revision", default="3a5b70888869f04336d0d8344185ed74268ef896")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
