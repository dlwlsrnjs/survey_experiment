from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _load_image(url: str):
    from PIL import Image

    if not url.startswith("data:") or ";base64," not in url:
        raise ValueError("Only base64 data URLs are accepted by the local server")
    encoded = url.split(";base64,", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


class Engine:
    def __init__(self, model_id: str, revision: str | None, max_pixels: int):
        import torch
        from transformers import (
            AutoConfig,
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
        )

        self.torch = torch
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=True, max_pixels=max_pixels,
        )
        architecture = (AutoConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=True,
        ).architectures or [""])[0]
        model_class = {
            "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLForConditionalGeneration,
            "Qwen3VLForConditionalGeneration": Qwen3VLForConditionalGeneration,
            "Qwen3VLMoeForConditionalGeneration": Qwen3VLMoeForConditionalGeneration,
        }.get(architecture)
        if model_class is None:
            raise ValueError(f"Unsupported local VLM architecture: {architecture}")
        self.model = model_class.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).eval()
        self.lock = threading.Lock()

    def _messages(self, payload: dict[str, Any]):
        converted = []
        for message in payload["messages"]:
            content = message.get("content", "")
            if isinstance(content, str):
                converted.append({"role": message["role"], "content": content})
                continue
            parts = []
            for part in content:
                if part.get("type") == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    parts.append({"type": "image", "image": _load_image(part["image_url"]["url"])})
            converted.append({"role": message["role"], "content": parts})
        return converted

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        messages = self._messages(payload)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)
        max_tokens = int(payload.get("max_tokens", 1200))
        temperature = float(payload.get("temperature", 0.0))
        seed = int(payload.get("seed", 20261125))
        self.torch.manual_seed(seed)
        generation = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "use_cache": True,
        }
        if temperature > 0:
            generation["temperature"] = temperature
            generation["top_p"] = float(payload.get("top_p", 1.0))
        with self.lock, self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation)
        trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        prompt_tokens = int(inputs.input_ids.shape[1])
        completion_tokens = int(trimmed[0].shape[0])
        return {
            "id": f"chatcmpl-local-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class Handler(BaseHTTPRequestHandler):
    engine: Engine

    def _json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path in {"/health", "/v1/models"}:
            if self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": self.engine.model_id, "object": "model"}]})
            else:
                self._json(200, {"status": "ok", "model": self.engine.model_id})
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self._json(200, self.engine.complete(payload))
        except Exception as exc:  # request-scoped error, retained by caller logs
            traceback.print_exc()
            self._json(500, {"error": {"type": type(exc).__name__, "message": str(exc)}})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal OpenAI-compatible Qwen2.5/3-VL local server")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-pixels", type=int, default=1_048_576)
    args = parser.parse_args(argv)
    Handler.engine = Engine(args.model, args.revision, args.max_pixels)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"event": "server_ready", "model": args.model, "host": args.host, "port": args.port}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

