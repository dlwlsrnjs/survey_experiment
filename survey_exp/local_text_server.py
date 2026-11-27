from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class Engine:
    def __init__(self, model_id: str, revision: str | None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
        self.lock = threading.Lock()

    @staticmethod
    def _text_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        converted = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                parts = [part.get("text", "") for part in content if part.get("type") == "text"]
                content = "\n".join(part for part in parts if part)
            converted.append({"role": message.get("role", "user"), "content": str(content)})
        return converted

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = self._text_messages(payload["messages"])
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        max_tokens = int(payload.get("max_tokens", 1200))
        temperature = float(payload.get("temperature", 0.0))
        seed = int(payload.get("seed", 20261127))
        self.torch.manual_seed(seed)
        generation: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "use_cache": True,
        }
        if temperature > 0:
            generation["temperature"] = temperature
            generation["top_p"] = float(payload.get("top_p", 1.0))
        with self.lock, self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        generated = output[:, inputs.input_ids.shape[1]:]
        text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        completion_tokens = int(generated.shape[1])
        prompt_tokens = int(inputs.input_ids.shape[1])
        return {
            "id": f"chatcmpl-local-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "length" if completion_tokens >= max_tokens else "stop",
            }],
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
            value = {"status": "ok", "model": self.engine.model_id}
            if self.path == "/v1/models":
                value = {"object": "list", "data": [{"id": self.engine.model_id, "object": "model"}]}
            self._json(200, value)
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
        except Exception as exc:
            traceback.print_exc()
            self._json(500, {"error": {"type": type(exc).__name__, "message": str(exc)}})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal OpenAI-compatible local text model server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    Handler.engine = Engine(args.model, args.revision)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"event": "server_ready", "model": args.model, "port": args.port}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
