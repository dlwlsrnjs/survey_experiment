from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ModelConfig
from .util import parse_json_object, stable_hash, utc_now


@dataclass
class ModelResult:
    text: str
    parsed: dict[str, Any] | None
    usage: dict[str, Any]
    latency_seconds: float
    request_id: str
    response_id: str | None
    error: dict[str, Any] | None
    raw: dict[str, Any] | None


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def _api_key(self) -> str:
        return os.environ.get(self.config.api_key_env, "local")

    def complete_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> ModelResult:
        request_id = str(uuid.uuid4())
        payload = {
            "model": self.config.name,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "seed": self.config.seed if seed is None else seed,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key()}",
                "X-Request-ID": request_id,
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
            text = raw["choices"][0]["message"]["content"]
            try:
                parsed = parse_json_object(text)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            return ModelResult(
                text=text,
                parsed=parsed,
                usage=raw.get("usage", {}),
                latency_seconds=time.monotonic() - started,
                request_id=request_id,
                response_id=raw.get("id"),
                error=None,
                raw=raw,
            )
        except urllib.error.HTTPError as exc:
            try:
                response_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                response_body = ""
            return ModelResult(
                text="",
                parsed=None,
                usage={},
                latency_seconds=time.monotonic() - started,
                request_id=request_id,
                response_id=None,
                error={"type": type(exc).__name__, "status": exc.code, "message": str(exc.reason), "response_body": response_body},
                raw=None,
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            detail = getattr(exc, "reason", None) or str(exc)
            return ModelResult(
                text="",
                parsed=None,
                usage={},
                latency_seconds=time.monotonic() - started,
                request_id=request_id,
                response_id=None,
                error={"type": type(exc).__name__, "message": str(detail)},
                raw=None,
            )

    def complete(self, system: str, user: str, image: Path | None = None) -> ModelResult:
        content: str | list[dict[str, Any]] = user
        if image is not None:
            content = [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {"url": image_data_url(image)}},
            ]
        return self.complete_messages([
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ])

    def fingerprint(self) -> dict[str, Any]:
        return {
            "backend": self.config.backend,
            "model": self.config.name,
            "revision": self.config.revision,
            "base_url_hash": stable_hash(self.config.base_url),
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_tokens": self.config.max_tokens,
            "captured_at": utc_now(),
        }


class MockClient(OpenAICompatibleClient):
    def complete_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> ModelResult:
        request_id = str(uuid.uuid4())
        value = {"status": "mock", "messages": len(messages)}
        text = json.dumps(value)
        return ModelResult(text, value, {"prompt_tokens": 0, "completion_tokens": 0}, 0.0, request_id, "mock", None, value)

    def complete(self, system: str, user: str, image: Path | None = None) -> ModelResult:
        request_id = str(uuid.uuid4())
        role = "unknown"
        for candidate in ("generation", "observation_planning", "evaluation_revision", "experience_use"):
            if candidate in system:
                role = candidate
                break
        value = {
            "role": role,
            "refusal": False,
            "status": "mock",
            "evidence": [],
            "output": "MOCK_OUTPUT",
            "format_version": "1.0",
        }
        text = json.dumps(value)
        return ModelResult(text, value, {"prompt_tokens": 0, "completion_tokens": 0}, 0.0, request_id, "mock", None, value)

