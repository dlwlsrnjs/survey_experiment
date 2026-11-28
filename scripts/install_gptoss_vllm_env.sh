#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
UV_BIN=${UV_BIN:-$(command -v uv)}
ENV_DIR=${GPTOSS_VLLM_ENV_DIR:-$PROJECT_ROOT/.venvs/gptoss-vllm}

"$UV_BIN" venv --python 3.12 "$ENV_DIR"
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" 'vllm==0.34.0'
"$ENV_DIR/bin/python" - <<'PY'
import torch
import vllm

print("gpt-oss vLLM environment ready")
print("vllm", vllm.__version__)
print("torch", torch.__version__)
PY
