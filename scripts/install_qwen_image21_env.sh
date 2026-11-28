#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
UV_BIN=${UV_BIN:-$(command -v uv)}
ENV_DIR=${QWEN21_ENV_DIR:-$PROJECT_ROOT/.venvs/qwen21}
DIFFUSERS_REVISION=644aa884c4e46c37c6998252ca7c96be939cc292

"$UV_BIN" venv --python 3.12 "$ENV_DIR"
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" \
  'torch==2.7.1' 'torchvision==0.22.1' 'accelerate==1.16.0' 'pillow==13.0.0' \
  'transformers==5.24.1' \
  "https://github.com/huggingface/diffusers/archive/${DIFFUSERS_REVISION}.zip"

"$ENV_DIR/bin/python" - <<'PY'
from diffusers import QwenImage21Pipeline
import diffusers
import transformers

print("Qwen-Image-2.1 environment ready")
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("pipeline", QwenImage21Pipeline.__name__)
PY
