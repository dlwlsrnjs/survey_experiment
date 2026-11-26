#!/bin/bash
set -euo pipefail

PROJECT=/home/ljk98/survey_experiment
UV=/home/ljk98/.local/bin/uv
PYTHON=/home/ljk98/.local/share/uv/python/cpython-3.10.21-linux-x86_64-gnu/bin/python3.10

mkdir -p "$PROJECT/.venvs"

if [ ! -x "$PROJECT/.venvs/modern/bin/python" ]; then
  "$UV" venv --python "$PYTHON" --link-mode hardlink "$PROJECT/.venvs/modern"
fi
"$UV" pip install --python "$PROJECT/.venvs/modern/bin/python" --link-mode hardlink -r "$PROJECT/envs/requirements-modern.txt"

if [ ! -x "$PROJECT/.venvs/legacy-vlm/bin/python" ]; then
  "$UV" venv --python "$PYTHON" --link-mode hardlink "$PROJECT/.venvs/legacy-vlm"
fi
"$UV" pip install --python "$PROJECT/.venvs/legacy-vlm/bin/python" --link-mode hardlink -r "$PROJECT/envs/requirements-legacy-vlm.txt"

"$PROJECT/.venvs/modern/bin/python" - <<'PY'
import accelerate, diffusers, faiss, openai, torch, transformers
print("modern", torch.__version__, transformers.__version__, diffusers.__version__)
PY

"$PROJECT/.venvs/legacy-vlm/bin/python" - <<'PY'
import accelerate, torch, transformers
print("legacy-vlm", torch.__version__, transformers.__version__)
PY
