#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROFILE=${1:-core}
PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_ROOT=${VENV_ROOT:-$PROJECT_ROOT/.venvs}

if [ "$PROFILE" != core ] && [ "$PROFILE" != full ]; then
  echo "profile must be core or full" >&2
  exit 2
fi

mkdir -p "$VENV_ROOT"

create_environment() {
  local name=$1
  local requirements=$2
  local environment="$VENV_ROOT/$name"
  if [ ! -x "$environment/bin/python" ]; then
    "$PYTHON_BIN" -m venv "$environment"
  fi
  if ! "$environment/bin/python" -m pip --version >/dev/null 2>&1; then
    "$environment/bin/python" -m ensurepip --upgrade
  fi
  "$environment/bin/python" -m pip install --upgrade pip wheel
  "$environment/bin/python" -m pip install -r "$requirements"
}

create_environment modern "$PROJECT_ROOT/envs/requirements-modern.txt"
if [ "$PROFILE" = full ]; then
  create_environment legacy-vlm "$PROJECT_ROOT/envs/requirements-legacy-vlm.txt"
fi

"$VENV_ROOT/modern/bin/python" -c 'import diffusers, huggingface_hub, openai, torch, transformers; print("modern", torch.__version__, transformers.__version__, diffusers.__version__)'
if [ "$PROFILE" = full ]; then
  "$VENV_ROOT/legacy-vlm/bin/python" -c 'import accelerate, torch, transformers; print("legacy-vlm", torch.__version__, transformers.__version__)'
fi
