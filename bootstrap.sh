#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROFILE=core
SKIP_ENVS=0
SKIP_REPOS=0
SKIP_DATA=0
SKIP_MODELS=0

usage() {
  cat <<'EOF'
Usage: ./bootstrap.sh [--profile core|full] [--skip-envs] [--skip-repos] [--skip-data] [--skip-models]

core: current controlled 32B experiment, SDXL generator, all datasets and pinned source repositories
full: core plus source-fidelity models and the legacy VLM environment

Environment variables:
  PYTHON_BIN  Python 3.10+ used to create virtual environments (default: python3)
  VENV_ROOT   virtual environment directory (default: <repo>/.venvs)
  HF_HOME     Hugging Face cache (default: <repo>/.cache/huggingface)
  HF_TOKEN    required after accepting gated dataset/model licenses
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) PROFILE=$2; shift 2 ;;
    --skip-envs) SKIP_ENVS=1; shift ;;
    --skip-repos) SKIP_REPOS=1; shift ;;
    --skip-data) SKIP_DATA=1; shift ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "$PROFILE" != core ] && [ "$PROFILE" != full ]; then
  echo "profile must be core or full" >&2
  exit 2
fi

export PROJECT_ROOT
export HF_HOME=${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}
export VENV_ROOT=${VENV_ROOT:-$PROJECT_ROOT/.venvs}
mkdir -p "$HF_HOME" "$PROJECT_ROOT/runs" "$PROJECT_ROOT/annotations"

if [ "$SKIP_ENVS" -eq 0 ]; then
  bash "$PROJECT_ROOT/scripts/setup_model_environments.sh" "$PROFILE"
fi
PYTHON="$VENV_ROOT/modern/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Missing modern environment: $PYTHON" >&2
  exit 2
fi

if [ "$SKIP_REPOS" -eq 0 ]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/prepare_repositories.py" --root "$PROJECT_ROOT" --lock "$PROJECT_ROOT/configs/assets.lock.json"
fi
if [ "$SKIP_DATA" -eq 0 ]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/prepare_data.py" --root "$PROJECT_ROOT" --lock "$PROJECT_ROOT/configs/assets.lock.json"
  cd "$PROJECT_ROOT"
  "$PYTHON" -m survey_exp.memjack \
    --dataset-root "$PROJECT_ROOT/data/raw/memjack" \
    --output "$PROJECT_ROOT/data/manifests/memjack_train_experience.jsonl" \
    --split train
  "$PYTHON" "$PROJECT_ROOT/scripts/prepare_hades_released.py" \
    --dataset-root "$PROJECT_ROOT/data/raw/hades" \
    --output-root "$PROJECT_ROOT/data/raw/hades/released_200" \
    --manifest "$PROJECT_ROOT/data/manifests/hades_released_200.jsonl"
  "$PYTHON" "$PROJECT_ROOT/scripts/prepare_figstep_manifest.py" \
    --csv "$PROJECT_ROOT/third_party/FigStep/data/question/safebench.csv" \
    --image-root "$PROJECT_ROOT/third_party/FigStep/data/images/SafeBench" \
    --output "$PROJECT_ROOT/data/manifests/figstep_safebench_200.jsonl" \
    --per-category 20
fi
if [ "$SKIP_MODELS" -eq 0 ]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/download_required_models.py" \
    --lock "$PROJECT_ROOT/configs/assets.lock.json" \
    --profile "$PROFILE" \
    --status "$PROJECT_ROOT/runs/model_download_status.json" \
    --cache-dir "$HF_HOME/hub"
fi

cd "$PROJECT_ROOT"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m survey_exp.doctor --profile "$PROFILE" --strict
echo "Bootstrap complete: profile=$PROFILE root=$PROJECT_ROOT"
