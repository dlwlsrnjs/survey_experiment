#!/usr/bin/env bash
#SBATCH --job-name=himrd-e2e-200
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$PROJECT_ROOT" || ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
  echo "Cannot resolve project root. Submit from the repository root or export PROJECT_ROOT." >&2
  exit 2
fi
PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd)
KEY_FILE=${OPENAI_KEY_FILE:-$HOME/.config/survey_experiment/openai.env}
if [[ ! -f "$KEY_FILE" || "$(stat -c '%a' "$KEY_FILE")" != "600" ]]; then
  echo "Missing OpenAI key file or permissions are not 600: $KEY_FILE" >&2
  exit 2
fi
. "$KEY_FILE"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is empty after loading $KEY_FILE" >&2
  exit 2
fi

export PROJECT_ROOT
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"

HIMRD_LIMIT=${HIMRD_LIMIT:-200}
HIMRD_OUTPUT_DIR=${HIMRD_OUTPUT_DIR:-$PROJECT_ROOT/runs/himrd_e2e_llava16_200_20261126}

"$PROJECT_ROOT/.venvs/modern/bin/python" -m survey_exp.himrd_end_to_end \
  --manifest "$PROJECT_ROOT/data/manifests/himrd_safebench_full.jsonl" \
  --prior-outputs "$PROJECT_ROOT/runs/himrd_gpt4omini_safebench200_20261126/outputs.jsonl" \
  --output-dir "$HIMRD_OUTPUT_DIR" \
  --limit "$HIMRD_LIMIT"
