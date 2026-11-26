#!/usr/bin/env bash
#SBATCH --job-name=himrd-4omini-200
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
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

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"
"$PROJECT_ROOT/.venvs/modern/bin/python" -m survey_exp.source_fidelity_runner \
  --manifest "$PROJECT_ROOT/data/manifests/himrd_safebench_full.jsonl" \
  --output-dir "$PROJECT_ROOT/runs/himrd_gpt4omini_safebench200_20261126" \
  --repeats 1 --limit 200 \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o-mini-2024-07-18 --revision 2024-07-18 \
  --api-key-env OPENAI_API_KEY --methods HIMRD \
  --model-substitution "HIMRD GPT-4o-mini/GPT-4o/o1 roles use gpt-4o-mini-2024-07-18; non-GPT components remain source-model stages" \
  --input-price-per-million 0.15 --output-price-per-million 0.60 \
  --no-image-generation
