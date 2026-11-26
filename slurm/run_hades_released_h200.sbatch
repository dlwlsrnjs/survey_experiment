#!/usr/bin/env bash
#SBATCH --job-name=hades-rel-200
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=5-00:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$PROJECT_ROOT" || ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
  echo "Cannot resolve project root." >&2
  exit 2
fi
PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd)
export PROJECT_ROOT
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/HADES:$PROJECT_ROOT/third_party/HADES/eval/evaluate/inference${PYTHONPATH:+:$PYTHONPATH}"

HADES_LIMIT=${HADES_LIMIT:-1200}
HADES_OUTPUT_DIR=${HADES_OUTPUT_DIR:-$PROJECT_ROOT/runs/hades_released_llava15_200goals_1200rows_20261126}
LLAVA_PATH="$HF_HOME/hub/models--liuhaotian--llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234"
BEAVER_PATH="$HF_HOME/hub/models--PKU-Alignment--beaver-dam-7b/snapshots/3a5b70888869f04336d0d8344185ed74268ef896"

cd "$PROJECT_ROOT"
"$PROJECT_ROOT/.venvs/legacy-vlm/bin/python" -m survey_exp.hades_released_eval \
  --manifest "$PROJECT_ROOT/data/manifests/hades_released_200.jsonl" \
  --output-dir "$HADES_OUTPUT_DIR" \
  --limit "$HADES_LIMIT" \
  --llava-path "$LLAVA_PATH" \
  --beaver-path "$BEAVER_PATH"
