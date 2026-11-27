#!/usr/bin/env bash
#SBATCH --job-name=ob-models
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-17%2
#SBATCH --output=runs/slurm-%x-%A_%a.out
#SBATCH --error=runs/slurm-%x-%A_%a.err
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$PROJECT_ROOT" || ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
  echo "Cannot resolve project root." >&2
  exit 2
fi
PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venvs/modern/bin/python}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/download_open_best_asset.py \
  --lock configs/assets.lock.json \
  --index "$SLURM_ARRAY_TASK_ID" \
  --status-dir runs/open_best_download_status
