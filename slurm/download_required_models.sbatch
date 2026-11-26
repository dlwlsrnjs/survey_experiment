#!/usr/bin/env bash
#SBATCH --job-name=vlmaux-models
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROFILE=${PROFILE:-core}
export HF_HOME=${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}
"$PROJECT_ROOT/.venvs/modern/bin/python" "$PROJECT_ROOT/scripts/download_required_models.py" \
  --lock "$PROJECT_ROOT/configs/assets.lock.json" --profile "$PROFILE" \
  --status "$PROJECT_ROOT/runs/model_download_status.json" --cache-dir "$HF_HOME/hub"
