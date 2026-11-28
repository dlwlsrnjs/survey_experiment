#!/usr/bin/env bash
#SBATCH --job-name=openbest-download
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$PROJECT_ROOT" || ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
  echo "Cannot resolve project root." >&2
  exit 2
fi
PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venvs/modern/bin/python}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
: "${REPO_ID:?REPO_ID is required}"
: "${REVISION:?REVISION is required}"

"$PYTHON_BIN" - "$REPO_ID" "$REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, revision = sys.argv[1:]
print(snapshot_download(repo_id=repo_id, revision=revision))
PY

