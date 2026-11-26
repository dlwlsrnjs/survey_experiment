#!/bin/bash
#SBATCH --job-name=vlmaux-models
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/home/ljk98/survey_experiment
#SBATCH --output=/home/ljk98/survey_experiment/runs/slurm-%x-%j.out
#SBATCH --error=/home/ljk98/survey_experiment/runs/slurm-%x-%j.err
set -euo pipefail

export HF_HOME=/home/ljk98/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

/home/ljk98/RESORT/.venv_q3vl/bin/python scripts/download_required_models.py \
  --status runs/model_download_status.json \
  --cache-dir /home/ljk98/.cache/huggingface/hub
