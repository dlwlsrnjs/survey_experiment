#!/bin/bash
#SBATCH --job-name=vlmaux-envs
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/ljk98/survey_experiment
#SBATCH --output=/home/ljk98/survey_experiment/runs/slurm-%x-%j.out
#SBATCH --error=/home/ljk98/survey_experiment/runs/slurm-%x-%j.err
set -euo pipefail

export UV_CACHE_DIR=/home/ljk98/.cache/uv
export PYTHONUNBUFFERED=1
bash scripts/setup_model_environments.sh
