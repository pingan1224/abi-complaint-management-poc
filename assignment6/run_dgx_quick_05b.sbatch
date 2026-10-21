#!/usr/bin/env bash
#SBATCH --job-name=a6-quick05b-lora
#SBATCH --partition=ai-lab
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/nas/students/yy6056/a6-quick05b-%j.log

set -Eeuo pipefail

if [ -x /opt/sps-ai-lab/venv-train/bin/python ]; then
    PYTHON=/opt/sps-ai-lab/venv-train/bin/python
    export PYTHONPATH="$HOME/a6-python${PYTHONPATH:+:$PYTHONPATH}"
elif [ -x /opt/sps-ai-lab/venv/bin/python ]; then
    PYTHON=/opt/sps-ai-lab/venv/bin/python
    export PYTHONPATH="$HOME/a6-4694-python${PYTHONPATH:+:$PYTHONPATH}"
else
    echo 'Neither verified course Python environment is present' >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "$HOME/assignment6-candidate"
printf 'UTC start: '
date -u +%FT%TZ
printf 'Host: %s\nJob: %s\nPython: %s\nCommit: ' "$(hostname)" "$SLURM_JOB_ID" "$PYTHON"
git rev-parse HEAD
nvidia-smi
"$PYTHON" -B assignment6/quick_lora_05b.py
printf 'UTC finish: '
date -u +%FT%TZ
