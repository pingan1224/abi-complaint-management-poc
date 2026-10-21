#!/usr/bin/env bash
#SBATCH --job-name=a6-32b-lora
#SBATCH --partition=ai-lab
#SBATCH --nodelist=spark-4694
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/nas/students/yy6056/a6-slurm-%j.log

set -Eeuo pipefail

PYTHON=/opt/sps-ai-lab/venv/bin/python
MODEL=/mnt/nas/models/approved/qwen2.5-32b-instruct-fp8
REPO="$HOME/assignment6-candidate"
RUN="$HOME/a6-run-$SLURM_JOB_ID"

export PYTHONPATH="$HOME/a6-4694-python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
printf 'UTC start: '
date -u +%FT%TZ
printf 'Host: %s\nJob: %s\nCommit: ' "$(hostname)" "$SLURM_JOB_ID"
git rev-parse HEAD
test -x "$PYTHON" || { printf 'Training Python is missing on %s: %s\n' "$(hostname)" "$PYTHON" >&2; exit 1; }
nvidia-smi

"$PYTHON" -c 'from transformers import CompressedTensorsConfig; c = CompressedTensorsConfig(dequantize=True); assert c.dequantize is True; print("FP8 dequantization flag: True")'
"$PYTHON" -B assignment6/lora_experiment.py preflight --model "$MODEL"
"$PYTHON" -B assignment6/lora_experiment.py tokenize --model "$MODEL"
"$PYTHON" -B assignment6/lora_experiment.py train --model "$MODEL" --output "$RUN"
"$PYTHON" -B assignment6/lora_experiment.py evaluate --model "$MODEL" --output "$RUN"

printf 'UTC finish: '
date -u +%FT%TZ
printf 'Run directory: %s\n' "$RUN"
