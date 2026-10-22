#!/usr/bin/env bash
# Slurm template for the authorized DGX environment; adjust resources and paths first.
#SBATCH --job-name=a6_route32b
#SBATCH --partition=ai-lab
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=a6_route32b_%j.log

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/path/to/dgx/python}"
PYTHONPATH="${PYTHONPATH:-/path/to/dgx/python-packages}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH

"${PYTHON_BIN}" "${SCRIPT_DIR}/project_route_lora_train.py" \
  --config "${SCRIPT_DIR}/project_route_lora_32b_config.json"
