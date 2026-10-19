#!/bin/bash
#SBATCH --partition=ai-lab
#SBATCH --gres=shard:gb10:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=abi6-lora-learning
set -euo pipefail
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
cd "$HOME/abi-assignment6-lora"
/opt/sps-ai-lab/venv-train/bin/python -u assignment6/lora_learning.py \
  --model /mnt/nas/models/approved/llama-3.1-8b-instruct \
  --csv "$HOME/abi-assignment5-inputs/ABI_Bank_Complaints_Development_8000.csv" \
  --manifest "$HOME/abi-assignment5-evidence/data_run/internal_split_manifest.csv" \
  --output assignment6/evidence/run1 \
  --steps 48
