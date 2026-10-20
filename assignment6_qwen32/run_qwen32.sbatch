#!/bin/bash
#SBATCH --partition=ai-lab
#SBATCH --nodelist=spark-4694
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:25:00
#SBATCH --job-name=abi6-qwen32-lora
set -euo pipefail
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONPATH="$HOME/abi-assignment6-qwen32/.dependencies${PYTHONPATH:+:$PYTHONPATH}"
cd "$HOME/abi-assignment6-qwen32"
/opt/sps-ai-lab/venv/bin/python -u assignment6_qwen32/lora_learning_qwen32.py \
 --model /mnt/nas/models/approved/qwen2.5-32b-instruct-fp8 \
 --csv "$HOME/abi-assignment5-inputs/ABI_Bank_Complaints_Development_8000.csv" \
 --manifest "$HOME/abi-assignment5-evidence/data_run/internal_split_manifest.csv" \
 --selection assignment6_qwen32/pinned_case_ids.json \
 --output assignment6_qwen32/evidence/run4 --steps 48
