#!/bin/bash
#SBATCH --job-name=team7_integrated
#SBATCH --partition=ai-lab
#SBATCH --nodelist=spark-4694
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --output=team7-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
PY=/opt/sps-ai-lab/venv/bin/python
export PYTHONPATH="${TEAM7_PACKAGES:-$HOME/abi-assignment6-qwen32/.dependencies}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
OUT="evidence-dgx-${SLURM_JOB_ID}"
mkdir "$OUT"
git rev-parse HEAD | tee "$OUT/commit.txt"
hostname | tee "$OUT/host.txt"
"$PY" -B -m unittest -v test_workflow > "$OUT/tests.txt" 2>&1
"$PY" -B -u model_service.py > "$OUT/model_service.log" 2>&1 &
SERVICE_PID=$!
trap 'kill "$SERVICE_PID" 2>/dev/null || true' EXIT
for i in $(seq 1 360); do
  if grep -q 'MODEL_READY' "$OUT/model_service.log"; then break; fi
  if ! kill -0 "$SERVICE_PID" 2>/dev/null; then cat "$OUT/model_service.log"; exit 1; fi
  sleep 1
done
grep -q 'MODEL_READY' "$OUT/model_service.log"
"$PY" -B run_evidence.py --output "$OUT/run" --endpoint http://127.0.0.1:18707/v1/chat/completions --model qwen2.5-32b-team7-base | tee "$OUT/summary.txt"
echo TEAM7_INTEGRATED_PASS
