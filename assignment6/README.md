# Assignment 6 candidate LoRA experiment

**Owner:** Yuchen Yuan. **Status:** the 32B checkpoint was loaded and decompressed on DGX, but training stopped before its first optimizer step; adapter-saving and reload evidence remain pending. This is an individual learning artifact, not a Group 2 model selection decision.

## Question and scope

The narrow skill is configuring, saving, and reloading a LoRA adapter for the 32B Qwen model discussed in class. The approved local checkpoint is `/mnt/nas/models/approved/qwen2.5-32b-instruct-fp8`; its `config.json` identifies a Qwen2 32B architecture with compressed-tensors FP8 weights. The experiment explicitly requests `CompressedTensorsConfig(dequantize=True)` and BF16 weights at load time for training. This is a change from the original plain-BF16-checkpoint plan and is recorded in `experiment.json`. The script never downloads weights automatically.

Eight hand-written synthetic card-complaint examples are included solely to exercise the adapter workflow. They are not the course Development Dataset, contain no real consumer narrative, and provide no credible estimate of bank task quality. `eval` here means four synthetic diagnostic examples; the course held-out dataset is never opened or used.

## Local check

From the repository root, with Python 3.10 or newer:

```bash
python -B assignment6/lora_experiment.py preflight
python -B -m unittest discover -s assignment6 -p 'test_*.py' -v
```

This validates exact field names, synthetic IDs, split separation, duplicate narratives, reference labels, and the frozen experiment configuration. With `--model /mnt/nas/models/approved/qwen2.5-32b-instruct-fp8`, it also checks the local 32B architecture and FP8 configuration using only Python's standard library. It does **not** load weights, validate GPU training, or validate adapter loading.

## DGX run status and evidence

The JupyterLab terminal has no GPU; training requires a separate Slurm job. GPU probe 209 confirmed an NVIDIA GB10 on `spark-4694`. The shared `/opt/sps-ai-lab/venv/bin/python` there has PyTorch 2.13.0, Transformers 5.17.0, and compressed-tensors 0.17.0. PEFT 0.21.2 and Accelerate 1.15.0 are installed only in the owner's `$HOME/a6-4694-python`, which the runner adds to `PYTHONPATH`. CPU probe 216 imported the training components and verified `CompressedTensorsConfig(dequantize=True)`. Import success does **not** prove that the 32B weights load or train.

The checked-in [`run_dgx_32b.sbatch`](run_dgx_32b.sbatch) requests one GB10 on `spark-4694`, four CPU cores, 100 GB of node memory, and one hour. It performs format and tokenizer checks before loading weights, then attempts training and adapter reload evaluation in sequence. The training step dequantizes the approved FP8 checkpoint to BF16, checks the attention projection dtype, and applies LoRA to `q_proj` and `v_proj` (`r=4`, `alpha=8`, four optimizer steps, one synthetic example per step, sequence cap 192, learning rate `2e-4`, seed `2026`). It masks prompt tokens from the loss, saves adapter weights and configuration, and writes `run_manifest.json`. Evaluation reloads the adapter and compares deterministic base and adapted outputs on four distinct synthetic diagnostic cases in `comparison.json`. Logs and artifacts stay in the owner's NAS home directory, outside Git.

An earlier job on `spark-4694` failed before model loading because that node lacks `/opt/sps-ai-lab/venv-train/bin/python`. A revised job pinned to `spark-1338` could not start because the node lacked enough unallocated memory; it was cancelled while still pending. Job **219**, from commit `1f90ef0`, started on `spark-4694`. Its log shows all 1219 weight chunks loaded and model decompression completed. It then failed with `Chat-template prefix mismatch for SYN-T01` while preparing the first training example. No optimizer step, adapter save, or evaluation occurred. The corrected encoder constructs each supervised sequence from the exact generation prompt tokens, answer tokens, and the tokenizer's assistant end token. It prepares all samples before weight loading so another tokenization error will fail quickly.

First run the lightweight `tokenize` command with the installed tokenizer on `spark-4694`; it does not load weights or require a GPU. Only after it succeeds should a new GPU job be submitted. The runner's log and artifacts will be named for the new job ID:

```bash
PYTHONPATH="$HOME/a6-4694-python" /opt/sps-ai-lab/venv/bin/python -B assignment6/lora_experiment.py tokenize --model /mnt/nas/models/approved/qwen2.5-32b-instruct-fp8
```

This command must run on a node where that Python path exists, such as a short Slurm CPU job on `spark-4694`, not in the JupyterHub terminal. Completion of a subsequent GPU run requires `run_manifest.json`, saved adapter files, and `comparison.json`, with log evidence that all steps finished. This cluster has Slurm accounting disabled, so preserve the job log as the primary evidence. Do not silently switch to QLoRA or 8B; that changes the question being tested.

Keep the adapter and all run evidence outside Git. Do not upload model weights, raw course data, credentials, or the held-out CSV. Copy only an approved, redacted run manifest and comparison summary into the assignment record. If package compatibility fails or the run is out of memory, stop and log the exact error; do not silently switch to QLoRA or 8B because that changes the experiment.

## Decision rule

Completion of four training steps and a reload check proves a narrow technical capability only. A higher exact-match count on four synthetic examples is **not** evidence that LoRA should enter the team's complaint PoC. The team should prefer its prompted baseline unless a frozen internal Development comparison shows a meaningful gain on the approved task, including the ambiguous and high-risk strata, while retaining the human review controls. Formal held-out evaluation must wait until prompts, model, adapter, metrics, and thresholds are frozen.

## Documentation used

- [Qwen 2.5 32B Instruct model card](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
- [PEFT LoRA configuration and model guide](https://huggingface.co/docs/peft/guides/peft_model_config)
- [Transformers PEFT training and checkpoint guide](https://huggingface.co/docs/transformers/peft)
- [Transformers chat template guidance](https://huggingface.co/docs/transformers/chat_templating)
- [Transformers compressed-tensors loading modes](https://huggingface.co/docs/transformers/quantization/compressed_tensors)

Version-sensitive settings remain conditional on the actual DGX package versions and controlled run.
