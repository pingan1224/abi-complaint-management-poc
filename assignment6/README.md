# Assignment 6 candidate LoRA experiment

**Owner:** Yuchen Yuan. **Status:** local format checks complete and a two-minute Slurm GPU probe succeeded; DGX LoRA training and adapter-loading evidence remain pending. This is an individual learning artifact, not a Group 2 model selection decision.

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

## DGX run, after the training environment is verified

JupyterLab Terminal access and Slurm GPU allocation have been confirmed: probe job 209 ran `nvidia-smi` on an NVIDIA GB10 at `spark-4694`. This was **not** a LoRA run. The JupyterHub Python lacks training packages. The course `/opt/sps-ai-lab/venv-train/bin/python` has `torch 2.13.0+cu130`, `transformers 5.19.0`, `peft 0.21.2`, and `accelerate 1.15.0`, but **not** `compressed-tensors`. The separate `/opt/sps-ai-lab/venv/bin/python` has `compressed-tensors 0.17.0` but lacks PEFT and Accelerate. Use a separate Slurm GPU job for training, since the JupyterLab terminal's own job has no GPU. Do not install packages into a shared environment.

Place the single missing package in personal storage and verify imports before using GPU time:

```bash
mkdir -p "$HOME/a6-python"
/opt/sps-ai-lab/venv-train/bin/python -m pip install --no-deps --target "$HOME/a6-python" 'compressed-tensors==0.17.0'
/opt/sps-ai-lab/venv-train/bin/python -m pip install --no-deps --target "$HOME/a6-python" loguru
/opt/sps-ai-lab/venv-train/bin/python -m pip install --target "$HOME/a6-python" 'pydantic>=2,<3'
PYTHONPATH="$HOME/a6-python" /opt/sps-ai-lab/venv-train/bin/python -c 'import torch, transformers, peft, accelerate, compressed_tensors; print("All five imports succeeded")'
```

The `--no-deps` flag avoids copying the course's large PyTorch stack. Import checks then exposed two declared compressed-tensors dependencies: missing `loguru` and `pydantic`. They are installed separately above; the pydantic command includes its own small dependencies. A successful import is a prerequisite, not proof that 32B weights fit or train. If installation or import fails again, record the error and stop before submitting a GPU training job.

Check out the candidate commit on DGX and choose a **new directory outside Git** for the run. The following commands belong **inside an allocated Slurm GPU job**, not in the JupyterLab terminal directly. Save the output in your personal NAS directory:

```bash
/opt/sps-ai-lab/venv-train/bin/python -B assignment6/lora_experiment.py preflight --model /mnt/nas/models/approved/qwen2.5-32b-instruct-fp8
/opt/sps-ai-lab/venv-train/bin/python -B assignment6/lora_experiment.py train --model /mnt/nas/models/approved/qwen2.5-32b-instruct-fp8 --output "$HOME/a6-run-01"
/opt/sps-ai-lab/venv-train/bin/python -B assignment6/lora_experiment.py evaluate --model /mnt/nas/models/approved/qwen2.5-32b-instruct-fp8 --output "$HOME/a6-run-01"
```

The training command dequantizes the approved FP8 checkpoint to BF16, verifies the loaded attention projection is BF16, and applies LoRA to `q_proj` and `v_proj` (`r=4`, `alpha=8`, dropout `0`, four optimizer steps, one synthetic example per step, sequence cap 192, learning rate `2e-4`, seed `2026`). It freezes the base, checks the trainable-parameter fraction, masks prompt tokens from the supervised loss, saves only adapter weights/configuration, and writes `run_manifest.json` with the Git commit, package versions, GPU, model path, file hashes, loss, and run time. The evaluation command **reloads the saved adapter** and compares deterministic base and adapted outputs on the same four synthetic diagnostic cases, writing `comparison.json`. This remains untested on the actual DGX packages and may exceed available memory.

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
