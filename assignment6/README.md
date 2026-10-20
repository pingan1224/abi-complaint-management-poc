# Assignment 6 candidate LoRA experiment

**Owner:** Yuchen Yuan. **Status:** local data/configuration preflight complete; DGX training and adapter-loading evidence pending access to an approved terminal or notebook. This is an individual learning artifact, not a Group 2 model selection decision.

## Question and scope

The narrow skill is configuring, saving, and reloading a LoRA adapter for the 32B Qwen model discussed in class. The DGX Open WebUI currently displays **Qwen 2.5 32B**, but that display name does not reveal the local Hugging Face checkpoint path, precision, or training permissions. The intended model in `experiment.json` is `Qwen/Qwen2.5-32B-Instruct`; the experiment must use the administrator-approved **local** checkpoint directory for that exact compatible model. It never downloads weights automatically.

Eight hand-written synthetic card-complaint examples are included solely to exercise the adapter workflow. They are not the course Development Dataset, contain no real consumer narrative, and provide no credible estimate of bank task quality. `eval` here means four synthetic diagnostic examples; the course held-out dataset is never opened or used.

## Local check

From the repository root, with Python 3.10 or newer:

```bash
python -B assignment6/lora_experiment.py preflight
python -B -m unittest discover -s assignment6 -p 'test_*.py' -v
```

This validates exact field names, synthetic IDs, split separation, duplicate narratives, reference labels, and the frozen experiment configuration. It does **not** validate GPU training or adapter loading.

## DGX run, after an approved training entrance is provided

Ask the course administrator for (1) a terminal/SSH/Jupyter training entrance, (2) the local full-precision or BF16 Hugging Face checkpoint directory corresponding to Qwen 2.5 32B, (3) permission and storage location for a personal adapter, and (4) the installed `torch`, `transformers`, `peft`, and `accelerate` versions. A chat-only Open WebUI account cannot execute these commands. Do not install packages into a shared environment without administrator guidance.

Check out the candidate commit on DGX and choose a **new directory outside Git** for the run. Replace the two placeholders with paths approved for your account:

```bash
python -B assignment6/lora_experiment.py preflight
python -B assignment6/lora_experiment.py train --model /approved/local/Qwen2.5-32B-Instruct --output /approved/personal/yuchen/a6-run-01
python -B assignment6/lora_experiment.py evaluate --model /approved/local/Qwen2.5-32B-Instruct --output /approved/personal/yuchen/a6-run-01
```

The training command applies ordinary BF16 LoRA to `q_proj` and `v_proj` (`r=4`, `alpha=8`, dropout `0`, four optimizer steps, one synthetic example per step, sequence cap 192, learning rate `2e-4`, seed `2026`). It freezes the base, checks the trainable-parameter fraction, masks prompt tokens from the supervised loss, saves only adapter weights/configuration, and writes `run_manifest.json` with the Git commit, package versions, GPU, model path, file hashes, loss, and run time. The evaluation command **reloads the saved adapter** and compares deterministic base and adapted outputs on the same four synthetic diagnostic cases, writing `comparison.json`.

Keep the adapter and all run evidence outside Git. Do not upload model weights, raw course data, credentials, or the held-out CSV. Copy only an approved, redacted run manifest and comparison summary into the assignment record. If the model path is quantized or the run is out of memory, stop and log the exact error; do not silently switch to QLoRA or 8B because that changes the experiment.

## Decision rule

Completion of four training steps and a reload check proves a narrow technical capability only. A higher exact-match count on four synthetic examples is **not** evidence that LoRA should enter the team's complaint PoC. The team should prefer its prompted baseline unless a frozen internal Development comparison shows a meaningful gain on the approved task, including the ambiguous and high-risk strata, while retaining the human review controls. Formal held-out evaluation must wait until prompts, model, adapter, metrics, and thresholds are frozen.

## Documentation used

- [Qwen 2.5 32B Instruct model card](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
- [PEFT LoRA configuration and model guide](https://huggingface.co/docs/peft/guides/peft_model_config)
- [Transformers PEFT training and checkpoint guide](https://huggingface.co/docs/transformers/peft)
- [Transformers chat template guidance](https://huggingface.co/docs/transformers/chat_templating)

Version-sensitive settings remain conditional on the actual DGX package versions and controlled run.
