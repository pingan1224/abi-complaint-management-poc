# Assignment 6 Submission Package

This package documents a corrected LoRA experiment using the instructor-directed 32B model class on the NYU DGX Spark environment. It is a candidate contribution for the team's employee-assisted credit-card dispute-routing prototype, not a production approval.

## Contents

- `LoRA_Learning_and_Experiment_Record.docx`: two-page learning record, independent recommendation, failure diagnosis, AI verification, handoff, and a 75–150 word reflection draft.
- `project_route_lora_train.py`: corrected training and evaluation implementation. Prompt and answer are tokenized jointly; only answer tokens contribute to loss. It compares sequence log-likelihood with a mean-token sensitivity analysis and checks adapter reload parity.
- `project_route_lora_32b_config.json`: portable configuration template. Replace the clearly marked paths for the authorized DGX model and Development split before reuse.
- `run_project_route_32b_dgx.sh`: Slurm job template; adjust partition/resource/path settings to the target DGX account.
- `dgx_32b_corrected_rerun_result.json`: sanitized aggregate metrics from completed Slurm job 253. It contains no record-level data, predictions, or user/NAS paths.
- `DGX_execution_evidence.txt`: concise record of the verified environment, run completion, scope, and evidence limitations.

## Actual DGX experiment

Job 253 ran on `spark-4694` with an NVIDIA GB10. It used the approved Qwen2.5-32B-Instruct FP8 checkpoint, dequantized to BF16 for training; Torch 2.13.0+cu130, Transformers 5.17.0, and PEFT 0.21.2. The existing narrative-grouped Development split contained 330 Clear-labeled training examples and 71 internal-validation examples. The official Held-out Evaluation split was not used. Mean epoch loss was 2.9943; 4,194,304 parameters were trainable out of 32,772,723,712. Runtime, including model load, baseline scoring, training, adapted scoring, and reload, was 659.0 seconds; peak CUDA allocation was 66,722,334,720 bytes (62.2 GiB). Adapter reload reproduced predictions exactly.

Under sequence log-likelihood, base and adapted both scored 61/71 (85.9%) and fraud/security recall was 6/14 (42.9%). Under mean-token scoring, both scored 60/71 (84.5%) and fraud/security recall was 5/14 (35.7%). Always predicting the majority billing route scores 57/71 (80.3%) but detects no fraud cases. The result therefore does not demonstrate a robust LoRA gain; the recommendation is not to adopt this adapter yet.

## Corrected failure analysis and transferable lesson

The earlier full run tokenized prompt and answer separately. Because BPE tokenization can merge across the textual boundary, 285/330 examples had boundary merges and answer loss masking was misaligned. The corrected script tokenizes the concatenated prompt and answer and masks using their shared token prefix. This boundary-safe method transfers to other causal language models and tokenizers. Earlier backward failures were addressed by disabling PEFT inference mode and enabling input gradients. A separate OOM came from loading compressed weights and then copying the full model to CUDA; direct device placement avoided that duplicate allocation.

## Data and evaluation boundaries

No complaint narratives, source CSVs, row-level predictions, adapter weights, or official holdout data are included. Two official held-out examples were inadvertently previewed earlier; they were not used for training or tuning. The team should account for this exposure before final evaluation. Labels are synthetic instructional references rather than adjudicated bank policy. Route suggestions require employee confirmation.

## Repository status

The candidate lives on the personal branch `louyinuo/a6-lora-candidate`. The repository has no configured remote, so this package does not claim a GitHub push. Commit information is recorded after the local candidate commit. Review the work before integrating it into the team prototype. The reflection is a draft; the student should personally review the evidence and revise the reflection to accurately describe their own learning before submission.
