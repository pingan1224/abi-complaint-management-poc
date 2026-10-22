# Assignment 6 Submission Package

This package documents a corrected LoRA experiment using the instructor-directed 32B model class on the NYU DGX Spark environment. It is a candidate contribution for the team's employee-assisted credit-card dispute-routing prototype, not a production approval.

## Contents

- `LoRA_Learning_and_Experiment_Record.docx`: two-page learning record, independent recommendation, failure diagnosis, AI verification, handoff, and a 75–150 word reflection.
- `project_route_lora_train.py`: corrected training and evaluation implementation. Prompt and answer are tokenized jointly; only answer tokens contribute to loss. It compares sequence log-likelihood with a mean-token sensitivity analysis and checks adapter reload parity.
- `project_route_lora_32b_config.json`: portable configuration template. Replace the clearly marked paths for the authorized DGX model and Development split before reuse.
- `run_project_route_32b_dgx.sh`: portable Slurm launcher, using the same GPU class and resource request as the verified DGX run.
- `dgx_32b_corrected_rerun_result.json`: sanitized aggregate metrics from the corrected Slurm run. It contains no record-level data, predictions, or personal NAS path.
- `DGX_execution_evidence.txt`: sanitized execution details and a short original Slurm-log excerpt.
- `assignment6_final_update.patch`: patch for the final candidate-branch commit; its full commit ID is recorded in `FINAL_COMMIT.txt`.

## Actual DGX experiment

Diagnostic Slurm job 255 completed on `spark-4694` in 971.4 seconds with exit code 0. It used the approved Qwen2.5-32B-Instruct FP8 checkpoint, dequantized to BF16; Torch 2.13.0+cu130, Transformers 5.17.0, and PEFT 0.21.2. The narrative-grouped Development split contained 330 Clear-labeled training examples and 71 internal-validation examples; the official Held-out Evaluation split was not used. Mean epoch loss was 2.9943; first 50 step mean was 2.8968 and last 50 was 2.9586, so loss did not trend down. Peak CUDA allocation of 66,722,334,720 bytes is 62.1 GiB and represents the whole 32B-model run, not LoRA-only memory. The pilot's 61.2 GiB figure was from a non-matched run and cannot establish an exact LoRA increment. The main incremental costs to weigh are training time and adapter lifecycle.

Under sequence log-likelihood, base and adapted both scored 61/71 (85.9%) and fraud/security recall was 6/14 (42.9%). The mean Billing-minus-Fraud sequence log-likelihood margin was 4.760 for both; the mean absolute per-example margin change was exactly 0 over 71 examples, with zero sign flips. A post-run CPU inspection found all 128 saved LoRA B tensors exactly zero. Together with the non-decreasing loss windows, this indicates the saved adapter had no measurable update in this run; the training/update path needs debugging before calling this a valid LoRA learning result. It does not establish that LoRA is generally ineffective. Always predicting the majority billing route scores 57/71 (80.3%) but detects no fraud cases. A secondary scoring-method check changed one validation prediction, which is not evidence about the LoRA-versus-base difference.

## Corrected failure analysis and transferable lesson

The earlier full run tokenized prompt and answer separately. Because BPE tokenization can merge across the textual boundary, 285/330 examples had boundary merges and answer loss masking was misaligned. The corrected script tokenizes the concatenated prompt and answer and masks using their shared token prefix. This boundary-safe method transfers to other causal language models and tokenizers. Earlier backward failures were addressed by disabling PEFT inference mode and enabling input gradients. A separate OOM came from loading compressed weights and then copying the full model to CUDA; direct device placement avoided that duplicate allocation.

## Data and evaluation boundaries

No complaint narratives, source CSVs, row-level predictions, adapter weights, or official holdout data are included. Two official held-out examples were inadvertently previewed earlier; they were not used for training or tuning. The team should account for this exposure before final evaluation. Labels are synthetic instructional references rather than adjudicated bank policy. Route suggestions require employee confirmation.

## Repository status

The candidate lives on the personal branch `louyinuo/a6-lora-candidate`. The package includes a patch and final full commit ID so the submitted files can be tied to the branch history. The portable config anonymizes personal NAS locations; model, split, hyperparameters, and DGX resource request match the recorded run. AI assisted implementation, debugging, and drafting; the report identifies what was verified and the limits of the experiment. Review this candidate before integrating it into the team prototype.
