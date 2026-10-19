# Assignment 6 LoRA learning candidate

This candidate learns to save and reload a LoRA adapter on the school DGX Spark.
It is a bounded experiment, pending team review. It is not an adopted prototype.

Use the school's existing `/opt/sps-ai-lab/venv-train/bin/python`, approved local
Llama 3.1 8B Instruct weights, Development CSV, and Assignment 5 grouped manifest.
No packages or weights are downloaded by this script.

Prepare the log directory and submit from the personal worktree:

```bash
cd "$HOME/abi-assignment6-lora"
mkdir -p assignment6/evidence
sbatch --output=assignment6/evidence/run1_slurm.log assignment6/run_experiment.sbatch
```

The experiment refuses to overwrite an existing output directory. A rerun needs
a new output directory and log file. Do not delete a failed run to hide it.
The Slurm script assumes this personal worktree location; adjust paths explicitly
when porting. A shard is shared GPU access, not guaranteed dedicated GPU memory.

The source is committed before execution. `frozen_config.json` records its commit,
script hash, versions, seed, LoRA settings, 48 steps, and the decision rule.
The run hashes the local weights and configuration rather than inferring a Hub
revision. Only assistant tokens receive training loss. No source text is silently
truncated; eligibility requires at most 512 tokens for the complete sequence.

The existing train/validation group boundaries are retained. The subset uses 48
Clear training records (16 per observed urgency class) and 12 validation records
(9 Clear and 3 uncertain), selected before model evaluation. No narrative group
crosses the boundary. The selected subset is short and class balanced, so its
scores are not population performance estimates. Low coverage remains absent.

Three conditions use the same base weights: A uses a short task prompt; B uses
explicit schema/uncertainty guidance; C uses the saved/reloaded adapter with B's
identical prompt. Generation is greedy with a 128-token output limit. Raw replies
are retained, including invalid JSON and truncation. Clear-reference agreement is
reported separately from schema validity and is not official bank-policy accuracy.
Uncertain cases are not scored against a purported gold label. Inspect content
manually before interpreting the frozen decision rule.

`reload_check.json` records exact adapter tensor equality after a fresh base and
adapter load, active adapter names, and greedy response equality on one probe.
Equality on one response is a bounded check, not universal determinism proof.
Adapter weights remain on DGX under `evidence/run1/adapter/` and are excluded from
Git. Source, configuration, timing, provenance, and response evidence are tracked.

Official references, checked October 19 2026:

- https://huggingface.co/docs/peft/package_reference/peft_model
- https://huggingface.co/docs/transformers/chat_templating
- https://build.nvidia.com/spark/pytorch-fine-tune/instructions
- https://slurm.schedmd.com/gres.html
