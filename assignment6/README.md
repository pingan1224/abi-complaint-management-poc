# Assignment 6: 32B LoRA learning experiment

**Owner:** Yuchen Yuan. **Status:** Completed individual candidate experiment; team adoption remains a separate decision.

## Question and scope

Can a small LoRA adapter be trained on the approved 32B Qwen checkpoint, saved, reloaded, and compared with the base model in the controlled course environment? This experiment tests that workflow. It does not establish better complaint handling.

The approved checkpoint uses compressed-tensors FP8 weights. The training run requested dequantization to BF16 before applying LoRA to the attention `q_proj` and `v_proj` modules. The checked-in [experiment configuration](experiment.json) freezes the principal settings: rank 4, alpha 8, four optimizer steps, sequence cap 192, learning rate `2e-4`, and seed 2026. Model weights are loaded from the approved local checkpoint; the script does not download them automatically.

Eight hand-written card-complaint examples exercise the workflow. Four are used for training and four distinct examples for a diagnostic comparison. They contain no real consumer narratives and are not the course Development or held-out datasets.

## Reproducible checks

From the repository root, with Python 3.10 or newer:

```bash
python -B assignment6/lora_experiment.py preflight
python -B -m unittest discover -s assignment6 -p 'test_*.py' -v
```

The preflight checks the example schema, synthetic IDs, split separation, duplicate narratives, reference labels, and the frozen configuration. The `tokenize` command checks supervised sequence construction with the installed model tokenizer before weight loading. These checks do not by themselves prove that GPU training or adapter reload works.

The [batch runner](run_dgx_32b.sbatch) runs preflight, tokenization, training, and adapter reload comparison in order. It writes a run manifest, adapter files, and a comparison record outside Git. The training code masks prompt tokens from the loss and checks the projection dtype after loading the model.

## Experiment result

The full DGX run completed four optimization steps, saved the adapter, and reloaded it for comparison. On the four **synthetic diagnostic cases**, the base model matched all four reference labels and the adapted model also matched all four. There was **no observed improvement** on these cases. The successful reload demonstrates a working LoRA artifact; the tiny synthetic comparison cannot estimate performance on real complaints.

An earlier attempt exposed a chat-template prefix mismatch while constructing supervised examples. The encoder was corrected to build training sequences from the exact generation prompt tokens, answer tokens, and assistant end token. A tokenizer check on the target environment then passed before the successful full run. This failure and correction are part of the experiment record.

The detailed run log, manifest, comparison record, and adapter remain in the controlled course environment. The individual assignment record cites that evidence and distinguishes workflow completion from evidence of model benefit. No model weights, raw course data, credentials, or held-out evaluation data are committed here.

## Professional judgment

The team should keep its prompted baseline unless a frozen comparison on the approved Development data shows a repeatable, meaningful gain, including ambiguous and high-risk cases, with human review controls retained. Formal held-out evaluation should occur only after prompts, model, adapter, metrics, and thresholds are frozen. This individual result does not select a model for the group.

## Documentation used

- [Qwen 2.5 32B Instruct model card](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
- [PEFT LoRA configuration and model guide](https://huggingface.co/docs/peft/guides/peft_model_config)
- [Transformers PEFT training and checkpoint guide](https://huggingface.co/docs/transformers/peft)
- [Transformers chat template guidance](https://huggingface.co/docs/transformers/chat_templating)
- [Transformers compressed-tensors loading modes](https://huggingface.co/docs/transformers/quantization/compressed_tensors)

Version-sensitive settings remain conditional on the target environment and controlled run.
