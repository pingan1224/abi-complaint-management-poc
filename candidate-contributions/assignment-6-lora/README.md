# Assignment 6 LoRA Candidate

This folder contains Lou's candidate contribution for the team's review. It is isolated on branch `louyinuo/a6-lora-candidate` and is not merged into `main`.

Initial contribution commit: `13a7180` (`Add Assignment 6 LoRA candidate contribution`).

## Contribution

- The experiment record documents the project-aligned DGX Spark run, independent recommendation, failure diagnosis, AI verification, team handoff, and reflection.
- `project_route_lora_train.py` and `project_route_lora_config.json` define the candidate LoRA training and Development-only internal validation.
- `prepare_project_development.py` reproduces the scope and split filtering when the authorized local Development source files are available.
- `project_route_experiment_result.json` contains aggregate results only. DGX job 194 ran on `spark-4694` (NVIDIA GB10); base internal-validation accuracy was 80.3% and adapted accuracy was 94.4% on 71 cases. Adapter reload reproduced identical predictions.
- `A6_project_route_lora_run.ipynb` is the DGX execution notebook used to launch the run.

## Data and evaluation boundary

No complaint narratives, raw CSVs, row-level predictions, or official Held-out Evaluation data are included in Git. The model input is the complaint narrative only; the target contains only Clear instructional labels. Internal validation is a Development-only diagnostic and is not the team's official held-out evaluation. Course labels are synthetic instructional references. The model only suggests a route; employee review remains required.

Two held-out rows were inadvertently previewed before the evaluation boundary was frozen. They were not used in training or tuning, and this is disclosed in the experiment record.

## Team handoff

Compare base, prompted, and adapted models on the same frozen team-approved held-out cases. Review per-route errors, escalation safety, latency, and cost before deciding whether to carry LoRA into the shared prototype. Do not merge this candidate without team review.
