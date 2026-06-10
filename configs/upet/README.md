# UPET / Metatrain Configs

Use `options.yaml` as the starting point for `mtt train`.

Important fields from the procedure note:

- Put source datasets under `data/processed/upet/` or `data/splits/`.
- Put pretrained or fine-tuned checkpoints under `models/pretrained/` or `models/checkpoints/`.
- For a new fine-tuning run, set `architecture.training.finetune.read_from`.
- For resuming into the same W&B run, update `wandb.resume`, `wandb.id`, and apply the source-code modifications described in `Fine-tuning Procedures.md`.

The local `scripts/run_upet.sh` wrapper handles the usual resume flow:

```bash
scripts/run_upet.sh configs/upet/options.yaml
```

It runs metatrain from `runs/upet/<run-id>/`, finds the newest `.ckpt` in that
run directory, and passes it back as
`architecture.training.finetune.read_from`. On the first run, no checkpoint is
found and the value in `options.yaml` is used. On later runs, the latest saved
fine-tuning checkpoint is used and W&B resumes with the same `wandb.id`.

`architecture.training.num_epochs` is the total target epoch count, not the
number of additional epochs. If a run finished at epoch 9 and `num_epochs` is
10, rerunning can continue through epoch 10; if the run already reached
`num_epochs`, raise the value before rerunning.

Useful overrides:

```bash
UPET_RUN_ID=my_run scripts/run_upet.sh configs/upet/options.yaml
UPET_RESUME_FROM=/path/to/model_12.ckpt scripts/run_upet.sh configs/upet/options.yaml
scripts/run_upet.sh configs/upet/options.yaml -r architecture.training.num_epochs=20
```
