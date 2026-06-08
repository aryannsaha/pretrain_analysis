# UPET / Metatrain Configs

Use `options.yaml` as the starting point for `mtt train`.

Important fields from the procedure note:

- Put source datasets under `data/processed/upet/` or `data/splits/`.
- Put pretrained or fine-tuned checkpoints under `models/pretrained/` or `models/checkpoints/`.
- For a new fine-tuning run, set `architecture.training.finetune.read_from`.
- For resuming into the same W&B run, update `wandb.resume`, `wandb.id`, and apply the source-code modifications described in `Fine-tuning Procedures.md`.
