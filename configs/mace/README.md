# MACE Configs

Use `configs.yaml` with:

```bash
mace_run_train --config=configs/mace/configs.yaml
```

Follow `Fine-tuning Procedures.md` for:

- choosing `foundational_model` or a local checkpoint,
- enabling stress logging with `compute_stress: true` and `loss: stress`,
- setting `atomic_numbers` and matching `E0s`,
- using a MACE fork if you want MAE logging to W&B.
