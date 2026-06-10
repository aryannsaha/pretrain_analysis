# UMA / FAIR-Chem Configs

The raw split files can live under `data/processed/...` as `.traj` files, but
UMA fine-tuning should point at FAIR-Chem generated ASE-LMDB directories. Edit
the constants in `scripts/1_UMA_prepare_uma_finetune_from_traj.py`, including
`LOCAL_CHECKPOINT`, then run:

```bash
python scripts/1_UMA_prepare_uma_finetune_from_traj.py
```

That script creates `train/`, `val/`, a generated data-task YAML, and a runnable
`uma_sm_finetune_template.yaml`, then copies the runnable YAMLs into
`configs/uma/`.

If preparing files manually, copy the FAIR-Chem generated files here after
running its dataset/config generation step:

- `uma_sm_finetune_template.yaml`
- `data_task_energy_force_stress.yaml` or the generated `data/uma_conserving_data_task_energy_force_stress.yaml`

Then apply the edits from `Fine-tuning Procedures.md`:

- Use a `job` section compatible with `della-gpu`.
- Set `job.run_dir` to `runs/uma/<run_name>/`.
- Set `runner.train_eval_unit.model.checkpoint_location` to a local file in `models/pretrained/`.
- Remove Hugging Face checkpoint download fields if they are present.
- Use absolute paths for training and validation dataset `src` fields.
- Adjust energy/force/stress loss coefficients as needed.
