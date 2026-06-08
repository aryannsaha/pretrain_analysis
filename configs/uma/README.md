# UMA / FAIR-Chem Configs

Copy the FAIR-Chem generated files here after running its dataset/config generation step:

- `uma_sm_finetune_template.yaml`
- `data_task_energy_force_stress.yaml` or the generated `data/uma_conserving_data_task_energy_force_stress.yaml`

Then apply the edits from `Fine-tuning Procedures.md`:

- Use a `job` section compatible with `della-gpu`.
- Set `job.run_dir` to `runs/uma/<run_name>/`.
- Set `runner.train_eval_unit.model.checkpoint_location` to a local file in `models/pretrained/`.
- Remove Hugging Face checkpoint download fields if they are present.
- Use absolute paths for training and validation dataset `src` fields.
- Adjust energy/force/stress loss coefficients as needed.
