# pretrain_analysis

Workspace for fine-tuning pre-trained atomistic models on `della-gpu`.

The detailed notes live in [Fine-tuning Procedures.md](Fine-tuning%20Procedures.md). This repository is organized so the UMA, UPET, and MACE workflows can share data, checkpoints, logs, and run outputs without mixing generated artifacts into source control.

## Environment

Create or update the conda environment:

```bash
conda env create -f environment.yml
conda activate pretrain_analysis_env
python -m pip install -e ".[dev,analysis]"
```

If the environment already exists:

```bash
conda env update -n pretrain_analysis_env -f environment.yml --prune
conda activate pretrain_analysis_env
python -m pip install -e ".[dev,analysis]"
```

Install workflow-specific packages only when you are ready to run that workflow, because FAIR-Chem, metatrain, MACE, PyTorch, and CUDA builds are cluster-specific.

## Where Things Go

- `Fine-tuning Procedures.md`: human reference notes for UMA, UPET, and MACE.
- `configs/uma/`: FAIR-Chem/UMA config files. Copy generated UMA YAMLs here, then edit paths/checkpoints according to the procedure note.
- `configs/upet/`: `metatrain`/UPET `options.yaml` files.
- `configs/mace/`: MACE `configs.yaml` files.
- `data/raw/`: original trajectories or source data. Do not commit large data.
- `data/interim/`: temporary conversions, filtered structures, or staging data.
- `data/processed/`: workflow-ready datasets.
- `data/splits/`: train/validation/test split files or split directories.
- `models/pretrained/`: downloaded base checkpoints, such as UMA, PET/UPET, or MACE foundation models.
- `models/checkpoints/`: fine-tuned checkpoints you want to keep.
- `models/external_repos/`: local editable clones such as patched `submitit`, `metatrain`, or MACE forks.
- `runs/uma/`, `runs/upet/`, `runs/mace/`: run outputs, checkpoints, and W&B-linked training artifacts.
- `logs/slurm/`: scheduler stdout/stderr files.
- `notebooks/`: exploratory notebooks.
- `scripts/`: small launch/setup utilities.
- `src/pretrain_analysis/`: lightweight Python helpers for this workspace.

## Basic Commands

UMA:

```bash
conda activate pretrain_analysis_env
bash scripts/setup_submitit_proxy.sh
bash scripts/run_uma.sh configs/uma/uma_sm_finetune_template.yaml
```

UPET:

```bash
conda activate pretrain_analysis_env
bash scripts/run_upet.sh configs/upet/options.yaml
```

MACE:

```bash
conda activate pretrain_analysis_env
bash scripts/run_mace.sh configs/mace/configs.yaml
```

Workspace check:

```bash
pytest
```

## Workflow Notes

For UMA, generate the FAIR-Chem fine-tuning YAMLs first, then copy them into `configs/uma/`. The procedure note says to set absolute paths in the data-task YAML and replace Hugging Face checkpoint download settings with a local checkpoint path under `models/pretrained/`.

For UPET, use `configs/upet/options.yaml` as the starting point. If resuming into the same W&B run, update both the checkpoint path and the W&B `resume`/`id` fields as described in the procedure note.

For MACE, use a config-driven launch with `mace_run_train --config=configs/mace/configs.yaml`. Put foundation checkpoints in `models/pretrained/` before launching jobs.
