# Workflow Package Installation Notes

Install these only after activating `pretrain_analysis_env` and deciding which workflow you will run. The exact PyTorch/CUDA build should match the cluster node and package documentation.

## UMA / FAIR-Chem

The procedure note assumes `fairchem` is installed, and then replaces `submitit` with a proxy-aware editable clone.

```bash
conda activate pretrain_analysis_env
# Install FAIR-Chem according to the official FAIR-Chem docs for your CUDA/PyTorch stack.
bash scripts/setup_submitit_proxy.sh
```

## UPET / Metatrain

```bash
conda activate pretrain_analysis_env
# Install metatrain according to the metatensor/metatrain docs.
# If resuming W&B runs from checkpoints, apply the source edits described in Fine-tuning Procedures.md.
```

## MACE

```bash
conda activate pretrain_analysis_env
# Install MACE or clone the fork mentioned in Fine-tuning Procedures.md into models/external_repos/.
```
