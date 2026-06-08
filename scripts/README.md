# Scripts

All scripts assume they are launched from the repository root after:

```bash
conda activate pretrain_analysis_env
```

- `run_uma.sh`: launches `fairchem -c <config>`.
- `run_upet.sh`: launches `mtt train <options.yaml>`.
- `run_mace.sh`: launches `mace_run_train --config=<config>`.
- `setup_submitit_proxy.sh`: installs the proxy-aware `submitit` fork into the active environment.
- `make_uma_data_paths_absolute.py`: updates UMA data-task YAML `src` fields to absolute paths.
