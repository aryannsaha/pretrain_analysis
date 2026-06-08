---
title: Fine-tuning Procedures
created at: Mon May 18 2026 14:24:56 GMT+0000 (Coordinated Universal Time)
updated at: Fri Jun 05 2026 19:47:53 GMT+0000 (Coordinated Universal Time)
---

# Fine-tuning Procedures

# Universal Model for Atoms

### References

1. https://fair-chem.github.io/fine-tuning

### Intro

The FAIR Chemistry team has already documented how to [fine-tune UMA models](https://fair-chem.github.io/fine-tuning/) . However, some parts of the instructions are specific to the FAIR cluster setup, so additional adjustments are necessary to follow them seamlessly on `della-gpu`.

### Enabling external network access

1. Here is some context that will help you understand this section:
   1. The `fairchem` package runs training or fine-tuning jobs by automatically submitting a bash script to the SLURM scheduler through the [submitit](https://github.com/facebookincubator/submitit) package. You do not need to know the internal details of this package — only that it generates the bash script and submits it on our behalf. While this is very convenient, modifying the generated bash script can be somewhat cumbersome.
   2. When running training or fine-tuning, the `fairchem` package defaults to using [Weights & Biases](https://wandb.ai/)\*\* \*\*for logging. This requires network access from compute nodes, but compute nodes on `della-gpu` do not have network access by default (as also mentioned above).
2. Therefore, we need to add `module load proxy/default` to the generated bash script so that compute nodes can access external networks. To do this, assuming you have already installed the `fairchem` package, first uninstall `submitit` so we can reinstall it from a local copy:

```bash
pip uninstall submitit
```

```
Then, proceed with either Steps 3-4 or Step 5.
```

1. Clone the `submitit` repository and move into it:

```bash
git clone https://github.com/facebookincubator/submitit.git
cd submitit
```

1. Edit `slurm/slurm.py` and insert the following line around line 530 to hard-code loading the proxy module:

```python
    lines += [
        "",
        "# command",
        "export SUBMITIT_EXECUTOR=slurm",
        # Add this line to load the proxy module
        "module load proxy/default",
        # The input "command" is supposed to be a valid shell command
        command,
        "",
    ]
```

1. Simply clone the following repository:

```bash
git clone https://github.com/sihoonchoi/submitit.git
cd submitit
```

1. Install the modified package locally in editable mode:

```bash
pip install -e .
```

### Modifying YAML files

When you follow the [Generating Training/Fine-tuning Datasets](https://fair-chem.github.io/fine-tuning/#generating-training-fine-tuning-datasets) section, you will eventually obtain `uma_sm_finetune_template.yaml` and `data/uma_conserving_data_task_energy_force_stress.yaml` in your specified `--output-dir`. These YAML files are used to configure fine-tuning.

1. `uma_sm_finetune_template.yaml`
   1. You will see the `job` section begins around line 4. This section defines parameters used to generate the SLURM submission script as well as the W\&B configuration. Below is an example setup compatible with `della-gpu`.

```yaml
job:
  device_type: CUDA
  scheduler:
    mode: SLURM
    ranks_per_node: 1
    num_nodes: 1
    slurm:
      timeout_hr: 6
      cpus_per_task: 6
      mem_gb: 32
      additional_parameters:
        time: 0-12:00:00
        gres: gpu:1
        constraint: intel&gpu80
        mail_user: <NetID>@princeton.edu
        mail_type: begin,end,fail
  debug: false
  run_dir: <your_dir_name>/
  run_name: <your_run_name>
  logger:
    _target_: fairchem.core.common.logger.WandBSingletonLogger.init_wandb
    _partial_: true
    entity: <your_wandb_team_name>
    project: <your_wandb_project_name>
```

```
1. Under `job.scheduler.slurm`, several required parameters must be defined: `timeout_hr`, `cpus_per_task`, and `mem_gb`. For `cpus_per_task` and `mem_gb`, I recommend using the values stated above. The value of `timeout_hr` can be set arbitrarily, since it will be overridden in the `additional_parameters` section.
1. The `additional_parameters` field allows you to specify extra SLURM directives (i.e., lines that appear after `#SBACTH` in the generated script). Use this section to customize the job configuration, such as wall time (`time`), GPU node type (`constraint`), and other SLURM options you typically use.
1. Set `job.debug` to `false` to enable W&B logging.
1. For `job.run_dir`, use a directory name of your choice. Logs and checkpoints for each run will be saved there.
1. `job.run_name` is used to label each run in your W&B project.
1. Fill in your team name and project name under `job.logger.entity` and `job.logger.project`, respectively. These should remain consistent across runs, especially when performing hyperparameter tuning, so that evaluation metrics appear in the same W&B project dashboard.
1. Under `runner.train_eval_unit.model.checkpoint_location`, you will see `_target_` and `model_name` are predefined. By default, this configuration attempts to download the checkpoint file from its Hugging Face repository. However, even after successfully loading the proxy module in the previous section, this download does not work as expected. Therefore, please delete (or comment out) these two parameters. Instead, I recommend directly specifying the `checkpoint_location` parameter by providing the local path to the UMA model checkpoint you intend to fine-tune. In this case, since the `base_model_name` parameter will no longer be used anywhere in the configuration file, you may also delete (or comment out) its declaration.
```

1. `data/uma_conserving_data_task_energy_force_stress.yaml`
   1. Change `train_dataset.splits.train.src` (around line 107) and `val_dataset.splits.val.src` (around line 117) to absolute paths. This allows you to run the command `fairchem -c uma_sm_finetune_template.yaml` from any directory.
   2. In this file, you can also modify the loss function coefficients for energy per atom, forces, and stresses. Look for the `coefficient` fields and adjust them as needed. The default ratio is **20:2:1**.

# UPET

### References

1. https://docs.metatensor.org/metatrain/latest/concepts/fine-tuning.html
2. https://docs.metatensor.org/metatrain/latest/generated\_examples/0-beginner/02-fine-tuning.html
3. https://atomistic-cookbook.org/examples/pet-finetuning/pet-ft.html
4. \[optional] https://lab-cosmo.github.io/upet/latest/fine-tuning.html

### Intro

Fine-tuning various PET models is also well documented in the references listed above. I recommend reading them in the order shown above to get a better sense of how to set everything up. The fourth reference is included to simply list all documents related to fine-tuning, but the information there is quite minimal, so it can be skipped.

The key to fine-tuning PET models is customizing the settings in `options.yaml` and passing it to `mtt train options.yaml` through the `metatrain` package. The first reference introduces the basic workflow, while the second provides more detailed guidance on how to configure `options.yaml`. The third reference covers fine-tuning in the most detail, but it may feel overwhelming without first reading the first two, so I recommend following that order.

There are two things I want to discuss in this section:

* How to prepare the data
* How to continue fine-tuning in the same W\&B run when a job finishes or is interrupted

### How to prepare the data

This is also well documented [here](https://docs.metatensor.org/metatrain/latest/generated_examples/0-beginner/01-data_preparation.html) . Here are a few additional tips:

* `ase.trajectory` also works as an input format
* In addition to `ase.trajectory`, any trajectory format that can be read with `ase.io.read` is supported, not just the XYZ format mentioned in the documentation
* If your dataset contains more than 1 million data points, you may want to skip the `DiskDataset` section and move directly to `MemmapDataset`

### How to continue fine-tuning in the same W\&B

You may want to continue fine-tuning within the same W\&B as a previous job, or there may be cases where a job is stopped by the scheduler (or for other reasons) and you want to resume it while keeping the training progress in a single run. In that case, simply updating `architecture.training.finetune.read_from` in `options.yaml` is not enough to continue the learning curve in the same W\&B run.

```yaml
architecture
  training:
    finetune:
      method: "full"
      read_from: /path/to/checkpoint.ckpt
```

This is because, by design, each training/fine-tuning job starts from epoch 0. As a result, the existing learning curve in W\&B will be overwritten by the new run. To properly resume training from the desired epoch, some modifications to the source code are needed so that the `metatrain` package load the epoch information from the checkpoint.

You need to modify 2 files:

1. `metatrain/src/metatrain/cli/train.py`

   In line 671, replace:

```python
trainer = Trainer(hypers["training"])
```

```
 with:
```

```python
if 'huggingface' in restart_from:
    trainer = Trainer(hypers["training"])
else:
    trainer = trainer_from_checkpoint(
        checkpoint=checkpoint,
        hypers=hypers["training"],
        context=training_context)
```

```
This is currently hard-coded in my setup, so if you find a cleaner solution, please let me know. The purpose of this modification is to allow the training process to start from epoch 0 when loading a checkpoint from Hugging Face (i.e., when starting a new fine-tuning run), while resuming from the saved epoch when loading from an existing fine-tuning checkpoint.
```

1. `metatrain/src/metatrain/pet/trainer.py`

   In line 752, replace:

```python
trainer.epoch = None
```

```
with:
```

```python
trainer.epoch = checkpoint["epoch"]
```

1. Finally, do not forget to update the `wandb` section in `options.yaml` as follows:

```yaml
wandb:
  entity: your_group_name
  project: your_project_name
  name: your_run_name
  resume: allow
  id: your_run_id
```

```
This allows W&B to recognize the resumed training as the same run instead of creating a new one.
```

# MACE

### References

1. https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html

### Intro

Instructions for fine-tuning various MACE models can be found in the reference above. However, there are a few modifications that can make the workflow more convenient and better aligned with our use cases. Simply cloning this [repo](https://github.com/sihoonchoi/mace) should address the issues discussed in the following sections.

### Configuring fine-tuning

In the reference documentation, many parameters are passed as command-line arguments. However, I recommend creating a separate `configs.yaml` file, similar to the workflow used for UPET fine-tuning, and launching training with:

```bash
mace_run_train --config=configs.yaml
```

An example `configs.yaml` file is attached below.

[\[configs.yaml (2KB)\]](media_Fine-tuning%20Procedures/M-bHmKgAyRQ9rN-configs.yaml)

Some notable arguments are:

* `foundational_model`: You can provide the name of a pre-trained model available [here](https://github.com/ACEsuit/mace/blob/main/mace/calculators/foundations_models.py) . Make sure to download the checkpoint on the login node before starting fine-tuning.
* `compute_stress` and `loss`: By default, MACE fine-tuning only logs energy and force errors. Since we are also interested in stress errors, set:

```yaml
compute_stress: True
loss: stress
```

```
This will enables stress calculations and allows stress errors to be reported through the loss function defined in `error_table`.
```

* `atomic_numbers` and `E0s`: You must provide the atomic numbers corresponding to all elements present in the dataset, along with their reference energies (`E0s`). An example is included in the configuration file above. Make sure that the length of `atomic_numbers` and `E0s` match. Based on my testing, it is acceptable to include elements that do not appear in the dataset, but in that case you must still provide the corresponding reference energies in `E0s`.

### Logging MAEs to W\&B

The default W\&B integration in the MACE source code logs RMSE values for energy and forces only, regardless of the metrics defined in `error_table`. In most of our applications, however, we are primarily interested in tracking MAEs for energy, forces, and stress.

This behavior has already been modified in the cloned repository mentioned above.

### How to continue fine-tuning in the same W\&B

Additional modifications have been made to support resuming fine-tuning within the same W\&B run. This functionality is also included in the cloned repository.

To resume training, make sure to configure the following arguments in `configs.yaml`:

* `foundational_model`: You can also provide the checkpoint of a previously fine-tuned model, either to resume fine-tuning or to start a new run with different hyperparameters. In this case, provide the path to a checkpoint ending with `*.model`\_ \_rather than `*.pt`.
* `restart_latest`: Setting this option to `True` resumes training from the latest epoch stored in the provided checkpoint. Otherwise, training will start from epoch 0.
* `wandb_id`: Make sure to use the same W\&B run ID. This allows W\&B to extend the existing learning curves when training resumes, rather than creating a separate run with a new set of curves.
