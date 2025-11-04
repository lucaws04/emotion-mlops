# AzureML Environments

This folder contains scripts for managing Azure Machine Learning (AzureML) environments and compute targets. These utilities support listing available resources, registering custom environments, and verifying the configuration needed to run ML pipelines.

## Scripts Overview

### `compute_targets.py`

Lists all compute targets in the connected AzureML workspace. Logs the following for each compute resource:
- Compute Name
- Compute Type

This helps in identifying available infrastructure before submitting jobs or pipelines.

### `list_environments.py`

Lists all registered environments in the AzureML workspace. Outputs include:

- Name
- Version
- Description

Useful for environment inventory or validation.

### `pipeline_environment.py`

Defines and registers a specific environment tailored for a machine learning pipeline.

- **Name:** `pipeline`
- **Version:** `1.0.5`
- **Base Image:** PyTorch 2.6 on CUDA 11.8
    `(mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch260:biweekly.202505.2.v1)`
- **Dependencies:** Defined in [`environment.yaml`](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp9/blob/main/environment.yaml) in the root of the repository.
- **Build Compute:** `adsai-lamda-1`
- **Description:** Custom environment for emotion classification pipeline

This script uses `ml_client.environments.create_or_update(...)` to register the environment in the connected workspace.

### `compute_targets.py`

Lists all compute targets available in the AzureML workspace. Outputs:

- Compute Name
- Compute Type

Useful for understanding available infrastructure for jobs and pipelines.

---

## Prerequisites

Ensure the following are set up before running the scripts:

- Python 3.7+
- AzureML SDK v2:

```bash
pip install azure-ai-ml
```

- Workspace connection is properly configured in azureml/scripts/connecting_workspace.py
- Logging is handled via a shared utils.logger utility.

- AzureML SDK v2

```bash
pip install azure-ai-ml
```

## Usage

Make sure the `PYTHONPATH` is set to the project root to enable proper module imports:

On Windows Powershell:

```bash
set PYTHONPATH=%CD%
```

On macOS/Linux:

```bash
export PYTHONPATH=$(pwd)
```
## Run the scripts

Each script can be run independently to register or list environments:

```bash
python azureml/environments/compute_target.py
python azureml/environments/list_environments.py
python azureml/environments/pipeline_environment.py
```

## Notes

- These scripts use the workspace connection defined in `azureml/scripts/connecting_workspace.py`, which authenticates via `DefaultAzureCredential`.
- Logs will print environment and compute information to the console using a consistent logger format.
- Update the `environment.yaml` file at the project root to change dependencies.
