# AzureML

This directory contains modular scripts and configurations for interacting with Azure Machine Learning (AzureML) services. It provides tools for managing workspaces, data assets, environments, compute resources, and job pipelines in a reusable and maintainable way.

## Structure

- `components/` — AzureML pipeline components for training, evaluation, etc.
- `data_assets/` — Scripts and configs to manage datasets in AzureML
- `datastore/` — Configuration and scripts related to AzureML datastores
- `environments/` — Scripts for registering and managing custom environments in AzureML.
- `jobs/ ` — Job definitions and job submission scripts
- `model/` — Model definitions, saving/loading utilities
- `scripts/` — Scripts for connecting to AzureML workspace, defining components and a wheel script

Navigate into each folder for detailed instructions on their respective usage.

##  Prerequisites

Before using these scripts, ensure the following tools and accounts are set up:

- [Poetry](https://python-poetry.org/docs/) for dependency management.
- Python 3.8 or later.
- Azure CLI (for authentication and workspace management).
- Access to an AzureML workspace with the required permissions.

Optional but recommended:

- A configured .env or config.json for workspace parameters.
- A GPU compute cluster registered in your AzureML workspace (for model training tasks).

---

## Getting Started

Navigate to the project root (where pyproject.toml is located) and run:

1. **Install dependencies:**

From the project root (where `pyproject.toml` is located), run:

```bash
poetry install
```

This will install all required packages into a virtual environment.

2. **Activate the Poetry virtual environment**

Running commands:

```bash
poetry run <your_command>
```

3. **Set `PYTHONPATH`**

Ensure local modules can be resolved by setting the `PYTHONPATH`:

Windows (PowerShell):

```powershell
$env:PYTHONPATH = (Get-Location).Path

macOS/Linux:

```bash
export PYTHONPATH=$(pwd)
```

## Running Scripts

To run any script from the project root (after setting `PYTHONPATH`), use:

```bash
python azureml/path/to_script.py
```

Example:

```bash
python azureml/jobs/submit_pipeline.py
```

## Environments & Components

To manage environments and register components:

- Use scripts in `azureml/environments/` to define and register Docker/Conda environments.
- Use scripts in `azureml/components/` to package and register modular pipeline components like training or evaluation steps.

## Data Assets
Use scripts in `azureml/data_assets/ `to:

- Upload local datasets to AzureML
- Register datasets with versioning support
- Reference them in pipeline jobs

## Job Submission
To submit the full pipeline, use:

```bash
python azureml/jobs/submit_pipeline.py
```

Make sure all required components, data assets, and environments are registered before submission.

## Additional Information
For in-depth guidance on pipeline orchestration, component YAML structure, environment management, or AzureML workspace configuration, refer to the subfolder-level `README.md` files or see [AzureML official documentation](https://learn.microsoft.com/en-us/azure/machine-learning/?view=azureml-api-2).
