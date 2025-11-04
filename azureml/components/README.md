# Components Folder

This directory contains Python scripts for registering reusable Azure Machine Learning (AzureML) components, including data ingestion, model training, and evaluation. Each script loads a component definition from a corresponding YAML file and registers it to the AzureML workspace.

---

## Overview of Components

- `evaluation_component.py` — Registers the model evaluation component defined in `azureml/scripts/evaluate.yml`.
- `register_component.py` — Registers the model registration defined in `azureml/scripts/register.yml`.
- `train_component_v2.py` — Registers the model training component defined in `azureml/scripts/train_v2.yml`.

## Prerequisites

Ensure you have the following in place:

- AzureML workspace connection (handled via `azureml.scripts.connecting_workspace`) for more information [see]("https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp9/tree/main/azureml/scripts#connect-to-azureml-workspace")
- Required YAML component definitions in the `azureml/scripts/` directory
- Azure ML SDK v2 installed

```bash
pip install azure-ai-ml
```
#### Ensure that the `src/` package are installed via wheel before running components:

The components depend on the `src/` package, which must be installed using the wheel file:

```bash
pip install setup-1.0.8-py3-none-any.whl
```

## Usage

### Set Python Path:

On Windows Powershell:

```bash
set PYTHONPATH=%CD%
```

On macOS/Linux:

```bash
export PYTHONPATH=$(pwd)
```

Setting the `PYTHONPATH` environment variable to the project root directory allows Python to locate and import the local `src/` package correctly when running the component registration scripts. Without this, Python may fail to find the `src` package, causing import errors and preventing the scripts from running.

**Key points:**

- You only need to set this once per shell session.
- It ensures that all component scripts can identify and import the `src/ `package properly.
- This is essential because the `src/` package is not installed in the global Python environment, but instead lives locally within the project folder.

### Running a Script

Each script can be run independently to register or update a component:

```bash
python azureml/components/data_component.py
python azureml/components/train_component.py
python azureml/components/evaluation_component.py
```

Upon execution, each script:

1. Connects to the AzureML workspace
2. Loads the component definition from YAML
3. Assigns a specific version
4. Registers (or updates) the component in AzureML
5. Logs the registration status

### Adding/Updating Components

To add a component:

1. Create a `.yml` script that defines the name, version, inputs, outputs, environment and command. Store it in scripts/
2. Create a `.py` file that contains the code for the component. Store it in scripts/
3. Create a `_component.py` in components/ that registers the `.yml` file.
4. Ensure the `src/` package is installed via the provided wheel (`setup-1.0.9-py3-none-any.whl`)
5. Run the registration script to register or update the component in AzureML:

```bash
Python azureml/components/<component_name>_component.py
```

### Submitting Components

- Once registered, components can be submitted and used in AzureML pipelines or experiments by referencing their registered versions.
- Ensure version numbers are incremented manually within the designated `.yml` file, before submitting to avoid immutable field conflicts.
- Use the registered component in your pipeline scripts or jobs by specifying the component's name.

## Logging

All scripts use a shared logger from `src.utils.logger` for consistent log output. Logging includes:

- Workspace connection confirmation
- Registration initiation
- Component name and version confirmation

## Notes

- Version numbers are manually incremented to bypass AzureML’s immutable field restriction.
- These scripts are run as part of the NLP emotion classification pipeline.
- The `src/` package must be installed via the wheel file before running any component registration scripts to ensure all dependencies are available.
