# AzureML Data Assets

This directory contains scripts for managing Azure Machine Learning (AzureML) data assets. These assets are versioned datasets registered in the AzureML workspace, which can be used across experiments and pipelines.

## Scripts Overview

### 1. `upload_data_asset.py`

Registers a new versioned data asset to the AzureML workspace.

- **Asset Name:** `processed_dataset_v3`
- **Description:** Cleaned and processed full dataset for emotion classification
- **Version:** `1.0.0`
- **Path:** `data/processed/processed_dataset.csv`
- **Type:** `URI_FILE`

This script is useful when updating or uploading new datasets to be consumed by AzureML components or jobs.

### 2. `list_data_assets.py`

Lists all data assets currently registered in the AzureML workspace, logging each asset's:

- Name
- Description
- Type

Useful for validation, inventory, or debugging purposes.

## Prerequisites

Ensure the following are available and correctly configured:

- AzureML workspace access (via `azureml.scripts.connecting_workspace`)
- File path to dataset must exist in either [`data/processed/`](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp9/tree/main/data/processed) or [`data/raw/`](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp9/tree/main/data/raw).
- The `src/` package must be installed via the wheel file `setup-1.0.8-py3-none-any.whl`. This is required as the scripts depend on modules within the `src/` package.

```bash
pip install setup-1.0.9-py3-none-any.whl
```

Ensure Azure ML SDK v2 is installed

```bash
pip install azure-ai-ml
```

## Usage

### Set Python path to the project root

To ensure Python can locate the local `src/` package and other project modules, set the PYTHONPATH environment variable to the root of the project.

On Windows Powershell:

```bash
set PYTHONPATH=%CD%
```

On macOS/Linux:

```bash
export PYTHONPATH=$(pwd)
```

Setting `PYTHONPATH` allows Python to properly find and import the src/ package and other local modules. Without this, import errors may occur when running the scripts.
You only need to run this command once per shell session.

## Run the scripts

Each script can be run independently to register, update or list data assts

```bash
# To upload/register a data asset
python azureml/data_asset/upload_data_asset.py

# To list all registered data assets
python azureml/data_asset/list_data_assets.py
```

## Logging

- All actions are logged using a centralized logger to ensure consistency.
- Log messages include asset metadata and registration confirmations.

## Notes

- Update the dataset version in `upload_data_asset.py` when making changes to ensure proper version tracking.
- Data asset names and paths should align with the naming convenstions and directory [see here](https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp9/blob/main/docs/standards.md).
