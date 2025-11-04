# AzureML Datastores

This folder contains utility script for inspecting Azure Machine Learning (AzureML) datastores associated with a given workspace. A datastore in AzureML is a storage abstraction layer used to securely connect to data storage services like Azure Blob, ADLS, or file shares.

## Script Overview

### `list_datastore.py`

This script retrieves and logs all datastores registered in the AzureML workspace.

For each datastore, it logs:

- **Name** — The datastore's unique identifier
- **Type** — The type of backing storage (e.g., `AzureBlob`, `AzureDataLakeGen2`, etc.)

## Prerequisites

Before running the script, ensure the following:

- AzureML workspace connection is set up via `azureml.scripts.connecting_workspace`
- Logging utility is available from `src.utils.logger`
- Azure ML SDK v2 is installed:

```bash
pip install azure-ai-ml
```

## Usage

### Set Python path to the project root:

On Windows Powershell:

```bash
set PYTHONPATH=%CD%
```

On macOS/Linux:

```bash
export PYTHONPATH=$(pwd)
```
## Run the scripts

To list all datastores:

```bash
python azureml/datastore/list_datastore.py
```
