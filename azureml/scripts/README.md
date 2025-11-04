# AzureML Scripts

This folder contains essential scripts for interacting with the Azure Machine Learning (AzureML) workspace, managing data assets, and running model training and evaluation pipelines.

## Prerequisites

- Python 3.7+
- AzureML SDK v2 and dependencies installed:

```bash
pip install azure-ai-ml mlflow transformers pandas
```

- A valid config/config.json file with your Azure subscription and workspace details:

```json
{
  "subscription_id": "<your-subscription-id>",
  "resource_group": "<your-resource-group>",
  "workspace_name": "<your-workspace-name>"
}
```

-Set the PYTHONPATH environment variable to the project root to allow proper module imports:

Windows PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

Linux/macOS Bash:

```bash
export PYTHONPATH=$(pwd)
```

## Scripts Overview & Usage

1. ### Connect to AzureML Workspace

`connecting_workspace.py`

This script creates an authenticated connection to your AzureML workspace using the `InteractiveBrowserCredential`. It reads the workspace details from `config/config.json`.

#### Usage:

```bash
python azureml/scripts/connecting_workspace.py
```

2. ### Model Evaluation

`evaluate.py`
This script performs the evaluation of a trained emotion classification model:

- Loads the dataset from the given AzureML data asset URI.
- Splits the dataset into test subsets.
- Loads the trained model and tokenizer.
- Evaluates model performance, logs metrics to MLflow.
- Generates and logs prediction outputs and classification reports.
- Logs a Hugging Face Transformers TextClassificationPipeline model artifact to MLflow.

#### Required Arguments:

- --`data_asset_uri`: URI of the dataset to evaluate.
- --`model_dir`: Directory containing the trained model artifacts.
- --`output_dir`: Directory where evaluation outputs (reports, predictions) will be saved.

#### Usage

This file is used in `evaluate.yml`. Do **not** run this file on its own.

3. ### Model Training

`train_v2.py`

This script trains a BERT emotion classifier:

- Loads and splits the dataset.
- Prepares datasets and tokenizer.
- Trains the model with hyperparameters (learning rate, epochs, batch size).
- Saves the trained model and tokenizer.
- Logs parameters, metrics, and the model pipeline artifact to MLflow.

#### Required Arguments:

- --data_asset_uri: URI of the training dataset.
- --model_output: Directory to save the trained model.
- --learning_rate: Learning rate for training.
- --num_epochs: Number of training epochs.
- --batch_size: Batch size for training.

#### Usage

This file is used in `train.yml`. Do **not** run this file on its own.

## Notes

- These scripts leverage MLflow for experiment tracking and artifact logging.
- The `train_v2.py` and `evaluate.py` scripts assume that dataset and model loading functions, tokenizers, and other utilities are available in the project’s source code folder (data, features, models, utils).
- Modify paths and hyperparameters as needed to suit your environment and requirements.
- Ensure your AzureML compute targets and environment configurations are properly set up to run training and evaluation jobs in Azure.

## Troubleshooting

- If you encounter import errors, verify that `PYTHONPATH` is set correctly to include the project root.
- Make sure you have authenticated access to AzureML workspace; `InteractiveBrowserCredential` will prompt for login.
- Check that the AzureML workspace and resource details in `config/config.json` are accurate.
- Review logs for detailed error messages; logging is configured to provide info and error details.
