# AzureML Pipeline

This folder contains the definition of an Azure Machine Learning pipeline for training and evaluating an emotion classification model.

---

## Script Overview

### `pipeline_v2.py`

Defines an ML pipeline named **`emotion_classifier_pipeline`** using the AzureML SDK v2 DSL.

- **Pipeline Name:** `emotion_classifier_pipeline`
- **Description:** Trains and evaluates an emotion classifier model
- **Compute Target:** `adsai-lambda-1` with GPU instances

#### Pipeline Steps

1. **Training Step**
   - Uses the registered `train_component` (version 1.3.0)
   - Inputs:
     - `data_asset_uri` (URI of the training dataset)
     - Hyperparameters: learning rate `0.0001`, batch size `16`, epochs `3`
   - Compute: GPU instance on `adsai-lambda-1`
   - Outputs: model artifact (`model_output`)

2. **Evaluation Step**
   - Uses the registered `evaluate_component` (version 1.2.3)
   - Inputs:
     - Same `data_asset_uri` as training
     - Model directory output from the training step
   - Compute: GPU instance on `adsai-lambda-1`
   - Outputs: evaluation results (`evaluation_output`)

#### Returns

- `model_output`: Path to the trained model artifact
- `evaluation_output`: Path to evaluation results

---

## Prerequisites

- Registered components named `train_component` and `evaluate_component` in the AzureML workspace.
- Dataset accessible as a URI (e.g., AzureML data asset).
- AzureML SDK v2 installed:

```bash
pip install azure-ai-ml
```

## Usage

This file is used in `submit_pipeline_v2.py`. Do not run this file on its own. This file can be imported to be submitted by importing the function:

```bash
from azureml.pipeline.pipeline_v2 import emotion_classifier_pipeline
```

## Notes

- Adjust hyperparameters and compute targets as needed.
- Ensure the compute target adsai-lambda-1 supports GPU and is accessible.
- Version numbers of components should be updated to match registered versions.
