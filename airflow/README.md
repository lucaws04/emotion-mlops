# Airflow-AzureML Data Processing Pipeline

This project defines a Python-based data pipeline that integrates Apache Airflow with Azure Machine Learning to automate data ingestion, preprocessing, validation, and uploading as a versioned Azure ML data asset.

## Features

- Downloads a dataset from Azure Blob Storage
- Loads and preprocesses emotion-labeled datasets
- Applies emotion mapping into broader categories
- Handles missing values and validates schema
- Uploads the final dataset as an Azure ML Data Asset with versioning
- Airflow-compatible Python functions for each pipeline stage

## Pipeline Stages

1. **Download Blob**
    - Uses Airflow connection `azure_ml_conn`
    - Downloads a CSV from the Azure ML blob store

2. **Preprocess Data**
    - Applies emotion mapping
    - Writes a processed CSV

3. **Validate Dataset**
    - Checks for missing values
    - Validates the presence of only `sentence` and `emotion` columns

4. **Register AzureML Data Asset**
    - Uploads processed data to AzureML
    - Auto-increments semantic version

## Usage

[Airflow Web UI](http://194.171.191.227:30319/dags/NLP9_data_pipeline_dag/grid?tab=graph)