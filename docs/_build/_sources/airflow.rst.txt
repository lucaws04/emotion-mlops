Data Pipeline DAG
=======================

Overview
--------

This Airflow DAG orchestrates a data pipeline that downloads a dataset from an Azure Blob Storage,
ingests it locally, preprocesses the data by mapping emotions to broader categories, validates the
preprocessed data, and finally creates an Azure ML data asset for use in ML pipelines.

The DAG runs on a weekly schedule and consists of five main tasks executed sequentially:

1. Download blob from Azure Blob Storage
2. Load and ingest the downloaded dataset
3. Preprocess the dataset (emotion mapping)
4. Validate the processed dataset (missing values and required columns)
5. Create a data asset in Azure Machine Learning workspace

Prerequisites
-------------

- Apache Airflow environment with `azure-identity`, `azure-ai-ml`, `azure-storage-blob`, and `pandas` installed.
- An Airflow connection named `azure_ml_conn` configured with the following JSON in the "Extra" field:

  .. code-block:: json

     {
       "tenant_id": "<YOUR_TENANT_ID>",
       "client_id": "<YOUR_CLIENT_ID>",
       "client_secret": "<YOUR_CLIENT_SECRET>",
       "subscription_id": "<YOUR_AZURE_SUBSCRIPTION_ID>",
       "resource_group": "<YOUR_RESOURCE_GROUP>",
       "workspace_name": "<YOUR_AZURE_ML_WORKSPACE>"
     }

- Access to the Azure Blob Storage container and blob specified in the DAG code.

- Local temporary directory `/tmp` for file storage (modifiable in code).

DAG Details
-----------

- **DAG ID**: NLP9_data_pipeline_dag
- **Owner**: NLP9
- **Schedule**: Weekly (`@weekly`)
- **Start Date**: June 19, 2025
- **Retries**: 0
- **Tags**: `nlp9`, `azureml`, `datastore`, `blob`, `download`

Task Descriptions
-----------------

- **download_blob**

  Downloads the dataset CSV file from Azure Blob Storage to a local path (`/tmp/processed_dataset.csv`).

- **loading_blob**

  Loads the downloaded CSV dataset into a Pandas DataFrame and saves it as `ingested.csv` in `/tmp`.

- **preprocess_blob**

  Applies emotion mapping to convert detailed emotion labels into broader categories.

- **validate_blob**

  Checks the processed dataset for missing values and required columns. If validation fails, the DAG stops.

- **create_data_asset**

  Uploads the validated dataset to Azure ML as a versioned data asset.

Installation
------------

1. Clone the repository containing this DAG (or copy the DAG file into your Airflow `dags/` folder).

2. Install the required Python packages:

   .. code-block:: bash

      pip install apache-airflow
      pip install azure-identity azure-ai-ml azure-storage-blob pandas

3. Configure your Airflow connection `azure_ml_conn` as described in the **Prerequisites** section.

4. Restart your Airflow scheduler and webserver to pick up the new DAG.

Usage
-----

Once the DAG is deployed and dependencies are installed:

1. Open the Airflow UI (usually available at `http://localhost:8080`).
2. Search for the DAG named **NLP9_data_pipeline_dag**.
3. Turn the DAG "on" using the toggle switch.
4. Trigger the DAG manually by clicking the **Trigger DAG** button or wait for the weekly schedule.

Examples
--------

You can test individual components locally using Python scripts. For example, to test the emotion mapping step:

.. code-block:: python

   import pandas as pd
   from src.data.data_processing import emotion_mapping

   df = pd.read_csv("/tmp/ingested.csv")
   df_mapped = emotion_mapping(df)
   print(df_mapped.head())

To manually upload a dataset to Azure Blob and run the DAG end-to-end:

1. Upload a dataset file (e.g., `raw_dataset.csv`) to your Azure Blob container.
2. Modify the blob name and container in the DAG file if needed.
3. Trigger the DAG and monitor logs via the Airflow Web UI.

Usage Functions
-----------------
.. currentmodule:: airflow.dags.demo_dag

.. autofunction:: download_blob
.. autofunction:: loading_blob
.. autofunction:: preprocess_blob
.. autofunction:: validate_blob
.. autofunction:: create_data_asset