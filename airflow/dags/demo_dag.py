import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd
from airflow.hooks.base import BaseHook
from airflow.operators.python_operator import PythonOperator
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from packaging import version

from airflow import DAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATASTORE_NAME = "workspaceblobstore"
BLOB_URL = (
    "https://y2d20254243635300.blob.core.windows.net/"
    "79e5593e-6e03-4cab-b624-88336162ef1a-azureml-blobstore/"
    "LocalUpload"
)
BLOB_NAME = "LocalUpload/0ad6147ee9f63498418f412fe6270ac1/processed_dataset.csv"

LOCAL_FILE_NAME = "processed_dataset.csv"

# === Local paths ===
LOCAL_PATH = f"/tmp/{LOCAL_FILE_NAME}"
PROCESSED_PATH = f"/tmp/processed_{LOCAL_FILE_NAME}"


def download_all_csv_blobs(folder_path: str, local_dir: str) -> list[str]:
    """
    Downloads all CSV blobs from a specified Azure Blob Storage folder to a
    local directory.
    Args:
        folder_path (str): The path in the blob storage to search for CSV files.
        local_dir (str): The local directory where the CSV files will be saved.
    Returns:
        list[str]: A list of local file paths to the downloaded CSV files.
    """

    try:
        print("Starting bulk blob download task...")

        conn = BaseHook.get_connection("azure_ml_conn")
        extras = conn.extra_dejson

        credential = ClientSecretCredential(
            tenant_id=extras["tenant_id"],
            client_id=extras["client_id"],
            client_secret=extras["client_secret"],
        )
        ml_client = MLClient(
            credential=credential,
            subscription_id=extras["subscription_id"],
            resource_group_name=extras["resource_group"],
            workspace_name=extras["workspace_name"],
        )

        datastore = ml_client.datastores.get(DATASTORE_NAME)
        account_name = datastore.account_name
        container_name = datastore.container_name

        blob_service_client = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=credential,
        )
        container_client = blob_service_client.get_container_client(container_name)

        print(f"Listing blobs in: {folder_path}")
        blob_list = container_client.list_blobs(name_starts_with=folder_path)
        for blob in blob_list:
            print(blob.name)
        local_files = []
        for blob in blob_list:
            if blob.name.endswith(".csv"):
                local_path = os.path.join(local_dir, os.path.basename(blob.name))
                os.makedirs(local_dir, exist_ok=True)

                print(f"Downloading blob: {blob.name}")
                with open(local_path, "wb") as f:
                    blob_data = container_client.get_blob_client(
                        blob.name
                    ).download_blob()
                    f.write(blob_data.readall())
                local_files.append(local_path)
        return local_files

    except Exception as e:
        print(f"Error during bulk blob download: {e}")
        raise


def load_all_csvs(filepaths: list[str]) -> pd.DataFrame:
    """
    Load multiple CSV files into a single pandas DataFrame.
    Args:
        filepaths (list[str]): List of file paths to CSV files.
    Returns:
        pd.DataFrame: Concatenated DataFrame containing data from all loaded CSV files.
    """

    frames = []
    for path in filepaths:
        try:
            df = pd.read_csv(path)

            logger.info(f"Loaded {path} with shape {df.shape}")
            frames.append(df)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_ingestion(folder_path: str, output_dir: str, data_source: str):
    """
    Ingests CSV files from a specified folder, combines them into a single DataFrame,
    and saves the result to an output directory.
    Args:
        folder_path (str): Path to the folder containing CSV files to ingest.
        output_dir (str): Directory where the ingested CSV file will be saved.
        data_source (str): Identifier for the data source (not used in this function).
    Returns:
        None
    """

    logger.info(f"Running ingestion from folder: {folder_path}")
    local_dir = "/tmp/csv_downloads"

    csv_files = download_all_csv_blobs(folder_path, local_dir)
    df = load_all_csvs(csv_files)
    for file in csv_files:
        print(f"file {file.name} loaded with shape {df.shape}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / "ingested.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Data ingestion complete. Saved to {output_path}")


def emotion_mapping(emotion: Union[pd.Series, list]) -> pd.Series:
    """
    Maps fine-grained emotion labels to broader emotion categories.
    Args:
        emotion (Union[pd.Series, list]): A pandas Series or list containing
        emotion labels to be mapped.
    Returns:
        pd.Series: A pandas Series with each label mapped to its corresponding broad
        emotion category, or None if not found.
    """

    mapping = {
        "happiness": [
            "joy",
            "optimism",
            "approval",
            "pride",
            "gratitude",
            "confidence",
            "satisfaction",
            "hope",
            "love",
            "excitement",
            "caring",
            "relief",
            "admiration",
            "amusement",
            "anticipation",
            "encouragement",
            "desire",
            "happiness",
        ],
        "sadness": [
            "disappointment",
            "nostalgia",
            "remorse",
            "pain",
            "stress",
            "regret",
            "resignation",
            "despair",
            "confusion",
            "uncertainty",
            "sadness",
        ],
        "anger": [
            "anger",
            "annoyance",
            "disapproval",
            "frustration",
            "disbelief",
            "warning",
            "rejection",
        ],
        "disgust": ["disgust"],
        "fear": [
            "fear",
            "nervousness",
            "worry",
            "anxiety",
            "doubt",
            "insecurity",
            "urgency",
            "panic",
        ],
        "surprise": ["surprise", "realization", "shock", "amazement", "wonder"],
        "neutral": [
            "neutral",
            "mixed",
            "trust",
            "agreement",
            "instruction",
            "suggestion",
            "request",
            "confirmation",
            "acceptance",
            "reassurance",
            "clarification",
            "understanding",
            "certainty",
            "curiosity",
        ],
    }
    reverse_lookup = {
        sub_emotion.lower(): emotion
        for emotion, sub_list in mapping.items()
        for sub_emotion in sub_list
    }

    def map_label(label):
        """
        Maps a given label to its corresponding value using the
        reverse_lookup dictionary.
        Args:
            label (str): The label to be mapped.
        Returns:
            The mapped value from reverse_lookup if found, otherwise None.
        """

        if pd.isna(label) or str(label).strip() == "":
            return None
        return reverse_lookup.get(str(label).lower().strip(), None)

    return pd.Series(emotion).apply(map_label)


def run_preprocessing(**kwargs):
    """
    Preprocesses input CSV data for further analysis.
    Args:
        **kwargs: Arbitrary keyword arguments passed from Airflow context.
    Returns:
        None
    Raises:
        FileNotFoundError: If the input file does not exist at LOCAL_PATH.
        KeyError: If the 'emotion' column is missing in the input data.
        Exception: For any other errors during preprocessing.
    """

    try:
        print("Starting preprocessing task...")
        if not os.path.exists(LOCAL_PATH):
            raise FileNotFoundError(f"Input file not found at {LOCAL_PATH}")

        df = pd.read_csv(LOCAL_PATH)
        print(f"Read CSV from {LOCAL_PATH}, shape: {df.shape}")

        if "emotion" not in df.columns:
            raise KeyError("Missing 'emotion' column in input data")

        df["emotion"] = emotion_mapping(df["emotion"])
        df.to_csv(PROCESSED_PATH, index=False)

        print(f"Preprocessed data saved to {PROCESSED_PATH}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise


def handle_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataframe by removing the entire row.
    Args:
        dataframe (pd.DataFrame): The input dataframe with missing values.
    Returns:
        pd.DataFrame: The dataframe with missing values handled.
    """
    logger.info("Removing missing values from dataset")
    return dataframe.dropna()


def check_required_columns(dataframe: pd.DataFrame) -> bool:
    """
    Check if the dataframe has exactly two columns: 'sentence' and 'emotion'.

    Args:
        dataframe (pd.DataFrame): The input dataframe.

    Returns:
        bool: True if the dataframe has exactly the required columns, False otherwise.
    """
    required_columns = {"sentence", "emotion"}
    actual_columns = set(dataframe.columns)

    if actual_columns != required_columns:
        logger.error(
            f"DataFrame must contain exactly the columns {required_columns}, "
            "but found {actual_columns}"
        )
        return False

    logger.info(
        "DataFrame contains exactly the required columns: 'sentence' and 'emotion'."
    )
    return True


def check_emotion_labels(dataframe: pd.DataFrame, allowed_labels=None) -> bool:
    """
    Check if all labels in the 'emotion' column are within the allowed set.

    Args:
        dataframe (pd.DataFrame): The input dataframe.
        allowed_labels (set or None): Allowed emotion labels. If None, uses default.

    Returns:
        bool: True if all labels are valid, False otherwise.
    """
    if allowed_labels is None:
        allowed_labels = {
            "happiness",
            "sadness",
            "anger",
            "disgust",
            "fear",
            "surprise",
            "neutral",
        }
    unique_labels = set(dataframe["emotion"].dropna().unique())
    invalid_labels = unique_labels - allowed_labels
    if invalid_labels:
        logger.error(f"Invalid emotion labels found: {invalid_labels}")
        return False
    logger.info("All emotion labels are valid.")
    logger.info(f"Label counts: {dataframe['emotion'].value_counts().to_dict()}")
    logger.info(f"Number of unique emotion labels: {len(unique_labels)}")
    return True


def run_validation(**kwargs):
    """
    Validates and preprocesses a dataset by handling missing values,
    checking emotion labels, and required columns.
    Args:
        **kwargs: Arbitrary keyword arguments passed from Airflow context.
    Returns:
        None. Saves the validated DataFrame to a CSV file.
        Logs errors if validation fails.
    """

    input_path = "/tmp/processed_processed_dataset.csv"
    output_path = "/tmp/preprocessed.csv"

    try:
        logger.info(f"Loading data from {input_path} for validation...")
        df = pd.read_csv(input_path)

        df = handle_missing_values(df)
        if not check_emotion_labels(df):
            logger.error("Validation failed due to invalid emotion labels.")
            return
        if not check_required_columns(df):
            logger.error("Validation failed due to incorrect columns.")
            return  # or raise ValueError("Invalid columns")

        df.to_csv(output_path, index=False)
        logger.info(
            f"New data asset contains {df.shape[0]} rows and {df.shape[1]} columns."
        )
        logger.info(f"Validated data saved to {output_path}")

    except Exception as e:
        logger.error(f"Validation failed: {e}")


def get_next_version(ml_client, asset_name, current_version=None):
    """
    Get the next semantic version for a given asset, using
    single-digit versioning (e.g., 1.2.3).
    Args:
        ml_client: The ML client object used to list asset versions.
        asset_name (str): The name of the asset to version.
        current_version (str, optional): A specific version to use if it is
        newer than existing versions.
    Returns:
        str: The next semantic version in the format 'X.Y.Z',
        where each part is a single digit.
    """
    try:
        # List all versions of the asset
        versions = [v.version for v in ml_client.data.list(name=asset_name)]

        if not versions:
            return "1.0.0"  # Initial version

        # Only keep versions with single digits in each part (e.g., 1.2.3)
        def is_single_digit_version(ver):
            parts = ver.split(".")
            return len(parts) == 3 and all(
                p.isdigit() and 0 <= int(p) <= 9 for p in parts
            )

        single_digit_versions = [v for v in versions if is_single_digit_version(v)]
        if not single_digit_versions:
            return "1.0.0"

        parsed_versions = [version.parse(v) for v in single_digit_versions]
        latest = max(parsed_versions)

        # Increment patch version by default, but keep single digits
        next_micro = latest.micro + 1
        next_minor = latest.minor
        next_major = latest.major

        if next_micro > 9:
            next_micro = 0
            next_minor += 1
        if next_minor > 9:
            next_minor = 0
            next_major += 1
        if next_major > 9:
            # Reset to 1.0.0 if major exceeds 9 (or handle as needed)
            return "1.0.0"

        if current_version and version.parse(current_version) > latest:
            return current_version

        return f"{next_major}.{next_minor}.{next_micro}"
    except Exception as e:
        print(f"Error determining next version: {e}")
        return "1.0.0"


def create_data_asset():
    """
    Creates a data asset in Azure Machine Learning workspace using processed data.
    Args:
        None
    Returns:
        None
    Raises:
        FileNotFoundError: If the processed data file does not exist at the
        specified path.
        Exception: For any errors encountered during data asset creation.
    """
    try:
        logger.info("Starting data asset creation...")
        conn = BaseHook.get_connection("azure_ml_conn")
        extras = conn.extra_dejson

        tenant_id = extras.get("tenant_id")
        client_id = extras.get("client_id")
        client_secret = extras.get("client_secret")
        subscription_id = extras.get("subscription_id")
        resource_group = extras.get("resource_group")
        workspace_name = extras.get("workspace_name")

        credential = ClientSecretCredential(
            tenant_id=tenant_id, client_id=client_id, client_secret=client_secret
        )

        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )

        if not os.path.exists(PROCESSED_PATH):
            raise FileNotFoundError(f"Processed file not found at {PROCESSED_PATH}")
        asset_name = "final_processed_dataset"
        next_version = get_next_version(ml_client, asset_name)

        data_asset = Data(
            path=PROCESSED_PATH,
            type="uri_file",
            description=f"Data asset created successfully with version {next_version}."
            "This dataset can be used in Azure ML pipelines."
            "It has been processed and split into train, validation,"
            "and test sets.",
            name=asset_name,
            version=next_version,
        )

        ml_client.data.create_or_update(data_asset)
        print(f"Data asset created successfully with version {next_version}.")
    except Exception as e:
        print(f"Error during data asset creation: {e}")
        raise


default_args = {
    "owner": "NLP9",
    "start_date": datetime(2025, 6, 19),
    "retries": 0,
}

with DAG(
    dag_id="NLP9_data_pipeline_dag",
    default_args=default_args,
    schedule_interval="@weekly",
    catchup=False,
    tags=["nlp9", "azureml", "datastore", "blob", "download"],
) as dag:

    download_task = PythonOperator(
        task_id="download_blob",
        python_callable=download_all_csv_blobs,
        op_kwargs={
            "folder_path": "LocalUpload",
            "local_dir": "/tmp",  # Local directory to save downloaded files
        },
    )

    loading_task = PythonOperator(
        task_id="loading_blob",
        python_callable=run_ingestion,
        op_kwargs={
            "folder_path": "LocalUpload",
            "output_dir": "/tmp",
            "data_source": "local",
        },
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_blob",
        python_callable=run_preprocessing,
    )

    validation_task = PythonOperator(
        task_id="validate_blob",
        python_callable=run_validation,
    )

    create_asset_task = PythonOperator(
        task_id="create_data_asset",
        python_callable=create_data_asset,
    )

    (
        download_task
        >> loading_task
        >> preprocess_task
        >> validation_task
        >> create_asset_task
    )
