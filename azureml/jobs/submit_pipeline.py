from azure.ai.ml import Input
from packaging.version import parse as parse_version

from azureml.pipeline.pipeline import emotion_classifier_pipeline_sweep
from azureml.scripts.connecting_workspace import ml_client
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Fetch latest version of dataset
logger.info("Fetching latest version of 'final_processed_dataset'...")

# Get all versions of the dataset
dataset_versions = ml_client.data.list(name="final_processed_dataset")

# Sort and pick the latest version
latest_version = sorted(dataset_versions, key=lambda d: parse_version(d.version))[
    -1
].version

logger.info(f"Latest dataset version: {latest_version}")

data_asset_uri = Input(
    type="uri_file", path=f"azureml:final_processed_dataset:{latest_version}"
)


# Submit the pipeline
def submit_sweep_pipeline():
    logger.info("Creating sweep pipeline...")

    pipeline = emotion_classifier_pipeline_sweep(data_asset_uri=data_asset_uri)

    logger.info("Submitting sweep pipeline...")
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline, experiment_name="emotion_classifier_sweep"
    )

    logger.info(f"Sweep pipeline submitted. Job name: {pipeline_job.name}")
    logger.info(f"Job URL: {pipeline_job.studio_url}")

    return pipeline_job


if __name__ == "__main__":
    submit_sweep_pipeline()
