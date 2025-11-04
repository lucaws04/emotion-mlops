from azure.ai.ml import Input

from azureml.pipeline.pipeline_v2 import emotion_classifier_pipeline
from azureml.scripts.connecting_workspace import ml_client
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Connect to Azure ML workspace
logger.info("Connecting to Azure ML workspace...")

# Input dataset
data_input = Input(type="uri_file", path="azureml:final_processed_dataset:1.0.8")

# Create pipeline job
pipeline_job = emotion_classifier_pipeline(data_asset_uri=data_input)

logger.info(
    "Submitting emotion classifier pipeline with training and evaluation steps..."
)
submitted = ml_client.jobs.create_or_update(pipeline_job)
logger.info(f"Pipeline submitted! Name: {submitted.name}")
print(f"Submitted pipeline job: {submitted.name}")
print(f"Monitor in Azure ML Studio: {submitted.studio_url}")
