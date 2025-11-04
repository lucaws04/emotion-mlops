from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data

from azureml.scripts.connecting_workspace import ml_client
from utils.logger import setup_logger

# setup logger
logger = setup_logger(__name__)

# Create and register the data asset
data_asset = Data(
    name="processed_dataset_v3",
    description="Cleaned and processed full dataset for emotion classification",
    version="1.0.0",  # Change version numbers later when updating
    path="data/processed/processed_dataset.csv",
    type=AssetTypes.URI_FILE,
)

registered = ml_client.data.create_or_update(data_asset)
logger.info(
    "Registered data asset: %s with description: %s",
    registered.name,
    registered.description,
)
