from azureml.scripts.connecting_workspace import ml_client
from utils.logger import setup_logger

# setup logger
logger = setup_logger(__name__)

logger.info("Listing all registered data assets:")
data_assets = ml_client.data.list()
for data_asset in data_assets:
    logger.info(
        "Name: %s, Description: %s, Type: %s",
        data_asset.name,
        data_asset.description,
        data_asset.type,
    )
