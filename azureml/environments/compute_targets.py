from azureml.scripts.connecting_workspace import ml_client
from utils.logger import setup_logger

logger = setup_logger(__name__)

# List all compute targets in the workspace
logger.info("Listing compute targets...")
compute_targets = ml_client.compute.list()

for compute_target in compute_targets:
    logger.info("Name: %s, Type: %s", compute_target.name, compute_target.type)
