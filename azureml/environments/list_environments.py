from azureml.scripts.connecting_workspace import ml_client
from utils.logger import setup_logger

# setup logger
logger = setup_logger(__name__)

# List all environments in the workspace
environments = ml_client.environments.list()
for env in environments:
    logger.info(
        "Name: %s, Version: %s, Description: %s", env.name, env.version, env.description
    )
