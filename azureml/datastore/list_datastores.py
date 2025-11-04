from azureml.scripts.connecting_workspace import ml_client
from utils.logger import setup_logger

# setup logger
logger = setup_logger(__name__)

# List all datastores
datastores = ml_client.datastores.list()
for ds in datastores:
    logger.info("Datastore name: %s, type: %s", ds.name, ds.type)
