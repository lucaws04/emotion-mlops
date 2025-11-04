import json

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from utils.logger import setup_logger

# Create a browser credential
credential = DefaultAzureCredential()

# setup logger
logger = setup_logger(__name__)

with open("config/config.json") as f:
    config = json.load(f)

ml_client = MLClient(
    credential=credential,
    subscription_id=config["subscription_id"],
    resource_group_name=config["resource_group"],
    workspace_name=config["workspace_name"],
)
logger.info("Connected to workspace: %s", ml_client.workspace_name)
