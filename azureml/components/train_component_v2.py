import os

from azure.ai.ml import load_component

from azureml.scripts.connecting_workspace import ml_client
from utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

logger.info("Workspace connected — continuing processing")

# Define path to component YAML
train_src_dir = "azureml/scripts"
train_component_path = os.path.join(train_src_dir, "train_v2.yml")

# Load and update the component
train_component = load_component(source=train_component_path)

logger.info("Registering component...")
registered_component = ml_client.components.create_or_update(train_component)

logger.info(
    f"Component {registered_component.name} with Version "
    f"{registered_component.version} is registered"
)
