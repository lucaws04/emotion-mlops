import os

from azure.ai.ml import load_component

from azureml.scripts.connecting_workspace import ml_client
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)
logger.info("Workspace connected — registering register_model component")

# Define path to component YAML
register_src_dir = "azureml/scripts"
register_component_path = os.path.join(register_src_dir, "register.yml")

# Load and register the component
register_model_component = load_component(source=register_component_path)

registered_component = ml_client.components.create_or_update(register_model_component)

logger.info(
    f"Component {registered_component.name} with Version "
    f"{registered_component.version} is registered"
)
