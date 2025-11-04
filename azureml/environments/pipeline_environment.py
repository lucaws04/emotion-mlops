from azure.ai.ml.entities import Environment

from azureml.scripts.connecting_workspace import ml_client
from utils.logger import setup_logger

logger = setup_logger(__name__)

custom_env_name = "pipeline"
version = "1.0.5"

env = Environment(
    name=custom_env_name,
    description="Custom environment for emotion classification pipeline",
    tags={"pytorch": "2.6"},
    conda_file="environment.yaml",
    image="mcr.microsoft.com/aifx/acpt/stable-ubuntu2204"
    "-cu118-py310-torch260:biweekly.202505.2.v1",
    version=version,
)

env.image_build_compute = "adsai-lamda-1"

# Register the environment
pipeline_env = ml_client.environments.create_or_update(env)

logger.info(
    "Environment with name %s is registered to workspace, "
    "the environment version is %s",
    pipeline_env.name,
    pipeline_env.version,
)
