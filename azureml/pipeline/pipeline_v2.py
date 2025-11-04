from azure.ai.ml import Input, dsl

from azureml.scripts.connecting_workspace import ml_client
from utils.logger import setup_logger

# setup logger
logger = setup_logger(__name__)

logger.info("Loading components for emotion classifier pipeline...")
train_component = ml_client.components.get(name="train_component")
evaluate_component = ml_client.components.get(name="evaluate_component")


@dsl.pipeline(
    name="emotion_classifier_pipeline",
    description="Pipeline to train and evaluate emotion classifier",
    compute="adsai-lambda-1",
)
def emotion_classifier_pipeline(data_asset_uri: str):
    """
    Defines an Azure ML pipeline to train and evaluate an emotion classifier.

    The pipeline consists of two steps:
    1. Training the emotion classifier model using the specified dataset.
    2. Evaluating the trained model on the same dataset.

    Args:
        data_asset_uri (str): The URI/path to the training dataset
        (Azure ML data asset).

    Returns:
        dict: Dictionary containing outputs of the pipeline steps:
            - "model_output": Path to the trained model output.
            - "evaluation_output": Path to the evaluation results output.
    """
    training_step = train_component(
        data_asset_uri=Input(type="uri_file", path=data_asset_uri),
        learning_rate=2e-5,
        batch_size=32,
        num_epochs=3,
    )
    training_step.compute = "adsai-lambda-1"
    training_step.resources = {"instance_type": "gpu-instance", "instance_count": 1}

    evaluation_step = evaluate_component(
        data_asset_uri=data_asset_uri, model_dir=training_step.outputs.model_output
    )
    evaluation_step.compute = "adsai-lambda-1"
    evaluation_step.resources = {"instance_type": "gpu-instance", "instance_count": 1}

    return {
        "model_output": training_step.outputs.model_output,
        "evaluation_output": evaluation_step.outputs.output_dir,
    }
