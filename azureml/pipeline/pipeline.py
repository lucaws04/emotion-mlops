from azure.ai.ml import Input, dsl
from azure.ai.ml.sweep import Choice, Uniform

from azureml.scripts.connecting_workspace import ml_client
from utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)
logger.info("Loading components for emotion classifier pipeline...")

train_component = ml_client.components.get(name="train_component")
evaluate_component = ml_client.components.get(name="evaluate_component")


@dsl.pipeline(
    name="emotion_classifier_pipeline_sweep",
    description="Pipeline with hyperparameter sweep for emotion classifier",
    compute="adsai-lambda-1",
)
def emotion_classifier_pipeline_sweep(data_asset_uri: Input):
    """
    Defines an Azure ML pipeline with hyperparameter sweep for emotion classifier.

    The pipeline consists of:
    1. A sweep job that trains multiple models with different hyperparameters
    2. An evaluation step that uses the best model from the sweep

    Args:
        data_asset_uri (Input): The Input object for the training dataset

    Returns:
        dict: Dictionary containing outputs of the pipeline steps
    """

    # Create the training step with hyperparameter sweep
    training_step = train_component(
        data_asset_uri=data_asset_uri,
        learning_rate=Uniform(min_value=1e-6, max_value=5e-5),
        batch_size=Choice(values=[8, 16, 32]),
        num_epochs=Choice(values=[3, 5, 7]),
    )

    # Configure the sweep
    sweep_step = training_step.sweep(
        primary_metric="eval_f1",
        goal="maximize",
        sampling_algorithm="random",
        compute="adsai-lambda-1",
    )

    # Set sweep limits
    sweep_step.set_limits(max_total_trials=10, max_concurrent_trials=2, timeout=7200)

    # Set compute resources for sweep trials
    sweep_step.resources = {"instance_type": "gpu-instance", "instance_count": 1}

    # Evaluation step runs on the best model output
    evaluation_step = evaluate_component(
        data_asset_uri=data_asset_uri,  # Use the same Input object
        model_dir=sweep_step.outputs.model_output,
    )
    evaluation_step.compute = "adsai-lambda-1"
    evaluation_step.resources = {"instance_type": "gpu-instance", "instance_count": 1}

    return {
        "model_output": sweep_step.outputs.model_output,
        "evaluation_output": evaluation_step.outputs.output_dir,
    }


# Script to submit the sweep pipeline
def submit_sweep_pipeline():
    """
    Submit the sweep pipeline to Azure ML
    """
    logger.info("Creating sweep pipeline...")

    # Create pipeline instance with proper Input object
    pipeline = emotion_classifier_pipeline_sweep(
        data_asset_uri=Input(
            type="uri_file", path="azureml:final_processed_dataset:1.0.8"
        )
    )

    # Submit the pipeline
    logger.info("Submitting sweep pipeline...")
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline, experiment_name="emotion_classifier_sweep"
    )

    logger.info(f"Sweep pipeline submitted. Job name: {pipeline_job.name}")
    logger.info(f"Job URL: {pipeline_job.studio_url}")

    return pipeline_job
