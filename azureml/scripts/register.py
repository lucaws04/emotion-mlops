import argparse
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

from utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_input
    logger.info(f"Registering model from path: {model_path}")

    # Connect to Azure ML
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id="0a94de80-6d3b-49f2-b3e9-ec5818862801",
        resource_group="buas-y2",
        workspace_name="NLP9-2025",
    )

    model = Model(
        path=model_path,
        name="emotion_classifier_model",
        type="custom_model",
        description="Model registered via SDK instead of CLI",
        tags={"stage": "staging", "framework": "transformers"},
    )

    registered_model = ml_client.models.create_or_update(model)
    logger.info(
        f"Model registered "
        f"successfully: {registered_model.name}:{registered_model.version}"
    )

    # Write output for pipeline output
    output_dir = os.environ.get("AZUREML_OUTPUT_model_name", "/tmp")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_name.txt")
    with open(output_path, "w") as f:
        f.write(f"{registered_model.name}:{registered_model.version}")
    logger.info(f"Model info written to output: {output_path}")


if __name__ == "__main__":
    main()
