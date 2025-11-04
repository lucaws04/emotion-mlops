import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.transformers
import pandas as pd
import seaborn as sns
from mlflow.models import infer_signature
from sklearn.metrics import confusion_matrix
from transformers import TextClassificationPipeline

from data.data_ingestion import load_dataset
from data.data_splitting import split_dataset
from features.feature_extraction import EmotionDataset, get_tokenizer
from models.model_evaluation import evaluate_model
from models.model_prediction import predict
from models.model_saving import load_emotion_model as load_model
from utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args():
    """
    Parses command-line arguments.

    Expected arguments:
        --data_asset_uri (str): URI/path to the input dataset.
        --model_dir (str): Directory path where the trained model is stored.
        --output_dir (str): Directory path to save evaluation outputs and reports.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_asset_uri", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    """
    Main script entry point to evaluate the emotion classification model.

    Workflow:
    - Loads dataset from the given data asset URI.
    - Splits dataset into train, validation, and test sets.
    - Prepares the test dataset for evaluation.
    - Loads the trained model and tokenizer.
    - Evaluates the model on the test set and logs metrics.
    - Saves classification report and predictions as artifacts.
    - Logs the model pipeline with MLflow for reproducibility.
    - Registers the trained model in MLflow/Azure ML model registry.

    Uses MLflow for experiment tracking and artifact logging.
    """
    args = parse_args()
    logger.info("Starting evaluation script...")

    # Initialize MLflow (Azure ML auto-configures tracking URI)
    with mlflow.start_run():

        logger.info(f"Loading dataset from {args.data_asset_uri}...")
        df = load_dataset(args.data_asset_uri)

        logger.info("Splitting dataset...")
        _, _, X_test, _, _, y_test = split_dataset(
            dataframe=df,
            target_column="emotion",
            test_size=0.15,
            val_size=0.15,
            random_state=42,
        )

        logger.info("Preparing test dataset...")
        tokenizer = get_tokenizer()
        test_dataset = EmotionDataset(
            X_test["sentence"].astype(str).tolist(), y_test, tokenizer
        )

        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Loading model from {args.model_dir}...")
        model, tokenizer, device = load_model(args.model_dir, args.model_dir)

        logger.info(f"Model loaded on device: {device}")
        logger.info("Evaluating model...")

        # Save evaluation report
        report_path = Path(args.output_dir) / "classification_report.txt"
        metrics = evaluate_model(model, test_dataset, save_report_path=str(report_path))
        logger.info(f"Classification report saved to {report_path}")

        mlflow.log_metrics(metrics, step=0)

        # Save predictions
        predictions = predict(model, test_dataset)
        pred_df = pd.DataFrame(
            {
                "text": X_test["sentence"].tolist(),
                "true_label": y_test,
                "predicted_label": predictions,
            }
        )
        pred_file_path = Path(args.output_dir) / "predictions.csv"
        pred_df.to_csv(pred_file_path, index=False)
        logger.info(f"Predictions saved to {pred_file_path}")

        # Logging confusion matrix
        logger.info("Generating confusion matrix...")
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        conf_path = Path(args.output_dir) / "confusion_matrix.png"
        plt.savefig(conf_path)
        plt.close()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_path = f"evaluation_outputs_{timestamp}"
        mlflow.log_artifacts(args.output_dir, artifact_path=artifact_path)

        # Log model pipeline (optional, if you want to re-log the model)
        pipeline = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
            return_all_scores=True,
        )

        input_example = pd.DataFrame({"text": ["I am happy", "I am sad"]})
        output_example = pipeline(input_example["text"].tolist())
        signature = infer_signature(input_example, output_example)

        mlflow.transformers.log_model(
            transformers_model=pipeline,
            artifact_path="model",
            task="text-classification",
            input_example=input_example,
            signature=signature,
        )

        # Register model (visible in Azure ML Studio)
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, "emotion-classifier-model")

        logger.info("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
