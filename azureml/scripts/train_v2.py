import os
from argparse import ArgumentParser
from pathlib import Path

import mlflow
from transformers import TextClassificationPipeline

from data.data_ingestion import load_dataset
from data.data_splitting import split_dataset
from features.feature_extraction import EmotionDataset, get_tokenizer
from models.model_definitions import get_model
from models.model_saving import save_model
from models.model_training import train_model
from utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args():
    """
    Parses command-line arguments for training script.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - data_asset_uri (str): Path or URI to the dataset.
            - model_output (str): Directory to save the trained model.
            - learning_rate (float): Learning rate for training.
            - num_epochs (int): Number of training epochs.
            - batch_size (int): Batch size during training.
    """
    parser = ArgumentParser()

    # For normal runs
    parser.add_argument("--data_asset_uri", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    # Handle sweep job parameters from environment variables
    # This allows Azure ML sweep to pass hyperparameters via environment
    if args.learning_rate is None:
        args.learning_rate = float(os.environ.get("AZUREML_SWEEP_learning_rate", 2e-5))
    if args.num_epochs is None:
        args.num_epochs = int(os.environ.get("AZUREML_SWEEP_num_epochs", 3))
    if args.batch_size is None:
        args.batch_size = int(os.environ.get("AZUREML_SWEEP_batch_size", 32))

    logger.info(
        f"Training parameters - LR: {args.learning_rate}, "
        f"Epochs: {args.num_epochs}, Batch size: {args.batch_size}"
    )

    return args


def log_metrics_for_sweep(metrics):
    """
    Log metrics in a format suitable for Azure ML sweep jobs.

    Args:
        metrics (dict): Dictionary containing training metrics
    """
    # Log metrics that can be used for sweep optimization
    if "eval_accuracy" in metrics:
        mlflow.log_metric("eval_accuracy", metrics["eval_accuracy"])
        # Also log as primary metric for sweep
        mlflow.log_metric("accuracy", metrics["eval_accuracy"])

    if "eval_loss" in metrics:
        mlflow.log_metric("eval_loss", metrics["eval_loss"])
        mlflow.log_metric("loss", metrics["eval_loss"])

    if "eval_f1" in metrics:
        mlflow.log_metric("eval_f1", metrics["eval_f1"])
        mlflow.log_metric("f1_score", metrics["eval_f1"])

    # Log all other metrics
    for key, value in metrics.items():
        if key not in ["eval_accuracy", "eval_loss", "eval_f1"]:
            mlflow.log_metric(key, value)


def main():
    """
    Main training routine for the emotion classification model.
    Enhanced for sweep job support.

    Steps:
    - Loads the dataset from the given data asset URI.
    - Splits the dataset into training, validation, and test sets.
    - Initializes tokenizer and prepares datasets.
    - Instantiates and trains the model using specified hyperparameters.
    - Saves the trained model and tokenizer.
    - Logs training parameters and model to MLflow for tracking and deployment.
    """
    args = parse_args()
    logger.info("Starting training script...")
    logger.info(f"Arguments: {vars(args)}")

    # Initialize MLflow (Azure ML automatically sets tracking URI)
    with mlflow.start_run():
        # Log hyperparameters early for sweep tracking
        mlflow.log_params(
            {
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
            }
        )

        logger.info("Loading dataset...")
        df = load_dataset(args.data_asset_uri)
        logger.info(f"Dataset loaded with {len(df)} records.")

        logger.info("Splitting dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            dataframe=df,
            target_column="emotion",
            test_size=0.15,
            val_size=0.15,
            random_state=42,
        )

        logger.info(
            f"Dataset split into {len(X_train)} training, "
            f"{len(X_val)} validation, and {len(X_test)} test records."
        )

        # Log dataset statistics
        mlflow.log_metrics(
            {
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "total_size": len(df),
                "num_classes": len(set(y_train)),
            }
        )

        logger.info("Initializing tokenizer and datasets...")
        tokenizer = get_tokenizer()
        train_dataset = EmotionDataset(
            X_train["sentence"].astype(str).tolist(), y_train, tokenizer
        )
        val_dataset = EmotionDataset(
            X_val["sentence"].astype(str).tolist(), y_val, tokenizer
        )
        logger.info(
            f"Training dataset size: {len(train_dataset)}, "
            f"Validation dataset size: {len(val_dataset)}"
        )

        logger.info("Training model...")
        model = get_model(num_labels=len(set(y_train)))

        try:
            model, metrics = train_model(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device_override="cuda",
            )

            logger.info("Model training completed.")
            logger.info(f"Training metrics: {metrics}")

            # Log metrics in sweep-compatible format
            log_metrics_for_sweep(metrics)

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            # Log failure metrics for sweep tracking
            mlflow.log_metrics(
                {"eval_accuracy": 0.0, "accuracy": 0.0, "training_failed": 1.0}
            )
            raise

        logger.info(f"Saving model to {args.model_output}...")
        output_dir = Path(args.model_output)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            save_model(model, tokenizer, str(output_dir))
            logger.info(f"Model saved to {output_dir}.")

            # Create and log the pipeline for inference
            pipeline = TextClassificationPipeline(
                model=model,
                tokenizer=tokenizer,
                return_all_scores=True,
                device=0 if model.device.type == "cuda" else -1,
            )

            # Log model with MLflow
            mlflow.transformers.log_model(
                transformers_model=pipeline,
                artifact_path="model",
                task="text-classification",
                registered_model_name=(
                    f"emotion_classifier_lr_{args.learning_rate}_bs_{args.batch_size}"
                ),
            )

            logger.info("Model logged to MLflow successfully.")

        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            mlflow.log_metric("model_save_failed", 1.0)
            raise

        logger.info("Training completed successfully.")

        # Final success metric
        mlflow.log_metric("training_completed", 1.0)


if __name__ == "__main__":
    main()
