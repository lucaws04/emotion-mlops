import os
import sys

from src.data.data_ingestion import load_dataset
from src.data.data_processing import emotion_mapping
from src.data.data_splitting import split_dataset
from src.data.data_validation import handle_missing_values
from src.features.feature_extraction import EmotionDataset, get_tokenizer
from src.models.model_definitions import get_model
from src.models.model_evaluation import evaluate_model
from src.models.model_prediction import predict
from src.models.model_saving import save_model
from src.models.model_training import train_model
from src.utils.logger import setup_logger

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "src"))

logger = setup_logger(__name__)
logger.info("🚀 Starting emotion classification training pipeline...")


def main():
    """
    Executes the end-to-end emotion classification training pipeline.
    """
    try:
        # Step 1: Load Data
        logger.info("Loading dataset from CSV...")
        df = load_dataset("data/processed/processed_dataset.csv")
        logger.info(f"Loaded dataset with {len(df)} records.")

        # Step 2: Handle missing values
        logger.info("🧹 Handling missing values...")
        handle_missing_values(df)
        logger.info("Missing values handled.")

        # Step 3: Map emotions
        logger.info("Mapping fine-grained emotions to main categories...")
        df["main_emotion"] = emotion_mapping(df["emotion"])
        main_emotions = df["main_emotion"].nunique()
        logger.info(f"Mapped to {main_emotions} main emotion categories.")

        # Step 4: Split dataset
        logger.info("Splitting dataset into train, validation, and test sets...")

        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            dataframe=df,
            target_column="main_emotion",
            test_size=0.1,
            val_size=0.1,
            random_state=42,
        )
        logger.info(
            f"Split complete: Train={len(X_train)}, "
            f"Val={len(X_val)}, Test={len(X_test)}"
        )

        # Step 5: Tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = get_tokenizer()
        logger.info("Tokenizer loaded.")

        # Step 6: Create EmotionDataset instances
        logger.info("Converting data into EmotionDataset format...")
        train_dataset = EmotionDataset(
            X_train["sentence"].astype(str).tolist(),
            y_train,
            tokenizer,
        )

        val_dataset = EmotionDataset(
            X_val["sentence"].astype(str).tolist(), y_val, tokenizer
        )
        test_dataset = EmotionDataset(
            X_test["sentence"].astype(str).tolist(), y_test, tokenizer
        )
        logger.info("Datasets ready.")

        # Step 7: Model Initialization & Training
        logger.info("Initializing model...")
        model = get_model(num_labels=len(set(y_train)))
        logger.info("Starting model training...")
        train_model(model, train_dataset, val_dataset)
        logger.info("Model training complete.")

        # Step 8: Evaluation
        logger.info("Evaluating model on test set...")
        evaluate_model(model, test_dataset)
        logger.info("Evaluation complete.")

        # Step 9: Inference
        logger.info("Running inference on test set...")
        predictions = predict(model, test_dataset)
        logger.info("Predictions generated.")

        # Step 10: Save the model
        logger.info("Saving model and tokenizer...")
        save_model(model, tokenizer, "models/bert_emotion_model")
        logger.info("Model saved to 'models/bert_emotion_model'.")

        logger.info("Training pipeline completed successfully.")
        return predictions

    except Exception as e:
        logger.exception(f"Pipeline failed due to error: {e}")
        raise


if __name__ == "__main__":
    main()
