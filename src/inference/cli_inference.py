import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from utils.logger import setup_logger

logger = setup_logger(__name__)
model_path = "models/bert_model"
tokenizer_path = "models/bert_model/tokenizer_config.json"


# Function to load the model and tokenizer
def load_emotion_model(model_path: str, tokenizer_path: str):
    """
    Load a pre-trained BERT-based emotion classification model and tokenizer.

    :param model_path: Path to the directory containing the saved model files.
    :param tokenizer_path: Path to the directory containing the saved tokenizer files.

    :returns: A tuple containing:
        - **model** (`BertForSequenceClassification`): The loaded emotion classification model.
        - **tokenizer** (`BertTokenizer`): The tokenizer used to preprocess input text.
        - **device** (`torch.device`): The device (CPU or GPU) the model is loaded on.
    """

    logger.info("Loading emotion prediction model...")
    tokenizer = BertTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, cache_dir=str(tokenizer_path)
    )
    model = BertForSequenceClassification.from_pretrained(
        str(model_path), local_files_only=True, cache_dir=str(model_path)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Emotion prediction model and tokenizer loaded.")
    return model, tokenizer, device


def predict_emotions(df, model, tokenizer, device, class_names):
    """
    Predict emotions for each sentence in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Sentence' column with text data.
        model: Loaded emotion classification model.
        tokenizer: Tokenizer corresponding to the model.
        device: Device where the model runs (CPU or GPU).
        class_names (list): List of emotion class names.

    Returns:
        pd.DataFrame: Original DataFrame with an additional
        'Emotion' column containing predictions.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame with a 'Sentence' column.")
    if "Sentence" not in df.columns:
        raise ValueError("DataFrame must contain a 'Sentence' column.")

    def predict(text):
        model.eval()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        return class_names[torch.argmax(logits).item()]

    tqdm.pandas(desc="Predicting Emotions")
    df["Emotion"] = df["Sentence"].progress_apply(predict)
    logger.debug("Received input text: %s", df["Sentence"])

    return df


def predict_emotion_single(text, model, tokenizer, device, class_names):
    """
    Predict emotion for a single text input.

    Args:
        text (str): Input text string.
        model: Loaded emotion classification model.
        tokenizer: Corresponding tokenizer.
        device: Device where model runs.
        class_names (list): List of emotion class names.

    Returns:
        str: Predicted emotion label.
    """
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    logger.debug("Received input text: %s", text)
    return class_names[torch.argmax(logits).item()]


def predict_from_input(input_data, model, tokenizer, device, class_names):
    """
    General-purpose prediction handler for CSV files, TXT files, or single string input.

    Args:
        input_data (str): Path to a .csv/.txt file or a single sentence.
        model, tokenizer, device, class_names: Required components for
        emotion prediction.

    Returns:
        pd.DataFrame or str: DataFrame with predictions (for files) or a
        single prediction (for strings).
    """
    # If it's a path to a file
    if os.path.isfile(input_data):
        ext = os.path.splitext(input_data)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(input_data)
            if "Sentence" not in df.columns:
                raise ValueError("CSV file must contain a 'Sentence' column.")
            return predict_emotions(df, model, tokenizer, device, class_names)

        elif ext == ".txt":
            try:
                with open(input_data, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
            except UnicodeDecodeError:
                with open(input_data, "r", encoding="latin1") as f:
                    lines = [line.strip() for line in f if line.strip()]

            df = pd.DataFrame(lines, columns=["Sentence"])
            return predict_emotions(df, model, tokenizer, device, class_names)

        else:
            raise ValueError(
                "Unsupported file type. Please provide a .csv or .txt file."
            )

    # Else assume it's a direct sentence string
    elif isinstance(input_data, str):
        logger.debug(
            "Received input text: %s", input_data
        )  # internal details for debugging

        return predict_emotion_single(input_data, model, tokenizer, device, class_names)

    else:
        raise TypeError("Input must be a file path or a string.")
