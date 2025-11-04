import os

import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def load_emotion_model(model_path: str, tokenizer_path: str):
    """
    Load a pre-trained BERT-based emotion classification model and tokenizer.

    Args:
        model_path (str): Path to the directory containing the saved model files.
        tokenizer_path (str): Path to the directory containing the saved tokenizer
            files.

    Returns:
        tuple:
            model (BertForSequenceClassification): The loaded emotion classification
                model.
            tokenizer (BertTokenizer): The corresponding tokenizer for preprocessing
                inputs.
            device (torch.device): The device (CPU or GPU) to which the model has been
                moved.
    """
    logger.info("Loading emotion prediction model...")

    # Validate paths
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not os.path.isdir(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")

    try:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        )
    except Exception as e:
        logger.error(f"Error loading model/tokenizer: {e}", exc_info=True)
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model loaded and moved to device: {device}")
    return model, tokenizer, device


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    filepath: str,
    overwrite: bool = True,
):
    """
    Save a Hugging Face Transformer model and tokenizer.

    Args:
        model (PreTrainedModel): The Hugging Face model to save.
        tokenizer (PreTrainedTokenizerFast): The corresponding tokenizer.
        filepath (str): Directory where the model and tokenizer will be saved.
        overwrite (bool): Whether to overwrite if the directory exists.
    Returns:
        dict: Paths to saved model and tokenizer files.
    """
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string.")

    if os.path.exists(filepath) and not overwrite:
        logger.warning(
            f"Directory '{filepath}' already exists and overwrite is set to False."
        )
        return None

    logger.info("Saving the model and tokenizer...")
    try:
        os.makedirs(filepath, exist_ok=True)
        model.save_pretrained(filepath)
        tokenizer.save_pretrained(filepath)
        logger.info(f"Model and tokenizer saved to '{filepath}'")
        return {
            "model_path": os.path.join(filepath, "pytorch_model.bin"),
            "tokenizer_path": os.path.join(filepath, "tokenizer_config.json"),
        }
    except Exception as e:
        logger.error(f"Failed to save model/tokenizer: {e}", exc_info=True)
        raise
