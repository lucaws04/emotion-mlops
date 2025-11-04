from transformers import BertForSequenceClassification

from utils.logger import setup_logger

logger = setup_logger(__name__)


def get_model(num_labels):
    """
    Load and return a pre-trained BERT model for sequence classification.

    Args:
        num_labels (int): Number of output labels/classes for classification.

    Returns:
        BertForSequenceClassification: Pre-trained BERT model
        configured for sequence classification.

    Notes:
        - The model used is "bert-base-uncased" by default.
        - You can replace this with another pretrained model if desired.
        - The model is set for single-label classification tasks.
    """
    # Load the BERT model for sequence classification
    # You can replace "bert-base-uncased" with any other pre-trained model
    logger.info("Loading Transformers model...")
    return BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        problem_type="single_label_classification",
    )
