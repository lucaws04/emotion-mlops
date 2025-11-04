from typing import Union

import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)


def emotion_mapping(emotion_column: Union[pd.Series, list]) -> pd.Series:
    """
    Maps a column of fine-grained emotion labels to their main emotion categories.

    Args:
        emotion_column (Union[pd.Series, list]): A column or list of emotion labels
                                                 (e.g., 'joy', 'fear', 'annoyance').

    Returns:
        pd.Series: A Series of main emotion categories
                   (e.g., 'happiness', 'anger', 'sadness', etc.).
    """

    logger.info("Mapping fine-grained emotions to main emotion categories...")

    # Define fine-grained to main emotion category mapping
    mapping = {
        "happiness": [
            "joy",
            "optimism",
            "approval",
            "pride",
            "gratitude",
            "confidence",
            "satisfaction",
            "hope",
            "love",
            "excitement",
            "caring",
            "relief",
            "admiration",
            "amusement",
            "anticipation",
            "encouragement",
            "desire",
            "happiness",
        ],
        "sadness": [
            "disappointment",
            "nostalgia",
            "remorse",
            "pain",
            "stress",
            "regret",
            "resignation",
            "despair",
            "confusion",
            "uncertainty",
            "sadness",
        ],
        "anger": [
            "anger",
            "annoyance",
            "disapproval",
            "frustration",
            "disbelief",
            "warning",
            "rejection",
        ],
        "disgust": ["disgust"],
        "fear": [
            "fear",
            "nervousness",
            "worry",
            "anxiety",
            "doubt",
            "insecurity",
            "urgency",
            "panic",
        ],
        "surprise": ["surprise", "realization", "shock", "amazement", "wonder"],
        "neutral": [
            "neutral",
            "mixed",
            "trust",
            "agreement",
            "instruction",
            "suggestion",
            "request",
            "confirmation",
            "acceptance",
            "reassurance",
            "clarification",
            "understanding",
            "certainty",
            "curiosity",
        ],
    }

    # Create reverse lookup dict for faster mapping
    reverse_lookup = {
        sub_emotion.lower(): main_emotion
        for main_emotion, sub_list in mapping.items()
        for sub_emotion in sub_list
    }

    def map_label(label):
        if pd.isna(label) or str(label).strip() == "":
            return None
        return reverse_lookup.get(str(label).lower().strip(), None)

    return pd.Series(emotion_column).apply(map_label)
