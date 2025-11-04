from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from src.inference import cli_inference

CLASS_NAMES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]


@pytest.fixture
def dummy_model():
    """Fixture returning a mock model with callable interface and eval method."""

    model = MagicMock()
    model.eval = MagicMock()
    model.__call__ = MagicMock()
    return model


@pytest.fixture
def dummy_tokenizer():
    """Fixture returning a mock tokenizer that returns dummy tensors on call."""

    tokenizer = MagicMock()
    tokenizer.__call__ = MagicMock(
        return_value={
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }
    )
    return tokenizer


@pytest.fixture
def dummy_logits():
    """Fixture providing dummy logits tensor for model outputs."""

    return torch.tensor([[0.1, 0.2, 0.3, 0.8, 0.1, 0.1, 0.2]])


# --- Test 1: predict_emotions input must be DataFrame
def test_predict_emotions_type_error(dummy_model, dummy_tokenizer):
    """
    Test that predict_emotions raises a TypeError if input is not a DataFrame.

    This ensures input validation enforces correct input types.
    """
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        cli_inference.predict_emotions(
            "not a dataframe", dummy_model, dummy_tokenizer, "cpu", CLASS_NAMES
        )


# --- Test 2: DataFrame missing 'Sentence' column
def test_predict_emotions_missing_column(dummy_model, dummy_tokenizer):
    """
    Test that predict_emotions raises ValueError if DataFrame lacks 'Sentence' column.

    This checks presence of required column in input data.
    """
    df = pd.DataFrame({"Text": ["I am sad"]})
    with pytest.raises(ValueError, match="must contain a 'Sentence' column"):
        cli_inference.predict_emotions(
            df, dummy_model, dummy_tokenizer, "cpu", CLASS_NAMES
        )


# --- Test 3: CSV file missing 'Sentence' column
def test_predict_from_input_csv_missing_column(tmp_path, dummy_model, dummy_tokenizer):
    """
    Test that predict_from_input raises ValueError if CSV file lacks 'Sentence' column.

    Validates CSV input schema before prediction.
    """
    file = tmp_path / "test.csv"
    file.write_text("Text\nHello world")

    with pytest.raises(ValueError, match="CSV file must contain a 'Sentence' column"):
        cli_inference.predict_from_input(
            str(file), dummy_model, dummy_tokenizer, "cpu", CLASS_NAMES
        )


# --- Test 4: TXT file read with fallback encoding
def test_predict_from_input_txt(tmp_path, dummy_model, dummy_tokenizer, dummy_logits):
    """
    Test prediction from a TXT file input.

    Ensures the function reads lines, handles encodings, and returns a DataFrame
    containing an 'Emotion' prediction column.
    """
    dummy_model.return_value = MagicMock(logits=dummy_logits)
    file = tmp_path / "test.txt"
    file.write_text("I am happy\n")

    df = cli_inference.predict_from_input(
        str(file), dummy_model, dummy_tokenizer, "cpu", CLASS_NAMES
    )
    assert isinstance(df, pd.DataFrame)
    assert "Emotion" in df.columns


# --- Test 5: Input is neither file nor string
def test_predict_from_input_invalid_type(dummy_model, dummy_tokenizer):
    """
    Test that predict_from_input raises TypeError
    if input is neither file path nor string.

    Validates type checking and input parameter requirements.
    """
    with pytest.raises(TypeError, match="Input must be a file path or a string."):
        cli_inference.predict_from_input(
            1234, dummy_model, dummy_tokenizer, "cpu", CLASS_NAMES
        )
