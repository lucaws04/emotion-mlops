import pandas as pd
import pytest
from transformers import BertForSequenceClassification, BertTokenizer

from src.inference import cli_inference


@pytest.fixture(scope="module")
def mock_model_components(tmp_path_factory):
    """
    Create temporary directories and save a pretrained BERT model and tokenizer.

    Returns:
        tuple: Paths to the saved model directory and tokenizer directory.
    """
    # Create temp paths
    model_dir = tmp_path_factory.mktemp("model_dir")
    tokenizer_dir = model_dir / "tokenizer"

    model_dir.mkdir(exist_ok=True)
    tokenizer_dir.mkdir(exist_ok=True)

    # Save a simple model and tokenizer
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(tokenizer_dir))

    return str(model_dir), str(tokenizer_dir)


def test_single_text_prediction(mock_model_components):
    """
    Test single text emotion prediction returns a valid emotion label.

    Args:
        mock_model_components (tuple): Paths to model and tokenizer directories.
    """
    model_path, tokenizer_path = mock_model_components
    model, tokenizer, device = cli_inference.load_emotion_model(
        model_path, tokenizer_path
    )
    class_names = ["joy", "anger", "sadness"]

    text = "I am happy today!"
    prediction = cli_inference.predict_emotion_single(
        text, model, tokenizer, device, class_names
    )
    assert prediction in class_names


def test_file_input_csv(tmp_path, mock_model_components):
    """
    Test prediction from CSV input file returns a DataFrame with emotion labels.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        mock_model_components (tuple): Paths to model and tokenizer directories.
    """
    model_path, tokenizer_path = mock_model_components
    model, tokenizer, device = cli_inference.load_emotion_model(
        model_path, tokenizer_path
    )
    class_names = ["joy", "anger", "sadness"]

    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"Sentence": ["This is great!", "I'm upset."]})
    df.to_csv(csv_path, index=False)

    result = cli_inference.predict_from_input(
        str(csv_path), model, tokenizer, device, class_names
    )
    assert isinstance(result, pd.DataFrame)
    assert "Emotion" in result.columns
    assert len(result) == 2


def test_file_input_txt(tmp_path, mock_model_components):
    """
    Test prediction from TXT input file returns a DataFrame with emotion labels.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        mock_model_components (tuple): Paths to model and tokenizer directories.
    """
    model_path, tokenizer_path = mock_model_components
    model, tokenizer, device = cli_inference.load_emotion_model(
        model_path, tokenizer_path
    )
    class_names = ["joy", "anger", "sadness"]

    txt_path = tmp_path / "test.txt"
    txt_path.write_text("I'm thrilled.\nThis is frustrating.")

    result = cli_inference.predict_from_input(
        str(txt_path), model, tokenizer, device, class_names
    )
    assert isinstance(result, pd.DataFrame)
    assert "Emotion" in result.columns
    assert len(result) == 2


def test_invalid_input_type(mock_model_components):
    """
    Test that passing invalid input type raises a TypeError.

    Args:
        mock_model_components (tuple): Paths to model and tokenizer directories.
    """
    model_path, tokenizer_path = mock_model_components
    model, tokenizer, device = cli_inference.load_emotion_model(
        model_path, tokenizer_path
    )
    class_names = ["joy", "anger", "sadness"]

    with pytest.raises(TypeError):
        cli_inference.predict_from_input(12345, model, tokenizer, device, class_names)
