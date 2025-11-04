from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from src.models.model_saving import load_emotion_model, save_model


# --- Fixtures ---
@pytest.fixture
def tmp_model_dir(tmp_path):
    """Creates a temporary model directory."""

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return str(model_dir)


@pytest.fixture
def tmp_tokenizer_dir(tmp_path):
    """Creates a temporary tokenizer directory."""

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    return str(tokenizer_dir)


# --- Tests for load_emotion_model ---


@mock.patch("src.models.model_saving.BertTokenizer.from_pretrained")
@mock.patch("src.models.model_saving.BertForSequenceClassification.from_pretrained")
def test_load_emotion_model_success(
    mock_model_load, mock_tokenizer_load, tmp_model_dir, tmp_tokenizer_dir
):
    """Test successful loading of both model and tokenizer."""

    mock_model = mock.MagicMock(spec=BertForSequenceClassification)
    mock_tokenizer = mock.MagicMock(spec=BertTokenizer)

    mock_model_load.return_value = mock_model
    mock_tokenizer_load.return_value = mock_tokenizer

    model, tokenizer, device = load_emotion_model(tmp_model_dir, tmp_tokenizer_dir)

    assert model == mock_model
    assert tokenizer == mock_tokenizer
    assert device in [torch.device("cuda"), torch.device("cpu")]
    mock_model.to.assert_called_once()


def test_load_emotion_model_missing_model_dir(tmp_tokenizer_dir):
    """Test error is raised when model directory does not exist."""

    with pytest.raises(FileNotFoundError, match="Model directory not found"):
        load_emotion_model("invalid_model_path", tmp_tokenizer_dir)


def test_load_emotion_model_missing_tokenizer_dir(tmp_model_dir):
    """Test error is raised when tokenizer directory does not exist."""

    with pytest.raises(FileNotFoundError, match="Tokenizer directory not found"):
        load_emotion_model(tmp_model_dir, "invalid_tokenizer_path")


@mock.patch(
    "src.models.model_saving.BertTokenizer.from_pretrained",
    side_effect=Exception("tokenizer load error"),
)
@mock.patch("src.models.model_saving.BertForSequenceClassification.from_pretrained")
def test_load_emotion_model_tokenizer_load_raises(
    mock_model_load, mock_tokenizer_load, tmp_model_dir, tmp_tokenizer_dir, caplog
):
    """Test error handling and logging when tokenizer loading fails."""

    with caplog.at_level("ERROR"):
        with pytest.raises(Exception, match="tokenizer load error"):
            load_emotion_model(tmp_model_dir, tmp_tokenizer_dir)
    assert any(
        "Error loading model/tokenizer" in record.message for record in caplog.records
    )


@mock.patch("src.models.model_saving.BertTokenizer.from_pretrained")
@mock.patch(
    "src.models.model_saving.BertForSequenceClassification.from_pretrained",
    side_effect=Exception("model load error"),
)
def test_load_emotion_model_model_load_raises(
    mock_model_load, mock_tokenizer_load, tmp_model_dir, tmp_tokenizer_dir, caplog
):
    """Test error handling and logging when model loading fails."""

    with caplog.at_level("ERROR"):
        with pytest.raises(Exception, match="model load error"):
            load_emotion_model(tmp_model_dir, tmp_tokenizer_dir)
    assert any(
        "Error loading model/tokenizer" in record.message for record in caplog.records
    )


# --- Tests for save_model ---


def test_save_model_success(tmp_path):
    """Test successful saving of model and tokenizer."""

    model = mock.MagicMock()
    tokenizer = mock.MagicMock()

    save_path = tmp_path / "save_test"
    result = save_model(model, tokenizer, str(save_path))

    assert "model_path" in result
    assert "tokenizer_path" in result
    model.save_pretrained.assert_called_once()
    tokenizer.save_pretrained.assert_called_once()


def test_save_model_invalid_path_type():
    """Test save_model raises TypeError for non-string path."""

    with pytest.raises(TypeError):
        save_model(mock.MagicMock(), mock.MagicMock(), filepath=123)


def test_save_model_no_overwrite(tmp_path):
    """Test save_model returns None when directory exists and overwrite is False."""

    path = tmp_path / "existing_dir"
    path.mkdir()
    result = save_model(mock.MagicMock(), mock.MagicMock(), str(path), overwrite=False)
    assert result is None


def test_save_model_raises_and_logs_error(caplog):
    """Test error and logging when model saving fails."""

    dummy_model = MagicMock()
    dummy_tokenizer = MagicMock()

    with patch("os.makedirs", return_value=None), patch.object(
        dummy_model, "save_pretrained", side_effect=Exception("model save failed")
    ), patch.object(dummy_tokenizer, "save_pretrained"):

        with pytest.raises(Exception, match="model save failed"):
            save_model(dummy_model, dummy_tokenizer, filepath="dummy/path")

        assert any(
            "Failed to save model/tokenizer" in record.message
            for record in caplog.records
        )


def test_save_tokenizer_raises_and_logs_error(caplog):
    """Test error and logging when tokenizer saving fails."""

    dummy_model = MagicMock()
    dummy_tokenizer = MagicMock()

    with patch("os.makedirs", return_value=None), patch.object(
        dummy_model, "save_pretrained"
    ), patch.object(
        dummy_tokenizer,
        "save_pretrained",
        side_effect=Exception("tokenizer save failed"),
    ):

        with pytest.raises(Exception, match="tokenizer save failed"):
            save_model(dummy_model, dummy_tokenizer, filepath="dummy/path")

        assert any(
            "Failed to save model/tokenizer" in record.message
            for record in caplog.records
        )
