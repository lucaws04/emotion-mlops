from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
import torch
from httpx import ASGITransport, AsyncClient

from app.API import app


@pytest.fixture(scope="module")
def test_app():
    with patch("src.models.model_saving.load_emotion_model") as mock_load:
        fake_model = MagicMock()
        fake_model.eval.return_value = None

        mock_outputs = MagicMock()
        mock_outputs.logits = MagicMock()
        mock_outputs.logits.__getitem__.return_value = [0.1] * 7
        fake_model.__call__ = MagicMock(return_value=mock_outputs)

        fake_tokenizer = MagicMock()
        mock_encoded = MagicMock()
        mock_encoded.to.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        fake_tokenizer.return_value = mock_encoded

        mock_load.return_value = (fake_model, fake_tokenizer, "cpu")
        yield app


@patch("app.API.get_model_tokenizer")
@pytest.mark.asyncio
async def test_predict_valid_input(mock_get_model_tokenizer, test_app):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_device = "cpu"

    mock_encoded = MagicMock()
    mock_encoded.to.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_tokenizer.return_value = mock_encoded

    mock_model.__call__ = MagicMock(
        return_value=MagicMock(
            logits=torch.tensor([[0.1, 0.1, 0.1, 5.0, 0.1, 0.1, 0.1]])
        )
    )

    mock_get_model_tokenizer.return_value = (mock_model, mock_tokenizer, mock_device)

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/predict", json={"sentences": ["I am very happy today!"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1


@pytest.mark.asyncio
async def test_predict_missing_sentences(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/predict", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_empty_sentence_list(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/predict", json={"sentences": []})
    assert response.status_code == 400
    assert response.json()["detail"] == "No sentences provided."


@pytest.mark.asyncio
async def test_root_endpoint(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the NLP API. Use /predict or /transcribe."
    }


@patch("app.API.get_model_tokenizer")
@patch("app.API.whisper.load_model")
@patch("app.API.yt_dlp.YoutubeDL")
@patch("app.API.os.path.exists", return_value=True)
@patch("app.API.shutil.rmtree")
@pytest.mark.asyncio
async def test_transcribe_endpoint(
    mock_rmtree,
    mock_exists,
    mock_ytdl,
    mock_whisper_model,
    mock_get_model_tokenizer,
    test_app,
):
    # Mock YouTube downloader
    mock_ytdl_instance = MagicMock()
    mock_ytdl.return_value.__enter__.return_value = mock_ytdl_instance
    mock_ytdl_instance.download.return_value = None

    # Mock whisper model
    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = {"text": "Hello world. This is a test."}
    mock_whisper_model.return_value = mock_whisper

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value.to.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    # Mock model output (logits as real tensor)
    def fake_forward(**kwargs):
        return type(
            "Output",
            (),
            {"logits": torch.tensor([[0.1, 0.1, 0.1, 5.0, 0.1, 0.1, 0.1]])},
        )()

    mock_model = MagicMock()
    mock_model.side_effect = fake_forward
    mock_device = "cpu"

    mock_get_model_tokenizer.return_value = (mock_model, mock_tokenizer, mock_device)

    # Make test request
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/transcribe", data={"url": "https://youtube.com/test"}
        )
        assert response.status_code == 200
        assert "transcription" in response.json()


@patch("app.API.get_model_tokenizer")
@patch("app.API.whisper.load_model")
@pytest.mark.asyncio
async def test_transcribe_upload_file(
    mock_whisper_model, mock_get_model_tokenizer, test_app
):
    # Mock whisper transcription
    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = {"text": "I feel great. What a day!"}
    mock_whisper_model.return_value = mock_whisper

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value.to.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    # Mock model output (logits as real tensor)
    def fake_forward(**kwargs):
        return type(
            "Output",
            (),
            {"logits": torch.tensor([[0.1, 0.1, 0.1, 5.0, 0.1, 0.1, 0.1]])},
        )()

    mock_model = MagicMock()
    mock_model.side_effect = fake_forward
    mock_device = "cpu"

    mock_get_model_tokenizer.return_value = (mock_model, mock_tokenizer, mock_device)

    fake_audio = BytesIO(b"fake audio content")
    files = {"file": ("test.mp3", fake_audio, "audio/mpeg")}

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/transcribe", files=files)
        assert response.status_code == 200
        assert "transcription" in response.json()
