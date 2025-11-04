import logging
import os
import re
import shutil
import uuid
from typing import List

import torch
import torch.nn.functional as F
import whisper
import yt_dlp
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from moviepy import VideoFileClip
from pydantic import BaseModel

from src.models.model_saving import load_emotion_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

# Initialize FastAPI app
app = FastAPI()

# Core emotions
CORE_EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

# Lazy global vars
_model = None
_tokenizer = None
_device = None


def get_model_tokenizer():
    """
    Load and cache the emotion classification model, tokenizer, and device.

    Returns:
        tuple: (model, tokenizer, device) loaded from the saved checkpoint.
    """
    global _model, _tokenizer, _device
    if _model is None or _tokenizer is None or _device is None:
        model_path = os.path.join("models", "bert_model")
        tokenizer_path = os.path.join("models", "bert_model")
        _model, _tokenizer, _device = load_emotion_model(model_path, tokenizer_path)
        _model.eval()
    return _model, _tokenizer, _device


# Input models
class PredictInput(BaseModel):
    sentences: List[str]


class URLInput(BaseModel):
    url: str


@app.get("/")
def read_root():
    """
    Root endpoint for health check or welcome message.

    Returns:
        dict: A welcome message with guidance on available endpoints.
    """
    return {"message": "Welcome to the NLP API. Use /predict or /transcribe."}


@app.post("/predict")
def predict(input: PredictInput):
    """
    Predict emotion labels for a list of input sentences.

    Args:
        input (PredictInput): A list of text sentences to classify.

    Returns:
        dict: A list of results, each containing the input sentence,
        predicted emotion, and confidence score.
    """
    logger.info(f"POST /predict called with {len(input.sentences)} sentence(s).")
    if not input.sentences:
        raise HTTPException(status_code=400, detail="No sentences provided.")

    model, tokenizer, device = get_model_tokenizer()
    results = []

    for sentence in input.sentences:
        if not sentence.strip():
            continue
        try:
            encoded = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            ).to(device)
            with torch.no_grad():
                outputs = model(**encoded)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).squeeze()
                predicted_index = torch.argmax(probs).item()
                predicted_emotion = CORE_EMOTIONS[predicted_index]
                confidence = float(round(probs[predicted_index].item() * 100, 2))
            results.append(
                {
                    "sentence": sentence,
                    "emotion": predicted_emotion,
                    "confidence": confidence,
                }
            )
        except Exception as e:
            logger.error(f"Failed to process sentence: {sentence}, error: {e}")
            results.append({"sentence": sentence, "error": "Failed to classify."})

    return {"results": results}


@app.post("/transcribe")
async def transcribe_video(
    url: str = Form(default=None), file: UploadFile = File(default=None)
):
    """
    Transcribe speech from an audio/video file or YouTube URL,
    then classify the emotion in each sentence.

    Args:
        url (str, optional): A YouTube URL to download and transcribe.
        file (UploadFile, optional): A locally uploaded audio or video file.

    Returns:
        dict: A list of sentences from the transcription,
        each with an emotion label and confidence score.

    Raises:
        HTTPException: If input is missing, file download fails,
        transcription fails, or emotion classification fails.
    """
    logger.info("POST /transcribe endpoint hit.")
    model, tokenizer, device = get_model_tokenizer()
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    unique_id = str(uuid.uuid4())
    mp3_path = os.path.join(temp_dir, f"{unique_id}.mp3")

    try:
        # Case 1: Transcribe from URL
        if url:
            logger.info(f"Processing from URL: {url}")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(temp_dir, f"{unique_id}.%(ext)s"),
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
                "keepvideo": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        # Case 2: Transcribe from uploaded file
        elif file:
            logger.info(f"Processing uploaded file: {file.filename}")
            suffix = os.path.splitext(file.filename)[-1]
            raw_path = os.path.join(temp_dir, f"{unique_id}{suffix}")
            with open(raw_path, "wb") as out_file:
                out_file.write(await file.read())

            # Convert to mp3 if video
            if suffix.lower() not in [".mp3", ".wav"]:
                logger.info("Converting uploaded video to mp3...")
                clip = VideoFileClip(raw_path)
                clip.audio.write_audiofile(mp3_path)
                clip.close()
            else:
                mp3_path = raw_path

        else:
            raise HTTPException(
                status_code=400, detail="Either 'url' or 'file' must be provided."
            )

        if not os.path.exists(mp3_path):
            raise FileNotFoundError("Audio extraction failed.")

        # Transcribe with whisper
        model_whisper = whisper.load_model("base", device=device)
        result = model_whisper.transcribe(mp3_path, language="en")
        full_text = result["text"]
        sentences = re.split(r"(?<=[a-z0-9\.][.!?])\s+", full_text)

        results = []
        for sentence in sentences:
            if sentence.strip():
                encoded = tokenizer(
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128,
                ).to(device)
                with torch.no_grad():
                    logits = model(**encoded).logits
                    probs = F.softmax(logits, dim=1).squeeze()
                    predicted_index = torch.argmax(probs).item()
                    predicted_emotion = CORE_EMOTIONS[predicted_index]
                    confidence = float(round(probs[predicted_index].item() * 100, 2))
                results.append(
                    {
                        "sentence": sentence,
                        "emotion": predicted_emotion,
                        "confidence": confidence,
                    }
                )

        return {"transcription": results}

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=500, detail="Transcription or classification failed."
        )

    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            logger.warning(f"Cleanup failed: {cleanup_error}")


if __name__ == "__main__":
    print("Swagger docs at http://127.0.0.1:8000/docs")

# Run with: uvicorn app.API:app --reload
