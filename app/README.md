# NLP API — Emotion Detection Interface

This FastAPI-based application provides a cloud-deployable backend for detecting emotions from text and audio inputs. It supports direct text prediction as well as transcription-based emotion detection from YouTube links and uploaded media files.

## Features

- **Text-Based Emotion Detection**: Send one or more typed sentences and receive emotion predictions.
- **Audio/Video Transcription**: Upload files or provide a YouTube link. The API transcribes speech using Whisper and returns sentence-level emotion predictions.
- **Model Integration**:
  - Audio transcription powered by **OpenAI Whisper**.
  - Emotion classification using a fine-tuned **BERT model**.
- **Core Emotions Supported**: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise.
- **FastAPI Interface**: Interactive API docs available at `/docs`.

## Installation Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/your-repo.git
   cd your-repo
   ```

2. **Install dependencies**:
   It's recommended to use a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download or place the models**:
   Place the fine-tuned BERT model and tokenizer files under:
   ```
   models/bert_model/
   ```

4. **Run the API locally**:
   ```bash
   uvicorn app.API:app --reload
   ```
   Then visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive testing.

## Usage Examples

### 1. Text Emotion Prediction

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "sentences": [
    "I'm feeling fantastic today!",
    "This is the worst thing ever."
  ]
}
```

**Response**:
```json
{
  "results": [
    {
      "sentence": "I'm feeling fantastic today!",
      "emotion": "happiness",
      "confidence": 96.2
    },
    {
      "sentence": "This is the worst thing ever.",
      "emotion": "anger",
      "confidence": 91.5
    }
  ]
}
```

### 2. Audio Transcription + Emotion Detection

**Endpoint**: `POST /transcribe`

You can either:
- Upload a file using `file` field (supports `.mp3`, `.wav`, `.mp4`, etc.)
- Or provide a YouTube URL using the `url` field.

Example (form-data):
```
file: your_audio.mp3
```
or
```
url: https://www.youtube.com/watch?v=example
```

**Response**:
```json
{
  "transcription": [
    {
      "sentence": "Hello and welcome.",
      "emotion": "neutral",
      "confidence": 88.4
    }
  ]
}
```

## Notes

- You must have internet access if using YouTube URL-based transcription (downloads the audio).
- Long audio files may take longer to process due to transcription and classification time.
- Error handling is built-in for invalid input, missing fields, or audio extraction failures.