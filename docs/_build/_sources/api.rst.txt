API Overview
============

The **FastAPI NLP Backend** serves as the core of the emotion detection and transcription system. It is designed for flexibility, supporting both text and audio inputs, including files and YouTube URLs.

App Structure
-------------

**`app/` Application Interface**

  This folder contains the main FastAPI application responsible for exposing API endpoints and orchestrating the NLP backend logic.

Main Features
-------------

- **Emotion Detection from Text**
  
  - Accepts plain text input via a REST endpoint.
  - Applies a fine-tuned BERT model to classify emotions:
    - *anger, disgust, fear, happiness, neutral, sadness, surprise*
  - Returns predictions with per-sentence confidence scores.

- **Emotion Detection from Audio**
  
  - Accepts audio and video file uploads (`.mp3`, `.wav`, `.mp4`, etc.).
  - Supports YouTube URLs by extracting audio using `yt_dlp`.
  - Converts video to `.mp3` using `moviepy`.
  - Transcribes speech to text using OpenAI Whisper model.
  - Feeds the transcribed text into the emotion classifier.

API Features
------------

- **Input Validation**
  
  - Uses Pydantic models to enforce schema validation and structure.
  - Supports batch inputs for text prediction.

- **Error Handling**
  
  - Returns clear, structured JSON error responses.
  - Provides informative logging for troubleshooting.

- **Interactive Docs**
  
  - OpenAPI documentation is automatically generated and available at: `/docs`

Use Cases
---------

This API supports a wide range of emotion recognition scenarios:

- Batch analysis of written text (e.g., chat logs, social media).
- Emotion detection in spoken language from uploaded media.
- Real-time emotion classification from YouTube videos.