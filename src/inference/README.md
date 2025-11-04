# Emotion Prediction Module

A powerful BERT-based emotion prediction module that supports emotion classification on both single text inputs and bulk file-based inputs (`.txt`, `.csv`). Ideal for pipelines that analyze the emotional tone of user content, feedback, or conversation logs.

---

## Features

- Load a pre-trained BERT model and tokenizer
- Predict emotions for:
    - A single sentence
    - A `.txt` file (one sentence per line)
    - A `.csv` file with a `Sentence` column
- Fast, GPU-accelerated predictions (if available)
- Returns one of the following emotion labels:
    - `anger`, `disgust`, `fear`, `happiness`, `neutral`, `sadness`, `surprise`
- Includes robust logging via `utils.logger`

---

## Project Structure

```bash
src/
├── inference
│ └── cli_inference.py
├── models/
│ └── bert_model/
│ ├── pytorch_model.bin
│ ├── config.json
│ └── tokenizer_config.json
└── utils/
  └── logger.py # Logging setup utility
```
