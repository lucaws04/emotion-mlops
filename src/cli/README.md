# Emotion Prediction CLI

A simple command-line interface (CLI) tool to detect emotions in text using a fine-tuned BERT-based model. Supports both direct text input and text files. Ideal for quick emotional analysis in scripts or terminal environments. Made using Typer.

---

## Features

- Predicts emotions like **happiness**, **sadness**, **anger**, **fear**, **disgust**, **surprise**, **neutral**.
- Accepts plain text via CLI or file input.
- Verbose mode for debugging.
- Reproducible predictions using a fixed random seed.

---

## Emotion Categories

The model supports the following emotion classes:

- `anger`
- `disgust`
- `fear`
- `happiness`
- `neutral`
- `sadness`
- `surprise`

---

### 1. Project Structure

```

project/
├── cli.py
├── models/
│ └── bert_model/tokenizer
│ ├── pytorch_model.bin
│ ├── config.json
│ └── tokenizer_config.json
├── src/
│ └── inference/
│ └── cli_inference.py
```

## Usage

```python cli.py predict --text "sentence" ```

```python cli.py predict --file path/to/file```