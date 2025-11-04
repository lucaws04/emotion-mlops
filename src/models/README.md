# Models Module

This module provides the core functionality for defining, training, evaluating, saving, and using machine learning models, specifically BERT-based models for emotion classification.

### Module Components

- `model_definitions.py`: Defines model architectures.
- `model_training.py`: Handles model training loops and validation.
- `model_evaluation.py`: Computes evaluation metrics like accuracy and F1 score.
- `model_prediction.py`: Generates predictions on new input data.
- `model_saving.py`: Saves and loads trained models.

### `model_definitions.py`
This script defines and retrieves the model architecture used in this project.

**Key Function:**
- `get_model(num_labels) -> BertForSequenceClassification`: Loads a pretrained BERT model (`bert-base-uncased`) for sequence classification with the specified number of output labels.

### `model_training.py`
This script trains a PyTorch model using the given training and validation datasets.

**Key Function:**
- `train_model(model, train_dataset, val_dataset, epochs, batch_size, learning_rate, device_override=None)`: Trains the model using AdamW optimizer and cross-entropy loss. Logs training and validation accuracy for each epoch.

### `model_evaluation.py`
This script evaluates the trained model on a test dataset.

**Key Function:**
- `evaluate_model(model, test_dataset)`:
  Computes and prints:
  - Test accuracy
  - Weighted F1 score
  - Detailed classification report (precision, recall, F1 for each class)

Uses `sklearn.metrics` for metric calculation and logs progress using the project’s logging utility.

### `model_prediction.py`
Generates predictions from a trained model for a list of input texts.

**Key Function:**
- `predict(model, dataloader, device='cuda' | 'cpu') -> List[int]`: Accepts a model and a list of input strings, returns a list of predicted labels. Logging is used to track input and output.

### `model_saving.py`
This script contains functions for saving and loading Keras models to/from disk. It's essential for persisting trained models and reusing them for predictions or further training.

**Key Functions:**
- `save_model(model, tokenizer, filepath, overwrite=True) -> dict`
  Saves model and tokenizer to disk. Creates directories if needed and handles overwriting.
- `load_emotion_model(model_path, tokenizer_path) -> Tuple[model, tokenizer, device]`:  Loads a model and tokenizer from specified local paths, returning them along with the proper device.

**Note:** These modules are designed to be used together as part of a larger machine learning pipeline for tasks such as emotion classification in NLP.
