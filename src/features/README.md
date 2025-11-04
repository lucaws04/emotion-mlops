# Features Module

Responsible for transforming raw data into features suitable for modeling.

- `feature_extraction.py`: Create domain-relevant features
- `feature_scaling.py`: Normalize or scale features

### `feature_extraction.py`
This script prepares textual data for training with transformer models, specifically BERT. It includes a custom PyTorch `Dataset` class for handling tokenized text and labels, as well as a utility function to load a pretrained tokenizer.

**Key Classes/Functions:**
- `EmotionDataset(texts, labels, tokenizer, max_len=128)`:
  A PyTorch Dataset that:
  - Tokenizes each input text using the provided BERT tokenizer.
  - Pads or truncates each sequence to a fixed length.
  - Returns input IDs, attention masks, and the corresponding label as tensors.

- `get_tokenizer() -> BertTokenizer`: Loads the pretrained BERT tokenizer (`bert-base-uncased`).

### `feature_scaling.py`
This script handles scaling of numerical features using either Standard Scaling (zero mean and unit variance) or Min-Max Scaling (rescaling between a defined range).

**Key Function:**
- `scale_numerical_features(dataframe: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame`:  Scales the specified columns in the dataframe.

**Note:** These modules are designed to be used together as part of a larger machine learning pipeline for tasks such as emotion classification in NLP.
