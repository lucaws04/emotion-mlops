import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from utils.logger import setup_logger

logger = setup_logger(__name__)


class EmotionDataset(Dataset):
    """
    PyTorch dataset class for emotion classification tasks.

    This dataset processes a list of texts and their corresponding emotion labels.
    It tokenizes the input texts using a provided tokenizer (e.g., BERT), and returns
    tokenized inputs suitable for training a transformer-based model.

    Args:
        texts (list of str): List of input sentences or texts.
        labels (list or array-like): List of corresponding emotion labels as integers.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for
        converting text to token IDs.
        max_len (int, optional): Maximum sequence length for tokenization.
        Default is 128.

    Returns:
        dict:
            - 'input_ids' (torch.LongTensor): Token IDs padded/truncated to max_len.
            - 'attention_mask' (torch.LongTensor): Attention mask
               indicating non-padding tokens.
            - 'label' (torch.LongTensor): Emotion label as an integer tensor.
    """

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_tokenizer():
    """
    Load and return a pre-trained BERT tokenizer.

    This function loads the BERT tokenizer from the Hugging Face Transformers library,
    specifically the 'bert-base-uncased' pretrained model by default.
    You can replace this model name with any other compatible pretrained tokenizer.

    Returns:
        BertTokenizer: The loaded BERT tokenizer instance.
    """
    # Load the BERT tokenizer
    # You can replace "bert-base-uncased" with any other pre-trained model
    logger.info("Loading Transformers tokenizer...")
    return BertTokenizer.from_pretrained("bert-base-uncased")
