import pytest
import torch
from transformers import BertTokenizer

from src.features.feature_extraction import EmotionDataset, get_tokenizer


@pytest.fixture
def sample_data():
    """
    Provides sample input text and corresponding labels for testing.
    """
    texts = ["I love this!", "This is terrible."]
    labels = [1, 0]
    return texts, labels


def test_emotion_dataset_len(sample_data):
    """
    Test that the length of the EmotionDataset matches the number of input samples.
    """
    texts, labels = sample_data
    tokenizer = get_tokenizer()
    dataset = EmotionDataset(texts, labels, tokenizer)

    assert len(dataset) == len(texts)


def test_emotion_dataset_getitem(sample_data):
    """
    Test that a single sample returned from EmotionDataset has all required keys
    and that the values are PyTorch tensors of expected dimensions.
    """
    texts, labels = sample_data
    tokenizer = get_tokenizer()
    dataset = EmotionDataset(texts, labels, tokenizer)

    sample = dataset[0]

    # Check keys
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "label" in sample

    # Check tensor types and shapes
    assert isinstance(sample["input_ids"], torch.Tensor)
    assert isinstance(sample["attention_mask"], torch.Tensor)
    assert isinstance(sample["label"], torch.Tensor)

    # input_ids and attention_mask should have shape (max_len,)
    assert sample["input_ids"].dim() == 1
    assert sample["attention_mask"].dim() == 1

    # Label should be a scalar tensor
    assert sample["label"].dim() == 0

    # Label value matches
    assert sample["label"].item() == labels[0]


def test_get_tokenizer_returns_berttokenizer():
    """
    Test that get_tokenizer returns an instance of BertTokenizer.
    """

    tokenizer = get_tokenizer()
    assert isinstance(tokenizer, BertTokenizer)
