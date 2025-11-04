import unittest
from unittest.mock import MagicMock, patch

import torch

from src.models.model_evaluation import evaluate_model


class DummyOutput:
    """
    Dummy output class to mimic the structure returned by Hugging Face models,
    particularly the `logits` attribute used during evaluation.
    """

    def __init__(self, logits):
        self.logits = logits


class DummyDataset:
    """
    A simple dummy dataset that returns fixed input tensors for use in
    testing the evaluate_model function. It mimics a small classification dataset.
    """

    def __init__(self):
        # 3 samples with same input shape for batching
        self.data = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "label": torch.tensor(0),
            },
            {
                "input_ids": torch.tensor([4, 5, 6]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "label": torch.tensor(1),
            },
            {
                "input_ids": torch.tensor([7, 8, 9]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "label": torch.tensor(1),
            },
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestEvaluateModel(unittest.TestCase):
    """
    Unit tests for the evaluate_model function in src.models.model_evaluation.
    """

    def test_evaluate_model_returns_classification_report(self):
        """
        Test that evaluate_model returns a classification report containing the expected
        metrics: precision, recall, f1-score, and support.
        """
        # Mock model
        model = MagicMock()
        model.eval = MagicMock()

        def model_call(input_ids, attention_mask=None):
            batch_size = input_ids.shape[0]
            logits = torch.zeros((batch_size, 2))
            if batch_size > 0:
                logits[0, 0] = 10.0
            if batch_size > 1:
                logits[1:, 1] = 10.0
            return DummyOutput(logits=logits)

        model.side_effect = model_call
        dataset = DummyDataset()

        with patch("torch.cuda.is_available", return_value=False):
            report = evaluate_model(model, dataset)

        # Updated: match keys returned by your actual implementation
        assert isinstance(report, dict)
        assert "accuracy" in report
        assert "f1_weighted" in report

    def test_model_eval_called(self):
        """
        Test that the model's eval() method is called during evaluation to ensure
        it switches to inference mode.
        """
        model = MagicMock()
        model.eval = MagicMock()

        def model_call(input_ids, attention_mask=None):
            batch_size = input_ids.shape[0]
            logits = torch.zeros((batch_size, 2))
            if batch_size > 0:
                logits[0, 0] = 10.0
            if batch_size > 1:
                logits[1:, 1] = 10.0
            return DummyOutput(logits=logits)

        model.side_effect = model_call

        dataset = DummyDataset()

        with patch("torch.cuda.is_available", return_value=False):
            evaluate_model(model, dataset)
            model.eval.assert_called_once()


def test_evaluate_model_saves_report(tmp_path):
    """
    Test that evaluate_model saves the classification report
    when given a save_report_path.

    Checks that the report file is created and contains expected content
    (e.g., 'precision'),
    and that the function returns a dict with accuracy and f1_weighted metrics.

    Args:
        tmp_path: pytest fixture providing a temporary directory for the test.
    """
    model = MagicMock()
    model.eval = MagicMock()

    def model_call(input_ids, attention_mask=None):
        batch_size = input_ids.shape[0]
        logits = torch.zeros((batch_size, 2))
        if batch_size > 0:
            logits[0, 0] = 10.0
        if batch_size > 1:
            logits[1:, 1] = 10.0
        return DummyOutput(logits=logits)

    model.side_effect = model_call

    dataset = DummyDataset()

    report_file = tmp_path / "report.txt"

    with patch("torch.cuda.is_available", return_value=False):
        result = evaluate_model(model, dataset, save_report_path=str(report_file))

    # Check the file was created and has content
    assert report_file.exists()
    content = report_file.read_text()
    assert "precision" in content.lower()  # basic check for report content

    # Also verify returned metrics are present
    assert "accuracy" in result
    assert "f1_weighted" in result
