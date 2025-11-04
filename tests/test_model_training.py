from unittest import mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.models.model_training import train_model

# --- Dummy Classes for Isolated Testing ---


class DummyDataset(Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 100, (8,)),
            "attention_mask": torch.ones(8),
            "label": torch.randint(0, 2, (1,)).item(),
        }


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 2)

    def forward(self, input_ids, attention_mask=None):
        logits = self.linear(input_ids.float())
        return type("Output", (object,), {"logits": logits})


# --- Fixtures ---


@pytest.fixture
def model():
    return DummyModel()


@pytest.fixture
def train_dataset():
    return DummyDataset(size=20)


@pytest.fixture
def val_dataset():
    return DummyDataset(size=10)


# --- Tests ---


def test_train_model_runs_and_returns_metrics(
    model, train_dataset, val_dataset, tmp_path
):
    """Ensure train_model runs and returns valid metrics."""

    with mock.patch("mlflow.log_metric"), mock.patch("torch.save"), mock.patch(
        "torch.load", return_value=model.state_dict()
    ):

        trained_model, metrics = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            device_override="cpu",
        )

    assert isinstance(trained_model, nn.Module)
    assert isinstance(metrics, dict)
    assert "eval_accuracy" in metrics
    assert "train_loss" in metrics
    assert metrics["eval_accuracy"] >= 0.0
    assert metrics["eval_f1"] >= 0.0


def test_train_model_invalid_device_raises(model, train_dataset, val_dataset):
    """Ensure train_model raises ValueError on invalid device string."""
    with pytest.raises(ValueError):
        train_model(
            model,
            train_dataset,
            val_dataset,
            epochs=1,
            batch_size=4,
            learning_rate=3e-5,
            device_override="invalid",
        )


def test_train_model_device_selection(model, train_dataset, val_dataset):
    """Ensure train_model supports both CPU and automatic device selection."""

    with mock.patch("mlflow.log_metric"), mock.patch("torch.save"), mock.patch(
        "torch.load", return_value=model.state_dict()
    ):

        # Explicit CPU
        train_model(
            model,
            train_dataset,
            val_dataset,
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            device_override="cpu",
        )
        # Auto selection (should use CPU if GPU is unavailable)
        train_model(
            model,
            train_dataset,
            val_dataset,
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            device_override=None,
        )


def test_early_stopping_triggered(monkeypatch, model, train_dataset, val_dataset):
    """Simulate bad validation to trigger early stopping."""

    # Patch accuracy_score to always return low accuracy to force early stopping
    monkeypatch.setattr(
        "src.models.model_training.accuracy_score", lambda y_true, y_pred: 0.5
    )
    monkeypatch.setattr(
        "src.models.model_training.precision_recall_fscore_support",
        lambda y_true, y_pred, average: (0.5, 0.5, 0.5, None),
    )

    with mock.patch("mlflow.log_metric"), mock.patch("torch.save"), mock.patch(
        "torch.load", return_value=model.state_dict()
    ):

        _, metrics = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=10,  # Should stop before this
            batch_size=4,
            learning_rate=1e-3,
            device_override="cpu",
        )

    assert metrics["eval_accuracy"] == 0.5


def test_train_model_logs_mlflow(monkeypatch, model, train_dataset, val_dataset):
    """Test that mlflow.log_metric is called properly."""

    mocked_log_metric = mock.Mock()
    monkeypatch.setattr("mlflow.log_metric", mocked_log_metric)

    with mock.patch("torch.save"), mock.patch(
        "torch.load", return_value=model.state_dict()
    ):
        train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            device_override="cpu",
        )

    assert mocked_log_metric.called
