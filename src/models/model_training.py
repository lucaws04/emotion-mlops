import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from utils.logger import setup_logger

logger = setup_logger(__name__)


def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs,
    batch_size,
    learning_rate,
    device_override=None,
):
    """
    Train a neural network classification model with evaluation and early stopping.

    This function performs model training over a specified number of epochs using
    mini-batch gradient descent and the AdamW optimizer. Model performance is
    evaluated on a validation dataset after each epoch, logging accuracy, precision,
    recall, and F1-score. Training stops early if validation accuracy does not improve
    for a specified number of epochs.

    Features:
        - Automatic device selection (GPU if available), with optional override.
        - Mixed precision training for faster computation on CUDA devices.
        - Learning rate scheduling with warmup.
        - Early stopping based on validation accuracy.
        - Best model checkpoint saving and restoration.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_dataset (torch.utils.data.Dataset): Dataset for model training.
        val_dataset (torch.utils.data.Dataset): Dataset for model validation.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training and validation.
        learning_rate (float): Learning rate for the AdamW optimizer.
        device_override (str, optional): Override for device selection
        ("cpu" or "cuda").
        If None, CUDA is used if available.

    Returns:
        torch.nn.Module: The model with weights from the epoch that achieved the
        highest validation accuracy.
        final_metrics (dict): Dictionary containing final training and validation
        metrics, including accuracy, precision, recall, F1-score, and average loss.

    Raises:
        ValueError: If device_override is not None, 'cpu', or 'cuda'.

    Example:
        trained_model = train_model(
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            epochs=5,
            batch_size=32,
            learning_rate=3e-5,
            device_override=None
        )
    """
    logger.info("Starting training process")

    # Device setup
    if device_override not in [None, "cpu", "cuda"]:
        raise ValueError("device_override must be one of None, 'cpu', or 'cuda'.")
    device = torch.device(
        device_override
        if device_override
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)

    # Enable mixed precision training if available
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    early_stop_counter = 0
    patience = 2

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            correct += (outputs.logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}"
        )
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)

        # Validation phase with full metrics
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate comprehensive metrics
        val_acc = accuracy_score(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="weighted"
        )

        logger.info(
            f"Epoch {epoch+1} - Val Acc: {val_acc:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )
        # Log validation metrics per epoch
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("val_precision", precision, step=epoch)
        mlflow.log_metric("val_recall", recall, step=epoch)
        mlflow.log_metric("val_f1", f1, step=epoch)

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_model.pt")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model before returning
    model.load_state_dict(torch.load("best_model.pt"))

    # Return final metrics for MLflow logging
    final_metrics = {
        "eval_accuracy": val_acc,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "train_accuracy": train_acc,
        "train_loss": avg_loss,
    }

    return model, final_metrics
