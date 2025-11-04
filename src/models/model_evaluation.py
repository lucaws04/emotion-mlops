import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from utils.logger import setup_logger

logger = setup_logger(__name__)


def evaluate_model(model, test_dataset, save_report_path=None):
    """
    Evaluate the model on the test dataset and return classification report.
    Optionally saves the report to disk.

    Args:
        model: The trained model to be evaluated.
        test_dataset: The test dataset.
        save_report_path (str, optional): Path to save the classification report.

    Returns:
        dict: Accuracy and F1 score.
    """
    logger.info("Evaluating model performance on test dataset...")

    test_loader = DataLoader(test_dataset, batch_size=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    report = classification_report(true_labels, predictions)

    logger.info(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    if save_report_path:
        with open(save_report_path, "w") as f:
            f.write(report)
        logger.info(f"Classification report saved to {save_report_path}")

    return {"accuracy": acc, "f1_weighted": f1}
