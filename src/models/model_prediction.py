import torch
import torch.nn.functional as F

from utils.logger import setup_logger

logger = setup_logger(__name__)


def predict(model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform emotion prediction on batches of data using a pre-trained model.

    Parameters:
        model (torch.nn.Module): Pre-trained PyTorch model for sequence classification.
        dataloader (torch.utils.data.DataLoader): DataLoader that
            yields batches of input data.
        device (str, optional): Device to run inference on ("cuda" or "cpu").
            Defaults to GPU if available.

    Returns:
        list: List of predicted class indices for all samples in the dataloader.
    """
    logger.info("Starting prediction...")
    model.eval()
    model.to(device)
    logger.info(f"Model moved to device: {device}")

    all_predictions = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            logger.debug(f"Processing batch {i + 1}")

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)

            all_predictions.extend(preds.cpu().numpy())

    logger.info("Prediction completed.")
    return all_predictions
