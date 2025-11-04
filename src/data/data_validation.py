import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)


def handle_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataframe by removing the entire row.
    Args:
        dataframe (pd.DataFrame): The input dataframe with missing values.
    Returns:
        pd.DataFrame: The dataframe with missing values handled.
    """
    logger.info("Removing missing values from dataset")
    return dataframe.dropna()
