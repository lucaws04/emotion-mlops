import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Read a CSV file and return a DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    try:
        logger.info(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file at {filepath}: {e}")
        return pd.DataFrame()
