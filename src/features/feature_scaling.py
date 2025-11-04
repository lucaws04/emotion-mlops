# standard_scaler(dataframe)
# min_max_scaler(dataframe, feature_range=(0, 1))

from typing import List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.logger import setup_logger

logger = setup_logger(__name__)


def scale_numerical_features(
    dataframe: pd.DataFrame, columns: List[str], method: str = "standard"
) -> pd.DataFrame:
    """
    Scale numerical features in the dataframe using standard scaling or min-max scaling.

    Args:
        dataframe (pd.DataFrame): The input dataframe with numerical features.
        columns (List[str]): List of columns to be scaled.
        method (str): The scaling method to use ('standard' or 'minmax').

    Returns:
        pd.DataFrame: The dataframe with numerical features scaled.

    Raises:
        ValueError: If an invalid scaling method is provided.
    """
    logger.info(f"Scaling columns: {columns} using method: {method}")
    # Select scaler based on method
    scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler()}

    if method not in scalers:
        raise ValueError("Invalid method. Choose 'standard' or 'minmax'.")

    scaler = scalers[method]
    scaled_columns = scaler.fit_transform(dataframe[columns])
    scaled_df = pd.DataFrame(scaled_columns, columns=columns)

    return pd.concat([dataframe.drop(columns, axis=1), scaled_df], axis=1)
