from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.logger import setup_logger

logger = setup_logger(__name__)


def split_dataset(
    dataframe: pd.DataFrame,
    target_column,
    test_size,
    val_size,
    random_state=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the DataFrame into training, validation, and test sets
    for both features and target.

    Args:
        dataframe (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple:
            X_train, X_val, X_test (pd.DataFrame): Features for training,
                validation, and test sets.
            y_train, y_val, y_test (pd.Series): Target values for training,
                validation, and test sets.
    """

    logger.info("Splitting dataset into train, validation, and test sets...")

    # Split features and target
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # First split: Train+Val vs Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state
    )

    # Second split: Train vs Val (from the remaining training portion)
    val_relative_size = val_size / (1 - test_size)  # adjust val size relative to X_temp
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
