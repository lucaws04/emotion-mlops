import pandas as pd

from src.data.data_splitting import split_dataset


def test_split_dataset():
    """
    Test the split_dataset function to ensure it correctly splits the data
    into training, validation, and test sets with non-zero lengths.
    """
    df = pd.DataFrame(
        {
            "sentence": ["I am happy", "I am sad", "I am angry"] * 10,
            "main_emotion": ["hapiness", "sadness", "anger"] * 10,
        }
    )
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df, "main_emotion", test_size=0.2, val_size=0.2, random_state=42
    )
    assert len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0
