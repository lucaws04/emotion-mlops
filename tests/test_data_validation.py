import pandas as pd

from src.data.data_validation import handle_missing_values


def test_handle_missing_values():
    """
    Test that handle_missing_values correctly drops rows with any missing values.

    This test verifies that:
    - All NaNs are removed.
    - The resulting DataFrame contains only complete rows.
    """
    df = pd.DataFrame(
        {"sentence": ["hello", None, "hi"], "emotion": ["happy", "sad", None]}
    )
    result_df = handle_missing_values(df)
    assert result_df.isnull().sum().sum() == 0
    assert len(result_df) == 1
