from unittest import mock

import pandas as pd

from src.data.data_ingestion import load_dataset


def test_load_dataset_success(tmp_path):
    """
    Test that load_dataset successfully loads a CSV file into a DataFrame.

    Creates a temporary CSV file, loads it, and checks that the DataFrame
    has the expected shape, columns, and values.
    """
    # Create a sample CSV file
    csv_content = "col1,col2\n1,2\n3,4"
    file_path = tmp_path / "test.csv"
    file_path.write_text(csv_content)

    df = load_dataset(str(file_path))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]
    assert df.iloc[0]["col1"] == 1


@mock.patch("src.data.data_ingestion.pd.read_csv")
def test_load_dataset_failure(mock_read_csv):
    """
    Test that load_dataset returns an empty DataFrame when pd.read_csv raises an error.

    Mocks pd.read_csv to raise FileNotFoundError and verifies
    that load_dataset handles the error gracefully.
    """
    # Simulate pd.read_csv raising an exception
    mock_read_csv.side_effect = FileNotFoundError("No such file")

    df = load_dataset("nonexistent.csv")

    assert isinstance(df, pd.DataFrame)
    assert df.empty
