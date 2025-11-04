import pandas as pd

from src.data.data_processing import emotion_mapping


def test_emotion_mapping_known_emotions():
    """
    Test that known fine-grained emotions are correctly mapped to main emotions.
    """
    fine_emotions = [
        "joy",
        "anger",
        "disgust",
        "fear",
        "surprise",
        "neutral",
        "nostalgia",
    ]
    expected_main = [
        "happiness",
        "anger",
        "disgust",
        "fear",
        "surprise",
        "neutral",
        "sadness",
    ]

    result = emotion_mapping(fine_emotions)

    assert list(result) == expected_main


def test_emotion_mapping_unknown_emotion():
    """
    Test that unknown emotions (not in mapping) are mapped to None.
    """
    # Unknown emotions should map to None
    fine_emotions = ["elation", "boredom", "confusionxyz"]
    result = emotion_mapping(fine_emotions)

    assert all(x is None for x in result)


def test_emotion_mapping_with_nan_and_empty():
    """
    Test that None, empty strings, whitespace-only strings, and NaN values
    are mapped to None.
    """
    fine_emotions = ["joy", None, "", "  ", float("nan"), "fear"]
    expected_main = ["happiness", None, None, None, None, "fear"]

    result = emotion_mapping(fine_emotions)

    assert list(result) == expected_main


def test_emotion_mapping_series_input():
    """
    Test that emotion_mapping works correctly when input is a pandas Series
    and returns a Series as output.
    """
    # Should also work if input is a pandas Series
    fine_emotions = pd.Series(["love", "anger", "surprise"])
    expected_main = ["happiness", "anger", "surprise"]

    result = emotion_mapping(fine_emotions)

    assert isinstance(result, pd.Series)
    assert list(result) == expected_main
