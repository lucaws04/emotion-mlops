import unittest
from unittest.mock import MagicMock, patch

from transformers import BertForSequenceClassification

from src.models.model_definitions import get_model


class TestGetModel(unittest.TestCase):
    """
    Unit test for the get_model function to ensure it returns a correctly initialized
    BertForSequenceClassification model with the specified number of labels.
    """

    @patch("src.models.model_definitions.BertForSequenceClassification.from_pretrained")
    def test_get_model_returns_bert_model(self, mock_from_pretrained):
        """
        Test that get_model correctly loads a pretrained BERT model with the
        specified number of labels and returns the model instance.
        """
        dummy_model = MagicMock(spec=BertForSequenceClassification)
        dummy_model.config = MagicMock()  # explicitly set config mock
        dummy_model.config.num_labels = 3  # set expected attribute

        mock_from_pretrained.return_value = dummy_model

        num_labels = 3
        model = get_model(num_labels)

        mock_from_pretrained.assert_called_once_with(
            "bert-base-uncased",
            num_labels=num_labels,
            problem_type="single_label_classification",
        )

        self.assertIs(model, dummy_model)
        self.assertEqual(model.config.num_labels, num_labels)
