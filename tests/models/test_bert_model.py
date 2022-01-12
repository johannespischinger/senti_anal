from src.models.bert_model import SentimentClassifier


class TestSentimentClassifier:
    def test_loadmodel(self):
        model = SentimentClassifier()
        # test if no error when model loaded
        assert model.parameters()
