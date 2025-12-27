import unittest
from ml_assist.mistral_feature_extractor import MistralFeatureExtractor

class TestMistralFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = MistralFeatureExtractor()

    def test_token_count_short(self):
        text = "Hello world!"
        count = self.extractor.get_token_count(text)
        self.assertTrue(isinstance(count, int))
        self.assertGreater(count, 0)

    def test_token_count_long(self):
        text = "This is a much longer message that should result in a higher token count. " * 5
        count = self.extractor.get_token_count(text)
        self.assertTrue(count > 10)

if __name__ == "__main__":
    unittest.main()
