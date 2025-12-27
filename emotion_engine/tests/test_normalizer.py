import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.feature_normalizer import FeatureNormalizer

class TestFeatureNormalizer(unittest.TestCase):
    def test_normalize_output_range(self):
        normalizer = FeatureNormalizer()
        features = normalizer.normalize({"text": "test", "metadata": {}, "context": {}})
        for k in ["valence", "arousal", "dominance", "confidence", "threat"]:
            self.assertGreaterEqual(features[k], 0.0)
            self.assertLessEqual(features[k], 1.0)

if __name__ == "__main__":
    unittest.main()
