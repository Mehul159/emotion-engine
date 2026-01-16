import unittest
from inference.text_inference import TextEmotionInference

class TestTextEmotionInference(unittest.TestCase):
    def setUp(self):
        self.infer = TextEmotionInference()
    def test_infer(self):
        emotions, vad, conf, reason = self.infer.infer("I am happy.")
        # TODO: Add assertions for output
        pass
if __name__ == "__main__":
    unittest.main()
