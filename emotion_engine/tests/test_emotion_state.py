import unittest
from core.emotion_state import EmotionState
from core.config import load_config

class TestEmotionState(unittest.TestCase):
    def setUp(self):
        self.config = load_config('config.yaml')
        self.state = EmotionState(self.config)
    def test_initial_vad(self):
        self.assertEqual(list(self.state.vad), self.config['vad_base'])
    def test_update(self):
        # TODO: Add update logic test
        pass
if __name__ == "__main__":
    unittest.main()
