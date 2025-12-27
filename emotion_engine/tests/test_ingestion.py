import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingestion.signal_ingestion import SignalIngestion

class TestSignalIngestion(unittest.TestCase):
    def test_valid_input(self):
        ingestion = SignalIngestion()
        result = ingestion.ingest("test", {"meta":1}, {"ctx":2})
        self.assertEqual(result["text"], "test")
        self.assertIn("metadata", result)
        self.assertIn("context", result)

    def test_invalid_input(self):
        ingestion = SignalIngestion()
        with self.assertRaises(ValueError):
            ingestion.ingest("", {}, {})

if __name__ == "__main__":
    unittest.main()
