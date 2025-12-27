"""
Feature Normalization Layer: Converts raw input to normalized features.
"""


import json
import os
from transformers import pipeline
from ml_assist.mistral_feature_extractor import MistralFeatureExtractor

class FeatureNormalizer:
    def __init__(self):
        # Load emotion values from config file
        config_path = os.path.join(os.path.dirname(__file__), '../config/emotion_values.json')
        with open(os.path.abspath(config_path), 'r') as f:
            self.emotion_values = json.load(f)
        # HuggingFace sentiment pipeline (can be swapped for emotion model)
        self.sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        # Mistral LLM feature extractor
        self.mistral_extractor = MistralFeatureExtractor()

    def normalize(self, input_data: dict) -> dict:
        """
        Map input to valence, arousal, dominance, confidence, threat (0-1) using HuggingFace sentiment as assistive signal, bounded by emotion_values.json.
        """
        text = input_data.get("text", "").lower()
        valence = 0.5
        arousal = 0.5
        dominance = 0.5
        confidence = 0.8
        threat = 0.2


        # Use HuggingFace sentiment as assistive feature
        if text.strip():
            result = self.sentiment_pipeline(text[:512])[0]
            if result['label'] == 'POSITIVE':
                valence = min(max(result['score'], 0.0), 1.0)
                arousal = 0.5 + 0.2 * result['score']  # Slightly higher arousal for strong positive
            elif result['label'] == 'NEGATIVE':
                valence = 1.0 - min(max(result['score'], 0.0), 1.0)
                arousal = 0.5 + 0.3 * result['score']  # Higher arousal for strong negative
                threat = 0.5 + 0.4 * result['score']

        # Use Mistral LLM token count as an assistive feature (e.g., longer input = higher arousal)
        token_count = self.mistral_extractor.get_token_count(text)
        if token_count > 50:
            arousal = min(arousal + 0.1, 1.0)
            confidence = min(confidence + 0.05, 1.0)

        # Use emotion_values.json for keyword mapping (overrides sentiment if match found)
        for entry in self.emotion_values:
            if entry["emotion"] in text:
                valence = entry["valence"]
                arousal = entry["arousal"]
                dominance = entry["dominance"]
                threat = entry["threat"]
                confidence = 0.9
                break

        # Bound all values to [0,1]
        valence = max(0.0, min(valence, 1.0))
        arousal = max(0.0, min(arousal, 1.0))
        dominance = max(0.0, min(dominance, 1.0))
        confidence = max(0.0, min(confidence, 1.0))
        threat = max(0.0, min(threat, 1.0))

        return {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "confidence": confidence,
            "threat": threat
        }
