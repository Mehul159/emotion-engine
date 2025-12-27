"""
Cognitive Appraisal Engine: Applies rule-based logic to normalized features.
"""

import json
import os

class CognitiveAppraisalEngine:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), '../config/emotion_values.json')
        with open(os.path.abspath(config_path), 'r') as f:
            self.emotion_values = json.load(f)

    def appraise(self, features: dict, context: dict) -> dict:
        """
        Apply human-readable rules to features, output emotion + explanation.
        Uses emotion_values.json for emotion lookup.
        """
        # Find the closest matching emotion by Euclidean distance in feature space
        min_dist = float('inf')
        best_emotion = None
        for entry in self.emotion_values:
            dist = (
                (features["valence"] - entry["valence"]) ** 2 +
                (features["arousal"] - entry["arousal"]) ** 2 +
                (features["dominance"] - entry["dominance"]) ** 2 +
                (features["threat"] - entry["threat"]) ** 2
            ) ** 0.5
            if dist < min_dist:
                min_dist = dist
                best_emotion = entry["emotion"]
        explanation = f"Closest match in emotion_values.json: {best_emotion} (distance={min_dist:.2f})"
        return {
            "emotion": best_emotion,
            "intensity": max(features.get("arousal", 0.5), 0.1),
            "explanation": explanation
        }
