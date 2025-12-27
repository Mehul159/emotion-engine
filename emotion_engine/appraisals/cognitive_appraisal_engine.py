"""
Cognitive Appraisal Engine: Applies rule-based logic to normalized features.
"""

class CognitiveAppraisalEngine:
    def appraise(self, features: dict, context: dict) -> dict:
        """
        Apply human-readable rules to features, output emotion + explanation.
        """
        # Example rule: Frustration if low valence, high arousal, low dominance
        if features["valence"] < 0.3 and features["arousal"] > 0.6 and features["dominance"] < 0.4:
            return {
                "emotion": "frustration",
                "intensity": 0.7,
                "explanation": "Negative valence, high arousal, low dominance. Rule: 'goal_blocked_low_control'"
            }
        # Default: Neutral
        return {
            "emotion": "neutral",
            "intensity": 0.5,
            "explanation": "No strong appraisal rule matched."
        }
