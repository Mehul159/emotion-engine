import json
from transformers import pipeline

from typing import Dict, Any, List

class MLAssist:
    """
    Assistive, deterministic, and explainable emotion signal analyzer.
    Provides bounded, machine-readable outputs for downstream rule-based appraisal.
    """
    def analyze(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze input and extract conservative emotion signals.
        Args:
            input_json (dict): Input with user_text, context, and system_state.
        Returns:
            dict: Bounded, explainable emotion signal output.
        """
        user_text: str = input_json.get("user_text", "")
        context: Dict[str, Any] = input_json.get("context", {})
        system_state: Dict[str, Any] = input_json.get("system_state", {})

        # Conservative defaults
        valence: float = 0.5
        arousal: float = 0.3
        dominance: float = 0.5
        confidence: float = 0.4
        threat: float = 0.2
        candidate_emotions: List[Dict[str, Any]] = [
            {"emotion": "neutral", "intensity": 0.4}
        ]
        explanation: str = (
            "Input is missing or ambiguous; defaulted to neutral valence and low arousal. "
            "Confidence reduced due to lack of clear signals. No strong contextual or textual evidence for elevated emotion."
        )
        uncertainty_notes: List[str] = [
            "Input data incomplete or unclear.",
            "Assumed neutral state due to insufficient evidence.",
            "No strong indicators of threat, arousal, or dominance."
        ]

        # Adjust based on context
        if context:
            if context.get("goal_blocked"):
                valence = 0.3
                arousal = 0.5
                threat = 0.5
                candidate_emotions = [{"emotion": "frustration", "intensity": 0.5}]
                explanation = "Goal blocked in context; valence and threat reduced, arousal moderate."
                confidence = 0.5
                uncertainty_notes.append("Assumed frustration due to goal blockage.")
            if context.get("failure_count", 0) > 2:
                arousal = min(arousal + 0.1, 1.0)
                threat = min(threat + 0.1, 1.0)
                explanation += " Multiple failures detected; slight increase in arousal and threat."
                uncertainty_notes.append("Failure count > 2.")
            if context.get("time_pressure", 0.0) > 0.7:
                arousal = min(arousal + 0.1, 1.0)
                explanation += " High time pressure; arousal increased."
                uncertainty_notes.append("High time pressure.")
            if "control_level" in context:
                dominance = max(min(context["control_level"], 1.0), 0.0)
                explanation += " Control level set from context."

        # Clamp all values
        valence = max(0.0, min(valence, 1.0))
        arousal = max(0.0, min(arousal, 1.0))
        dominance = max(0.0, min(dominance, 1.0))
        confidence = max(0.0, min(confidence, 1.0))
        threat = max(0.0, min(threat, 1.0))
        candidate_emotions = candidate_emotions[:3]

        return {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "confidence": confidence,
            "threat": threat,
            "candidate_emotions": candidate_emotions,
            "explanation": explanation,
            "uncertainty_notes": uncertainty_notes
        }

# Optional: HuggingFace Transformers integration for assistive, explainable NLP tasks only.
# This is strictly for extracting features (e.g., sentiment, keywords) to support bounded, rule-based appraisal.
# No end-to-end emotion prediction or final decision-making is allowed.

class HFAnalyzer:
    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        self.sentiment_pipeline = pipeline('sentiment-analysis', model=model_name)

    def get_sentiment_score(self, text):
        """
        Returns sentiment score in [0.0, 1.0] (positive) or [0.0, 1.0] (negative),
        mapped to valence for the Emotion Engine.
        """
        if not text.strip():
            return 0.5  # Neutral if empty
        result = self.sentiment_pipeline(text[:512])[0]
        if result['label'] == 'POSITIVE':
            return min(max(result['score'], 0.0), 1.0)
        elif result['label'] == 'NEGATIVE':
            return 1.0 - min(max(result['score'], 0.0), 1.0)
        return 0.5
