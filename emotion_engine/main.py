"""
Main entry point for the Emotion Engine backend (no frontend).
"""

from ingestion.signal_ingestion import SignalIngestion
from core.feature_normalizer import FeatureNormalizer
from appraisals.cognitive_appraisal_engine import CognitiveAppraisalEngine
from ml_assist.ml_assist import MLAssist
from core.emotion_state_manager import EmotionStateManager
from modulation.personality_modulator import PersonalityModulator
from policy.policy_guardrails import PolicyGuardrails
from mapping.emotion_intent_mapper import EmotionIntentMapper
from feedback.feedback_loop import FeedbackLoop

# Example: CLI or windowed interface for demonstration
if __name__ == "__main__":
    print("Emotion Engine backend initialized. Ready for input.")
    # Example input (replace with actual window/CLI logic)
    text = input("Enter text: ")
    metadata = {"user_id": "anon-123", "event_type": "demo", "timestamp": "2025-12-27T12:34:56Z"}
    context = {"channel": "cli", "priority": "normal"}

    ingestion = SignalIngestion()
    features = FeatureNormalizer().normalize(ingestion.ingest(text, metadata, context))
    appraisal = CognitiveAppraisalEngine().appraise(features, context)

    # Prepare input for MLAssist
    system_state = {
        "previous_emotions": [appraisal["emotion"]],
        "emotion_intensity": appraisal["intensity"]
    }
    mlassist_input = {
        "user_text": text,
        "context": context,
        "system_state": system_state
    }
    mlassist_result = MLAssist().analyze(mlassist_input)

    # Use MLAssist output for further processing if needed
    state = EmotionStateManager().update_state(
        appraisal["emotion"],
        appraisal["intensity"],
        0
    )
    modulated = PersonalityModulator().modulate(appraisal["emotion"], appraisal["intensity"], {})
    policy = PolicyGuardrails().enforce(appraisal["emotion"], modulated, state)
    mapping = EmotionIntentMapper().map(policy["emotion"], policy["intensity"], context)
    FeedbackLoop().record("demo-audit-id", mapping)
    print(f"Output: {mapping}\nExplanation: {appraisal['explanation']}\nMLAssist: {mlassist_result}")
