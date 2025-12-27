"""
Emotion State Manager: Maintains persistent, blended, decaying emotion state.
"""
import math

class EmotionStateManager:
    def __init__(self):
        self.current_state = {"emotion": "neutral", "intensity": 0.5, "timestamp": 0}

    def update_state(self, new_emotion: str, new_intensity: float, timestamp: float):
        """
        Blend new emotion, apply decay, prevent abrupt jumps.
        """
        prev = self.current_state
        # Exponential decay
        decay_lambda = 0.1
        dt = timestamp - prev["timestamp"] if timestamp > prev["timestamp"] else 1
        decayed = prev["intensity"] * math.exp(-decay_lambda * dt)
        # Blending
        alpha = 0.3
        blended = alpha * new_intensity + (1 - alpha) * decayed
        self.current_state = {"emotion": new_emotion, "intensity": blended, "timestamp": timestamp}
        return self.current_state
