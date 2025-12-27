"""
Policy & Guardrails Layer: Enforces limits, forbidden combos, ethical checks.
"""

class PolicyGuardrails:
    def enforce(self, emotion: str, intensity: float, state: dict) -> dict:
        """
        Apply policy checks, cap intensity, block forbidden combos.
        """
        # Cap intensity
        safe_intensity = min(intensity, 0.9)
        # Forbidden combo: joy + threat
        if emotion == "joy" and state.get("threat", 0) > 0.5:
            return {"emotion": "neutral", "intensity": 0.5, "policy_status": "neutralized"}
        return {"emotion": emotion, "intensity": safe_intensity, "policy_status": "compliant"}
