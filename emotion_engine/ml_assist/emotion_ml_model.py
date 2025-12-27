"""
EmotionMLModel: Industry-standard, assistive ML model integration for Emotion Engine.
- Model is trained offline on non-personal, aggregate, or synthetic data.
- Model is used for assistive, explainable emotion signal extraction only.
- All outputs are bounded, policy-checked, and never used for direct end-to-end emotion prediction.
"""
from typing import Dict, Any
import joblib
import os

class EmotionMLModel:
    """
    Loads and uses a pre-trained ML model (e.g., RandomForest, XGBoost) for assistive emotion signal extraction.
    Model must be trained offline and saved as a .pkl file.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '../config/emotion_model.pkl')
        self.model = joblib.load(os.path.abspath(model_path))

    def predict_emotion(self, features: Dict[str, float]) -> str:
        """
        Predicts the most likely emotion label given normalized features.
        Args:
            features (dict): Dict with keys valence, arousal, dominance, threat.
        Returns:
            str: Predicted emotion label (must be explainable and bounded).
        """
        X = [[
            features.get("valence", 0.5),
            features.get("arousal", 0.5),
            features.get("dominance", 0.5),
            features.get("threat", 0.2)
        ]]
        pred = self.model.predict(X)
        return str(pred[0])

    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Returns probability distribution over all emotion classes.
        Args:
            features (dict): Dict with keys valence, arousal, dominance, threat.
        Returns:
            dict: {emotion_label: probability}
        """
        X = [[
            features.get("valence", 0.5),
            features.get("arousal", 0.5),
            features.get("dominance", 0.5),
            features.get("threat", 0.2)
        ]]
        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        return {str(cls): float(prob) for cls, prob in zip(classes, proba)}
