"""
Text-based Emotion Inference (ML Pipeline)
"""
import numpy as np
import pickle
import os

class TextEmotionInference:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/emotion_model.pkl'))
        with open(model_path, 'rb') as f:
            obj = pickle.load(f)
        self.vectorizer = obj['vectorizer']
        self.model = obj['model']
        self.emotions = obj['emotions']

    def infer(self, text):
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)
        scores = np.array([p[0][1] if p[0].shape[0]>1 else 0.0 for p in probs])
        emotion_scores = dict(zip(self.emotions, scores))
        vad = [float(scores.mean()), float(scores.std()), float(scores.max())]
        confidence = float(scores.max())
        reason = f"Top emotion: {self.emotions[np.argmax(scores)]}"
        return emotion_scores, vad, confidence, reason
