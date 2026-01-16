"""
Text-based Emotion Inference (ML Pipeline)
"""
import numpy as np
import pickle
import os
from transformers import pipeline

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

# --- DistilBERT-based inference ---

class DistilBERTEmotionInference:
    def __init__(self, model_name="bhadresh-savani/distilbert-base-uncased-emotion"):
        self.classifier = pipeline("text-classification", model=model_name, return_all_scores=True)
        self.emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    def infer(self, text):
        preds = self.classifier(text)[0]
        emotion_scores = {p['label']: p['score'] for p in preds}
        # Map to VAD (simple mapping for demo)
        vad_map = {
            "joy": [0.8, 0.6, 0.7],
            "sadness": [-0.7, 0.3, -0.5],
            "anger": [-0.6, 0.8, 0.5],
            "fear": [-0.8, 0.9, -0.7],
            "love": [0.7, 0.5, 0.6],
            "surprise": [0.2, 0.9, 0.2],
        }
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        vad = vad_map.get(top_emotion, [0.0, 0.0, 0.0])
        confidence = emotion_scores[top_emotion]
        reason = f"DistilBERT: Top emotion {top_emotion} with confidence {confidence:.2f}"
        return emotion_scores, vad, confidence, reason
