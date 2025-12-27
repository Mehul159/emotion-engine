import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_assist.emotion_ml_model import EmotionMLModel

# Example features (replace with real input)
features = {"valence": 0.5, "arousal": 0.7, "dominance": 0.3, "threat": 0.6}

# Load model and predict
model = EmotionMLModel()
emotion = model.predict_emotion(features)
proba = model.predict_proba(features)

print("Predicted emotion:", emotion)
print("Probabilities:", proba)
