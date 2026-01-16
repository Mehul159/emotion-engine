"""
Emotion Engine CLI Entry Point
"""
import sys
from core import emotion_state, emotion_memory, config

import os
import sys
import json
import datetime
from core.config import load_config
from core.emotion_state import EmotionState
from utils.logger import setup_logger
from utils.file_io import save_json
import pickle
import numpy as np
from visualization.confusion_matrix import plot_confusion_matrix, plot_confusion_matrix_img
from sklearn.metrics import confusion_matrix

def load_model(model_path):
    with open(model_path, 'rb') as f:
        obj = pickle.load(f)
    return obj['vectorizer'], obj['model'], obj['emotions']

def infer_emotions(text, vectorizer, model, emotions):
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)
    # MultiOutputClassifier returns list of arrays
    scores = np.array([p[0][1] if p[0].shape[0]>1 else 0.0 for p in probs])
    emotion_scores = dict(zip(emotions, scores))
    # VAD mapping: dummy for now (could use a lookup table)
    vad = [float(scores.mean()), float(scores.std()), float(scores.max())]
    confidence = float(scores.max())
    reason = f"Top emotion: {emotions[np.argmax(scores)]}"
    return emotion_scores, vad, confidence, reason

def main():
    print("Emotion Engine CLI - Windows Terminal Edition")
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    log_dir = config.get('log_dir', 'logs')
    output_dir = config.get('output_dir', 'outputs')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(log_dir)
    state = EmotionState(config)
    model_path = os.path.join(os.path.dirname(__file__), '../models/emotion_model.pkl')
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        sys.exit(1)
    vectorizer, model, emotions = load_model(model_path)
    print("Type text to analyze emotions. Type 'exit' to quit.\n")
    # For confusion matrix
    y_true = []
    y_pred = []
    while True:
        text = input("Input: ").strip()
        if text.lower() in ("exit", "quit"): break
        emotion_scores, vad, confidence, reason = infer_emotions(text, vectorizer, model, emotions)
        state.update(vad, emotion_scores)
        result = {
            'timestamp': datetime.datetime.now().isoformat(),
            'text': text,
            'emotions': emotion_scores,
            'vad': vad,
            'confidence': confidence,
            'reason': reason,
            'stability': state.stability
        }
        # Logging
        logger.info(json.dumps(result))
        # Save output
        out_file = os.path.join(output_dir, f"result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json")
        save_json(result, out_file)
        # Print summary
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        print(f"Emotion: {top_emotion}")
        print(f"Confidence: {confidence:.2f}")
        print(f"VAD: ({vad[0]:+.2f}, {vad[1]:+.2f}, {vad[2]:+.2f})")
        print(f"Reason: {reason}")
        print(f"Stability: {state.stability:.2f}\n")
        # Ask user for true label
        print(f"Available emotion labels: {', '.join(emotions)}")
        true_label = input("Enter the TRUE emotion label for this input (or leave blank to skip): ").strip().upper()
        if true_label in emotions:
            y_true.append(true_label)
            y_pred.append(top_emotion)
        else:
            print("Skipped adding to confusion matrix (invalid or blank label).\n")
        # Show confusion matrix if at least 2 samples
        if len(y_true) > 1:
            cm = confusion_matrix(y_true, y_pred, labels=emotions)
            plot_confusion_matrix_img(cm, emotions)

if __name__ == "__main__":
    main()
