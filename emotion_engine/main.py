"""
Emotion Engine CLI Entry Point (User-selectable: DistilBERT or Classic ML)
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
import numpy as np
from visualization.confusion_matrix import plot_confusion_matrix_img
from sklearn.metrics import confusion_matrix
from inference.text_inference import DistilBERTEmotionInference, TextEmotionInference

def main():
    print("Emotion Engine CLI - Windows Terminal Edition")
    print("Choose inference engine:")
    print("  1. DistilBERT (transformer, SOTA)")
    print("  2. Classic ML (TF-IDF + LogisticRegression)")
    engine = None
    while engine not in ("1", "2"):
        engine = input("Enter 1 or 2: ").strip()
    use_distilbert = (engine == "1")
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    log_dir = config.get('log_dir', 'logs')
    output_dir = config.get('output_dir', 'outputs')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(log_dir)
    state = EmotionState(config)
    if use_distilbert:
        infer = DistilBERTEmotionInference()
        emotions = infer.emotions
        print("[DistilBERT] Loaded.")
    else:
        model_path = os.path.join(os.path.dirname(__file__), '../models/emotion_model.pkl')
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Please train the model first.")
            sys.exit(1)
        infer = TextEmotionInference(model_path)
        emotions = infer.emotions
        print("[Classic ML] Loaded.")
    print("Type text to analyze emotions. Type 'exit' to quit.\n")
    y_true = []
    y_pred = []
    while True:
        text = input("Input: ").strip()
        if text.lower() in ("exit", "quit"): break
        emotion_scores, vad, confidence, reason = infer.infer(text)
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
        logger.info(json.dumps(result))
        out_file = os.path.join(output_dir, f"result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json")
        save_json(result, out_file)
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        print(f"Emotion: {top_emotion}")
        print(f"Confidence: {confidence:.2f}")
        print(f"VAD: ({vad[0]:+.2f}, {vad[1]:+.2f}, {vad[2]:+.2f})")
        print(f"Reason: {reason}")
        print(f"Stability: {state.stability:.2f}\n")
        print(f"Available emotion labels: {', '.join(emotions)}")
        true_label = input("Enter the TRUE emotion label for this input (or leave blank to skip): ").strip().lower()
        if true_label in emotions:
            y_true.append(true_label)
            y_pred.append(top_emotion)
        else:
            print("Skipped adding to confusion matrix (invalid or blank label).\n")
        if len(y_true) > 1:
            cm = confusion_matrix(y_true, y_pred, labels=emotions)
            plot_confusion_matrix_img(cm, emotions)

if __name__ == "__main__":
    main()
