"""
Batch pipeline: Reads emotion scenarios from a file, infers HuggingFace emotion labels, and trains the model automatically.
- Input: emotions_dataset.txt (one scenario per line)
- Output: Trains the session model using LLM labels as ground truth
"""
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from .mistral_feature_extractor import MistralFeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import joblib
import os

class HuggingFaceLLMWrapper:
    def __init__(self):
        self.emotion_pipeline = pipeline(
            'text-classification',
            model='j-hartmann/emotion-english-distilroberta-base',
            top_k=None
        )
        self.labels = self._get_sorted_labels()

    def _get_sorted_labels(self):
        dummy = self.emotion_pipeline("Hello world")[0]
        return [d['label'] for d in sorted(dummy, key=lambda x: x['label'])]

    def infer(self, text):
        scores = self.emotion_pipeline(text[:512])[0]
        return sorted(scores, key=lambda x: x['label'])

def batch_train_from_file(dataset_path, model_path):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    mistral_extractor = MistralFeatureExtractor()
    llm = HuggingFaceLLMWrapper()
    X, y = [], []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith('#'):
                continue
            llm_result = llm.infer(text)
            emotion_scores = [d['score'] for d in llm_result]
            embedding = embedder.encode(text, normalize_embeddings=True)
            token_count = mistral_extractor.get_token_count(text)
            features = list(emotion_scores) + list(embedding) + [token_count, len(text)]
            label = llm.labels[np.argmax(emotion_scores)]
            X.append(features)
            y.append(label)
    if len(X) >= 5:
        clf = RandomForestClassifier()
        clf.fit(X, y)
        joblib.dump(clf, model_path)
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds, labels=llm.labels)
        print(f"Model accuracy: {acc:.2f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y, preds, labels=llm.labels, zero_division=0))
    else:
        print("Not enough data to train.")

if __name__ == '__main__':
    dataset_path = os.path.join(os.path.dirname(__file__), 'emotions_dataset.txt')
    model_path = os.path.join(os.path.dirname(__file__), '../config/session_emotion_model.pkl')
    batch_train_from_file(dataset_path, model_path)
