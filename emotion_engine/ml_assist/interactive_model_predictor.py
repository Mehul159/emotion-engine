"""
Interactive predictor: Loads the trained model, takes user text input, predicts emotion, and shows visualizations.
"""
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from .mistral_feature_extractor import MistralFeatureExtractor
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt

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

def main():
    model_path = os.path.join(os.path.dirname(__file__), '../config/session_emotion_model.pkl')
    if not os.path.exists(model_path):
        print("Trained model not found. Please run the batch trainer first.")
        return
    clf = joblib.load(model_path)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    mistral_extractor = MistralFeatureExtractor()
    llm = HuggingFaceLLMWrapper()
    X, y, preds, acc_history = [], [], [], []
    fig, axs = None, None
    plt.ion()
    print("Enter text (blank line to exit):")
    while True:
        text = input("Enter text: ")
        if not text.strip():
            break
        llm_result = llm.emotion_pipeline(text[:512])[0]
        llm_result = sorted(llm_result, key=lambda x: x['label'])
        emotion_scores = [d['score'] for d in llm_result]
        embedding = embedder.encode(text, normalize_embeddings=True)
        token_count = mistral_extractor.get_token_count(text)
        features = list(emotion_scores) + list(embedding) + [token_count, len(text)]
        pred = clf.predict([features])[0]
        print(f"Predicted emotion: {pred}")
        X.append(features)
        y.append(llm.labels[np.argmax(emotion_scores)])
        preds.append(pred)
        acc = np.mean([p == t for p, t in zip(preds, y)])
        acc_history.append(acc)
        print(f"Session accuracy: {acc:.2f}")
        # Visualization
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y, preds, labels=llm.labels)
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y, preds, labels=llm.labels, zero_division=0))
        if fig is None or axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].clear()
        axs[0].plot(acc_history, marker='o')
        axs[0].set_title('Accuracy Over Time')
        axs[0].set_xlabel('Sample')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_ylim(0, 1.05)
        axs[1].clear()
        im = axs[1].imshow(cm, cmap='Blues')
        axs[1].set_title('Confusion Matrix')
        axs[1].set_xlabel('Predicted')
        axs[1].set_ylabel('True')
        axs[1].set_xticks(np.arange(len(llm.labels)))
        axs[1].set_yticks(np.arange(len(llm.labels)))
        axs[1].set_xticklabels(llm.labels, rotation=45, ha='right')
        axs[1].set_yticklabels(llm.labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axs[1].text(j, i, str(cm[i, j]), ha='center', va='center', color='red')
        plt.tight_layout()
        plt.pause(0.1)

if __name__ == '__main__':
    main()