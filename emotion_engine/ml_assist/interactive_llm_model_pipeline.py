"""
Interactive pipeline: HuggingFace LLM assistive inference, session-based model training, and evaluation.
- LLM output is used as ground truth for session model.
- Model is trained only on session data (no persistent or personal data).
- Console shows LLM output, model output, accuracy, and errors.
"""
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from .mistral_feature_extractor import MistralFeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

class HuggingFaceLLMWrapper:
    def __init__(self):
        # Multi-class emotion model from HuggingFace
        self.emotion_pipeline = pipeline(
            'text-classification',
            model='j-hartmann/emotion-english-distilroberta-base',
            top_k=None
        )
        # Get sorted emotion labels for consistent feature order
        self.labels = self._get_sorted_labels()

    def _get_sorted_labels(self):
        # Run a dummy inference to get all possible labels
        dummy = self.emotion_pipeline("Hello world")[0]
        return [d['label'] for d in sorted(dummy, key=lambda x: x['label'])]

    def infer(self, text):
        # Returns list of dicts: [{label, score}, ...]
        scores = self.emotion_pipeline(text[:512])[0]
        # Sort by label for consistency
        return sorted(scores, key=lambda x: x['label'])

def main():
    # Load sentence embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Load Mistral feature extractor
    mistral_extractor = MistralFeatureExtractor()
    model_path = os.path.join(os.path.dirname(__file__), '../config/session_emotion_model.pkl')
    X, y = [], []
    clf = None
    llm = HuggingFaceLLMWrapper()
    acc_history = []
    fig, axs = None, None
    plt.ion()
    while True:
        text = input("Enter text: ")
        if not text.strip():
            break
        llm_result = llm.infer(text)
        print("HuggingFace LLM output:", llm_result)
        # Feature vector: emotion scores, sentence embedding, token count, text length
        emotion_scores = [d['score'] for d in llm_result]
        embedding = embedder.encode(text, normalize_embeddings=True)
        token_count = mistral_extractor.get_token_count(text)
        features = list(emotion_scores) + list(embedding) + [token_count, len(text)]
        # Label: highest scoring emotion
        label = llm.labels[np.argmax(emotion_scores)]
        X.append(features)
        y.append(label)
        if len(X) >= 5:
            print("Training model on collected data...")
            clf = RandomForestClassifier()
            clf.fit(X, y)
            joblib.dump(clf, model_path)
        if clf:
            print("Try with model:")
            pred = clf.predict([features])[0]
            print("Model output:", pred)
            preds = clf.predict(X)
            acc = accuracy_score(y, preds)
            acc_history.append(acc)
            print(f"Model accuracy on session data: {acc:.2f}")
            errors = sum(1 for yp, yt in zip(preds, y) if yp != yt)
            print(f"Errors: {errors}")
            # Show confusion matrix and classification report
            cm = confusion_matrix(y, preds, labels=llm.labels)
            print("Confusion Matrix:")
            print(cm)
            print("Classification Report:")
            print(classification_report(y, preds, labels=llm.labels, zero_division=0))

            # --- Visualization ---
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
            # Add text labels
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axs[1].text(j, i, str(cm[i, j]), ha='center', va='center', color='red')
            plt.tight_layout()
            plt.pause(0.1)

if __name__ == '__main__':
    main()
