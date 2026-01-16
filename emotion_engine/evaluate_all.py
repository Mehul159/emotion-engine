"""
Evaluate both Classic ML and DistilBERT models on a labeled test set.
Prints accuracy, recall, precision, F1-score for each class.
Also plots a bar chart comparing F1-scores for each model.
"""
import pandas as pd
from sklearn.metrics import classification_report
from inference.model_training import evaluate_model
from inference.text_inference import DistilBERTEmotionInference, TextEmotionInference
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "data/synthetic_emotions.csv"
MODEL_PATH = "models/emotion_model.pkl"

print("[1] Classic ML Model Evaluation:")
def get_classic_metrics():
    df = pd.read_csv(DATA_PATH)
    emotions = [col for col in df.columns if col != 'text']
    X = df['text']
    Y = df[emotions]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    vectorizer.fit(X_train)
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf = MultiOutputClassifier(LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1))
    clf.fit(X_train_vec, Y_train)
    preds = clf.predict(X_test_vec)
    report = classification_report(Y_test, preds, target_names=emotions, output_dict=True)
    print(classification_report(Y_test, preds, target_names=emotions))
    metrics = {label: vals for label, vals in report.items() if label in emotions}
    return metrics

classic_metrics = get_classic_metrics()

print("\n[2] DistilBERT Model Evaluation (sample of 100):")
df = pd.read_csv(DATA_PATH)
test_texts = df['text'].tolist()
test_labels = df.drop('text', axis=1).idxmax(axis=1).tolist()
infer = DistilBERTEmotionInference()
y_true = []
y_pred = []
SAMPLE_SIZE = 100
for text, true_label in zip(test_texts[:SAMPLE_SIZE], test_labels[:SAMPLE_SIZE]):
    emotion_scores, vad, confidence, reason = infer.infer(text)
    top_emotion = max(emotion_scores, key=emotion_scores.get)
    y_true.append(true_label.lower())
    y_pred.append(top_emotion)
from sklearn.metrics import classification_report as cr
report = cr(y_true, y_pred, output_dict=True)
print(cr(y_true, y_pred))

def plot_comparison(classic_metrics, distilbert_metrics, metric='f1-score'):
    labels = sorted(set(classic_metrics.keys()) & set(distilbert_metrics.keys()))
    classic_vals = [classic_metrics[l][metric] for l in labels]
    distilbert_vals = [distilbert_metrics[l][metric] for l in labels]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, classic_vals, width, label='Classic ML')
    plt.bar(x + width/2, distilbert_vals, width, label='DistilBERT')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Emotion Label')
    plt.title(f'Comparison of {metric.capitalize()} by Model')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'outputs/model_comparison_{metric}.png')
    plt.show()

distilbert_metrics = {k: v for k, v in report.items() if k in classic_metrics}
plot_comparison(classic_metrics, distilbert_metrics, metric='f1-score')
plot_comparison(classic_metrics, distilbert_metrics, metric='precision')
plot_comparison(classic_metrics, distilbert_metrics, metric='recall')
