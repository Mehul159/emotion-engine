"""
Model Training & Evaluation
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_model(data_path, model_save_path):
    df = pd.read_csv(data_path)
    emotions = [col for col in df.columns if col != 'text']
    X = df['text']
    Y = df[emotions]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf = MultiOutputClassifier(LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1))
    clf.fit(X_train_vec, Y_train)
    # Save model and vectorizer
    with open(model_save_path, 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'model': clf, 'emotions': emotions}, f)
    print(f"Model saved to {model_save_path}")
    # Evaluate
    preds = clf.predict(X_test_vec)
    print(classification_report(Y_test, preds, target_names=emotions))
    print(f"Mean accuracy: {accuracy_score(Y_test, preds):.3f}")

def evaluate_model(model_path, test_data_path):
    with open(model_path, 'rb') as f:
        obj = pickle.load(f)
    vectorizer = obj['vectorizer']
    model = obj['model']
    emotions = obj['emotions']
    df = pd.read_csv(test_data_path)
    X = df['text']
    Y = df[emotions]
    X_vec = vectorizer.transform(X)
    preds = model.predict(X_vec)
    print(classification_report(Y, preds, target_names=emotions))
    print(f"Mean accuracy: {accuracy_score(Y, preds):.3f}")
