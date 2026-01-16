"""
Generate synthetic emotion data and train a multi-label classifier.
Saves data to data/synthetic_emotions.csv and model to models/emotion_model.pkl
"""
import os
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import pickle

# Define emotions (Ekman + Plutchik, subset for demo)
EMOTIONS = [
    'HAPPY', 'SAD', 'ANGRY', 'FEAR', 'SURPRISE', 'DISGUST',
    'TRUST', 'ANTICIPATION', 'JOY', 'LOVE', 'REMORSE', 'OPTIMISM',
    'AGGRESSIVE', 'CONTENT', 'BORED', 'SHAME', 'PRIDE', 'ENVY', 'GUILT', 'AWE'
]

# Simple templates for synthetic data
templates = [
    "I feel {emotion} today.",
    "This makes me so {emotion}...",
    "Why am I always so {emotion}?",
    "Such a {emotion} moment!",
    "{emotion} is all I know right now.",
    "Can't believe how {emotion} this is.",
    "{emotion} overwhelms me.",
    "It's a {emotion} kind of day.",
    "{emotion} fills my mind.",
    "{emotion} and nothing else."
]

# Generate synthetic dataset
def generate_data(n=300000, out_path="data/synthetic_emotions.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = []
    for _ in range(n):
        num_labels = random.choices([1,2,3], weights=[0.7,0.2,0.1])[0]
        chosen = random.sample(EMOTIONS, num_labels)
        text = random.choice(templates).format(emotion=chosen[0].lower())
        labels = [1 if e in chosen else 0 for e in EMOTIONS]
        data.append([text] + labels)
    columns = ['text'] + EMOTIONS
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(out_path, index=False)
    print(f"Synthetic data saved to {out_path}")
    return df

def train_model(data_path="data/synthetic_emotions.csv", model_path="models/emotion_model.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("Loading data for training...")
    df = pd.read_csv(data_path)
    X = df['text']
    Y = df[EMOTIONS]
    print(f"Total samples: {len(df)}")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    vectorizer = TfidfVectorizer(max_features=500)
    print("Fitting TF-IDF vectorizer...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("Training MultiOutputClassifier (LogisticRegression, saga, max_iter=1000)...")
    clf = MultiOutputClassifier(LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1, verbose=1))
    clf.fit(X_train_vec, Y_train)
    score = clf.score(X_test_vec, Y_test)
    print(f"Test accuracy (mean over labels): {score:.3f}")
    # Save model and vectorizer
    with open(model_path, "wb") as f:
        pickle.dump({'vectorizer': vectorizer, 'model': clf, 'emotions': EMOTIONS}, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # To generate a 300k+ sample dataset for advanced use, just run this script.
    # For smaller datasets, call generate_data(n=10000) etc.
    # For quick testing, use a smaller n (e.g., n=10000)
    df = generate_data(n=300000)
    train_model()
