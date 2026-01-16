"""
Demo: Run Hugging Face Transformers text classification (DistilBERT) on Windows
"""
from transformers import pipeline

if __name__ == "__main__":
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    text = "I love using Hugging Face Transformers!"
    result = classifier(text)
    print(f"Input: {text}\nResult: {result}")
