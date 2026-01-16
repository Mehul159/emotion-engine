"""
Emotion Memory Module
Handles short-term and long-term emotion memory, persistence
"""
import json
import os


class EmotionMemory:
    def __init__(self, short_term_limit=100):
        self.short_term = []  # List of recent (text, emotions)
        self.long_term = {}   # {emotion: count}
        self.short_term_limit = short_term_limit

    def store(self, text, emotions):
        # Store in short-term memory
        self.short_term.append({'text': text, 'emotions': emotions})
        if len(self.short_term) > self.short_term_limit:
            self.short_term.pop(0)
        # Update long-term emotion counts
        for emo in emotions:
            self.long_term[emo] = self.long_term.get(emo, 0) + 1

    def save(self, path):
        mem = {
            'short_term': self.short_term,
            'long_term': self.long_term
        }
        with open(path, 'w') as f:
            json.dump(mem, f, indent=2)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                mem = json.load(f)
            self.short_term = mem.get('short_term', [])
            self.long_term = mem.get('long_term', {})
