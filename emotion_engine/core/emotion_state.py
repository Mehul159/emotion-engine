"""
Emotion State Core Module
Maintains VAD vector, active emotions, momentum, stability
"""
import numpy as np
import yaml
import json
from .emotion_memory import EmotionMemory

class EmotionState:
    def __init__(self, config):
        self.vad = np.array(config.get('vad_base', [0.0, 0.0, 0.0]), dtype=float)
        self.active_emotions = {}  # {emotion: weight}
        self.momentum = np.zeros(3)
        self.stability = 1.0
        self.memory = EmotionMemory()
        self.decay_rate = config.get('decay_rate', 0.05)
        self.inertia = config.get('inertia', 0.85)
        self.vad_base = np.array(config.get('vad_base', [0.0, 0.0, 0.0]), dtype=float)

    def update(self, vad_new, emotions_new):
        # Blend VAD with inertia and decay
        vad_new = np.array(vad_new, dtype=float)
        self.momentum = self.inertia * self.momentum + (1 - self.inertia) * (vad_new - self.vad)
        self.vad += self.momentum
        self.vad += -self.decay_rate * (self.vad - self.vad_base)
        # Update active emotions with decay
        for emo in list(self.active_emotions.keys()):
            self.active_emotions[emo] *= (1 - self.decay_rate)
            if self.active_emotions[emo] < 0.01:
                del self.active_emotions[emo]
        for emo, w in emotions_new.items():
            self.active_emotions[emo] = max(self.active_emotions.get(emo, 0.0), w)
        # Stability: 1 - norm of momentum
        self.stability = float(1.0 - np.linalg.norm(self.momentum) / (np.linalg.norm(self.vad) + 1e-6))

    def save(self, path):
        state = {
            'vad': self.vad.tolist(),
            'active_emotions': self.active_emotions,
            'momentum': self.momentum.tolist(),
            'stability': self.stability
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    @staticmethod
    def load_or_init(path=None, config=None):
        if path is not None:
            try:
                with open(path, 'r') as f:
                    state = json.load(f)
                obj = EmotionState(config)
                obj.vad = np.array(state['vad'], dtype=float)
                obj.active_emotions = state['active_emotions']
                obj.momentum = np.array(state['momentum'], dtype=float)
                obj.stability = state['stability']
                return obj
            except Exception:
                pass
        return EmotionState(config)
