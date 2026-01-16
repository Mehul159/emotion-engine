"""
Emotion Dynamics Engine
Handles force/spring-based transitions, decay, reinforcement, suppression
"""
import numpy as np

def apply_dynamics(state, vad_input, emotions_input, config):
    # Spring/force-based VAD update
    vad_input = np.array(vad_input, dtype=float)
    vad_force = vad_input - state.vad
    inertia = config.get('inertia', 0.85)
    decay = config.get('decay_rate', 0.05)
    state.momentum = inertia * state.momentum + (1 - inertia) * vad_force
    new_vad = state.vad + state.momentum - decay * (state.vad - state.vad_base)
    # Blend emotions with decay and reinforcement
    new_emotions = state.active_emotions.copy()
    for emo in list(new_emotions.keys()):
        new_emotions[emo] *= (1 - decay)
        if new_emotions[emo] < 0.01:
            del new_emotions[emo]
    for emo, w in emotions_input.items():
        new_emotions[emo] = max(new_emotions.get(emo, 0.0), w)
    return new_vad, new_emotions
