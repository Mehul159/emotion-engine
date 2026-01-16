"""
Emotion Reasoning Engine
Trigger detection, cause explanation, duration estimation, behavioral implication
"""
import numpy as np

def reason_emotion(text, emotions, vad):
    """
    Generate a human-readable explanation for the current emotion state.
    Args:
        text (str): Input text
        emotions (dict): {emotion: score}
        vad (list): [valence, arousal, dominance]
    Returns:
        str: Explanation string
    """
    if not emotions:
        return "No emotions detected."
    top_emotion = max(emotions, key=emotions.get)
    intensity = emotions[top_emotion]
    # Trigger detection (simple keyword)
    trigger = None
    keywords = {
        'HAPPY': ['happy', 'joy', 'pleased', 'delighted'],
        'SAD': ['sad', 'down', 'unhappy', 'depressed'],
        'ANGRY': ['angry', 'mad', 'furious', 'irritated'],
        'FEAR': ['afraid', 'scared', 'fear', 'terrified'],
        'SURPRISE': ['surprised', 'amazed', 'astonished'],
        'DISGUST': ['disgust', 'gross', 'nausea'],
        # Add more as needed
    }
    for emo, words in keywords.items():
        if any(w in text.lower() for w in words):
            trigger = emo
            break
    # Cause explanation
    cause = f"Detected emotion '{top_emotion}' with intensity {intensity:.2f}."
    if trigger:
        cause += f" Triggered by keywords related to '{trigger}'."
    # Duration estimation (simple rule)
    if intensity > 0.8:
        duration = "likely to persist for a while"
    elif intensity > 0.5:
        duration = "may fade soon"
    else:
        duration = "likely to be brief"
    # Behavioral implication
    if vad[0] < -0.3:
        behavior = "May withdraw or seek comfort."
    elif vad[0] > 0.3:
        behavior = "May express positivity or engage socially."
    elif vad[1] > 0.5:
        behavior = "May act impulsively or energetically."
    else:
        behavior = "Likely to remain calm."
    explanation = f"{cause} Duration: {duration}. {behavior}"
    return explanation
