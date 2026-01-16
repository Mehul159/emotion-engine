"""
Run All: Orchestrates the full Emotion Engine pipeline in sequence.
- Trains model
- Evaluates model
- Visualizes dataset distribution
- Runs inference and reasoning
- Plots VAD and emotion timeline
"""
import os
import pandas as pd
from inference import model_training
from inference.text_inference import TextEmotionInference
from inference.explainability import explain_prediction
from reasoning.emotion_reasoning import reason_emotion
from visualization.dataset_distribution import plot_distribution
from visualization.vad_plot import plot_vad_2d, plot_vad_3d, plot_timeline

DATA_PATH = "data/synthetic_emotions.csv"
MODEL_PATH = "models/emotion_model.pkl"
VAD_PNG = "outputs/vad_2d.png"
VAD3D_PNG = "outputs/vad_3d.png"
TIMELINE_PNG = "outputs/emotion_timeline.png"
DIST_PNG = "outputs/dataset_distribution.png"

# 1. Train model
print("[1] Training model...")
model_training.train_model(DATA_PATH, MODEL_PATH)

# 2. Evaluate model
print("[2] Evaluating model...")
model_training.evaluate_model(MODEL_PATH, DATA_PATH)

# 3. Visualize dataset distribution
print("[3] Visualizing dataset distribution...")
df = pd.read_csv(DATA_PATH)
plot_distribution(df, DIST_PNG)
print(f"Saved: {DIST_PNG}")

# 4. Run inference and reasoning on a few samples
print("[4] Running inference and reasoning...")
infer = TextEmotionInference(MODEL_PATH)
vad_history = []
emotion_history = []
for i, row in df.head(20).iterrows():
    text = row['text']
    emotions, vad, conf, reason = infer.infer(text)
    vad_history.append(vad)
    emotion_history.append(emotions)
    explanation = reason_emotion(text, emotions, vad)
    print(f"Text: {text}\n  Emotions: {emotions}\n  VAD: {vad}\n  Reason: {reason}\n  Explanation: {explanation}\n")
    # Optionally, explain features
    # print("  Features:", explain_prediction(infer.model, infer.vectorizer, text))

# 5. Plot VAD and emotion timeline
print("[5] Plotting VAD and emotion timeline...")
plot_vad_2d(vad_history, VAD_PNG)
plot_vad_3d(vad_history, VAD3D_PNG)
plot_timeline(emotion_history, TIMELINE_PNG)
print(f"Saved: {VAD_PNG}, {VAD3D_PNG}, {TIMELINE_PNG}")

print("\nAll steps completed.")
