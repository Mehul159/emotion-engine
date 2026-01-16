"""
VAD Space and Timeline Visualization
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def plot_vad_2d(vad_history, save_path):
    vad_history = np.array(vad_history)
    plt.figure(figsize=(7, 6))
    plt.scatter(vad_history[:,0], vad_history[:,1], c=range(len(vad_history)), cmap='viridis', s=40)
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title('VAD Space (2D)')
    plt.colorbar(label='Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_vad_3d(vad_history, save_path):
    vad_history = np.array(vad_history)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(vad_history[:,0], vad_history[:,1], vad_history[:,2], c=range(len(vad_history)), cmap='viridis', s=40)
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    ax.set_title('VAD Space (3D)')
    fig.colorbar(p, label='Time')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_timeline(emotion_history, save_path):
    # ...existing code...
    df = pd.DataFrame(emotion_history)
    plt.figure(figsize=(10, 5))
    df.plot(ax=plt.gca())
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.title('Emotion Timeline')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
