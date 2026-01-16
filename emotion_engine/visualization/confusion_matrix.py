"""
Confusion Matrix Visualization
"""

def plot_confusion_matrix(cm, labels):
    """
    Print a confusion matrix in a readable text format in the terminal.
    Args:
        cm (array-like): Confusion matrix (2D array)
        labels (list): List of class labels
    """
    # Print header
    header = "{:>10} ".format("") + " ".join([f"{l:>8}" for l in labels])
    print(header)
    for i, row in enumerate(cm):
        row_str = "{:>10} ".format(labels[i]) + " ".join([f"{v:8d}" for v in row])
        print(row_str)
    print("\n")

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix_img(cm, labels):
    """
    Display a confusion matrix as an image (matplotlib window).
    Args:
        cm (array-like): Confusion matrix (2D array)
        labels (list): List of class labels
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import numpy as np
    # Demo: 5-class confusion matrix
    labels = ['HAPPY', 'SAD', 'ANGRY', 'FEAR', 'SURPRISE']
    cm = np.array([
        [50, 2, 1, 0, 0],
        [3, 45, 2, 1, 0],
        [0, 2, 48, 0, 0],
        [0, 1, 0, 47, 2],
        [0, 0, 1, 2, 49]
    ])
    plot_confusion_matrix(cm, labels)
    plot_confusion_matrix_img(cm, labels)
