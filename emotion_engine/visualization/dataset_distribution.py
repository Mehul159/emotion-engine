"""
Dataset Distribution Visualization
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_distribution(data, save_path):
    """
    Plot and save the distribution of emotion labels in the dataset.
    Args:
        data: Can be a pandas DataFrame or dict of label counts
        save_path: Path to save the PNG
    """
    if isinstance(data, dict):
        labels, counts = zip(*data.items())
        df = pd.DataFrame({'label': labels, 'count': counts})
    elif isinstance(data, pd.DataFrame):
        # Assume one-hot columns for each label
        label_cols = [col for col in data.columns if col != 'text']
        counts = data[label_cols].sum().sort_values(ascending=False)
        df = pd.DataFrame({'label': counts.index, 'count': counts.values})
    else:
        raise ValueError('data must be dict or DataFrame')
    plt.figure(figsize=(10, 5))
    sns.barplot(x='label', y='count', data=df, palette='viridis')
    plt.title('Dataset Emotion Label Distribution')
    plt.xlabel('Emotion Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
