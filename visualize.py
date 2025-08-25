from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(history: Dict, embeddings_data: Dict, metrics: Dict):
    """
    Visualize training results and evaluation metrics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    epochs = range(1, len(history['train_losses']) + 1)

    # Training loss
    axes[0, 0].plot(epochs, history['train_losses'], label='Training Loss', marker='o')
    axes[0, 0].plot(epochs, history['val_losses'], label='Validation Loss', marker='s')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Training accuracy
    axes[0, 1].plot(epochs, history['train_accuracies'], label='Training Accuracy', marker='o')
    axes[0, 1].plot(epochs, history['val_accuracies'], label='Validation Accuracy', marker='s')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # User embeddings distribution (first 2 dimensions)
    user_embeddings = embeddings_data['user_embeddings'][:1000]  # Sample for visualization
    axes[0, 2].scatter(user_embeddings[:, 0], user_embeddings[:, 1], alpha=0.6, s=10)
    axes[0, 2].set_title('User Embeddings (First 2D)')
    axes[0, 2].set_xlabel('Dimension 1')
    axes[0, 2].set_ylabel('Dimension 2')
    axes[0, 2].grid(True)

    # Item embeddings distribution (first 2 dimensions)
    item_embeddings = embeddings_data['item_embeddings'][:1000]  # Sample for visualization
    axes[1, 0].scatter(item_embeddings[:, 0], item_embeddings[:, 1], alpha=0.6, s=10, color='orange')
    axes[1, 0].set_title('Item Embeddings (First 2D)')
    axes[1, 0].set_xlabel('Dimension 1')
    axes[1, 0].set_ylabel('Dimension 2')
    axes[1, 0].grid(True)

    # Evaluation metrics
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    bars = axes[1, 1].bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'salmon'])
    axes[1, 1].set_title('Evaluation Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, v in zip(bars, metric_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{v:.3f}', ha='center', va='bottom')

    # Embedding similarity heatmap
    user_sample = user_embeddings[:50]
    item_sample = item_embeddings[:50]
    similarity_matrix = np.dot(user_sample, item_sample.T)

    im = axes[1, 2].imshow(similarity_matrix, cmap='coolwarm', aspect='auto')
    axes[1, 2].set_title('User-Item Similarity Heatmap (Sample)')
    axes[1, 2].set_xlabel('Items')
    axes[1, 2].set_ylabel('Users')
    plt.colorbar(im, ax=axes[1, 2])

    plt.tight_layout()
    plt.show()