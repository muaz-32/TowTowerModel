from datetime import datetime
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
plot_saving_path = os.getenv("PLOT_SAVING_PATH")
timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_saving_path = os.path.join(plot_saving_path, timestamp_dir)
os.makedirs(plot_saving_path, exist_ok=True)


def create_unique_filename(base_name: str) -> str:
    """Generate unique filename with timestamp for each run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.png"


def save_plot(fig, filename: str):
    """Save plot with unique naming."""
    full_path = os.path.join(plot_saving_path, create_unique_filename(filename))
    fig.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {full_path}")
    plt.close(fig)


def plot_training_loss(history: Dict):
    """Plot training and validation loss over epochs."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(history['train_losses']) + 1)

    ax.plot(epochs, history['train_losses'], label='Training Loss', marker='o', linewidth=2)
    ax.plot(epochs, history['val_losses'], label='Validation Loss', marker='s', linewidth=2)
    ax.set_title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.5)

    # Add value annotations on last epoch
    ax.annotate(f'{history["train_losses"][-1]:.4f}',
                xy=(epochs[-1], history['train_losses'][-1]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax.annotate(f'{history["val_losses"][-1]:.4f}',
                xy=(epochs[-1], history['val_losses'][-1]),
                xytext=(5, -15), textcoords='offset points', fontsize=9)

    save_plot(fig, "training_loss")


def plot_training_accuracy(history: Dict):
    """Plot training and validation accuracy over epochs."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(history['train_accuracies']) + 1)

    ax.plot(epochs, history['train_accuracies'], label='Training Accuracy', marker='o', linewidth=2)
    ax.plot(epochs, history['val_accuracies'], label='Validation Accuracy', marker='s', linewidth=2)
    ax.set_title('Training and Validation Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.5)
    ax.set_ylim(0, 100)

    # Add value annotations on last epoch
    ax.annotate(f'{history["train_accuracies"][-1]:.2f}%',
                xy=(epochs[-1], history['train_accuracies'][-1]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax.annotate(f'{history["val_accuracies"][-1]:.2f}%',
                xy=(epochs[-1], history['val_accuracies'][-1]),
                xytext=(5, -15), textcoords='offset points', fontsize=9)

    save_plot(fig, "training_accuracy")


def plot_metrics_comparison(metrics_k5: Dict, metrics_k10: Dict, metrics_k5_cs: Dict, metrics_k10_cs: Dict):
    """Plot evaluation metrics comparison for k=5 and k=10, with and without candidate sampling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Regular evaluation metrics
    metric_names = ['Precision', 'Recall', 'NDCG', 'F1']
    k5_values = [metrics_k5.get('precision', 0), metrics_k5.get('recall', 0),
                 metrics_k5.get('ndcg', 0), metrics_k5.get('f1', 0)]
    k10_values = [metrics_k10.get('precision', 0), metrics_k10.get('recall', 0),
                  metrics_k10.get('ndcg', 0), metrics_k10.get('f1', 0)]

    x = np.arange(len(metric_names))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, k5_values, width, label='k=5', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, k10_values, width, label='k=10', color='lightcoral', alpha=0.8)

    ax1.set_title('Evaluation Metrics Comparison (Regular Evaluation)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Metrics', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Add value labels on bars
    for bar, value in zip(bars1, k5_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    for bar, value in zip(bars2, k10_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    # Candidate sampling evaluation metrics
    k5_cs_values = [metrics_k5_cs.get('precision', 0), metrics_k5_cs.get('recall', 0),
                    metrics_k5_cs.get('ndcg', 0), metrics_k5_cs.get('f1', 0)]
    k10_cs_values = [metrics_k10_cs.get('precision', 0), metrics_k10_cs.get('recall', 0),
                     metrics_k10_cs.get('ndcg', 0), metrics_k10_cs.get('f1', 0)]

    bars3 = ax2.bar(x - width / 2, k5_cs_values, width, label='k=5', color='lightgreen', alpha=0.8)
    bars4 = ax2.bar(x + width / 2, k10_cs_values, width, label='k=10', color='orange', alpha=0.8)

    ax2.set_title('Evaluation Metrics Comparison (Candidate Sampling)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Metrics', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Add value labels on bars
    for bar, value in zip(bars3, k5_cs_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    for bar, value in zip(bars4, k10_cs_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_plot(fig, "metrics_comparison")


def plot_embeddings_pca(embeddings_data: Dict):
    """Plot PCA visualization of user and item embeddings."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    user_embeddings = embeddings_data['user_embeddings']
    item_embeddings = embeddings_data['item_embeddings']

    # User embeddings PCA
    pca_user = PCA(n_components=2)
    user_pca = pca_user.fit_transform(user_embeddings)

    scatter1 = ax1.scatter(user_pca[:, 0], user_pca[:, 1], alpha=0.6, s=20, c='blue')
    ax1.set_title(f'User Embeddings PCA\n({len(user_embeddings)} users, {user_embeddings.shape[1]}D → 2D)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca_user.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca_user.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Item embeddings PCA
    pca_item = PCA(n_components=2)
    item_pca = pca_item.fit_transform(item_embeddings)

    scatter2 = ax2.scatter(item_pca[:, 0], item_pca[:, 1], alpha=0.6, s=20, c='orange')
    ax2.set_title(f'Item Embeddings PCA\n({len(item_embeddings)} items, {item_embeddings.shape[1]}D → 2D)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca_item.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax2.set_ylabel(f'PC2 ({pca_item.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "embeddings_pca")


def plot_embeddings_tsne(embeddings_data: Dict):
    """Plot t-SNE visualization of user and item embeddings."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    user_embeddings = embeddings_data['user_embeddings']
    item_embeddings = embeddings_data['item_embeddings']

    # Sample for t-SNE if dataset is too large
    max_samples = 5000
    user_sample = user_embeddings[:max_samples] if len(user_embeddings) > max_samples else user_embeddings
    item_sample = item_embeddings[:max_samples] if len(item_embeddings) > max_samples else item_embeddings

    # User embeddings t-SNE
    tsne_user = TSNE(n_components=2, random_state=42, perplexity=min(30, len(user_sample) - 1))
    user_tsne = tsne_user.fit_transform(user_sample)

    scatter1 = ax1.scatter(user_tsne[:, 0], user_tsne[:, 1], alpha=0.6, s=20, c='blue')
    ax1.set_title(f'User Embeddings t-SNE\n({len(user_sample)} users, {user_embeddings.shape[1]}D → 2D)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE 1', fontsize=12)
    ax1.set_ylabel('t-SNE 2', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Item embeddings t-SNE
    tsne_item = TSNE(n_components=2, random_state=42, perplexity=min(30, len(item_sample) - 1))
    item_tsne = tsne_item.fit_transform(item_sample)

    scatter2 = ax2.scatter(item_tsne[:, 0], item_tsne[:, 1], alpha=0.6, s=20, c='orange')
    ax2.set_title(f'Item Embeddings t-SNE\n({len(item_sample)} items, {item_embeddings.shape[1]}D → 2D)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE 1', fontsize=12)
    ax2.set_ylabel('t-SNE 2', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "embeddings_tsne")


def plot_similarity_heatmap(embeddings_data: Dict):
    """Create comprehensive similarity heatmap for research insights."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    user_embeddings = embeddings_data['user_embeddings']
    item_embeddings = embeddings_data['item_embeddings']

    # Sample for heatmap (use more samples for better research insights)
    sample_size = min(100, len(user_embeddings), len(item_embeddings))
    user_sample = user_embeddings[:sample_size]
    item_sample = item_embeddings[:sample_size]

    # User-Item similarity matrix
    user_item_similarity = np.dot(user_sample, item_sample.T)
    im1 = ax1.imshow(user_item_similarity, cmap='RdYlBu_r', aspect='auto')
    ax1.set_title(f'User-Item Similarity Matrix\n({sample_size}×{sample_size})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Items')
    ax1.set_ylabel('Users')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # User-User similarity matrix
    user_user_similarity = np.dot(user_sample, user_sample.T)
    im2 = ax2.imshow(user_user_similarity, cmap='RdYlBu_r', aspect='auto')
    ax2.set_title(f'User-User Similarity Matrix\n({sample_size}×{sample_size})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Users')
    ax2.set_ylabel('Users')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Item-Item similarity matrix
    item_item_similarity = np.dot(item_sample, item_sample.T)
    im3 = ax3.imshow(item_item_similarity, cmap='RdYlBu_r', aspect='auto')
    ax3.set_title(f'Item-Item Similarity Matrix\n({sample_size}×{sample_size})', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Items')
    ax3.set_ylabel('Items')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Similarity distribution histogram
    all_similarities = np.concatenate([
        user_item_similarity.flatten(),
        user_user_similarity[np.triu_indices_from(user_user_similarity, k=1)],
        item_item_similarity[np.triu_indices_from(item_item_similarity, k=1)]
    ])

    ax4.hist(all_similarities, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_title('Similarity Score Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Cosine Similarity')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(np.mean(all_similarities), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_similarities):.3f}')
    ax4.legend()

    plt.tight_layout()
    save_plot(fig, "similarity_heatmap")


def plot_embedding_statistics(embeddings_data: Dict):
    """Plot embedding statistics and distributions for research insights."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    user_embeddings = embeddings_data['user_embeddings']
    item_embeddings = embeddings_data['item_embeddings']

    # Embedding norm distribution
    user_norms = np.linalg.norm(user_embeddings, axis=1)
    item_norms = np.linalg.norm(item_embeddings, axis=1)

    ax1.hist(user_norms, bins=50, alpha=0.7, label='User Embeddings', color='blue')
    ax1.hist(item_norms, bins=50, alpha=0.7, label='Item Embeddings', color='orange')
    ax1.set_title('Embedding Norm Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('L2 Norm')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Dimension-wise variance
    user_var = np.var(user_embeddings, axis=0)
    item_var = np.var(item_embeddings, axis=0)

    dimensions = range(len(user_var))
    ax2.plot(dimensions, user_var, label='User Embeddings', alpha=0.8)
    ax2.plot(dimensions, item_var, label='Item Embeddings', alpha=0.8)
    ax2.set_title('Dimension-wise Variance', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Embedding Dimension')
    ax2.set_ylabel('Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mean activation per dimension
    user_mean = np.mean(user_embeddings, axis=0)
    item_mean = np.mean(item_embeddings, axis=0)

    ax3.plot(dimensions, user_mean, label='User Embeddings', alpha=0.8)
    ax3.plot(dimensions, item_mean, label='Item Embeddings', alpha=0.8)
    ax3.set_title('Mean Activation per Dimension', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Embedding Dimension')
    ax3.set_ylabel('Mean Activation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Sparsity analysis
    user_sparsity = np.mean(np.abs(user_embeddings) < 0.01, axis=1) * 100
    item_sparsity = np.mean(np.abs(item_embeddings) < 0.01, axis=1) * 100

    ax4.hist(user_sparsity, bins=30, alpha=0.7, label='User Embeddings', color='blue')
    ax4.hist(item_sparsity, bins=30, alpha=0.7, label='Item Embeddings', color='orange')
    ax4.set_title('Embedding Sparsity Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Sparsity (%)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "embedding_statistics")


def visualize_results(history: Dict, embeddings_data: Dict, metrics_k5: Dict, metrics_k10: Dict, metrics_k5_cs: Dict = None, metrics_k10_cs: Dict = None):
    """
    Main visualization function that creates comprehensive research-quality plots.
    Each plot is saved as a separate image with unique naming for each run.
    """
    print("Generating comprehensive visualizations...")
    print(f"Saving plots to: {plot_saving_path}")

    # Create directory if it doesn't exist
    os.makedirs(plot_saving_path, exist_ok=True)

    # Plot training metrics
    plot_training_loss(history)
    plot_training_accuracy(history)

    # Plot evaluation metrics comparison
    if metrics_k5_cs is not None and metrics_k10_cs is not None:
        plot_metrics_comparison(metrics_k5, metrics_k10, metrics_k5_cs, metrics_k10_cs)
    else:
        # If no candidate sampling metrics, create a simpler comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        metric_names = ['Precision', 'Recall', 'NDCG', 'F1']
        k5_values = [metrics_k5.get('precision', 0), metrics_k5.get('recall', 0),
                     metrics_k5.get('ndcg', 0), metrics_k5.get('f1', 0)]
        k10_values = [metrics_k10.get('precision', 0), metrics_k10.get('recall', 0),
                      metrics_k10.get('ndcg', 0), metrics_k10.get('f1', 0)]

        x = np.arange(len(metric_names))
        width = 0.35

        bars1 = ax.bar(x - width / 2, k5_values, width, label='k=5', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width / 2, k10_values, width, label='k=10', color='lightcoral', alpha=0.8)

        ax.set_title('Evaluation Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        for bar, value in zip(bars1, k5_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        for bar, value in zip(bars2, k10_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        save_plot(fig, "metrics_comparison")

    # Plot embedding visualizations
    plot_embeddings_pca(embeddings_data)
    plot_embeddings_tsne(embeddings_data)
    plot_similarity_heatmap(embeddings_data)
    plot_embedding_statistics(embeddings_data)

    print("All visualizations completed and saved!")
    print(f"Generated {6 if metrics_k5_cs is not None else 6} research-quality plots.")