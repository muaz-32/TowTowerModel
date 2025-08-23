"""
Two-Tower Model Implementation with Evaluation Metrics using PyTorch
A recommendation system model that learns separate embeddings for users and items.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TwoTowerDataset(Dataset):
    """
    Custom Dataset for Two-Tower model training.
    """
    def __init__(self, user_features, item_features, labels):
        self.user_features = torch.FloatTensor(user_features)
        self.item_features = torch.FloatTensor(item_features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'user_features': self.user_features[idx],
            'item_features': self.item_features[idx],
            'labels': self.labels[idx]
        }

class TwoTowerModel(nn.Module):
    """
    Two-Tower Model for recommendation systems using PyTorch.

    The model consists of two separate neural networks (towers):
    - User Tower: Encodes user features into embeddings
    - Item Tower: Encodes item features into embeddings

    The similarity between users and items is computed using cosine similarity.
    """

    def __init__(self, user_features_dim: int, item_features_dim: int,
                 embedding_dim: int = 64, hidden_layers: List[int] = [128, 64],
                 dropout_rate: float = 0.3):
        super(TwoTowerModel, self).__init__()

        self.user_features_dim = user_features_dim
        self.item_features_dim = item_features_dim
        self.embedding_dim = embedding_dim

        # User Tower
        user_layers = []
        input_dim = user_features_dim

        for hidden_dim in hidden_layers:
            user_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        # Final embedding layer for user tower
        user_layers.append(nn.Linear(input_dim, embedding_dim))
        user_layers.append(nn.ReLU())

        self.user_tower = nn.Sequential(*user_layers)

        # Item Tower
        item_layers = []
        input_dim = item_features_dim

        for hidden_dim in hidden_layers:
            item_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        # Final embedding layer for item tower
        item_layers.append(nn.Linear(input_dim, embedding_dim))
        item_layers.append(nn.ReLU())

        self.item_tower = nn.Sequential(*item_layers)

    def forward(self, user_features, item_features):
        """Forward pass through both towers."""
        # Get embeddings from both towers
        user_embeddings = self.user_tower(user_features)
        item_embeddings = self.item_tower(item_features)

        # L2 normalize embeddings for cosine similarity
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
        item_embeddings = F.normalize(item_embeddings, p=2, dim=1)

        # Compute cosine similarity (dot product of normalized vectors)
        similarity = torch.sum(user_embeddings * item_embeddings, dim=1)

        return similarity, user_embeddings, item_embeddings

    def get_user_embeddings(self, user_features):
        """Get user embeddings only."""
        with torch.no_grad():
            user_embeddings = self.user_tower(user_features)
            return F.normalize(user_embeddings, p=2, dim=1)

    def get_item_embeddings(self, item_features):
        """Get item embeddings only."""
        with torch.no_grad():
            item_embeddings = self.item_tower(item_features)
            return F.normalize(item_embeddings, p=2, dim=1)

def generate_synthetic_data(n_users: int = 10000, n_items: int = 5000,
                          n_interactions: int = 100000) -> Dict:
    """
    Generate synthetic data for training and evaluation.

    Returns:
        Dictionary containing user features, item features, and interactions
    """
    print("Generating synthetic data...")

    # User features (age, income, location, preferences)
    user_features = np.random.randn(n_users, 10)  # 10 user features
    user_ids = np.arange(n_users)

    # Item features (price, category, brand, ratings)
    item_features = np.random.randn(n_items, 8)   # 8 item features
    item_ids = np.arange(n_items)

    # Generate interactions (some positive, some negative)
    interactions = []

    # Positive interactions (users with similar preferences to items)
    for _ in range(n_interactions // 2):
        user_id = np.random.choice(user_ids)
        # Create positive bias by selecting items that align with user preferences
        item_id = np.random.choice(item_ids)
        similarity_score = np.dot(user_features[user_id][:5], item_features[item_id][:5])
        if similarity_score > 0:  # Positive interaction
            interactions.append([user_id, item_id, 1])

    # Negative interactions (random selection)
    for _ in range(n_interactions // 2):
        user_id = np.random.choice(user_ids)
        item_id = np.random.choice(item_ids)
        interactions.append([user_id, item_id, 0])

    interactions_df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'label'])

    print(f"Generated {len(interactions_df)} interactions")
    print(f"Positive interactions: {sum(interactions_df['label'])}")
    print(f"Negative interactions: {len(interactions_df) - sum(interactions_df['label'])}")

    return {
        'user_features': user_features,
        'item_features': item_features,
        'interactions': interactions_df,
        'n_users': n_users,
        'n_items': n_items
    }

def prepare_training_data(data: Dict, test_size: float = 0.2) -> Tuple[Dict, Dict]:
    """
    Prepare training and test datasets from the synthetic data.
    """
    interactions = data['interactions']
    user_features = data['user_features']
    item_features = data['item_features']

    # Split interactions into train and test
    train_interactions, test_interactions = train_test_split(
        interactions, test_size=test_size, random_state=42, stratify=interactions['label']
    )

    # Prepare training data
    train_user_feats = user_features[train_interactions['user_id'].values]
    train_item_feats = item_features[train_interactions['item_id'].values]
    train_labels = train_interactions['label'].values

    # Prepare test data
    test_user_feats = user_features[test_interactions['user_id'].values]
    test_item_feats = item_features[test_interactions['item_id'].values]
    test_labels = test_interactions['label'].values

    # Normalize features
    user_scaler = StandardScaler()
    item_scaler = StandardScaler()

    train_user_feats = user_scaler.fit_transform(train_user_feats)
    train_item_feats = item_scaler.fit_transform(train_item_feats)
    test_user_feats = user_scaler.transform(test_user_feats)
    test_item_feats = item_scaler.transform(test_item_feats)

    train_data = {
        'user_features': train_user_feats,
        'item_features': train_item_feats,
        'labels': train_labels,
        'interactions': train_interactions,
        'scalers': {'user_scaler': user_scaler, 'item_scaler': item_scaler}
    }

    test_data = {
        'user_features': test_user_feats,
        'item_features': test_item_feats,
        'labels': test_labels,
        'interactions': test_interactions
    }

    return train_data, test_data

def train_model(model: TwoTowerModel, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 50, learning_rate: float = 0.001) -> Dict:
    """
    Train the two-tower model.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in train_pbar:
            user_features = batch['user_features'].to(device)
            item_features = batch['item_features'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            similarities, _, _ = model(user_features, item_features)
            loss = criterion(similarities, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            predictions = torch.sigmoid(similarities) > 0.5
            train_correct += (predictions == labels.bool()).sum().item()
            train_total += labels.size(0)

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * train_correct / train_total:.2f}%'
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for batch in val_pbar:
                user_features = batch['user_features'].to(device)
                item_features = batch['item_features'].to(device)
                labels = batch['labels'].to(device)

                similarities, _, _ = model(user_features, item_features)
                loss = criterion(similarities, labels)

                val_loss += loss.item()

                # Calculate accuracy
                predictions = torch.sigmoid(similarities) > 0.5
                val_correct += (predictions == labels.bool()).sum().item()
                val_total += labels.size(0)

                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.0 * val_correct / val_total:.2f}%'
                })

        scheduler.step()

        # Record metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print()

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

class EvaluationMetrics:
    """
    Evaluation metrics for recommendation systems.
    """

    @staticmethod
    def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 5) -> float:
        """Calculate Precision@K"""
        if len(y_scores) < k:
            k = len(y_scores)

        # Get top-k predictions
        top_k_indices = np.argsort(y_scores)[-k:]
        top_k_true = y_true[top_k_indices]

        precision = np.sum(top_k_true) / k
        return precision

    @staticmethod
    def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 5) -> float:
        """Calculate Recall@K"""
        if np.sum(y_true) == 0:
            return 0.0

        if len(y_scores) < k:
            k = len(y_scores)

        # Get top-k predictions
        top_k_indices = np.argsort(y_scores)[-k:]
        top_k_true = y_true[top_k_indices]

        recall = np.sum(top_k_true) / np.sum(y_true)
        return recall

    @staticmethod
    def ndcg_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 5) -> float:
        """Calculate NDCG@K (Normalized Discounted Cumulative Gain)"""
        def dcg_at_k(scores: np.ndarray, k: int) -> float:
            scores = scores[:k]
            if len(scores) == 0:
                return 0.0
            return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))

        if len(y_scores) < k:
            k = len(y_scores)

        # Get top-k predictions
        top_k_indices = np.argsort(y_scores)[-k:][::-1]  # Descending order
        top_k_true = y_true[top_k_indices]

        # Calculate DCG
        dcg = dcg_at_k(top_k_true, k)

        # Calculate IDCG (Ideal DCG)
        ideal_scores = np.sort(y_true)[::-1]
        idcg = dcg_at_k(ideal_scores, k)

        if idcg == 0:
            return 0.0

        ndcg = dcg / idcg
        return ndcg

def evaluate_model(model: TwoTowerModel, test_data: Dict, original_data: Dict,
                   k: int = 5, n_users_sample: int = 100) -> Dict[str, float]:
    """
    Evaluate the two-tower model using various metrics.
    """
    print(f"Evaluating model with k={k}...")

    model.eval()
    model.to(device)

    # Get unique users from test data
    test_interactions = test_data['interactions']
    unique_users = test_interactions['user_id'].unique()[:n_users_sample]

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    # Prepare normalized features for all items
    user_scaler = test_data.get('user_scaler') or StandardScaler().fit(original_data['user_features'])
    item_scaler = test_data.get('item_scaler') or StandardScaler().fit(original_data['item_features'])

    all_item_features_norm = item_scaler.transform(original_data['item_features'])
    all_user_features_norm = user_scaler.transform(original_data['user_features'])

    # Convert to tensors
    all_item_features_tensor = torch.FloatTensor(all_item_features_norm).to(device)
    all_user_features_tensor = torch.FloatTensor(all_user_features_norm).to(device)

    with torch.no_grad():
        for user_id in tqdm(unique_users, desc="Evaluating users"):
            # Get user interactions
            user_interactions = test_interactions[test_interactions['user_id'] == user_id]

            if len(user_interactions) < 2:  # Skip users with too few interactions
                continue

            # Get ground truth items for this user
            true_items = user_interactions[user_interactions['label'] == 1]['item_id'].values

            if len(true_items) == 0:  # Skip users with no positive interactions
                continue

            # Get user features
            user_features = all_user_features_tensor[user_id:user_id+1]

            # Repeat user features for all items
            user_features_repeated = user_features.repeat(len(all_item_features_tensor), 1)

            # Calculate similarities with all items
            similarities, _, _ = model(user_features_repeated, all_item_features_tensor)
            similarities = torch.sigmoid(similarities).cpu().numpy()

            # Create ground truth array
            y_true = np.zeros(len(similarities))
            y_true[true_items] = 1

            # Calculate metrics
            precision = EvaluationMetrics.precision_at_k(y_true, similarities, k)
            recall = EvaluationMetrics.recall_at_k(y_true, similarities, k)
            ndcg = EvaluationMetrics.ndcg_at_k(y_true, similarities, k)

            precision_scores.append(precision)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)

    # Calculate average metrics
    avg_precision = np.mean(precision_scores) if precision_scores else 0.0
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

    print(f"Average Precision@{k}: {avg_precision:.4f}")
    print(f"Average Recall@{k}: {avg_recall:.4f}")
    print(f"Average NDCG@{k}: {avg_ndcg:.4f}")

    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'ndcg': avg_ndcg
    }

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

def main():
    """
    Main function to run the two-tower model training and evaluation.
    """
    print("Two-Tower Model Implementation with Evaluation (PyTorch)")
    print("=" * 60)

    # Generate synthetic data
    data = generate_synthetic_data(n_users=5000, n_items=2000, n_interactions=50000)

    # Prepare training and test data
    train_data, test_data = prepare_training_data(data)

    # Create datasets and data loaders
    train_dataset = TwoTowerDataset(
        train_data['user_features'],
        train_data['item_features'],
        train_data['labels']
    )

    test_dataset = TwoTowerDataset(
        test_data['user_features'],
        test_data['item_features'],
        test_data['labels']
    )

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
    val_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

    # Initialize the model
    model = TwoTowerModel(
        user_features_dim=train_data['user_features'].shape[1],
        item_features_dim=train_data['item_features'].shape[1],
        embedding_dim=64,
        hidden_layers=[128, 64],
        dropout_rate=0.3
    )

    print(f"\nModel Architecture:")
    print(f"User features dimension: {train_data['user_features'].shape[1]}")
    print(f"Item features dimension: {train_data['item_features'].shape[1]}")
    print(f"Embedding dimension: 64")
    print(f"Hidden layers: [128, 64]")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train the model
    print("\nTraining the model...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        learning_rate=0.001
    )

    # Generate embeddings for visualization
    print("\nGenerating embeddings for visualization...")
    model.eval()
    model.to(device)

    # Sample data for embedding visualization
    sample_user_features = torch.FloatTensor(train_data['user_features'][:5000]).to(device)
    sample_item_features = torch.FloatTensor(train_data['item_features'][:5000]).to(device)

    with torch.no_grad():
        user_embeddings = model.get_user_embeddings(sample_user_features).cpu().numpy()
        item_embeddings = model.get_item_embeddings(sample_item_features).cpu().numpy()

    embeddings_data = {
        'user_embeddings': user_embeddings,
        'item_embeddings': item_embeddings
    }

    # Add scalers to test_data for evaluation
    test_data['user_scaler'] = train_data['scalers']['user_scaler']
    test_data['item_scaler'] = train_data['scalers']['item_scaler']

    # Evaluate the model
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    metrics_k5 = evaluate_model(model, test_data, data, k=5, n_users_sample=50)
    metrics_k10 = evaluate_model(model, test_data, data, k=10, n_users_sample=50)

    print(f"\nFinal Results:")
    print(f"Metrics@5: Precision={metrics_k5['precision']:.4f}, "
          f"Recall={metrics_k5['recall']:.4f}, NDCG={metrics_k5['ndcg']:.4f}")
    print(f"Metrics@10: Precision={metrics_k10['precision']:.4f}, "
          f"Recall={metrics_k10['recall']:.4f}, NDCG={metrics_k10['ndcg']:.4f}")

    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(history, embeddings_data, metrics_k5)

    print("\n" + "=" * 60)
    print("Training and evaluation completed!")
    print("The PyTorch two-tower model has been successfully trained and evaluated.")
    print("\nKey insights:")
    print("- The model learns separate embeddings for users and items using PyTorch")
    print("- Cosine similarity is used to measure user-item compatibility")
    print("- BCEWithLogitsLoss is used for binary classification")
    print("- Adam optimizer with learning rate scheduling")
    print("- Evaluation metrics show the model's recommendation performance")
    print("=" * 60)

if __name__ == "__main__":
    main()