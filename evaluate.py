from typing import Dict
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from model import TwoTowerModel
from utils import device

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

    all_item_features = original_data['item_topics']
    all_user_features = original_data['user_topics']

    # Convert to tensors
    all_item_features_tensor = torch.FloatTensor(all_item_features).to(device)
    all_user_features_tensor = torch.FloatTensor(all_user_features).to(device)

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