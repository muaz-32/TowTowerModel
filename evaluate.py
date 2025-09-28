from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm
from model import TwoTowerModel
from utils import device
import os
from dotenv import load_dotenv

load_dotenv()
interaction_threshold = int(os.getenv("SKIP_USER_INTERACTION_THRESHOLD"))

class EvaluationMetrics:
    """
    Evaluation metrics for recommendation systems.
    """

    @staticmethod
    def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
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

    @staticmethod
    def f1_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 5) -> float:
        """Calculate F1@K score"""
        precision = EvaluationMetrics.precision_at_k(y_true, y_scores, k)
        recall = EvaluationMetrics.recall_at_k(y_true, y_scores, k)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

def evaluate_model(model: TwoTowerModel, test_data: Dict, test_user_ids: np.array, test_item_ids: np.array, k: int) -> Dict[str, float]:
    """
    Evaluate the two-tower model using various metrics.
    """
    print(f"Evaluating model with k={k}...")

    model.eval()
    model.to(device)

    # Get unique users from test data
    test_interactions = test_data['interactions']
    unique_users = test_interactions['user_id'].unique()

    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    f1_scores = []

    all_user_features = test_data['user_topic_embeddings']
    all_item_features = test_data['item_topic_embeddings']

    # Convert to tensors
    all_user_features_tensor = torch.FloatTensor(all_user_features).to(device)
    all_item_features_tensor = torch.FloatTensor(all_item_features).to(device)

    with torch.no_grad():
        for user_id in tqdm(unique_users, desc="Evaluating users"):
            # Get user interactions
            user_interactions = test_interactions[test_interactions['user_id'] == user_id]

            if len(user_interactions) < interaction_threshold:  # Skip users with too few interactions
                continue

            user_index = list(test_user_ids).index(user_id)

            # Get user features
            user_features = all_user_features_tensor[user_index:user_index+1]

            # Repeat user features for all items
            user_features_repeated = user_features.repeat(len(all_item_features_tensor), 1)

            # Calculate similarities with all items
            similarities, _, _ = model(user_features_repeated, all_item_features_tensor)
            similarities = torch.sigmoid(similarities).cpu().numpy()

            # Get ground truth items for this user
            true_items = user_interactions[user_interactions['label'] == 1]['item_id'].values
            if len(true_items) == 0:  # Skip users with no positive interactions
                continue

            # Create ground truth array
            y_true = np.zeros(len(similarities))

            for true_item in true_items:
                true_item_index = list(test_item_ids).index(true_item)
                y_true[true_item_index] = 1

            # Calculate metrics
            precision = EvaluationMetrics.precision_at_k(y_true, similarities, k)
            recall = EvaluationMetrics.recall_at_k(y_true, similarities, k)
            ndcg = EvaluationMetrics.ndcg_at_k(y_true, similarities, k)
            f1 = EvaluationMetrics.f1_at_k(y_true, similarities, k)

            precision_scores.append(precision)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
            f1_scores.append(f1)

    # Calculate average metrics
    avg_precision = np.mean(precision_scores) if precision_scores else 0.0
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0

    print(f"Average Precision@{k}: {avg_precision:.4f}")
    print(f"Average Recall@{k}: {avg_recall:.4f}")
    print(f"Average NDCG@{k}: {avg_ndcg:.4f}")
    print(f"Average F1@{k}: {avg_f1:.4f}")

    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'ndcg': avg_ndcg,
        'f1': avg_f1
    }


def evaluate_model_candidate_sampling(model: TwoTowerModel, test_data: Dict, test_user_ids: np.array, test_item_ids: np.array, k: int) -> Dict[str, float]:
    """
    Improved evaluation with candidate sampling.
    """
    model.eval()
    model.to(device)

    test_interactions = test_data['interactions']
    unique_users = test_interactions['user_id'].unique()

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    # Use candidate sampling instead of all items
    num_candidates = 100  # Sample 100 items instead of all 18,730

    with torch.no_grad():
        for user_id in tqdm(unique_users, desc="Evaluating users"):
            user_interactions = test_interactions[test_interactions['user_id'] == user_id]

            if len(user_interactions) < interaction_threshold:
                continue

            # Get positive items for this user
            positive_items = user_interactions[user_interactions['label'] == 1]['item_id'].values
            if len(positive_items) == 0:
                continue

            # Sample negative candidates
            all_item_indices = list(range(len(test_item_ids)))
            positive_indices = [list(test_item_ids).index(item) for item in positive_items if item in test_item_ids]
            negative_indices = [idx for idx in all_item_indices if idx not in positive_indices]

            # Sample candidates (positive + random negatives)
            num_negatives = min(num_candidates - len(positive_indices), len(negative_indices))
            sampled_negatives = np.random.choice(negative_indices, num_negatives, replace=False)
            candidate_indices = positive_indices + list(sampled_negatives)

            # Get user and candidate item features
            user_index = list(test_user_ids).index(user_id)
            user_features = torch.FloatTensor(test_data['user_topic_embeddings'][user_index:user_index + 1]).to(device)
            candidate_features = torch.FloatTensor(test_data['item_topic_embeddings'][candidate_indices]).to(device)

            # Calculate similarities
            user_features_repeated = user_features.repeat(len(candidate_indices), 1)
            similarities, _, _ = model(user_features_repeated, candidate_features)
            similarities = torch.sigmoid(similarities).cpu().numpy()

            # Create ground truth for candidates
            y_true = np.zeros(len(candidate_indices))
            y_true[:len(positive_indices)] = 1  # First items are positive

            # Calculate metrics
            precision = EvaluationMetrics.precision_at_k(y_true, similarities, k)
            recall = EvaluationMetrics.recall_at_k(y_true, similarities, k)
            ndcg = EvaluationMetrics.ndcg_at_k(y_true, similarities, k)

            precision_scores.append(precision)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)

    return {
        'precision': np.mean(precision_scores) if precision_scores else 0.0,
        'recall': np.mean(recall_scores) if recall_scores else 0.0,
        'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0.0
    }
