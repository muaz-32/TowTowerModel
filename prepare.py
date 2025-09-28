from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
import numpy as np
import os
from dotenv import load_dotenv
from embed import convert_to_embedding
from calculate_activeness import append_activeness_features_to_embeddings
from calculate_contribution import append_reputation_features_to_embeddings

# Load environment variables
load_dotenv()
bronze_badge_weight = int(os.getenv("BRONZE_BADGE_WEIGHT"))
silver_badge_weight = int(os.getenv("SILVER_BADGE_WEIGHT"))
gold_badge_weight = int(os.getenv("GOLD_BADGE_WEIGHT"))

def prepare_data(data: Dict, test_size: float) -> Tuple[Dict, Dict]:
    """
    Prepare training and test datasets from the synthetic data with topic lists.
    """
    interactions = data['interactions']
    user_topics = data['user_topics']
    item_topics = data['item_topics']
    user_badges = data['user_badges']
    topic_vocab = data['topic_vocab']

    # Badge weights
    badge_weights = {
        'bronze': bronze_badge_weight,
        'silver': silver_badge_weight,
        'gold': gold_badge_weight
    }

    # Split interactions into train and test
    train_interactions, test_interactions = train_test_split(
        interactions, test_size=test_size, random_state=42, stratify=interactions['label']
    )

    # Create topic-to-index mappings
    topic_to_idx = {topic: idx + 1 for idx, topic in enumerate(topic_vocab)}  # Start from 1, reserve 0 for padding

    def convert_to_multihot_encoding(topic_lists: List[List[str]], badge_lists:List[List[Tuple[str, str, int]]] | None, vocab_size: int, topic_to_idx: Dict) -> np.ndarray:
        """Convert variable-length topic lists to multi-hot encoded vectors."""
        multihot_vectors = []

        if badge_lists is not None:
            for topics, badges in zip(topic_lists, badge_lists):
                # Create zero vector of vocab size
                multihot = np.zeros(vocab_size, dtype=np.float32)

                # Set 1 for each topic present
                for topic in topics:
                    if topic in topic_to_idx:
                        multihot[topic_to_idx[topic] - 1] = 1.0  # -1 because we start indexing from 1

                for badge in badges:
                    # Since badge name = topic name, direct mapping
                    if badge[0] in topic_to_idx:
                        idx = topic_to_idx[badge[0]] - 1  # -1 because we start indexing from 1
                        badge_rank = badge[1]  # badge is a tuple (name, rank, count)
                        badge_count = badge[2]

                        weight = badge_weights.get(badge_rank, 0) * badge_count
                        multihot[idx] += weight  # Increment by weight

                multihot_vectors.append(multihot)
        else:
            for topics in topic_lists:
                # Create zero vector of vocab size
                multihot = np.zeros(vocab_size, dtype=np.float32)

                # Set 1 for each topic present
                for topic in topics:
                    if topic in topic_to_idx:
                        multihot[topic_to_idx[topic] - 1] = 1.0  # -1 because we start indexing from 1

                multihot_vectors.append(multihot)

        return np.array(multihot_vectors)

    user_topic_values = convert_to_multihot_encoding(user_topics, vocab_size=len(topic_vocab), topic_to_idx=topic_to_idx, badge_lists=user_badges)
    item_topic_values = convert_to_multihot_encoding(item_topics, vocab_size=len(topic_vocab), topic_to_idx=topic_to_idx, badge_lists=None)

    user_topic_embeddings = convert_to_embedding(user_topic_values, topic_to_idx)
    item_topic_embeddings = convert_to_embedding(item_topic_values, topic_to_idx)

    # Add Activeness features
    user_topic_activeness_embeddings = append_activeness_features_to_embeddings(user_topic_embeddings)
    # Add Contribution features
    user_topic_activeness_contribution_embeddings = append_reputation_features_to_embeddings(user_topic_activeness_embeddings)

    # Prepare training data
    train_user_topic_embeddings = np.array([user_topic_activeness_contribution_embeddings[uid] for uid in train_interactions['user_id'].values])
    train_item_topic_embeddings = np.array([item_topic_embeddings[iid] for iid in train_interactions['item_id'].values])
    train_labels = train_interactions['label'].values

    # Prepare test data
    test_user_ids = test_interactions['user_id'].values
    test_item_ids = test_interactions['item_id'].values
    test_user_topic_embeddings = np.array([user_topic_activeness_contribution_embeddings[uid] for uid in test_user_ids])
    test_item_topic_embeddings = np.array([item_topic_embeddings[iid] for iid in test_item_ids])
    test_labels = test_interactions['label'].values

    train_data = {
        'user_topic_embeddings': train_user_topic_embeddings,
        'item_topic_embeddings': train_item_topic_embeddings,
        'labels': train_labels,
        'interactions': train_interactions,
        'topic_to_idx': topic_to_idx,
        'vocab_size': len(topic_vocab)
    }

    test_data = {
        'user_topic_embeddings': test_user_topic_embeddings,
        'item_topic_embeddings': test_item_topic_embeddings,
        'labels': test_labels,
        'interactions': test_interactions,
        'user_ids': test_user_ids,
        'item_ids': test_item_ids
    }

    return train_data, test_data