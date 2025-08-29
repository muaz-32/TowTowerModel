from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_data(data: Dict, test_size: float = 0.2) -> Tuple[Dict, Dict, Dict]:
    """
    Prepare training and test datasets from the synthetic data with topic lists.
    """
    interactions = data['interactions']
    user_topics = data['user_topics']
    item_topics = data['item_topics']
    topic_vocab = data['topic_vocab']

    # Split interactions into train and test
    train_interactions, test_interactions = train_test_split(
        interactions, test_size=test_size, random_state=42, stratify=interactions['label']
    )

    # Create topic-to-index mappings
    topic_to_idx = {topic: idx + 1 for idx, topic in enumerate(topic_vocab)}  # Start from 1, reserve 0 for padding

    def convert_to_multihot_encoding(topic_lists: List[List[str]], vocab_size: int, topic_to_idx: Dict) -> np.ndarray:
        """Convert variable-length topic lists to multi-hot encoded vectors."""
        multihot_vectors = []

        for topics in topic_lists:
            # Create zero vector of vocab size
            multihot = np.zeros(vocab_size, dtype=np.float32)

            # Set 1 for each topic present
            for topic in topics:
                if topic in topic_to_idx:
                    multihot[topic_to_idx[topic] - 1] = 1.0  # -1 because we start indexing from 1

            multihot_vectors.append(multihot)

        return np.array(multihot_vectors)

    user_topic_indices = convert_to_multihot_encoding(user_topics, vocab_size=len(topic_vocab), topic_to_idx=topic_to_idx)
    item_topic_indices = convert_to_multihot_encoding(item_topics, vocab_size=len(topic_vocab), topic_to_idx=topic_to_idx)

    # Prepare training data
    train_user_topics = np.array([user_topic_indices[uid] for uid in train_interactions['user_id'].values])
    train_item_topics = np.array([item_topic_indices[iid] for iid in train_interactions['item_id'].values])
    train_labels = train_interactions['label'].values

    # Prepare test data
    test_user_topics = np.array([user_topic_indices[uid] for uid in test_interactions['user_id'].values])
    test_item_topics = np.array([item_topic_indices[iid] for iid in test_interactions['item_id'].values])
    test_labels = test_interactions['label'].values

    train_data = {
        'user_topics': train_user_topics,
        'item_topics': train_item_topics,
        'labels': train_labels,
        'interactions': train_interactions,
        'topic_to_idx': topic_to_idx,
        'vocab_size': len(topic_vocab)
    }

    test_data = {
        'user_topics': test_user_topics,
        'item_topics': test_item_topics,
        'labels': test_labels,
        'interactions': test_interactions
    }

    original_data = {
        'user_topics': user_topic_indices,
        'item_topics': item_topic_indices,
        'topic_to_idx': topic_to_idx,
        'vocab_size': len(topic_vocab)
    }

    return train_data, test_data, original_data