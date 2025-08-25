import numpy as np
import pandas as pd
from typing import Dict
import random

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