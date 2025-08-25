import numpy as np
import pandas as pd
from typing import Dict, List
import random

def generate_synthetic_data(n_users: int = 10000, n_items: int = 5000,
                            n_interactions: int = 100000) -> Dict:
    """
    Generate synthetic data with topic lists for users and items.

    Returns:
        Dictionary containing user topics, item topics, and interactions
    """
    print("Generating synthetic data with topics...")

    # Define topic pools
    topics = [
        'technology', 'sports', 'music', 'movies', 'cooking', 'travel',
        'fitness', 'gaming', 'art', 'fashion', 'photography',
        'science', 'politics', 'business', 'health', 'education', 'nature',
        'electronics', 'clothing', 'home_decor', 'books', 'sports_equipment',
        'kitchen_appliances', 'beauty', 'automotive', 'toys', 'music_instruments',
        'outdoor_gear', 'health_supplements', 'art_supplies', 'pet_supplies',
        'office_supplies', 'furniture', 'jewelry', 'food_beverages'
    ]

    # Generate user topic lists (each user interested in 3-7 topics)
    user_topic_lists = []
    for _ in range(n_users):
        n_topics = random.randint(3, 7)
        user_topics_sample = random.sample(topics, n_topics)
        user_topic_lists.append(user_topics_sample)

    # Generate item topic lists (each item belongs to 2-5 categories)
    item_topic_lists = []
    for _ in range(n_items):
        n_categories = random.randint(2, 5)
        item_topics_sample = random.sample(topics, n_categories)
        item_topic_lists.append(item_topics_sample)

    # Generate interactions based on topic overlap
    interactions = []

    # Positive interactions (users and items with topic overlap)
    for _ in range(n_interactions // 2):
        user_id = random.randint(0, n_users - 1)
        item_id = random.randint(0, n_items - 1)

        # Check for topic similarity (can be customized)
        user_topics_set = set(user_topic_lists[user_id])
        item_topics_set = set(item_topic_lists[item_id])

        # Create some overlap bias for positive interactions
        if len(user_topics_set.intersection(item_topics_set)) > 0 or random.random() < 0.3:
            interactions.append([user_id, item_id, 1])

    # Negative interactions (mostly random, some with no overlap)
    for _ in range(n_interactions // 2):
        user_id = random.randint(0, n_users - 1)
        item_id = random.randint(0, n_items - 1)
        interactions.append([user_id, item_id, 0])

    interactions_df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'label'])

    print(f"Generated {len(interactions_df)} interactions")
    print(f"Positive interactions: {sum(interactions_df['label'])}")
    print(f"Negative interactions: {len(interactions_df) - sum(interactions_df['label'])}")

    return {
        'user_topics': user_topic_lists,
        'item_topics': item_topic_lists,
        'interactions': interactions_df,
        'n_users': n_users,
        'n_items': n_items,
        'topic_vocab': topics
    }