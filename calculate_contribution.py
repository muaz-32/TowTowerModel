import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

users_file = './data/api/users.csv'


def extract_reputation_features(reputation: float, reputation_change_yearly: float) -> Dict[str, float]:
    """
    Extract user reputation features from reputation data

    Args:
        reputation: Current user reputation
        reputation_change_yearly: Yearly change in reputation

    Returns:
        Dictionary of reputation features
    """
    if pd.isna(reputation):
        reputation = 0
    if pd.isna(reputation_change_yearly):
        reputation_change_yearly = 0

    # Basic reputation metrics
    log_reputation = np.log1p(max(reputation, 0))  # log(1 + reputation) to handle 0 values

    # Reputation growth rate (normalized)
    reputation_growth_rate = reputation_change_yearly / max(reputation, 1)

    # Reputation categories (you can adjust these thresholds)
    reputation_tier = 0
    if reputation >= 10000:
        reputation_tier = 4  # Expert
    elif reputation >= 3000:
        reputation_tier = 3  # Advanced
    elif reputation >= 1000:
        reputation_tier = 2  # Intermediate
    elif reputation >= 100:
        reputation_tier = 1  # Beginner
    else:
        reputation_tier = 0  # New user

    # Growth momentum indicators
    is_growing = 1 if reputation_change_yearly > 0 else 0
    is_declining = 1 if reputation_change_yearly < 0 else 0
    is_stable = 1 if reputation_change_yearly == 0 else 0

    # High growth indicator
    high_growth = 1 if reputation_change_yearly > 500 else 0

    return {
        'reputation': float(reputation),
        'log_reputation': float(log_reputation),
        'reputation_change_yearly': float(reputation_change_yearly),
        'reputation_growth_rate': float(reputation_growth_rate),
        'reputation_tier': float(reputation_tier),
        'is_growing': float(is_growing),
        'is_declining': float(is_declining),
        'is_stable': float(is_stable),
        'high_growth': float(high_growth)
    }


def get_reputation_data(users_file_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Load reputation data from users CSV file

    Args:
        users_file_path: Path to the users CSV file

    Returns:
        Dictionary mapping user_id to (reputation, reputation_change_yearly) tuple
    """
    try:
        users_df = pd.read_csv(users_file_path)

        # Ensure required columns exist
        if 'reputation' not in users_df.columns or 'reputation_change_yearly' not in users_df.columns:
            print("Warning: Required reputation columns not found in users.csv")
            return {}

        reputation_data = {}
        for idx, row in users_df.iterrows():
            user_id = str(idx)  # Using index as user_id
            reputation = row['reputation']
            reputation_change = row['reputation_change_yearly']
            reputation_data[user_id] = (reputation, reputation_change)

        print(f"Loaded reputation data for {len(reputation_data)} users")
        return reputation_data

    except FileNotFoundError:
        print(f"Warning: Users CSV file not found at {users_file_path}")
        return {}
    except Exception as e:
        print(f"Error loading reputation data: {e}")
        return {}


def append_reputation_features_to_embeddings(user_topic_embeddings: np.ndarray) -> np.ndarray:
    """
    Append reputation features to user topic embeddings

    Args:
        user_topic_embeddings: numpy array with user embeddings

    Returns:
        Updated numpy array with reputation features
    """
    reputation_features = []
    reputation_data = get_reputation_data(users_file)

    for user_id in range(user_topic_embeddings.shape[0]):
        user_reputation_info = reputation_data.get(str(user_id), (0, 0))
        reputation, reputation_change = user_reputation_info

        features = extract_reputation_features(reputation, reputation_change)
        reputation_features.append(list(features.values()))

    # Convert to numpy array
    reputation_array = np.array(reputation_features)

    # Concatenate with existing embeddings
    enhanced_embeddings = np.concatenate([user_topic_embeddings, reputation_array], axis=1)

    return enhanced_embeddings

