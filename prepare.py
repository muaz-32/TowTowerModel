from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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