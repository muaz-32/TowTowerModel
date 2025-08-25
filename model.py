from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

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