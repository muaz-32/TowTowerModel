from torch.utils.data import Dataset
import torch

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