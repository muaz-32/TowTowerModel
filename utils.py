import numpy as np
import torch

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across all libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")