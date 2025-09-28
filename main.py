"""
Two-Tower Model Implementation with Evaluation Metrics using PyTorch
A recommendation system model that learns separate embeddings for users and items.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
from dataset import TwoTowerDataset
from model import TwoTowerModel
from prepare import prepare_data
from evaluate import evaluate_model, evaluate_model_candidate_sampling
from visualize import visualize_results
from train import train_model
from utils import set_random_seeds, device
from extract import extract_data
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
test_size = float(os.getenv("TEST_DATA_SIZE"))

warnings.filterwarnings('ignore')

def main():
    """
    Main function to run the two-tower model training and evaluation.
    """
    print("Two-Tower Model Implementation with Evaluation (PyTorch)")
    print("=" * 60)

    set_random_seeds(42)

    # Get data
    data = extract_data()

    # Prepare training and test data
    train_data, test_data = prepare_data(data, test_size)

    # Create datasets and data loaders
    train_dataset = TwoTowerDataset(
        train_data['user_topic_embeddings'],
        train_data['item_topic_embeddings'],
        train_data['labels']
    )

    test_dataset = TwoTowerDataset(
        test_data['user_topic_embeddings'],
        test_data['item_topic_embeddings'],
        test_data['labels']
    )

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
    val_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

    # Initialize the model
    model = TwoTowerModel(
        user_features_dim=train_data['user_topic_embeddings'].shape[1],
        item_features_dim=train_data['item_topic_embeddings'].shape[1],
        embedding_dim=64,
        hidden_layers=[128, 64],
        dropout_rate=0.3
    )

    print(f"\nModel Architecture:")
    print(f"User topics dimension: {train_data['user_topic_embeddings'].shape[1]}")
    print(f"Item topics dimension: {train_data['item_topic_embeddings'].shape[1]}")
    print(f"Embedding dimension: 64")
    print(f"Hidden layers: [128, 64]")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train the model
    print("\nTraining the model...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        learning_rate=0.001
    )

    # Generate embeddings for visualization
    print("\nGenerating embeddings for visualization...")
    model.eval()
    model.to(device)

    # data for embedding visualization
    user_features = torch.FloatTensor(train_data['user_topic_embeddings']).to(device)
    item_features = torch.FloatTensor(train_data['item_topic_embeddings']).to(device)

    with torch.no_grad():
        user_embeddings = model.get_user_embeddings(user_features).cpu().numpy()
        item_embeddings = model.get_item_embeddings(item_features).cpu().numpy()

    embeddings_data = {
        'user_embeddings': user_embeddings,
        'item_embeddings': item_embeddings
    }

    # Evaluate the model
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    metrics_k5 = evaluate_model(model, test_data, test_user_ids=test_data['user_ids'], test_item_ids=test_data['item_ids'], k=5)
    metrics_k10 = evaluate_model(model, test_data, test_user_ids=test_data['user_ids'], test_item_ids=test_data['item_ids'], k=10)

    metrics_k5_candidate_sampling = evaluate_model_candidate_sampling(model, test_data, test_user_ids=test_data['user_ids'], test_item_ids=test_data['item_ids'], k=5)
    metrics_k10_candidate_sampling = evaluate_model_candidate_sampling(model, test_data, test_user_ids=test_data['user_ids'], test_item_ids=test_data['item_ids'], k=10)

    print(f"\nFinal Results:")
    print(f"Metrics@5: Precision={metrics_k5['precision']:.4f}, "
          f"Recall={metrics_k5['recall']:.4f}, NDCG={metrics_k5['ndcg']:.4f}")
    print(f"Metrics@10: Precision={metrics_k10['precision']:.4f}, "
          f"Recall={metrics_k10['recall']:.4f}, NDCG={metrics_k10['ndcg']:.4f}")

    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(history, embeddings_data, metrics_k5)
    visualize_results(history, embeddings_data, metrics_k5_candidate_sampling)

    print("\n" + "=" * 60)
    print("Training and evaluation completed!")
    print("The PyTorch two-tower model has been successfully trained and evaluated.")
    print("\nKey insights:")
    print("- The model learns separate embeddings for users and items using PyTorch")
    print("- Cosine similarity is used to measure user-item compatibility")
    print("- BCEWithLogitsLoss is used for binary classification")
    print("- Adam optimizer with learning rate scheduling")
    print("- Evaluation metrics show the model's recommendation performance")
    print("=" * 60)

if __name__ == "__main__":
    main()