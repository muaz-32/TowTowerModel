import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
from model import TwoTowerModel
from utils import device

def train_model(model: TwoTowerModel, train_loader: DataLoader, val_loader: DataLoader, epochs: int, learning_rate: float = 0.001) -> Dict:
    """
    Train the two-tower model.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in train_pbar:
            user_features = batch['user_topic_embeddings'].to(device)
            item_features = batch['item_topic_embeddings'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            similarities, _, _ = model(user_features, item_features)
            loss = criterion(similarities, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            predictions = torch.sigmoid(similarities) > 0.5
            train_correct += (predictions == labels.bool()).sum().item()
            train_total += labels.size(0)

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * train_correct / train_total:.2f}%'
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for batch in val_pbar:
                user_features = batch['user_topic_embeddings'].to(device)
                item_features = batch['item_topic_embeddings'].to(device)
                labels = batch['labels'].to(device)

                similarities, _, _ = model(user_features, item_features)
                loss = criterion(similarities, labels)

                val_loss += loss.item()

                # Calculate accuracy
                predictions = torch.sigmoid(similarities) > 0.5
                val_correct += (predictions == labels.bool()).sum().item()
                val_total += labels.size(0)

                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.0 * val_correct / val_total:.2f}%'
                })

        scheduler.step()

        # Record metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print()

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }