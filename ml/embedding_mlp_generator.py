#!/usr/bin/env python3
"""
Embedding-to-Embedding MLP Generator
Implements Experiment 1: Train a small MLP to predict compatible item embeddings from query embeddings.

This is a fast baseline for generative fashion recommendation that learns to map
query embeddings to compatible target embeddings using MSE + cosine similarity loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import faiss
import json
import os
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeneratorConfig:
    """Configuration for the embedding generator."""
    embedding_dim: int = 512
    hidden_dims: List[int] = None
    dropout_rate: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 50
    validation_split: float = 0.2
    cosine_weight: float = 1.0
    mse_weight: float = 0.5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [self.embedding_dim * 2, self.embedding_dim * 2, self.embedding_dim]

class PairDataset(Dataset):
    """Dataset for query-target embedding pairs."""
    
    def __init__(self, query_embeddings: np.ndarray, target_embeddings: np.ndarray):
        """
        Args:
            query_embeddings: Query embeddings (N, embedding_dim)
            target_embeddings: Target compatible embeddings (N, embedding_dim)
        """
        assert len(query_embeddings) == len(target_embeddings), "Query and target must have same length"
        
        self.query_embeddings = query_embeddings.astype(np.float32)
        self.target_embeddings = target_embeddings.astype(np.float32)
        
        # Normalize embeddings
        self.query_embeddings = self._normalize(self.query_embeddings)
        self.target_embeddings = self._normalize(self.target_embeddings)
        
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return embeddings / norms
    
    def __len__(self) -> int:
        return len(self.query_embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.query_embeddings[idx]),
            torch.from_numpy(self.target_embeddings[idx])
        )

class GenNet(nn.Module):
    """Enhanced MLP generator for embedding-to-embedding mapping."""
    
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        
        layers = []
        input_dim = config.embedding_dim
        
        for hidden_dim in config.hidden_dims[:-1]:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Final layer without activation
        layers.append(nn.Linear(input_dim, config.hidden_dims[-1]))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Query embeddings (batch_size, embedding_dim)
            
        Returns:
            Generated target embeddings (batch_size, embedding_dim)
        """
        output = self.net(x)
        # L2 normalize output
        return nn.functional.normalize(output, p=2, dim=1)

class EmbeddingMLPGenerator:
    """Main class for training and using the embedding MLP generator."""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.model = GenNet(config).to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined MSE + cosine similarity loss."""
        mse_loss = nn.functional.mse_loss(pred, target)
        cosine_sim = nn.functional.cosine_similarity(pred, target, dim=1).mean()
        cosine_loss = 1 - cosine_sim  # Maximize cosine similarity
        
        total_loss = (
            self.config.mse_weight * mse_loss + 
            self.config.cosine_weight * cosine_loss
        )
        
        return total_loss, mse_loss, cosine_loss
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_cosine = 0
        
        for query, target in tqdm(dataloader, desc="Training"):
            query = query.to(self.config.device)
            target = target.to(self.config.device)
            
            self.optimizer.zero_grad()
            pred = self.model(query)
            
            loss, mse_loss, cosine_loss = self.compute_loss(pred, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_cosine += cosine_loss.item()
        
        num_batches = len(dataloader)
        return {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'cosine': total_cosine / num_batches
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_cosine = 0
        
        with torch.no_grad():
            for query, target in tqdm(dataloader, desc="Validation"):
                query = query.to(self.config.device)
                target = target.to(self.config.device)
                
                pred = self.model(query)
                loss, mse_loss, cosine_loss = self.compute_loss(pred, target)
                
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_cosine += cosine_loss.item()
        
        num_batches = len(dataloader)
        return {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'cosine': total_cosine / num_batches
        }
    
    def train(self, dataset: PairDataset) -> None:
        """Train the model."""
        # Split dataset
        val_size = int(len(dataset) * self.config.validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        logger.info(f"Training on {train_size} samples, validating on {val_size} samples")
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Track losses
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_model('best_embedding_generator.pth')
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"Train Cosine: {1-train_metrics['cosine']:.4f} - "
                f"Val Cosine: {1-val_metrics['cosine']:.4f}"
            )
    
    def generate_embeddings(self, query_embeddings: np.ndarray) -> np.ndarray:
        """Generate compatible embeddings for given queries."""
        self.model.eval()
        
        query_tensor = torch.from_numpy(query_embeddings.astype(np.float32))
        query_tensor = query_tensor.to(self.config.device)
        
        with torch.no_grad():
            generated = self.model(query_tensor)
        
        return generated.cpu().numpy()
    
    def evaluate_with_faiss(self, 
                           query_embeddings: np.ndarray, 
                           target_embeddings: np.ndarray,
                           item_database: np.ndarray,
                           k: int = 10) -> Dict[str, float]:
        """Evaluate generated embeddings using FAISS nearest neighbor search."""
        # Generate embeddings
        generated_embeddings = self.generate_embeddings(query_embeddings)
        
        # Build FAISS index
        index = faiss.IndexFlatIP(item_database.shape[1])  # Inner product for cosine similarity
        # Normalize database embeddings
        normalized_db = item_database / np.linalg.norm(item_database, axis=1, keepdims=True)
        index.add(normalized_db.astype(np.float32))
        
        # Search for nearest neighbors
        _, indices = index.search(generated_embeddings.astype(np.float32), k)
        
        # Compute hit rate (simplified evaluation)
        # In practice, you'd need ground truth compatible items
        hit_rate = self._compute_hit_rate(indices, target_embeddings, normalized_db)
        
        return {
            'hit_rate@10': hit_rate,
            'num_queries': len(query_embeddings)
        }
    
    def _compute_hit_rate(self, indices: np.ndarray, targets: np.ndarray, database: np.ndarray) -> float:
        """Compute hit rate by checking if generated embeddings are close to targets."""
        hits = 0
        for i, target in enumerate(targets):
            retrieved_items = database[indices[i]]
            # Check if any retrieved item is similar to target (cosine similarity > 0.8)
            similarities = np.dot(retrieved_items, target)
            if np.any(similarities > 0.8):
                hits += 1
        
        return hits / len(targets)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_curves(self, save_path: str = 'training_curves.png') -> None:
        """Plot training and validation curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Training curves saved to {save_path}")

def create_synthetic_data(num_samples: int = 10000, embedding_dim: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic query-target embedding pairs for testing."""
    # Generate random query embeddings
    query_embeddings = np.random.randn(num_samples, embedding_dim).astype(np.float32)
    
    # Generate correlated target embeddings (with some noise)
    noise = np.random.randn(num_samples, embedding_dim) * 0.3
    target_embeddings = query_embeddings + noise
    
    # Add some random transformations to make it more realistic
    rotation_matrix = np.random.randn(embedding_dim, embedding_dim)
    rotation_matrix, _ = np.linalg.qr(rotation_matrix)  # Orthogonal matrix
    target_embeddings = target_embeddings @ rotation_matrix.T
    
    return query_embeddings, target_embeddings

def main():
    """Main training script."""
    # Configuration
    config = GeneratorConfig(
        embedding_dim=512,
        hidden_dims=[1024, 1024, 512],
        dropout_rate=0.1,
        learning_rate=1e-4,
        batch_size=64,
        num_epochs=30,
        validation_split=0.2
    )
    
    logger.info(f"Using device: {config.device}")
    
    # Create synthetic data (replace with real data)
    logger.info("Creating synthetic data...")
    query_embeddings, target_embeddings = create_synthetic_data(10000, config.embedding_dim)
    
    # Create dataset
    dataset = PairDataset(query_embeddings, target_embeddings)
    
    # Initialize generator
    generator = EmbeddingMLPGenerator(config)
    
    # Train
    logger.info("Starting training...")
    generator.train(dataset)
    
    # Plot training curves
    generator.plot_training_curves()
    
    # Save final model
    generator.save_model('final_embedding_generator.pth')
    
    # Test generation
    logger.info("Testing generation...")
    test_queries = query_embeddings[:100]
    generated = generator.generate_embeddings(test_queries)
    
    # Compute similarity with targets
    targets = target_embeddings[:100]
    similarities = np.sum(generated * targets, axis=1)  # Cosine similarity (normalized)
    avg_similarity = np.mean(similarities)
    
    logger.info(f"Average cosine similarity with targets: {avg_similarity:.4f}")
    
    # Evaluate with FAISS (using synthetic database)
    database = np.random.randn(5000, config.embedding_dim).astype(np.float32)
    metrics = generator.evaluate_with_faiss(test_queries, targets, database)
    logger.info(f"FAISS evaluation metrics: {metrics}")

if __name__ == "__main__":
    main()