#!/usr/bin/env python3
"""
Training script for the embedding generator model.
This script trains the MLP generator to produce compatible embeddings.
"""

import torch
import numpy as np
import argparse
import logging
from pathlib import Path
import json
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import matplotlib.pyplot as plt

from embedding_mlp_generator import EmbeddingMLPGenerator, GeneratorConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from file.
    Expected format: JSON with 'query_embeddings' and 'target_embeddings' arrays.
    """
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        query_embeddings = np.array(data['query_embeddings'], dtype=np.float32)
        target_embeddings = np.array(data['target_embeddings'], dtype=np.float32)
        
        logger.info(f"Loaded {len(query_embeddings)} training pairs")
        logger.info(f"Query embedding shape: {query_embeddings.shape}")
        logger.info(f"Target embedding shape: {target_embeddings.shape}")
        
        return query_embeddings, target_embeddings
        
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise

def generate_synthetic_data(num_samples: int = 1000, embedding_dim: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for testing purposes.
    """
    logger.info(f"Generating {num_samples} synthetic training pairs")
    
    # Generate random query embeddings
    query_embeddings = np.random.randn(num_samples, embedding_dim).astype(np.float32)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    
    # Generate compatible target embeddings with some correlation
    noise = np.random.randn(num_samples, embedding_dim).astype(np.float32) * 0.3
    target_embeddings = query_embeddings + noise
    target_embeddings = target_embeddings / np.linalg.norm(target_embeddings, axis=1, keepdims=True)
    
    return query_embeddings, target_embeddings

def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training and validation loss curves.
    """
    plt.figure(figsize=(12, 4))
    
    # Plot total loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot cosine loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_cosine_loss'], label='Training Cosine Loss')
    plt.plot(history['val_cosine_loss'], label='Validation Cosine Loss')
    plt.title('Cosine Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train embedding generator model')
    parser.add_argument('--data-path', type=str, help='Path to training data JSON file')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data for training')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of synthetic samples')
    parser.add_argument('--embedding-dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--hidden-dims', nargs='+', type=int, default=[1024, 512], help='Hidden layer dimensions')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--dropout-rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--cosine-weight', type=float, default=1.0, help='Weight for cosine loss')
    parser.add_argument('--mse-weight', type=float, default=0.5, help='Weight for MSE loss')
    parser.add_argument('--output-dir', type=str, default='./models', help='Output directory for trained model')
    parser.add_argument('--plot', action='store_true', help='Plot training history')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or generate training data
    if args.data_path:
        query_embeddings, target_embeddings = load_training_data(args.data_path)
    elif args.synthetic:
        query_embeddings, target_embeddings = generate_synthetic_data(
            args.num_samples, args.embedding_dim
        )
    else:
        logger.error("Either --data-path or --synthetic must be specified")
        return
    
    # Create configuration
    config = GeneratorConfig(
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        cosine_weight=args.cosine_weight,
        mse_weight=args.mse_weight
    )
    
    logger.info(f"Training configuration: {config}")
    
    # Initialize generator
    generator = EmbeddingMLPGenerator(config)
    
    # Train the model
    logger.info("Starting training...")
    history = generator.train(query_embeddings, target_embeddings)
    
    # Save the trained model
    model_path = output_dir / "embedding_generator.pth"
    generator.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'embedding_dim': config.embedding_dim,
            'hidden_dims': config.hidden_dims,
            'dropout_rate': config.dropout_rate,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'cosine_weight': config.cosine_weight,
            'mse_weight': config.mse_weight
        }, f, indent=2)
    logger.info(f"Configuration saved to {config_path}")
    
    # Plot training history if requested
    if args.plot:
        plot_path = output_dir / "training_history.png"
        plot_training_history(history, str(plot_path))
    
    # Print final metrics
    logger.info("Training completed!")
    logger.info(f"Final training loss: {history['train_loss'][-1]:.6f}")
    logger.info(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    logger.info(f"Best validation loss: {min(history['val_loss']):.6f}")

if __name__ == "__main__":
    main()