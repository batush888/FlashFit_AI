#!/usr/bin/env python3
"""
Fashion Encoder Fine-tuning Script
Implements multi-task learning with style attention mechanisms
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.enhanced_fashion_encoder import (
    EnhancedFashionEncoder, 
    FashionEncoderTrainer,
    FashionDataset
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


class FashionEncoderFinetuner:
    """
    Comprehensive fine-tuning pipeline for Enhanced Fashion Encoder
    """
    
    def __init__(self, 
                 data_dir: str,
                 model_save_dir: str = "models/fashion_encoder",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 device: Optional[str] = None):
        """
        Initialize the fine-tuning pipeline
        
        Args:
            data_dir: Directory containing fashion dataset
            model_save_dir: Directory to save trained models
            batch_size: Training batch size
            num_workers: Number of data loading workers
            device: Device to use for training
        """
        self.data_dir = Path(data_dir)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dataset statistics (will be computed from data)
        self.category_mapping = {}
        self.num_categories = 0
        
        logger.info(f"FashionEncoderFinetuner initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Model save directory: {self.model_save_dir}")
        logger.info(f"Device: {self.device}")
    
    def prepare_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare training, validation, and test datasets
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        logger.info("Preparing datasets...")
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        full_dataset = FashionDataset(
            data_dir=str(self.data_dir),
            transform=train_transform
        )
        
        # Get category mapping
        self.category_mapping = full_dataset.category_mapping
        self.num_categories = len(self.category_mapping)
        
        logger.info(f"Found {len(full_dataset)} samples across {self.num_categories} categories")
        logger.info(f"Categories: {list(self.category_mapping.keys())}")
        
        # Split dataset (80% train, 10% val, 10% test)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Update transforms for validation and test sets
        val_dataset.dataset.transform = val_test_transform
        test_dataset.dataset.transform = val_test_transform
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        logger.info(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def create_model_and_trainer(self, 
                                learning_rate: float = 1e-4,
                                weight_decay: float = 1e-5) -> Tuple[EnhancedFashionEncoder, FashionEncoderTrainer]:
        """
        Create model and trainer instances
        
        Args:
            learning_rate: Learning rate for training
            weight_decay: Weight decay for regularization
            
        Returns:
            Tuple of (model, trainer)
        """
        logger.info("Creating model and trainer...")
        
        # Initialize model
        model = EnhancedFashionEncoder(
            num_categories=self.num_categories,
            embedding_dim=512
        )
        
        # Initialize trainer
        trainer = FashionEncoderTrainer(
            model=model,
            device=self.device,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model, trainer
    
    def train_model(self,
                   trainer: FashionEncoderTrainer,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   num_epochs: int = 50,
                   early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """
        Train the fashion encoder model
        
        Args:
            trainer: FashionEncoderTrainer instance
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        # Define save path
        save_path = self.model_save_dir / "best_fashion_encoder.pth"
        
        # Train the model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            save_path=str(save_path)
        )
        
        # Save training history
        history_path = self.model_save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training completed. Best model saved to: {save_path}")
        logger.info(f"Training history saved to: {history_path}")
        
        return history
    
    def evaluate_model(self,
                      model: EnhancedFashionEncoder,
                      test_loader: DataLoader) -> Dict[str, float]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_embeddings = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['category_label'].to(self.device)
                
                outputs = model(images)
                
                # Get predictions
                _, predicted = torch.max(outputs['category_logits'], 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_embeddings.append(outputs['compatibility_embedding'].cpu())
        
        # Calculate classification metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # Calculate compatibility metrics
        all_embeddings = torch.cat(all_embeddings, dim=0)
        compatibility_metrics = self._calculate_compatibility_metrics(
            all_embeddings, torch.tensor(all_labels)
        )
        
        metrics = {
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1_score': f1,
            **compatibility_metrics
        }
        
        logger.info("Test Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def _calculate_compatibility_metrics(self, 
                                       embeddings: torch.Tensor, 
                                       labels: torch.Tensor) -> Dict[str, float]:
        """
        Calculate style compatibility metrics
        
        Args:
            embeddings: Fashion embeddings
            labels: Category labels
            
        Returns:
            Compatibility metrics
        """
        # Calculate cosine similarities
        similarities = torch.nn.functional.cosine_similarity(
            embeddings.unsqueeze(1), 
            embeddings.unsqueeze(0), 
            dim=2
        )
        
        # Create masks for same/different categories
        same_category_mask = (labels.unsqueeze(1) == labels.unsqueeze(0))
        different_category_mask = ~same_category_mask
        
        # Remove diagonal (self-similarity)
        mask = ~torch.eye(len(labels), dtype=torch.bool)
        same_category_mask = same_category_mask & mask
        different_category_mask = different_category_mask & mask
        
        # Calculate metrics
        same_category_sim = similarities[same_category_mask].mean().item()
        different_category_sim = similarities[different_category_mask].mean().item()
        
        # Compatibility score (higher is better)
        compatibility_score = same_category_sim - different_category_sim
        
        return {
            'compatibility_score': compatibility_score,
            'same_category_similarity': same_category_sim,
            'different_category_similarity': different_category_sim,
            'embedding_separation': compatibility_score
        }
    
    def plot_training_history(self, history: Dict[str, List[float]]) -> None:
        """
        Plot training history
        
        Args:
            history: Training history dictionary
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Fashion Encoder Training History', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Compatibility score plot (if available)
        if 'compatibility_score' in history and history['compatibility_score']:
            axes[1, 0].plot(history['compatibility_score'], label='Compatibility Score', color='green')
            axes[1, 0].set_title('Style Compatibility Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'Compatibility Score\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Learning rate plot (placeholder)
        axes[1, 1].text(0.5, 0.5, 'Learning Rate\nScheduling Info', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.model_save_dir / "training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to: {plot_path}")
        
        plt.show()
    
    def run_full_pipeline(self,
                         num_epochs: int = 50,
                         learning_rate: float = 1e-4,
                         early_stopping_patience: int = 10) -> Dict[str, float]:
        """
        Run the complete fine-tuning pipeline
        
        Args:
            num_epochs: Maximum number of training epochs
            learning_rate: Learning rate for training
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Final evaluation metrics
        """
        logger.info("Starting full fine-tuning pipeline...")
        
        try:
            # 1. Prepare datasets
            train_loader, val_loader, test_loader = self.prepare_datasets()
            
            # 2. Create model and trainer
            model, trainer = self.create_model_and_trainer(
                learning_rate=learning_rate
            )
            
            # 3. Train model
            history = self.train_model(
                trainer=trainer,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                early_stopping_patience=early_stopping_patience
            )
            
            # 4. Load best model for evaluation
            best_model_path = self.model_save_dir / "best_fashion_encoder.pth"
            checkpoint = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 5. Evaluate model
            metrics = self.evaluate_model(model, test_loader)
            
            # 6. Plot training history
            self.plot_training_history(history)
            
            # 7. Save final results
            results = {
                'training_history': history,
                'evaluation_metrics': metrics,
                'model_info': {
                    'num_categories': self.num_categories,
                    'category_mapping': self.category_mapping,
                    'embedding_dim': 512,
                    'device': self.device
                }
            }
            
            results_path = self.model_save_dir / "final_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Full pipeline completed successfully!")
            logger.info(f"Final results saved to: {results_path}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise


def main():
    """
    Main training script
    """
    parser = argparse.ArgumentParser(description='Fine-tune Enhanced Fashion Encoder')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing fashion dataset')
    parser.add_argument('--model_save_dir', type=str, default='models/fashion_encoder',
                       help='Directory to save trained models')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for training')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for training (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    finetuner = FashionEncoderFinetuner(
        data_dir=args.data_dir,
        model_save_dir=args.model_save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device
    )
    
    # Run full pipeline
    final_metrics = finetuner.run_full_pipeline(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience
    )
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()