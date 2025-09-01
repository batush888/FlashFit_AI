#!/usr/bin/env python3
"""
BLIP+CLIP Fusion Training Script

This script provides comprehensive training for the BLIP+CLIP fusion model
with enhanced fashion vocabulary, improved text-image alignment, and
advanced fusion mechanisms.

Author: FlashFit AI Team
Date: 2024
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.blip_clip_fusion import (
    BLIPCLIPFusionModel,
    BLIPCLIPTrainer,
    FashionVocabularyExpander,
    FashionCaptionDataset
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BLIPCLIPFinetuner:
    """
    Comprehensive fine-tuning pipeline for BLIP+CLIP fusion model
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize components
        self.vocabulary_expander = FashionVocabularyExpander()
        self.model = None
        self.trainer = None
        
        logger.info(f"Initialized BLIPCLIPFinetuner on device: {self.device}")
    
    def prepare_datasets(self, data_dir: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare training, validation, and test datasets
        
        Args:
            data_dir: Directory containing image and caption data
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        logger.info("Preparing datasets...")
        
        # Define image transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load or create sample data
        if os.path.exists(os.path.join(data_dir, 'annotations.json')):
            image_paths, captions = self._load_real_data(data_dir)
        else:
            logger.warning("No real data found, creating sample dataset")
            image_paths, captions = self._create_sample_data(data_dir)
        
        # Create full dataset
        full_dataset = FashionCaptionDataset(
            image_paths=image_paths,
            captions=captions,
            vocabulary_expander=self.vocabulary_expander,
            transform=train_transform,
            enhance_captions=True
        )
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # Update validation dataset transform
        val_dataset.dataset.transform = val_transform
        test_dataset.dataset.transform = val_transform
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def _load_real_data(self, data_dir: str) -> Tuple[List[str], List[str]]:
        """
        Load real fashion dataset
        
        Args:
            data_dir: Directory containing the dataset
            
        Returns:
            Tuple of (image_paths, captions)
        """
        annotations_path = os.path.join(data_dir, 'annotations.json')
        
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        image_paths = []
        captions = []
        
        for item in annotations:
            image_path = os.path.join(data_dir, 'images', item['image'])
            if os.path.exists(image_path):
                image_paths.append(image_path)
                captions.append(item['caption'])
        
        logger.info(f"Loaded {len(image_paths)} real data samples")
        return image_paths, captions
    
    def _create_sample_data(self, data_dir: str) -> Tuple[List[str], List[str]]:
        """
        Create sample fashion data for testing
        
        Args:
            data_dir: Directory to save sample data
            
        Returns:
            Tuple of (image_paths, captions)
        """
        os.makedirs(data_dir, exist_ok=True)
        images_dir = os.path.join(data_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Sample fashion captions
        sample_captions = [
            "A stylish black leather jacket with silver zippers",
            "Elegant blue denim jeans with distressed details",
            "Comfortable white cotton t-shirt with minimalist design",
            "Trendy red high-heeled shoes with pointed toe",
            "Classic navy blue blazer with gold buttons",
            "Casual gray hoodie with kangaroo pocket",
            "Sophisticated black dress with lace details",
            "Modern white sneakers with mesh upper",
            "Vintage brown leather boots with lace-up closure",
            "Chic floral print skirt with A-line silhouette"
        ]
        
        image_paths = []
        captions = []
        
        for i, caption in enumerate(sample_captions):
            # Create a simple colored image as placeholder
            img = Image.new('RGB', (224, 224), color=(i*25 % 255, (i*50) % 255, (i*75) % 255))
            image_path = os.path.join(images_dir, f'sample_{i:03d}.jpg')
            img.save(image_path)
            
            image_paths.append(image_path)
            captions.append(caption)
        
        # Save annotations
        annotations = [
            {'image': f'sample_{i:03d}.jpg', 'caption': caption}
            for i, caption in enumerate(sample_captions)
        ]
        
        with open(os.path.join(data_dir, 'annotations.json'), 'w') as f:
            json.dump(annotations, f, indent=2)
        
        logger.info(f"Created {len(image_paths)} sample data items")
        return image_paths, captions
    
    def create_model_and_trainer(self) -> Tuple[BLIPCLIPFusionModel, BLIPCLIPTrainer]:
        """
        Create and initialize the fusion model and trainer
        
        Returns:
            Tuple of (model, trainer)
        """
        logger.info("Creating BLIP+CLIP fusion model...")
        
        # Create model
        model = BLIPCLIPFusionModel(
            blip_model_name=self.config['model']['blip_model_name'],
            clip_model_name=self.config['model']['clip_model_name'],
            fusion_dim=self.config['model']['fusion_dim'],
            dropout_rate=self.config['model']['dropout_rate']
        ).to(self.device)
        
        # Create trainer
        trainer = BLIPCLIPTrainer(
            model=model,
            device=self.device,
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        self.model = model
        self.trainer = trainer
        
        return model, trainer
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train the BLIP+CLIP fusion model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call create_model_and_trainer() first.")
        
        logger.info("Starting BLIP+CLIP fusion training...")
        
        # Create save directory
        save_dir = Path(self.config['paths']['model_save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / "blip_clip_fusion_best.pth"
        
        # Train the model
        history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.config['training']['epochs'],
            save_path=str(save_path)
        )
        
        logger.info("Training completed successfully")
        return history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the trained model
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call create_model_and_trainer() first.")
        
        logger.info("Evaluating model on test set...")
        
        # Load best model
        save_path = Path(self.config['paths']['model_save_dir']) / "blip_clip_fusion_best.pth"
        
        if save_path.exists():
            checkpoint = torch.load(save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        # Evaluate
        metrics = self.trainer.validate(test_loader)
        
        logger.info("Test Results:")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  Text-to-Image Retrieval: {metrics['retrieval_accuracy']['text_to_image']:.4f}")
        logger.info(f"  Image-to-Text Retrieval: {metrics['retrieval_accuracy']['image_to_text']:.4f}")
        logger.info(f"  Average Retrieval: {metrics['retrieval_accuracy']['average']:.4f}")
        
        return metrics
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            history: Training history dictionary
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot retrieval accuracy
        axes[1].plot(history['retrieval_accuracy'], label='Retrieval Accuracy', color='green')
        axes[1].set_title('Retrieval Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Main training function
    """
    parser = argparse.ArgumentParser(description='Train BLIP+CLIP Fusion Model')
    parser.add_argument('--config', type=str, default='ml/config/blip_clip_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data/fashion_captions',
                       help='Directory containing fashion image-caption data')
    parser.add_argument('--model_dir', type=str, default='models/blip_clip',
                       help='Directory to save trained models')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Default configuration
        config = {
            'model': {
                'blip_model_name': 'Salesforce/blip-image-captioning-base',
                'clip_model_name': 'openai/clip-vit-base-patch32',
                'fusion_dim': 512,
                'dropout_rate': 0.1
            },
            'training': {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'weight_decay': 1e-6,
                'num_workers': 4
            },
            'paths': {
                'data_dir': args.data_dir,
                'model_save_dir': args.model_dir
            },
            'device': args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        }
        logger.info("Using default configuration")
    
    # Override config with command line arguments
    config['training']['batch_size'] = args.batch_size
    config['training']['epochs'] = args.epochs
    config['training']['learning_rate'] = args.learning_rate
    config['paths']['data_dir'] = args.data_dir
    config['paths']['model_save_dir'] = args.model_dir
    if args.device != 'auto':
        config['device'] = args.device
    
    # Initialize finetuner
    finetuner = BLIPCLIPFinetuner(config)
    
    # Prepare datasets
    train_loader, val_loader, test_loader = finetuner.prepare_datasets(config['paths']['data_dir'])
    
    # Create model and trainer
    model, trainer = finetuner.create_model_and_trainer()
    
    # Train the model
    history = finetuner.train(train_loader, val_loader)
    
    # Evaluate on test set
    test_metrics = finetuner.evaluate(test_loader)
    
    # Plot training history
    plot_save_path = os.path.join(config['paths']['model_save_dir'], 'training_history.png')
    finetuner.plot_training_history(history, plot_save_path)
    
    # Save final results
    results = {
        'config': config,
        'training_history': history,
        'test_metrics': test_metrics,
        'vocabulary_coverage': finetuner.vocabulary_expander.calculate_vocabulary_coverage(
            [item['caption'] for item in train_loader.dataset.dataset.captions[:100]]
        )
    }
    
    results_path = os.path.join(config['paths']['model_save_dir'], 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Training results saved to {results_path}")
    logger.info("BLIP+CLIP fusion training completed successfully!")


if __name__ == "__main__":
    main()