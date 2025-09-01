#!/usr/bin/env python3
"""
BLIP+CLIP Fusion Training Runner

Comprehensive script for training and evaluating the enhanced BLIP+CLIP fusion model
with expanded fashion vocabulary and improved text-image alignment.

Usage:
    python scripts/run_blip_clip_training.py --config ml/config/blip_clip_config.yaml
    python scripts/run_blip_clip_training.py --quick-test  # For quick testing
    python scripts/run_blip_clip_training.py --evaluate-only --model-path models/blip_clip/best.pth

Author: FlashFit AI Team
Date: 2024
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml.train_blip_clip_fusion import BLIPCLIPFinetuner, load_config
from ml.blip_clip_fusion import (
    BLIPCLIPFusionModel,
    BLIPCLIPTrainer,
    FashionVocabularyExpander,
    FashionCaptionDataset
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BLIPCLIPTrainingRunner:
    """
    Comprehensive training runner for BLIP+CLIP fusion model
    """
    
    def __init__(self, config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None):
        self.config_path = config_path
        self.args = args
        self.config = self._load_configuration()
        self.finetuner = None
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        self._setup_logging()
        
        logger.info("BLIPCLIPTrainingRunner initialized")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """
        Load and merge configuration from file and command line arguments
        
        Returns:
            Merged configuration dictionary
        """
        # Load base configuration
        if self.config_path and os.path.exists(self.config_path):
            config = load_config(self.config_path)
            logger.info(f"Loaded configuration from {self.config_path}")
        else:
            config = self._get_default_config()
            logger.info("Using default configuration")
        
        # Override with command line arguments
        if self.args:
            self._override_config_with_args(config)
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'model': {
                'blip_model_name': 'Salesforce/blip-image-captioning-base',
                'clip_model_name': 'openai/clip-vit-base-patch32',
                'fusion_dim': 512,
                'dropout_rate': 0.1
            },
            'training': {
                'batch_size': 16,
                'epochs': 50,
                'learning_rate': 1e-5,
                'weight_decay': 1e-6,
                'num_workers': 4
            },
            'paths': {
                'data_dir': 'data/fashion_captions',
                'model_save_dir': 'models/blip_clip',
                'log_dir': 'logs/blip_clip'
            },
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'experiment': {
                'name': 'blip_clip_fusion_enhanced',
                'seed': 42
            }
        }
    
    def _override_config_with_args(self, config: Dict[str, Any]):
        """
        Override configuration with command line arguments
        
        Args:
            config: Configuration dictionary to modify
        """
        if hasattr(self.args, 'data_dir') and self.args.data_dir:
            config['paths']['data_dir'] = self.args.data_dir
        
        if hasattr(self.args, 'model_dir') and self.args.model_dir:
            config['paths']['model_save_dir'] = self.args.model_dir
        
        if hasattr(self.args, 'batch_size') and self.args.batch_size:
            config['training']['batch_size'] = self.args.batch_size
        
        if hasattr(self.args, 'epochs') and self.args.epochs:
            config['training']['epochs'] = self.args.epochs
        
        if hasattr(self.args, 'learning_rate') and self.args.learning_rate:
            config['training']['learning_rate'] = self.args.learning_rate
        
        if hasattr(self.args, 'device') and self.args.device != 'auto':
            config['device'] = self.args.device
        
        # Quick test mode
        if hasattr(self.args, 'quick_test') and self.args.quick_test:
            config['training']['epochs'] = 2
            config['training']['batch_size'] = 4
            logger.info("Quick test mode enabled - reduced epochs and batch size")
    
    def _setup_directories(self):
        """
        Setup required directories
        """
        directories = [
            self.config['paths']['model_save_dir'],
            self.config['paths']['log_dir'],
            self.config['paths'].get('results_dir', 'results/blip_clip'),
            self.config['paths'].get('plots_dir', 'plots/blip_clip')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """
        Setup enhanced logging
        """
        log_file = Path(self.config['paths']['log_dir']) / f"training_{int(time.time())}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    def create_sample_dataset(self, data_dir: str, num_samples: int = 100):
        """
        Create a sample fashion dataset for testing
        
        Args:
            data_dir: Directory to create the dataset
            num_samples: Number of samples to create
        """
        logger.info(f"Creating sample dataset with {num_samples} samples...")
        
        os.makedirs(data_dir, exist_ok=True)
        images_dir = os.path.join(data_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Extended fashion vocabulary for sample captions
        fashion_items = [
            "black leather jacket", "blue denim jeans", "white cotton t-shirt", "red high heels",
            "navy blazer", "gray hoodie", "black dress", "white sneakers", "brown boots",
            "floral skirt", "striped shirt", "wool sweater", "silk blouse", "cargo pants",
            "denim jacket", "maxi dress", "ankle boots", "running shoes", "winter coat",
            "summer dress", "formal suit", "casual shorts", "knit cardigan", "leather boots"
        ]
        
        colors = ["black", "white", "blue", "red", "green", "yellow", "purple", "pink", "brown", "gray"]
        styles = ["casual", "formal", "vintage", "modern", "elegant", "sporty", "bohemian", "classic"]
        materials = ["cotton", "leather", "denim", "silk", "wool", "polyester", "linen", "cashmere"]
        
        annotations = []
        
        for i in range(num_samples):
            # Generate diverse captions
            item = np.random.choice(fashion_items)
            color = np.random.choice(colors)
            style = np.random.choice(styles)
            material = np.random.choice(materials)
            
            # Create varied caption formats
            caption_formats = [
                f"A {style} {color} {item} made of {material}",
                f"{color.title()} {item} with {style} design",
                f"Trendy {color} {item} perfect for {style} occasions",
                f"Comfortable {material} {item} in {color} color",
                f"{style.title()} {color} {item} with premium {material}"
            ]
            
            caption = np.random.choice(caption_formats)
            
            # Create a colorful image as placeholder
            color_rgb = (
                (i * 37) % 255,
                (i * 73) % 255,
                (i * 109) % 255
            )
            
            img = Image.new('RGB', (224, 224), color=color_rgb)
            image_filename = f'sample_{i:04d}.jpg'
            image_path = os.path.join(images_dir, image_filename)
            img.save(image_path)
            
            annotations.append({
                'image': image_filename,
                'caption': caption,
                'metadata': {
                    'item': item,
                    'color': color,
                    'style': style,
                    'material': material
                }
            })
        
        # Save annotations
        annotations_path = os.path.join(data_dir, 'annotations.json')
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        logger.info(f"Created sample dataset with {len(annotations)} items")
        logger.info(f"Dataset saved to: {data_dir}")
    
    def run_training(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Returns:
            Training results dictionary
        """
        logger.info("Starting BLIP+CLIP fusion training pipeline...")
        
        # Set random seed for reproducibility
        if 'experiment' in self.config and 'seed' in self.config['experiment']:
            seed = self.config['experiment']['seed']
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            logger.info(f"Set random seed to {seed}")
        
        # Create sample dataset if needed
        data_dir = self.config['paths']['data_dir']
        if not os.path.exists(os.path.join(data_dir, 'annotations.json')):
            logger.info("No existing dataset found, creating sample dataset...")
            num_samples = 100 if not (hasattr(self.args, 'quick_test') and self.args.quick_test) else 20
            self.create_sample_dataset(data_dir, num_samples)
        
        # Initialize finetuner
        self.finetuner = BLIPCLIPFinetuner(self.config)
        
        # Prepare datasets
        train_loader, val_loader, test_loader = self.finetuner.prepare_datasets(data_dir)
        
        # Create model and trainer
        model, trainer = self.finetuner.create_model_and_trainer()
        
        # Log model information
        self._log_model_info(model)
        
        # Train the model
        start_time = time.time()
        history = self.finetuner.train(train_loader, val_loader)
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on test set
        test_metrics = self.finetuner.evaluate(test_loader)
        
        # Generate and save plots
        self._save_training_plots(history)
        
        # Prepare results
        results = {
            'config': self.config,
            'training_history': history,
            'test_metrics': test_metrics,
            'training_time': training_time,
            'model_info': self._get_model_info(model),
            'vocabulary_coverage': self._calculate_vocabulary_coverage(train_loader)
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def run_evaluation_only(self, model_path: str) -> Dict[str, Any]:
        """
        Run evaluation only on a pre-trained model
        
        Args:
            model_path: Path to the trained model
            
        Returns:
            Evaluation results
        """
        logger.info(f"Running evaluation on model: {model_path}")
        
        # Initialize finetuner
        self.finetuner = BLIPCLIPFinetuner(self.config)
        
        # Prepare test dataset
        data_dir = self.config['paths']['data_dir']
        if not os.path.exists(os.path.join(data_dir, 'annotations.json')):
            self.create_sample_dataset(data_dir, 50)
        
        _, _, test_loader = self.finetuner.prepare_datasets(data_dir)
        
        # Create model and load weights
        model, trainer = self.finetuner.create_model_and_trainer()
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.finetuner.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {model_path}")
        
        # Evaluate
        test_metrics = self.finetuner.evaluate(test_loader)
        
        results = {
            'model_path': model_path,
            'test_metrics': test_metrics,
            'model_info': self._get_model_info(model)
        }
        
        return results
    
    def _log_model_info(self, model: nn.Module):
        """
        Log detailed model information
        
        Args:
            model: The model to analyze
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info("Model Information:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
        logger.info(f"  Device: {next(model.parameters()).device}")
    
    def _get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get model information dictionary
        
        Args:
            model: The model to analyze
            
        Returns:
            Model information dictionary
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'device': str(next(model.parameters()).device)
        }
    
    def _calculate_vocabulary_coverage(self, train_loader: DataLoader) -> float:
        """
        Calculate vocabulary coverage on training data
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Vocabulary coverage percentage
        """
        # Sample some captions for coverage calculation
        sample_captions = []
        for i, batch in enumerate(train_loader):
            if i >= 10:  # Sample from first 10 batches
                break
            sample_captions.extend(batch['caption'])
        
        if sample_captions:
            coverage = self.finetuner.vocabulary_expander.calculate_vocabulary_coverage(sample_captions)
            logger.info(f"Fashion vocabulary coverage: {coverage:.2%}")
            return coverage
        
        return 0.0
    
    def _save_training_plots(self, history: Dict[str, List[float]]):
        """
        Save training plots
        
        Args:
            history: Training history dictionary
        """
        plots_dir = Path(self.config['paths'].get('plots_dir', 'plots/blip_clip'))
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = plots_dir / 'training_history.png'
        self.finetuner.plot_training_history(history, str(plot_path))
    
    def _save_results(self, results: Dict[str, Any]):
        """
        Save training results
        
        Args:
            results: Results dictionary to save
        """
        results_dir = Path(self.config['paths'].get('results_dir', 'results/blip_clip'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_path = results_dir / f"training_results_{int(time.time())}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_path}")
        
        # Save summary
        summary = {
            'experiment_name': self.config.get('experiment', {}).get('name', 'blip_clip_fusion'),
            'final_test_metrics': results['test_metrics'],
            'training_time': results['training_time'],
            'model_parameters': results['model_info']['total_parameters'],
            'vocabulary_coverage': results['vocabulary_coverage']
        }
        
        summary_path = results_dir / 'latest_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary saved to: {summary_path}")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='BLIP+CLIP Fusion Training Runner')
    parser.add_argument('--config', type=str, default='ml/config/blip_clip_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data/fashion_captions',
                       help='Directory containing fashion image-caption data')
    parser.add_argument('--model-dir', type=str, default='models/blip_clip',
                       help='Directory to save trained models')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced parameters')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only run evaluation on existing model')
    parser.add_argument('--model-path', type=str,
                       help='Path to model for evaluation-only mode')
    parser.add_argument('--create-sample-data', action='store_true',
                       help='Create sample dataset and exit')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples for sample dataset')
    
    args = parser.parse_args()
    
    # Create sample data only
    if args.create_sample_data:
        runner = BLIPCLIPTrainingRunner(args=args)
        runner.create_sample_dataset(args.data_dir, args.num_samples)
        logger.info("Sample dataset created successfully")
        return
    
    # Initialize runner
    runner = BLIPCLIPTrainingRunner(config_path=args.config, args=args)
    
    try:
        if args.evaluate_only:
            if not args.model_path:
                logger.error("--model-path is required for evaluation-only mode")
                return
            
            results = runner.run_evaluation_only(args.model_path)
            logger.info("Evaluation completed successfully")
            
        else:
            results = runner.run_training()
            logger.info("Training completed successfully")
        
        # Print final summary
        logger.info("\n" + "="*50)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("="*50)
        
        if 'test_metrics' in results:
            metrics = results['test_metrics']
            logger.info(f"Test Loss: {metrics['loss']:.4f}")
            logger.info(f"Text-to-Image Retrieval: {metrics['retrieval_accuracy']['text_to_image']:.4f}")
            logger.info(f"Image-to-Text Retrieval: {metrics['retrieval_accuracy']['image_to_text']:.4f}")
            logger.info(f"Average Retrieval: {metrics['retrieval_accuracy']['average']:.4f}")
        
        if 'training_time' in results:
            logger.info(f"Training Time: {results['training_time']:.2f} seconds")
        
        if 'vocabulary_coverage' in results:
            logger.info(f"Vocabulary Coverage: {results['vocabulary_coverage']:.2%}")
        
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()