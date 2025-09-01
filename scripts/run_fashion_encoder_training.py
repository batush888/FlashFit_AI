#!/usr/bin/env python3
"""
Script to run Fashion Encoder fine-tuning with configuration support
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml.train_fashion_encoder import FashionEncoderFinetuner
from utils.logging_config import get_logger

logger = get_logger(__name__)


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


def create_sample_dataset(data_dir: Path, num_samples: int = 1000) -> None:
    """
    Create a sample fashion dataset for testing
    
    Args:
        data_dir: Directory to create sample dataset
        num_samples: Number of sample images to create
    """
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    
    logger.info(f"Creating sample dataset with {num_samples} samples...")
    
    # Create directory structure
    categories = [
        'tops', 'bottoms', 'dresses', 'shoes', 'bags', 'accessories',
        'jackets', 'sweaters', 'skirts', 'pants', 'shirts', 'blouses',
        'coats', 'boots', 'sneakers', 'sandals', 'jewelry', 'hats',
        'scarves', 'belts'
    ]
    
    for category in categories:
        category_dir = data_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample images for each category
        samples_per_category = max(1, num_samples // len(categories))
        
        for i in range(samples_per_category):
            # Generate random image (224x224x3)
            np.random.seed(hash(f"{category}_{i}") % 2**32)
            
            # Create different patterns for different categories
            if 'top' in category or 'shirt' in category:
                # Solid colors for tops
                color = np.random.randint(0, 255, 3)
                image_array = np.full((224, 224, 3), color, dtype=np.uint8)
            elif 'bottom' in category or 'pant' in category:
                # Striped pattern for bottoms
                image_array = np.zeros((224, 224, 3), dtype=np.uint8)
                for y in range(224):
                    if (y // 10) % 2 == 0:
                        image_array[y, :, :] = [100, 100, 150]
                    else:
                        image_array[y, :, :] = [50, 50, 100]
            elif 'dress' in category:
                # Gradient for dresses
                image_array = np.zeros((224, 224, 3), dtype=np.uint8)
                for y in range(224):
                    intensity = int(255 * y / 224)
                    image_array[y, :, :] = [intensity, 100, 200]
            else:
                # Random noise for accessories
                image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Convert to PIL Image and save
            image = Image.fromarray(image_array)
            image_path = category_dir / f"{category}_{i:04d}.jpg"
            image.save(image_path)
    
    logger.info(f"Sample dataset created at: {data_dir}")
    logger.info(f"Categories: {categories}")


def run_training_with_config(config_path: str, 
                           data_dir: str = None,
                           create_sample_data: bool = False) -> Dict[str, float]:
    """
    Run fashion encoder training with configuration
    
    Args:
        config_path: Path to configuration file
        data_dir: Override data directory from config
        create_sample_data: Whether to create sample dataset
        
    Returns:
        Final evaluation metrics
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override data directory if provided
    if data_dir:
        config['paths']['data_dir'] = data_dir
    
    data_path = Path(config['paths']['data_dir'])
    
    # Create sample dataset if requested
    if create_sample_data:
        create_sample_dataset(data_path, num_samples=1000)
    
    # Check if data directory exists
    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_path}")
        logger.info("Use --create-sample-data to create a sample dataset")
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Initialize fine-tuner with config
    finetuner = FashionEncoderFinetuner(
        data_dir=str(data_path),
        model_save_dir=config['paths']['model_save_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        device=config['hardware']['device'] if config['hardware']['device'] != 'auto' else None
    )
    
    # Run training pipeline
    logger.info("Starting fashion encoder fine-tuning...")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Data directory: {data_path}")
    
    final_metrics = finetuner.run_full_pipeline(
        num_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    return final_metrics


def main():
    """
    Main function for running fashion encoder training
    """
    parser = argparse.ArgumentParser(
        description='Run Fashion Encoder Fine-tuning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='ml/config/fashion_encoder_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default=None,
        help='Override data directory from config'
    )
    
    parser.add_argument(
        '--create-sample-data', 
        action='store_true',
        help='Create sample dataset for testing'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Print configuration and exit without training'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    try:
        # Load and display configuration
        config = load_config(str(config_path))
        
        logger.info("=" * 60)
        logger.info("FASHION ENCODER FINE-TUNING")
        logger.info("=" * 60)
        logger.info(f"Configuration file: {config_path}")
        logger.info(f"Experiment: {config.get('experiment', {}).get('name', 'unnamed')}")
        logger.info(f"Description: {config.get('experiment', {}).get('description', 'N/A')}")
        
        if args.dry_run:
            logger.info("\nConfiguration:")
            print(yaml.dump(config, default_flow_style=False, indent=2))
            logger.info("Dry run completed.")
            return 0
        
        # Run training
        final_metrics = run_training_with_config(
            config_path=str(config_path),
            data_dir=args.data_dir,
            create_sample_data=args.create_sample_data
        )
        
        # Display final results
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("Final Evaluation Metrics:")
        
        for metric, value in final_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)