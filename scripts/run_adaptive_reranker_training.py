#!/usr/bin/env python3
"""
Adaptive Fusion Reranker Training Runner

This script provides a comprehensive interface for training and evaluating
the adaptive fusion reranker with meta-learning capabilities.

Features:
- Multiple training modes (full, quick test, evaluation only)
- Synthetic data generation and real data integration
- Comprehensive logging and monitoring
- Model evaluation and comparison
- Integration with existing FlashFit pipeline

Usage:
    python scripts/run_adaptive_reranker_training.py --mode train
    python scripts/run_adaptive_reranker_training.py --mode evaluate --model-path models/best_reranker.pth
    python scripts/run_adaptive_reranker_training.py --mode quick-test

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
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml.train_adaptive_reranker import AdaptiveRerankerTrainer, SyntheticFeedbackGenerator
from ml.adaptive_fusion_reranker import AdaptiveFusionReranker, UserFeedback, RecommendationContext

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaptiveRerankerRunner:
    """
    Main runner class for adaptive reranker training and evaluation
    """
    
    def __init__(self, config_path: str, output_dir: str):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.trainer = None
        self.reranker = None
        
        logger.info(f"AdaptiveRerankerRunner initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Configuration loaded from: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or create default
        
        Returns:
            Configuration dictionary
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        else:
            # Create default configuration
            config = self._create_default_config()
            logger.warning(f"Configuration file not found. Using defaults.")
        
        return config
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'model': {
                'embedding_dim': 512,
                'num_models': 3,
                'hidden_dim': 256,
                'num_attention_heads': 8,
                'dropout_rate': 0.1
            },
            'training': {
                'meta_learning_rate': 1e-4,
                'batch_size': 32,
                'epochs': 100,
                'validation_split': 0.2,
                'early_stopping': {
                    'patience': 15,
                    'min_delta': 1e-4
                }
            },
            'data': {
                'num_users': 100,
                'num_items': 1000,
                'sessions_per_user': 10
            },
            'evaluation': {
                'metrics': ['ndcg@5', 'ndcg@10', 'map', 'precision@5'],
                'eval_frequency': 5
            },
            'hardware': {
                'device': 'auto',
                'num_workers': 4
            },
            'paths': {
                'model_save_dir': 'models/adaptive_reranker',
                'results_dir': 'results/adaptive_reranker',
                'logs_dir': 'logs/adaptive_reranker'
            }
        }
    
    def _setup_logging(self):
        """
        Setup comprehensive logging
        """
        # Create logs directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup file handler
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Setup formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        logger.info(f"Logging setup complete. Log file: {log_file}")
    
    def create_sample_data(self, save_path: Optional[str] = None) -> List[UserFeedback]:
        """
        Create sample synthetic data for testing
        
        Args:
            save_path: Optional path to save the generated data
            
        Returns:
            List of synthetic feedback samples
        """
        logger.info("Creating synthetic feedback data...")
        
        # Initialize feedback generator
        generator = SyntheticFeedbackGenerator(
            num_users=self.config['data']['num_users'],
            num_items=self.config['data']['num_items']
        )
        
        # Generate feedback data
        feedback_data = generator.generate_training_dataset(
            num_sessions_per_user=self.config['data']['sessions_per_user']
        )
        
        logger.info(f"Generated {len(feedback_data)} feedback samples")
        
        # Save data if requested
        if save_path:
            data_dir = Path(save_path).parent
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            serializable_data = []
            for feedback in feedback_data:
                serializable_data.append({
                    'user_id': feedback.user_id,
                    'item_id': feedback.item_id,
                    'feedback_type': feedback.feedback_type,
                    'feedback_value': feedback.feedback_value,
                    'timestamp': feedback.timestamp.isoformat(),
                    'context': feedback.context,
                    'model_confidence': feedback.model_confidence,
                    'recommendation_rank': feedback.recommendation_rank
                })
            
            with open(save_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"Synthetic data saved to {save_path}")
        
        return feedback_data
    
    def train_model(self, mode: str = 'full') -> Dict[str, Any]:
        """
        Train the adaptive reranker model
        
        Args:
            mode: Training mode ('full', 'quick', 'debug')
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting model training in '{mode}' mode")
        
        # Adjust config based on mode
        if mode == 'quick':
            self.config['data']['num_users'] = 20
            self.config['data']['num_items'] = 100
            self.config['data']['sessions_per_user'] = 3
            self.config['training']['epochs'] = 10
            self.config['training']['batch_size'] = 8
            logger.info("Quick mode: Reduced dataset and training parameters")
        elif mode == 'debug':
            self.config['data']['num_users'] = 5
            self.config['data']['num_items'] = 20
            self.config['data']['sessions_per_user'] = 2
            self.config['training']['epochs'] = 3
            self.config['training']['batch_size'] = 2
            logger.info("Debug mode: Minimal dataset for testing")
        
        # Initialize trainer
        self.trainer = AdaptiveRerankerTrainer(self.config)
        
        # Setup model
        self.reranker = self.trainer.setup_reranker()
        
        # Generate or load training data
        data_path = self.output_dir / "synthetic_feedback.json"
        if data_path.exists() and mode == 'full':
            logger.info(f"Loading existing synthetic data from {data_path}")
            feedback_data = self._load_feedback_data(str(data_path))
        else:
            feedback_data = self.create_sample_data(str(data_path))
        
        # Split data
        split_idx = int(0.8 * len(feedback_data))
        train_feedback = feedback_data[:split_idx]
        test_feedback = feedback_data[split_idx:]
        
        logger.info(f"Training set: {len(train_feedback)} samples")
        logger.info(f"Test set: {len(test_feedback)} samples")
        
        # Train the model
        start_time = time.time()
        training_history = self.trainer.train_meta_learner(
            train_feedback, 
            self.config['training']['epochs']
        )
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate the model
        logger.info("Evaluating trained model...")
        test_results = self.trainer.evaluate_reranker(test_feedback)
        
        # Compile results
        results = {
            'config': self.config,
            'training_history': training_history,
            'test_results': test_results,
            'training_time': training_time,
            'mode': mode,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save model and results
        self._save_training_results(results)
        
        return results
    
    def evaluate_model(self, model_path: str, test_data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a pre-trained model
        
        Args:
            model_path: Path to the trained model
            test_data_path: Optional path to test data
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating model from {model_path}")
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint.get('config', self.config)
        
        # Initialize trainer and model
        self.trainer = AdaptiveRerankerTrainer(model_config)
        self.reranker = self.trainer.setup_reranker()
        self.reranker.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("Model loaded successfully")
        
        # Load or generate test data
        if test_data_path and os.path.exists(test_data_path):
            test_feedback = self._load_feedback_data(test_data_path)
            logger.info(f"Loaded test data from {test_data_path}")
        else:
            logger.info("Generating synthetic test data...")
            test_feedback = self.create_sample_data()
        
        # Evaluate
        start_time = time.time()
        evaluation_results = self.trainer.evaluate_reranker(test_feedback)
        evaluation_time = time.time() - start_time
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        
        # Compile results
        results = {
            'model_path': model_path,
            'evaluation_results': evaluation_results,
            'evaluation_time': evaluation_time,
            'test_samples': len(test_feedback),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save evaluation results
        eval_results_path = self.output_dir / "evaluation_results.json"
        with open(eval_results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {eval_results_path}")
        
        return results
    
    def compare_models(self, model_paths: List[str]) -> Dict[str, Any]:
        """
        Compare multiple trained models
        
        Args:
            model_paths: List of paths to trained models
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(model_paths)} models")
        
        # Generate common test data
        test_feedback = self.create_sample_data()
        logger.info(f"Using {len(test_feedback)} samples for comparison")
        
        comparison_results = {}
        
        for i, model_path in enumerate(model_paths):
            logger.info(f"Evaluating model {i+1}/{len(model_paths)}: {model_path}")
            
            try:
                # Load and evaluate model
                results = self.evaluate_model(model_path)
                model_name = Path(model_path).stem
                comparison_results[model_name] = results['evaluation_results']
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_path}: {e}")
                comparison_results[Path(model_path).stem] = {'error': str(e)}
        
        # Create comparison summary
        summary = self._create_comparison_summary(comparison_results)
        
        # Save comparison results
        comparison_path = self.output_dir / "model_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump({
                'comparison_results': comparison_results,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        logger.info(f"Model comparison saved to {comparison_path}")
        
        return {'results': comparison_results, 'summary': summary}
    
    def _load_feedback_data(self, data_path: str) -> List[UserFeedback]:
        """
        Load feedback data from JSON file
        
        Args:
            data_path: Path to the data file
            
        Returns:
            List of UserFeedback objects
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        feedback_list = []
        for item in data:
            feedback = UserFeedback(
                user_id=item['user_id'],
                item_id=item['item_id'],
                feedback_type=item['feedback_type'],
                feedback_value=item['feedback_value'],
                timestamp=datetime.fromisoformat(item['timestamp']),
                context=item['context'],
                model_confidence=item['model_confidence'],
                recommendation_rank=item['recommendation_rank']
            )
            feedback_list.append(feedback)
        
        return feedback_list
    
    def _save_training_results(self, results: Dict[str, Any]):
        """
        Save training results and artifacts
        
        Args:
            results: Training results dictionary
        """
        # Save model
        model_dir = self.output_dir / "models"
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "adaptive_reranker_best.pth"
        self.trainer.save_model(str(model_path))
        
        # Save results
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save configuration
        config_path = self.output_dir / "config_used.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Plot training history
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plot_path = plots_dir / "training_history.png"
        self.trainer.plot_training_history(str(plot_path))
        
        logger.info(f"Training artifacts saved:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Results: {results_path}")
        logger.info(f"  Config: {config_path}")
        logger.info(f"  Plots: {plot_path}")
    
    def _create_comparison_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create summary of model comparison
        
        Args:
            comparison_results: Results from model comparison
            
        Returns:
            Summary dictionary
        """
        summary = {
            'best_model': None,
            'metrics_summary': {},
            'rankings': {}
        }
        
        # Extract metrics
        metrics = ['ndcg_mean', 'map_mean', 'weight_diversity_mean']
        
        for metric in metrics:
            metric_values = {}
            for model_name, results in comparison_results.items():
                if 'error' not in results and metric in results:
                    metric_values[model_name] = results[metric]
            
            if metric_values:
                # Find best model for this metric
                best_model = max(metric_values.items(), key=lambda x: x[1])
                summary['metrics_summary'][metric] = {
                    'best_model': best_model[0],
                    'best_value': best_model[1],
                    'all_values': metric_values
                }
        
        # Overall ranking (based on NDCG)
        if 'ndcg_mean' in summary['metrics_summary']:
            summary['best_model'] = summary['metrics_summary']['ndcg_mean']['best_model']
        
        return summary
    
    def run_integration_test(self) -> Dict[str, Any]:
        """
        Run integration test with existing FlashFit components
        
        Returns:
            Integration test results
        """
        logger.info("Running integration test...")
        
        try:
            # Test model initialization
            self.trainer = AdaptiveRerankerTrainer(self.config)
            self.reranker = self.trainer.setup_reranker()
            
            # Test synthetic data generation
            feedback_data = self.create_sample_data()
            
            # Test training (minimal)
            mini_config = self.config.copy()
            mini_config['data']['num_users'] = 5
            mini_config['data']['num_items'] = 20
            mini_config['training']['epochs'] = 2
            
            mini_trainer = AdaptiveRerankerTrainer(mini_config)
            mini_reranker = mini_trainer.setup_reranker()
            
            # Quick training test
            mini_feedback = feedback_data[:10]
            history = mini_trainer.train_meta_learner(mini_feedback, 2)
            
            # Test evaluation
            eval_results = mini_trainer.evaluate_reranker(mini_feedback)
            
            results = {
                'status': 'success',
                'components_tested': [
                    'model_initialization',
                    'data_generation',
                    'training_pipeline',
                    'evaluation_pipeline'
                ],
                'training_history': history,
                'evaluation_results': eval_results,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Integration test completed successfully")
            
        except Exception as e:
            results = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"Integration test failed: {e}")
        
        # Save test results
        test_results_path = self.output_dir / "integration_test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results


def main():
    """
    Main function for running adaptive reranker training
    """
    parser = argparse.ArgumentParser(description='Adaptive Fusion Reranker Training Runner')
    
    # Main arguments
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'compare', 'create-data', 'integration-test', 'quick-test'],
                       help='Operation mode')
    
    # Configuration
    parser.add_argument('--config', type=str, 
                       default='ml/config/adaptive_reranker_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, 
                       default='results/adaptive_reranker',
                       help='Output directory for results')
    
    # Training arguments
    parser.add_argument('--training-mode', type=str, default='full',
                       choices=['full', 'quick', 'debug'],
                       help='Training mode')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    
    # Evaluation arguments
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model for evaluation')
    parser.add_argument('--test-data-path', type=str,
                       help='Path to test data')
    
    # Comparison arguments
    parser.add_argument('--model-paths', type=str, nargs='+',
                       help='Paths to models for comparison')
    
    # Data generation
    parser.add_argument('--data-output-path', type=str,
                       default='data/synthetic_feedback.json',
                       help='Path to save generated data')
    
    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = AdaptiveRerankerRunner(args.config, args.output_dir)
    
    # Override config with command line arguments
    if args.epochs:
        runner.config['training']['epochs'] = args.epochs
    if args.batch_size:
        runner.config['training']['batch_size'] = args.batch_size
    if args.device != 'auto':
        runner.config['hardware']['device'] = args.device
    
    # Execute based on mode
    try:
        if args.mode == 'train' or args.mode == 'quick-test':
            training_mode = 'quick' if args.mode == 'quick-test' else args.training_mode
            results = runner.train_model(mode=training_mode)
            
            logger.info("\n" + "="*60)
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Final NDCG: {results['test_results']['ndcg_mean']:.4f}")
            logger.info(f"Final MAP: {results['test_results']['map_mean']:.4f}")
            logger.info(f"Training Time: {results['training_time']:.2f}s")
            logger.info("="*60)
            
        elif args.mode == 'evaluate':
            if not args.model_path:
                raise ValueError("--model-path required for evaluation mode")
            
            results = runner.evaluate_model(args.model_path, args.test_data_path)
            
            logger.info("\n" + "="*60)
            logger.info("EVALUATION COMPLETED")
            logger.info("="*60)
            logger.info(f"NDCG: {results['evaluation_results']['ndcg_mean']:.4f}")
            logger.info(f"MAP: {results['evaluation_results']['map_mean']:.4f}")
            logger.info(f"Evaluation Time: {results['evaluation_time']:.2f}s")
            logger.info("="*60)
            
        elif args.mode == 'compare':
            if not args.model_paths:
                raise ValueError("--model-paths required for comparison mode")
            
            results = runner.compare_models(args.model_paths)
            
            logger.info("\n" + "="*60)
            logger.info("MODEL COMPARISON COMPLETED")
            logger.info("="*60)
            if results['summary']['best_model']:
                logger.info(f"Best Model: {results['summary']['best_model']}")
            logger.info("="*60)
            
        elif args.mode == 'create-data':
            feedback_data = runner.create_sample_data(args.data_output_path)
            
            logger.info("\n" + "="*60)
            logger.info("DATA GENERATION COMPLETED")
            logger.info("="*60)
            logger.info(f"Generated {len(feedback_data)} feedback samples")
            logger.info(f"Saved to: {args.data_output_path}")
            logger.info("="*60)
            
        elif args.mode == 'integration-test':
            results = runner.run_integration_test()
            
            logger.info("\n" + "="*60)
            logger.info("INTEGRATION TEST COMPLETED")
            logger.info("="*60)
            logger.info(f"Status: {results['status'].upper()}")
            if results['status'] == 'success':
                logger.info(f"Components Tested: {len(results['components_tested'])}")
            else:
                logger.error(f"Error: {results.get('error', 'Unknown error')}")
            logger.info("="*60)
            
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)
    
    logger.info(f"All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()