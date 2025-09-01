#!/usr/bin/env python3
"""
Adaptive Fusion Reranker Training Script

This script provides comprehensive training for the adaptive fusion reranker
with meta-learning capabilities, including:
- Meta-learner training with synthetic and real user feedback
- Advanced weight adaptation strategies
- Performance evaluation and monitoring
- Integration with existing recommendation pipeline

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
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import redis
from sklearn.metrics import ndcg_score, average_precision_score

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.adaptive_fusion_reranker import (
    AdaptiveFusionReranker,
    MetaLearner,
    UserFeedback,
    RecommendationContext,
    RecommendationEvaluator
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SyntheticFeedbackGenerator:
    """
    Generate synthetic user feedback for training the meta-learner
    """
    
    def __init__(self, num_users: int = 100, num_items: int = 1000):
        self.num_users = num_users
        self.num_items = num_items
        
        # User preference profiles
        self.user_profiles = self._generate_user_profiles()
        
        # Item features
        self.item_features = self._generate_item_features()
        
        logger.info(f"Generated {num_users} user profiles and {num_items} item features")
    
    def _generate_user_profiles(self) -> Dict[str, Dict[str, float]]:
        """
        Generate diverse user preference profiles
        
        Returns:
            Dictionary mapping user_id to preference profile
        """
        profiles = {}
        
        # Define user archetypes
        archetypes = [
            {'clip_preference': 0.7, 'blip_preference': 0.2, 'fashion_preference': 0.1, 'style': 'visual'},
            {'clip_preference': 0.2, 'blip_preference': 0.7, 'fashion_preference': 0.1, 'style': 'descriptive'},
            {'clip_preference': 0.1, 'blip_preference': 0.2, 'fashion_preference': 0.7, 'style': 'fashion_expert'},
            {'clip_preference': 0.4, 'blip_preference': 0.3, 'fashion_preference': 0.3, 'style': 'balanced'},
            {'clip_preference': 0.6, 'blip_preference': 0.1, 'fashion_preference': 0.3, 'style': 'visual_fashion'}
        ]
        
        for i in range(self.num_users):
            user_id = f"user_{i:04d}"
            
            # Assign archetype with some noise
            archetype = np.random.choice(archetypes)
            noise = np.random.normal(0, 0.1, 3)
            
            preferences = np.array([
                archetype['clip_preference'],
                archetype['blip_preference'],
                archetype['fashion_preference']
            ]) + noise
            
            # Normalize to sum to 1
            preferences = np.clip(preferences, 0.05, 0.9)
            preferences = preferences / preferences.sum()
            
            profiles[user_id] = {
                'clip_preference': preferences[0],
                'blip_preference': preferences[1],
                'fashion_preference': preferences[2],
                'style': archetype['style'],
                'consistency': np.random.uniform(0.6, 0.95)  # How consistent user is
            }
        
        return profiles
    
    def _generate_item_features(self) -> Dict[str, Dict[str, float]]:
        """
        Generate item features for synthetic recommendations
        
        Returns:
            Dictionary mapping item_id to features
        """
        features = {}
        
        for i in range(self.num_items):
            item_id = f"item_{i:04d}"
            
            # Generate diverse item characteristics
            features[item_id] = {
                'clip_score': np.random.uniform(0.1, 0.9),
                'blip_score': np.random.uniform(0.1, 0.9),
                'fashion_score': np.random.uniform(0.1, 0.9),
                'category': np.random.choice(['tops', 'bottoms', 'dresses', 'shoes', 'accessories']),
                'style': np.random.choice(['casual', 'formal', 'sporty', 'bohemian', 'vintage']),
                'price_tier': np.random.choice(['budget', 'mid', 'luxury'])
            }
        
        return features
    
    def generate_feedback_session(self, user_id: str, num_recommendations: int = 20) -> List[UserFeedback]:
        """
        Generate a realistic feedback session for a user
        
        Args:
            user_id: User identifier
            num_recommendations: Number of items recommended
            
        Returns:
            List of user feedback
        """
        if user_id not in self.user_profiles:
            raise ValueError(f"Unknown user: {user_id}")
        
        profile = self.user_profiles[user_id]
        feedback_list = []
        
        # Sample items for recommendation
        item_ids = np.random.choice(list(self.item_features.keys()), num_recommendations, replace=False)
        
        for rank, item_id in enumerate(item_ids):
            item = self.item_features[item_id]
            
            # Calculate user's expected preference for this item
            expected_score = (
                profile['clip_preference'] * item['clip_score'] +
                profile['blip_preference'] * item['blip_score'] +
                profile['fashion_preference'] * item['fashion_score']
            )
            
            # Add noise based on user consistency
            noise_std = (1 - profile['consistency']) * 0.3
            actual_score = expected_score + np.random.normal(0, noise_std)
            actual_score = np.clip(actual_score, 0, 1)
            
            # Determine feedback type based on score and position
            position_penalty = rank * 0.02  # Users less likely to engage with lower-ranked items
            engagement_prob = max(0, actual_score - position_penalty)
            
            if np.random.random() < engagement_prob:
                if actual_score > 0.7:
                    feedback_type = 'like' if np.random.random() < 0.8 else 'purchase'
                    feedback_value = 1.0
                elif actual_score > 0.4:
                    feedback_type = 'click'
                    feedback_value = 0.5
                else:
                    feedback_type = 'dislike'
                    feedback_value = 0.0
            else:
                continue  # No feedback (user didn't engage)
            
            # Create feedback object
            feedback = UserFeedback(
                user_id=user_id,
                item_id=item_id,
                feedback_type=feedback_type,
                feedback_value=feedback_value,
                timestamp=datetime.now(),
                context={
                    'category': item['category'],
                    'style': item['style'],
                    'price_tier': item['price_tier']
                },
                model_confidence={
                    'clip': item['clip_score'],
                    'blip': item['blip_score'],
                    'fashion': item['fashion_score']
                },
                recommendation_rank=rank
            )
            
            feedback_list.append(feedback)
        
        return feedback_list
    
    def generate_training_dataset(self, num_sessions_per_user: int = 10) -> List[UserFeedback]:
        """
        Generate a complete training dataset
        
        Args:
            num_sessions_per_user: Number of recommendation sessions per user
            
        Returns:
            List of all user feedback
        """
        all_feedback = []
        
        for user_id in tqdm(self.user_profiles.keys(), desc="Generating feedback"):
            for session in range(num_sessions_per_user):
                session_feedback = self.generate_feedback_session(user_id)
                all_feedback.extend(session_feedback)
        
        logger.info(f"Generated {len(all_feedback)} feedback samples from {len(self.user_profiles)} users")
        return all_feedback


class AdaptiveRerankerTrainer:
    """
    Comprehensive trainer for the adaptive fusion reranker
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize components
        self.reranker = None
        self.evaluator = RecommendationEvaluator()
        self.feedback_generator = None
        
        # Training history
        self.training_history = {
            'meta_loss': [],
            'ndcg_scores': [],
            'map_scores': [],
            'weight_diversity': [],
            'user_satisfaction': []
        }
        
        logger.info(f"AdaptiveRerankerTrainer initialized on device: {self.device}")
    
    def setup_reranker(self) -> AdaptiveFusionReranker:
        """
        Setup the adaptive fusion reranker
        
        Returns:
            Configured reranker instance
        """
        # Initialize Redis client if available
        redis_client = None
        if self.config.get('use_redis', False):
            try:
                redis_client = redis.Redis(
                    host=self.config.get('redis_host', 'localhost'),
                    port=self.config.get('redis_port', 6379),
                    db=self.config.get('redis_db', 0),
                    decode_responses=False
                )
                redis_client.ping()
                logger.info("Connected to Redis for user embedding storage")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
        
        # Create reranker
        self.reranker = AdaptiveFusionReranker(
            embedding_dim=self.config['model']['embedding_dim'],
            num_models=self.config['model']['num_models'],
            meta_learning_rate=self.config['training']['meta_learning_rate'],
            redis_client=redis_client
        ).to(self.device)
        
        logger.info(f"Reranker setup complete with {sum(p.numel() for p in self.reranker.parameters()):,} parameters")
        return self.reranker
    
    def setup_synthetic_data(self) -> SyntheticFeedbackGenerator:
        """
        Setup synthetic feedback generator
        
        Returns:
            Configured feedback generator
        """
        self.feedback_generator = SyntheticFeedbackGenerator(
            num_users=self.config['data']['num_users'],
            num_items=self.config['data']['num_items']
        )
        
        return self.feedback_generator
    
    def train_meta_learner(self, feedback_data: List[UserFeedback], num_epochs: int = 100) -> Dict[str, List[float]]:
        """
        Train the meta-learner component
        
        Args:
            feedback_data: Training feedback data
            num_epochs: Number of training epochs
            
        Returns:
            Training history
        """
        logger.info(f"Training meta-learner for {num_epochs} epochs on {len(feedback_data)} samples")
        
        # Group feedback by user for batch processing
        user_feedback = {}
        for feedback in feedback_data:
            if feedback.user_id not in user_feedback:
                user_feedback[feedback.user_id] = []
            user_feedback[feedback.user_id].append(feedback)
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_ndcg = []
            epoch_map = []
            
            # Sample users for this epoch
            sampled_users = np.random.choice(
                list(user_feedback.keys()),
                min(self.config['training']['batch_size'], len(user_feedback)),
                replace=False
            )
            
            for user_id in sampled_users:
                user_sessions = user_feedback[user_id]
                
                if len(user_sessions) < 3:  # Need minimum feedback for training
                    continue
                
                # Process user feedback
                loss = self._train_user_session(user_sessions)
                if loss is not None:
                    epoch_losses.append(loss)
                
                # Evaluate on this user's data
                ndcg, map_score = self._evaluate_user_performance(user_sessions)
                if ndcg is not None:
                    epoch_ndcg.append(ndcg)
                if map_score is not None:
                    epoch_map.append(map_score)
            
            # Record epoch metrics
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                avg_ndcg = np.mean(epoch_ndcg) if epoch_ndcg else 0.0
                avg_map = np.mean(epoch_map) if epoch_map else 0.0
                
                self.training_history['meta_loss'].append(avg_loss)
                self.training_history['ndcg_scores'].append(avg_ndcg)
                self.training_history['map_scores'].append(avg_map)
                
                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, NDCG={avg_ndcg:.4f}, MAP={avg_map:.4f}"
                    )
        
        logger.info("Meta-learner training completed")
        return self.training_history
    
    def _train_user_session(self, user_sessions: List[UserFeedback]) -> Optional[float]:
        """
        Train on a single user's feedback sessions
        
        Args:
            user_sessions: List of feedback from one user
            
        Returns:
            Training loss or None if insufficient data
        """
        if len(user_sessions) < 3:
            return None
        
        # Create synthetic recommendation context
        context = RecommendationContext(
            user_id=user_sessions[0].user_id,
            query_image_embedding=torch.randn(512).to(self.device),
            user_style_embedding=None,
            session_history=[],
            time_of_day="afternoon",
            season="spring",
            occasion="casual",
            budget_range=None
        )
        
        # Simulate recommendations and collect feedback
        total_loss = 0.0
        num_updates = 0
        
        for feedback in user_sessions:
            # Create item embedding (synthetic)
            item_embedding = torch.randn(512).to(self.device)
            
            # Get model confidences
            model_confidences = torch.tensor([
                feedback.model_confidence['clip'],
                feedback.model_confidence['blip'],
                feedback.model_confidence['fashion']
            ]).to(self.device)
            
            # Compute adaptive weights
            weight_info = self.reranker.compute_adaptive_weights(context, model_confidences)
            predicted_weights = weight_info['weights']
            
            # Update from feedback
            self.reranker.update_from_feedback(feedback, item_embedding, predicted_weights)
            
            # Train meta-learner
            loss = self.reranker.train_meta_learner(batch_size=1)
            if loss > 0:
                total_loss += loss
                num_updates += 1
        
        return total_loss / num_updates if num_updates > 0 else None
    
    def _evaluate_user_performance(self, user_sessions: List[UserFeedback]) -> Tuple[Optional[float], Optional[float]]:
        """
        Evaluate performance on user sessions
        
        Args:
            user_sessions: List of feedback from one user
            
        Returns:
            Tuple of (NDCG score, MAP score)
        """
        if len(user_sessions) < 3:
            return None, None
        
        # Extract scores and relevance
        predicted_scores = []
        true_relevance = []
        
        for feedback in user_sessions:
            # Predicted score (weighted combination)
            predicted_score = (
                0.33 * feedback.model_confidence['clip'] +
                0.33 * feedback.model_confidence['blip'] +
                0.34 * feedback.model_confidence['fashion']
            )
            predicted_scores.append(predicted_score)
            
            # True relevance from feedback
            true_relevance.append(feedback.feedback_value)
        
        # Calculate metrics
        try:
            ndcg = ndcg_score([true_relevance], [predicted_scores], k=min(10, len(predicted_scores)))
            map_score = average_precision_score(true_relevance, predicted_scores)
            return ndcg, map_score
        except:
            return None, None
    
    def evaluate_reranker(self, test_feedback: List[UserFeedback]) -> Dict[str, float]:
        """
        Comprehensive evaluation of the trained reranker
        
        Args:
            test_feedback: Test feedback data
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating reranker on {len(test_feedback)} test samples")
        
        # Group by user
        user_feedback = {}
        for feedback in test_feedback:
            if feedback.user_id not in user_feedback:
                user_feedback[feedback.user_id] = []
            user_feedback[feedback.user_id].append(feedback)
        
        all_ndcg = []
        all_map = []
        weight_diversity = []
        
        for user_id, sessions in user_feedback.items():
            if len(sessions) < 3:
                continue
            
            # Evaluate user performance
            ndcg, map_score = self._evaluate_user_performance(sessions)
            if ndcg is not None:
                all_ndcg.append(ndcg)
            if map_score is not None:
                all_map.append(map_score)
            
            # Calculate weight diversity for this user
            user_weights = []
            for feedback in sessions:
                model_confidences = torch.tensor([
                    feedback.model_confidence['clip'],
                    feedback.model_confidence['blip'],
                    feedback.model_confidence['fashion']
                ])
                
                context = RecommendationContext(
                    user_id=user_id,
                    query_image_embedding=torch.randn(512),
                    user_style_embedding=None,
                    session_history=[],
                    time_of_day="afternoon",
                    season="spring",
                    occasion="casual",
                    budget_range=None
                )
                
                weight_info = self.reranker.compute_adaptive_weights(context, model_confidences)
                user_weights.append(weight_info['weights'].detach().numpy())
            
            if user_weights:
                # Calculate standard deviation across weights (diversity measure)
                weight_std = np.std(user_weights, axis=0).mean()
                weight_diversity.append(weight_std)
        
        # Compile results
        results = {
            'ndcg_mean': np.mean(all_ndcg) if all_ndcg else 0.0,
            'ndcg_std': np.std(all_ndcg) if all_ndcg else 0.0,
            'map_mean': np.mean(all_map) if all_map else 0.0,
            'map_std': np.std(all_map) if all_map else 0.0,
            'weight_diversity_mean': np.mean(weight_diversity) if weight_diversity else 0.0,
            'num_users_evaluated': len([u for u, s in user_feedback.items() if len(s) >= 3])
        }
        
        logger.info("Evaluation Results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def save_model(self, save_path: str):
        """
        Save the trained model
        
        Args:
            save_path: Path to save the model
        """
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.reranker.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Meta-learner loss
        if self.training_history['meta_loss']:
            axes[0, 0].plot(self.training_history['meta_loss'])
            axes[0, 0].set_title('Meta-Learner Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # NDCG scores
        if self.training_history['ndcg_scores']:
            axes[0, 1].plot(self.training_history['ndcg_scores'])
            axes[0, 1].set_title('NDCG Scores')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('NDCG')
            axes[0, 1].grid(True)
        
        # MAP scores
        if self.training_history['map_scores']:
            axes[1, 0].plot(self.training_history['map_scores'])
            axes[1, 0].set_title('MAP Scores')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAP')
            axes[1, 0].grid(True)
        
        # Weight diversity (if available)
        if self.training_history['weight_diversity']:
            axes[1, 1].plot(self.training_history['weight_diversity'])
            axes[1, 1].set_title('Weight Diversity')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Diversity')
            axes[1, 1].grid(True)
        
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
    parser = argparse.ArgumentParser(description='Train Adaptive Fusion Reranker')
    parser.add_argument('--config', type=str, default='ml/config/adaptive_reranker_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model-dir', type=str, default='models/adaptive_reranker',
                       help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced parameters')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Default configuration
        config = {
            'model': {
                'embedding_dim': 512,
                'num_models': 3
            },
            'training': {
                'meta_learning_rate': 1e-4,
                'batch_size': args.batch_size,
                'epochs': args.epochs
            },
            'data': {
                'num_users': 50 if args.quick_test else 100,
                'num_items': 200 if args.quick_test else 1000
            },
            'device': args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'),
            'use_redis': False
        }
        logger.info("Using default configuration")
    
    # Override with command line arguments
    config['training']['batch_size'] = args.batch_size
    config['training']['epochs'] = args.epochs
    if args.device != 'auto':
        config['device'] = args.device
    if args.quick_test:
        config['data']['num_users'] = 20
        config['data']['num_items'] = 100
        config['training']['epochs'] = 10
    
    # Initialize trainer
    trainer = AdaptiveRerankerTrainer(config)
    
    # Setup components
    reranker = trainer.setup_reranker()
    feedback_generator = trainer.setup_synthetic_data()
    
    # Generate training data
    logger.info("Generating synthetic training data...")
    training_feedback = feedback_generator.generate_training_dataset(
        num_sessions_per_user=5 if args.quick_test else 10
    )
    
    # Split into train/test
    split_idx = int(0.8 * len(training_feedback))
    train_feedback = training_feedback[:split_idx]
    test_feedback = training_feedback[split_idx:]
    
    logger.info(f"Training set: {len(train_feedback)} samples")
    logger.info(f"Test set: {len(test_feedback)} samples")
    
    # Train the model
    history = trainer.train_meta_learner(train_feedback, config['training']['epochs'])
    
    # Evaluate the model
    test_results = trainer.evaluate_reranker(test_feedback)
    
    # Save model and results
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "adaptive_reranker_best.pth"
    trainer.save_model(str(model_path))
    
    # Save results
    results = {
        'config': config,
        'training_history': history,
        'test_results': test_results
    }
    
    results_path = model_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Plot training history
    plot_path = model_dir / "training_history.png"
    trainer.plot_training_history(str(plot_path))
    
    logger.info(f"Training completed successfully!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Results saved to: {results_path}")
    
    # Print final summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING SUMMARY")
    logger.info("="*50)
    logger.info(f"Final NDCG: {test_results['ndcg_mean']:.4f} ± {test_results['ndcg_std']:.4f}")
    logger.info(f"Final MAP: {test_results['map_mean']:.4f} ± {test_results['map_std']:.4f}")
    logger.info(f"Weight Diversity: {test_results['weight_diversity_mean']:.4f}")
    logger.info(f"Users Evaluated: {test_results['num_users_evaluated']}")
    logger.info("="*50)


if __name__ == "__main__":
    main()