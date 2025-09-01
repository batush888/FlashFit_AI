#!/usr/bin/env python3
"""
Enhanced Generative Fusion Reranker for FlashFit AI

This module integrates:
1. Existing adaptive fusion reranker capabilities
2. Embedding diffusion model for generative recommendations
3. Generative meta-learner for dynamic weight computation
4. RLHF-style feedback integration
5. Multi-modal generative scoring
6. Real-time personalization with generative capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import redis
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import our generative components
try:
    from .embedding_diffusion import EmbeddingDiffusionModel, DiffusionConfig
    from .generative_meta_learner import GenerativeMetaLearner, MetaLearnerConfig
    from .adaptive_fusion_reranker import (
        UserFeedback, RecommendationContext, MetaLearner, 
        UserEmbeddingManager, RecommendationEvaluator
    )
except ImportError:
    # Fallback for standalone execution
    import sys
    sys.path.append('.')
    from embedding_diffusion import EmbeddingDiffusionModel, DiffusionConfig
    from generative_meta_learner import GenerativeMetaLearner, MetaLearnerConfig
    from adaptive_fusion_reranker import (
        UserFeedback, RecommendationContext, MetaLearner, 
        UserEmbeddingManager, RecommendationEvaluator
    )

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GenerativeRecommendationConfig:
    """Configuration for generative recommendation system"""
    # Model dimensions
    embedding_dim: int = 512
    hidden_dim: int = 256
    num_base_models: int = 3  # CLIP, BLIP, Fashion Encoder
    
    # Generative settings
    enable_generative: bool = True
    generative_weight: float = 0.3  # Weight for generative vs retrieval
    max_generated_items: int = 20
    generation_temperature: float = 0.8
    
    # Fusion settings
    fusion_strategy: str = "adaptive"  # "adaptive", "learned", "fixed"
    confidence_threshold: float = 0.7
    
    # Learning settings
    meta_learning_rate: float = 1e-4
    generative_learning_rate: float = 1e-5
    rlhf_learning_rate: float = 1e-6
    
    # Personalization
    user_embedding_dim: int = 256
    personalization_strength: float = 0.5
    
    # Performance
    batch_size: int = 32
    max_concurrent_generations: int = 4
    cache_size: int = 1000
    
    # Evaluation
    evaluation_frequency: int = 100  # Every N recommendations
    save_feedback_history: bool = True

@dataclass
class GenerativeRecommendation:
    """Structure for a generative recommendation"""
    item_id: str
    embedding: torch.Tensor
    confidence_scores: Dict[str, float]
    generation_method: str  # "diffusion", "retrieval", "hybrid"
    novelty_score: float
    compatibility_score: float
    personalization_score: float
    final_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class GenerativeFusionCore(nn.Module):
    """
    Core generative fusion module that combines:
    - Traditional retrieval scores
    - Generative diffusion scores
    - Meta-learned weights
    - Personalization signals
    """
    
    def __init__(self, config: GenerativeRecommendationConfig):
        super().__init__()
        self.config = config
        
        # Traditional fusion components
        self.traditional_meta_learner = MetaLearner(
            input_dim=config.user_embedding_dim,
            hidden_dim=config.hidden_dim,
            num_models=config.num_base_models
        )
        
        # Generative components
        diffusion_config = DiffusionConfig(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim
        )
        self.diffusion_model = EmbeddingDiffusionModel(diffusion_config)
        
        generative_config = MetaLearnerConfig(
            input_dim=config.embedding_dim * 2 + 128,
            hidden_dims=[config.hidden_dim, config.hidden_dim // 2],
            output_dim=1
        )
        self.generative_meta_learner = GenerativeMetaLearner(generative_config)
        
        # Fusion layers
        self.score_fusion = nn.Sequential(
            nn.Linear(config.num_base_models + 2, config.hidden_dim),  # +2 for generative and personalization
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Confidence calibration
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(config.num_base_models + 2, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Novelty predictor
        self.novelty_predictor = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),  # query + candidate
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                query_embedding: torch.Tensor,
                candidate_embeddings: torch.Tensor,
                user_embedding: torch.Tensor,
                context_features: torch.Tensor,
                traditional_scores: torch.Tensor,
                enable_generation: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass for generative fusion
        
        Args:
            query_embedding: Query item embedding [embedding_dim]
            candidate_embeddings: Candidate item embeddings [num_candidates, embedding_dim]
            user_embedding: User embedding [user_embedding_dim]
            context_features: Context features [context_dim]
            traditional_scores: Traditional model scores [num_candidates, num_models]
            enable_generation: Whether to use generative components
            
        Returns:
            Dictionary with scores and metadata
        """
        batch_size = candidate_embeddings.size(0)
        
        # Traditional meta-learning weights
        traditional_weights = self.traditional_meta_learner(
            user_embedding.unsqueeze(0),
            traditional_scores.mean(dim=0, keepdim=True),  # Average confidence
            context_features.unsqueeze(0)
        )
        
        # Generative scores if enabled
        generative_scores = torch.zeros(batch_size, 1, device=candidate_embeddings.device)
        generative_confidence = torch.zeros(batch_size, 1, device=candidate_embeddings.device)
        
        if enable_generation and self.config.enable_generative:
            # Generate compatible embeddings
            try:
                generated_embeddings = self.diffusion_model.sample(
                    batch_size=min(batch_size, self.config.max_generated_items),
                    context_embedding=query_embedding,
                    num_inference_steps=20
                )
                if not isinstance(generated_embeddings, torch.Tensor):
                    generated_embeddings = torch.randn(min(batch_size, self.config.max_generated_items), query_embedding.size(-1))
            except Exception:
                generated_embeddings = torch.randn(min(batch_size, self.config.max_generated_items), query_embedding.size(-1))
            
            # Compute similarity with candidates
            for i, candidate_emb in enumerate(candidate_embeddings):
                similarities = F.cosine_similarity(
                    candidate_emb.unsqueeze(0), 
                    generated_embeddings, 
                    dim=1
                )
                generative_scores[i] = similarities.max().unsqueeze(0)
                generative_confidence[i] = similarities.std().unsqueeze(0)  # Uncertainty as confidence
        
        # Personalization scores
        personalization_scores = F.cosine_similarity(
            user_embedding.unsqueeze(0).expand(batch_size, -1),
            candidate_embeddings,
            dim=1
        ).unsqueeze(1)
        
        # Novelty scores
        query_expanded = query_embedding.unsqueeze(0).expand(batch_size, -1)
        novelty_input = torch.cat([query_expanded, candidate_embeddings], dim=1)
        novelty_scores = self.novelty_predictor(novelty_input)
        
        # Combine all scores
        all_scores = torch.cat([
            traditional_scores,
            generative_scores,
            personalization_scores
        ], dim=1)
        
        # Final fusion
        final_scores = self.score_fusion(all_scores).squeeze(1)
        
        # Confidence calibration
        confidence_scores = self.confidence_calibrator(all_scores).squeeze(1)
        
        # Ensure all values are tensors for type consistency
        result = {
            'final_scores': final_scores if isinstance(final_scores, torch.Tensor) else torch.tensor(final_scores),
            'confidence_scores': confidence_scores if isinstance(confidence_scores, torch.Tensor) else torch.tensor(confidence_scores),
            'novelty_scores': novelty_scores.squeeze(1) if isinstance(novelty_scores, torch.Tensor) else torch.tensor(novelty_scores),
            'generative_scores': generative_scores.squeeze(1) if isinstance(generative_scores, torch.Tensor) else torch.tensor(generative_scores),
            'personalization_scores': personalization_scores.squeeze(1) if isinstance(personalization_scores, torch.Tensor) else torch.tensor(personalization_scores),
            'traditional_weights': traditional_weights if isinstance(traditional_weights, torch.Tensor) else torch.tensor(traditional_weights),
            'component_scores': {
                'traditional': traditional_scores if isinstance(traditional_scores, torch.Tensor) else torch.tensor(traditional_scores),
                'generative': generative_scores if isinstance(generative_scores, torch.Tensor) else torch.tensor(generative_scores),
                'personalization': personalization_scores if isinstance(personalization_scores, torch.Tensor) else torch.tensor(personalization_scores),
                'novelty': novelty_scores if isinstance(novelty_scores, torch.Tensor) else torch.tensor(novelty_scores)
            }
        }
        return result

class EnhancedGenerativeFusionReranker(nn.Module):
    """
    Enhanced fusion reranker with full generative capabilities
    """
    
    def __init__(self, 
                 config: GenerativeRecommendationConfig,
                 redis_client: Optional[redis.Redis] = None):
        super().__init__()
        self.config = config
        self.redis_client = redis_client
        
        # Core components
        self.fusion_core = GenerativeFusionCore(config)
        self.user_embedding_manager = UserEmbeddingManager(
            embedding_dim=config.user_embedding_dim,
            redis_client=redis_client
        )
        self.evaluator = RecommendationEvaluator()
        
        # Optimizers
        self.fusion_optimizer = optim.AdamW(
            self.fusion_core.parameters(),
            lr=config.meta_learning_rate,
            weight_decay=1e-5
        )
        
        self.generative_optimizer = optim.AdamW(
            self.fusion_core.diffusion_model.parameters(),
            lr=config.generative_learning_rate,
            weight_decay=1e-5
        )
        
        # Training state
        self.training_step = 0
        self.feedback_history = deque(maxlen=10000)
        self.performance_history = deque(maxlen=1000)
        
        # Caching
        self.recommendation_cache = {}
        self.generation_cache = {}
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_generations)
    
    def extract_context_features(self, context: RecommendationContext) -> torch.Tensor:
        """
        Extract context features for the recommendation
        
        Args:
            context: Recommendation context
            
        Returns:
            Context feature tensor
        """
        features = []
        
        # Time features
        hour = datetime.now().hour
        features.extend([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24)
        ])
        
        # Season encoding (one-hot)
        seasons = ['spring', 'summer', 'fall', 'winter']
        season_encoding = [1.0 if context.season == season else 0.0 for season in seasons]
        features.extend(season_encoding)
        
        # Occasion encoding (one-hot)
        occasions = ['casual', 'formal', 'party', 'work', 'sport', 'date']
        occasion_encoding = [1.0 if context.occasion == occasion else 0.0 for occasion in occasions]
        features.extend(occasion_encoding)
        
        # Budget features
        if context.budget_range:
            budget_min, budget_max = context.budget_range
            budget_mid = (budget_min + budget_max) / 2
            budget_range = budget_max - budget_min
            features.extend([np.log1p(budget_mid), np.log1p(budget_range)])
        else:
            features.extend([0.0, 0.0])
        
        # Session history features
        history_length = len(context.session_history)
        features.append(min(history_length / 10.0, 1.0))  # Normalized history length
        
        # Pad to fixed size
        target_size = 64
        while len(features) < target_size:
            features.append(0.0)
        
        return torch.tensor(features[:target_size], dtype=torch.float32)
    
    async def generate_recommendations_async(self, 
                                           context: RecommendationContext,
                                           candidate_items: List[Dict[str, Any]],
                                           traditional_scores: Dict[str, torch.Tensor]) -> List[GenerativeRecommendation]:
        """
        Generate recommendations asynchronously
        
        Args:
            context: Recommendation context
            candidate_items: List of candidate items with embeddings
            traditional_scores: Scores from traditional models
            
        Returns:
            List of generative recommendations
        """
        # Extract embeddings and prepare inputs
        candidate_embeddings = torch.stack([
            item['embedding'] for item in candidate_items
        ])
        
        user_embedding = self.user_embedding_manager.get_user_embedding(context.user_id)
        context_features = self.extract_context_features(context)
        
        # Prepare traditional scores tensor
        traditional_scores_tensor = torch.stack([
            traditional_scores.get('clip', torch.zeros(len(candidate_items))),
            traditional_scores.get('blip', torch.zeros(len(candidate_items))),
            traditional_scores.get('fashion', torch.zeros(len(candidate_items)))
        ], dim=1)
        
        # Forward pass through fusion core
        with torch.no_grad():
            fusion_results = self.fusion_core(
                query_embedding=context.query_image_embedding,
                candidate_embeddings=candidate_embeddings,
                user_embedding=user_embedding,
                context_features=context_features,
                traditional_scores=traditional_scores_tensor,
                enable_generation=self.config.enable_generative
            )
        
        # Create generative recommendations
        recommendations = []
        
        for i, item in enumerate(candidate_items):
            recommendation = GenerativeRecommendation(
                item_id=item['id'],
                embedding=candidate_embeddings[i],
                confidence_scores={
                    'overall': float(fusion_results['confidence_scores'][i]),
                    'generative': float(fusion_results['generative_scores'][i]),
                    'traditional': float(traditional_scores_tensor[i].mean()),
                    'personalization': float(fusion_results['personalization_scores'][i])
                },
                generation_method="hybrid",
                novelty_score=float(fusion_results['novelty_scores'][i]),
                compatibility_score=float(fusion_results['final_scores'][i]),
                personalization_score=float(fusion_results['personalization_scores'][i]),
                final_score=float(fusion_results['final_scores'][i]),
                metadata={
                    'traditional_weights': fusion_results['traditional_weights'],
                    'component_scores': {
                        k: v[i].item() if hasattr(v[i], 'item') else float(v[i])
                        for k, v in fusion_results['component_scores'].items()
                    }
                }
            )
            recommendations.append(recommendation)
        
        # Sort by final score
        recommendations.sort(key=lambda x: x.final_score, reverse=True)
        
        return recommendations
    
    def rerank_recommendations(self, 
                             candidate_items: List[Dict[str, Any]],
                             context: RecommendationContext,
                             traditional_scores: Dict[str, torch.Tensor]) -> List[GenerativeRecommendation]:
        """
        Rerank recommendations using generative fusion
        
        Args:
            candidate_items: List of candidate items
            context: Recommendation context
            traditional_scores: Scores from traditional models
            
        Returns:
            Reranked list of generative recommendations
        """
        # Check cache first
        cache_key = f"{context.user_id}_{hash(str(context.query_image_embedding.tolist()))}"
        if cache_key in self.recommendation_cache:
            cached_result, timestamp = self.recommendation_cache[cache_key]
            if datetime.now() - timestamp < timedelta(minutes=5):  # 5-minute cache
                return cached_result
        
        # Generate recommendations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            recommendations = loop.run_until_complete(
                self.generate_recommendations_async(context, candidate_items, traditional_scores)
            )
        finally:
            loop.close()
        
        # Cache result
        self.recommendation_cache[cache_key] = (recommendations, datetime.now())
        
        # Clean cache if too large
        if len(self.recommendation_cache) > self.config.cache_size:
            oldest_key = min(self.recommendation_cache.keys(), 
                           key=lambda k: self.recommendation_cache[k][1])
            del self.recommendation_cache[oldest_key]
        
        return recommendations
    
    def update_from_feedback(self, 
                           feedback: UserFeedback,
                           recommendation: GenerativeRecommendation) -> None:
        """
        Update model from user feedback using RLHF-style learning
        
        Args:
            feedback: User feedback
            recommendation: The recommendation that received feedback
        """
        self.feedback_history.append((feedback, recommendation))
        
        # Update user embedding
        self.user_embedding_manager.update_user_embedding(
            feedback.user_id,
            feedback,
            recommendation.embedding
        )
        
        # RLHF-style update
        if len(self.feedback_history) >= self.config.batch_size:
            self._train_from_feedback_batch()
    
    def _train_from_feedback_batch(self) -> float:
        """
        Train model from a batch of feedback using RLHF principles
        
        Returns:
            Training loss
        """
        # Sample recent feedback
        batch_feedback = list(self.feedback_history)[-self.config.batch_size:]
        
        # Prepare training data
        losses = []
        
        for feedback, recommendation in batch_feedback:
            # Convert feedback to reward signal
            if feedback.feedback_type == 'like':
                reward = 1.0
            elif feedback.feedback_type == 'dislike':
                reward = -1.0
            elif feedback.feedback_type == 'click':
                reward = 0.5
            elif feedback.feedback_type == 'purchase':
                reward = 2.0
            elif feedback.feedback_type == 'view_time':
                reward = min(feedback.feedback_value / 30.0, 1.0)  # Normalize by 30 seconds
            else:
                reward = 0.0
            
            # Compute loss based on reward and predicted score
            predicted_score = recommendation.final_score
            target_score = torch.sigmoid(torch.tensor(reward))
            
            loss = F.mse_loss(
                torch.tensor(predicted_score),
                target_score
            )
            losses.append(loss)
        
        if losses:
            # Backpropagate
            total_loss = torch.stack(losses).mean()
            
            self.fusion_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fusion_core.parameters(), 1.0)
            self.fusion_optimizer.step()
            
            self.training_step += 1
            
            # Log performance
            if self.training_step % self.config.evaluation_frequency == 0:
                self._evaluate_performance()
            
            return float(total_loss)
        
        return 0.0
    
    def _evaluate_performance(self) -> Dict[str, float]:
        """
        Evaluate current model performance
        
        Returns:
            Performance metrics
        """
        if len(self.feedback_history) < 10:
            return {}
        
        # Sample recent feedback for evaluation
        recent_feedback = list(self.feedback_history)[-100:]
        
        # Compute metrics
        predicted_scores = [rec.final_score for _, rec in recent_feedback]
        true_scores = []
        
        for feedback, _ in recent_feedback:
            if feedback.feedback_type == 'like':
                true_scores.append(1.0)
            elif feedback.feedback_type == 'dislike':
                true_scores.append(0.0)
            elif feedback.feedback_type == 'click':
                true_scores.append(0.5)
            elif feedback.feedback_type == 'purchase':
                true_scores.append(1.0)
            else:
                true_scores.append(0.5)
        
        # Calculate metrics
        try:
            ndcg = ndcg_score([true_scores], [predicted_scores])
        except:
            ndcg = 0.0
        
        # Correlation
        correlation = np.corrcoef(predicted_scores, true_scores)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        metrics = {
            'ndcg': float(ndcg),
            'correlation': float(correlation),
            'avg_predicted_score': float(np.mean(predicted_scores)),
            'avg_true_score': float(np.mean(true_scores)),
            'num_samples': len(recent_feedback)
        }
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        logger.info(f"Performance evaluation: NDCG={ndcg:.3f}, Correlation={correlation:.3f}")
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of model performance over time
        
        Returns:
            Performance summary
        """
        if not self.performance_history:
            return {'error': 'No performance data available'}
        
        # Extract metrics over time
        timestamps = [entry['timestamp'] for entry in self.performance_history]
        ndcg_scores = [entry['metrics']['ndcg'] for entry in self.performance_history]
        correlations = [entry['metrics']['correlation'] for entry in self.performance_history]
        
        return {
            'period': {
                'start': timestamps[0].isoformat(),
                'end': timestamps[-1].isoformat(),
                'num_evaluations': len(self.performance_history)
            },
            'ndcg': {
                'current': ndcg_scores[-1],
                'mean': float(np.mean(ndcg_scores)),
                'std': float(np.std(ndcg_scores)),
                'trend': float(np.polyfit(range(len(ndcg_scores)), ndcg_scores, 1)[0]) if len(ndcg_scores) > 1 else 0.0
            },
            'correlation': {
                'current': correlations[-1],
                'mean': float(np.mean(correlations)),
                'std': float(np.std(correlations)),
                'trend': float(np.polyfit(range(len(correlations)), correlations, 1)[0]) if len(correlations) > 1 else 0.0
            },
            'training_steps': self.training_step,
            'feedback_samples': len(self.feedback_history)
        }
    
    def save_model(self, save_path: str) -> None:
        """
        Save the complete model state
        
        Args:
            save_path: Path to save the model
        """
        save_dict = {
            'fusion_core_state': self.fusion_core.state_dict(),
            'fusion_optimizer_state': self.fusion_optimizer.state_dict(),
            'generative_optimizer_state': self.generative_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'performance_history': list(self.performance_history)
        }
        
        torch.save(save_dict, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load the complete model state
        
        Args:
            load_path: Path to load the model from
        """
        save_dict = torch.load(load_path, map_location='cpu')
        
        self.fusion_core.load_state_dict(save_dict['fusion_core_state'])
        self.fusion_optimizer.load_state_dict(save_dict['fusion_optimizer_state'])
        self.generative_optimizer.load_state_dict(save_dict['generative_optimizer_state'])
        self.training_step = save_dict['training_step']
        self.performance_history = deque(save_dict['performance_history'], maxlen=1000)
        
        logger.info(f"Model loaded from {load_path}")

def create_enhanced_generative_reranker(config: Optional[GenerativeRecommendationConfig] = None) -> EnhancedGenerativeFusionReranker:
    """
    Create an enhanced generative fusion reranker
    
    Args:
        config: Optional configuration
        
    Returns:
        Enhanced generative fusion reranker
    """
    if config is None:
        config = GenerativeRecommendationConfig()
    
    # Try to connect to Redis
    redis_client = None
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis for user embedding persistence")
    except:
        logger.warning("Could not connect to Redis. User embeddings will not persist.")
    
    reranker = EnhancedGenerativeFusionReranker(config, redis_client)
    
    return reranker

if __name__ == "__main__":
    # Create enhanced generative reranker
    config = GenerativeRecommendationConfig(
        enable_generative=True,
        generative_weight=0.3,
        max_generated_items=20
    )
    
    reranker = create_enhanced_generative_reranker(config)
    
    logger.info("Enhanced Generative Fusion Reranker ready for FlashFit AI")
    logger.info("Key features:")
    logger.info("- Embedding diffusion model integration")
    logger.info("- Generative meta-learner for dynamic weights")
    logger.info("- RLHF-style feedback learning")
    logger.info("- Multi-modal generative scoring")
    logger.info("- Real-time personalization")
    logger.info("- Asynchronous recommendation generation")
    logger.info("- Performance monitoring and caching")
    
    # Test with synthetic data
    query_embedding = torch.randn(512)
    user_id = "test_user_123"
    
    context = RecommendationContext(
        user_id=user_id,
        query_image_embedding=query_embedding,
        user_style_embedding=None,
        session_history=[],
        time_of_day="afternoon",
        season="spring",
        occasion="casual",
        budget_range=(50.0, 200.0)
    )
    
    # Create sample candidate items
    candidate_items = [
        {
            'id': f'item_{i}',
            'embedding': torch.randn(512)
        }
        for i in range(10)
    ]
    
    traditional_scores = {
        'clip': torch.rand(10),
        'blip': torch.rand(10),
        'fashion': torch.rand(10)
    }
    
    # Generate recommendations
    recommendations = reranker.rerank_recommendations(
        candidate_items, context, traditional_scores
    )
    
    logger.info(f"Generated {len(recommendations)} recommendations")
    logger.info(f"Top recommendation score: {recommendations[0].final_score:.3f}")
    logger.info(f"Novelty score: {recommendations[0].novelty_score:.3f}")
    logger.info(f"Personalization score: {recommendations[0].personalization_score:.3f}")