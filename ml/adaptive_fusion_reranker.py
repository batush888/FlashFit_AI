#!/usr/bin/env python3
"""
Adaptive Fusion Reranker with Meta-Learning for Phase 2

This module implements:
1. Meta-learner for adaptive weight computation based on user feedback
2. Dynamic weight vector [w_clip, w_blip, w_fashion] per recommendation
3. Integration with existing fusion reranker architecture
4. Online learning with historical user feedback
5. Confidence-based weight adjustments
6. Performance tracking with NDCG, MAP metrics
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
from dataclasses import dataclass
import redis
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class UserFeedback:
    """Structure for user feedback data"""
    user_id: str
    item_id: str
    feedback_type: str  # 'like', 'dislike', 'click', 'purchase', 'view_time'
    feedback_value: float  # 1.0 for like, 0.0 for dislike, seconds for view_time
    timestamp: datetime
    context: Dict[str, Any]  # Additional context (style preferences, etc.)
    model_confidence: Dict[str, float]  # Confidence scores from each model
    recommendation_rank: int  # Position in recommendation list

@dataclass
class RecommendationContext:
    """Context for a recommendation request"""
    user_id: str
    query_image_embedding: torch.Tensor
    user_style_embedding: Optional[torch.Tensor]
    session_history: List[str]  # Recent item interactions
    time_of_day: str
    season: str
    occasion: str
    budget_range: Optional[Tuple[float, float]]

class MetaLearner(nn.Module):
    """
    Meta-learner for adaptive weight computation
    """
    
    def __init__(self, 
                 input_dim: int = 512,
                 hidden_dim: int = 128,
                 num_models: int = 3,  # CLIP, BLIP, Fashion Encoder
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.num_models = num_models
        
        # Input processing layers
        self.user_embedding_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.confidence_layer = nn.Sequential(
            nn.Linear(num_models, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.context_layer = nn.Sequential(
            nn.Linear(64, hidden_dim // 2),  # Context features
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention mechanism for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout_rate
        )
        
        # Weight prediction layers
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_models),
            nn.Softmax(dim=-1)  # Ensure weights sum to 1
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Uncertainty score [0, 1]
        )
        
        logger.info(f"MetaLearner initialized with {num_models} models")
    
    def forward(self, 
                user_embedding: torch.Tensor,
                model_confidences: torch.Tensor,
                context_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to predict adaptive weights
        
        Args:
            user_embedding: User preference embedding [batch_size, input_dim]
            model_confidences: Model confidence scores [batch_size, num_models]
            context_features: Context features [batch_size, 64]
            
        Returns:
            Dictionary with predicted weights and uncertainty
        """
        batch_size = user_embedding.size(0)
        
        # Process inputs
        user_features = self.user_embedding_layer(user_embedding)
        confidence_features = self.confidence_layer(model_confidences)
        context_features = self.context_layer(context_features)
        
        # Combine confidence and context features
        combined_features = torch.cat([confidence_features, context_features], dim=-1)
        
        # Apply attention between user features and combined features
        user_seq = user_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        combined_seq = combined_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        attended_user, _ = self.attention(
            query=user_seq, key=combined_seq, value=combined_seq
        )
        attended_combined, _ = self.attention(
            query=combined_seq, key=user_seq, value=user_seq
        )
        
        # Remove sequence dimension and concatenate
        attended_user = attended_user.squeeze(1)
        attended_combined = attended_combined.squeeze(1)
        final_features = torch.cat([attended_user, attended_combined], dim=-1)
        
        # Predict weights and uncertainty
        weights = self.weight_predictor(final_features)
        uncertainty = self.uncertainty_head(final_features)
        
        return {
            'weights': weights,
            'uncertainty': uncertainty,
            'features': final_features
        }

class UserEmbeddingManager:
    """
    Manages user embeddings with online learning
    """
    
    def __init__(self, 
                 embedding_dim: int = 512,
                 learning_rate: float = 0.01,
                 decay_factor: float = 0.95,
                 redis_client: Optional[redis.Redis] = None):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.redis_client = redis_client
        
        # In-memory storage for user embeddings
        self.user_embeddings: Dict[str, torch.Tensor] = {}
        self.user_feedback_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        logger.info(f"UserEmbeddingManager initialized with embedding_dim={embedding_dim}")
    
    def get_user_embedding(self, user_id: str) -> torch.Tensor:
        """
        Get user embedding, creating new one if doesn't exist
        
        Args:
            user_id: User identifier
            
        Returns:
            User embedding tensor
        """
        if user_id not in self.user_embeddings:
            # Initialize with small random values
            self.user_embeddings[user_id] = torch.randn(self.embedding_dim) * 0.1
            
            # Try to load from Redis if available
            if self.redis_client:
                try:
                    stored_embedding = self.redis_client.get(f"user_embedding:{user_id}")
                    if stored_embedding:
                        embedding_data = pickle.loads(stored_embedding)
                        self.user_embeddings[user_id] = torch.tensor(embedding_data)
                        logger.info(f"Loaded user embedding for {user_id} from Redis")
                except Exception as e:
                    logger.warning(f"Failed to load user embedding from Redis: {e}")
        
        return self.user_embeddings[user_id].clone()
    
    def update_user_embedding(self, 
                            user_id: str, 
                            feedback: UserFeedback,
                            item_embedding: torch.Tensor) -> None:
        """
        Update user embedding based on feedback
        
        Args:
            user_id: User identifier
            feedback: User feedback data
            item_embedding: Embedding of the item that received feedback
        """
        current_embedding = self.get_user_embedding(user_id)
        
        # Calculate update direction based on feedback
        if feedback.feedback_type in ['like', 'purchase']:
            # Move towards liked items
            update_direction = item_embedding - current_embedding
            update_strength = feedback.feedback_value * self.learning_rate
        elif feedback.feedback_type == 'dislike':
            # Move away from disliked items
            update_direction = current_embedding - item_embedding
            update_strength = self.learning_rate
        elif feedback.feedback_type == 'view_time':
            # Update based on engagement time (normalized)
            normalized_time = min(feedback.feedback_value / 30.0, 1.0)  # 30s = full engagement
            update_direction = item_embedding - current_embedding
            update_strength = normalized_time * self.learning_rate * 0.5
        else:
            return  # Unknown feedback type
        
        # Apply update with decay
        updated_embedding = (
            current_embedding * self.decay_factor + 
            update_direction * update_strength
        )
        
        # Normalize to prevent embedding drift
        updated_embedding = F.normalize(updated_embedding, dim=0)
        
        self.user_embeddings[user_id] = updated_embedding
        
        # Store feedback history
        self.user_feedback_history[user_id].append(feedback)
        
        # Save to Redis if available
        if self.redis_client:
            try:
                self.redis_client.set(
                    f"user_embedding:{user_id}",
                    pickle.dumps(updated_embedding.numpy()),
                    ex=86400 * 30  # 30 days expiration
                )
            except Exception as e:
                logger.warning(f"Failed to save user embedding to Redis: {e}")
        
        logger.debug(f"Updated embedding for user {user_id} based on {feedback.feedback_type}")
    
    def get_user_style_preferences(self, user_id: str) -> Dict[str, float]:
        """
        Extract style preferences from user feedback history
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of style preferences
        """
        feedback_history = self.user_feedback_history[user_id]
        
        if not feedback_history:
            return {}
        
        # Analyze feedback patterns
        style_scores = defaultdict(float)
        total_feedback = len(feedback_history)
        
        for feedback in feedback_history:
            if 'style' in feedback.context:
                style = feedback.context['style']
                if feedback.feedback_type in ['like', 'purchase']:
                    style_scores[style] += feedback.feedback_value
                elif feedback.feedback_type == 'dislike':
                    style_scores[style] -= 0.5
        
        # Normalize scores
        if style_scores:
            max_score = max(abs(score) for score in style_scores.values())
            if max_score > 0:
                style_scores = {k: v / max_score for k, v in style_scores.items()}
        
        return dict(style_scores)

class AdaptiveFusionReranker(nn.Module):
    """
    Adaptive fusion reranker with meta-learning
    """
    
    def __init__(self, 
                 embedding_dim: int = 512,
                 num_models: int = 3,
                 meta_learning_rate: float = 1e-4,
                 redis_client: Optional[redis.Redis] = None):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_models = num_models
        
        # Meta-learner for adaptive weights
        self.meta_learner = MetaLearner(
            input_dim=embedding_dim,
            num_models=num_models
        )
        
        # User embedding manager
        self.user_manager = UserEmbeddingManager(
            embedding_dim=embedding_dim,
            redis_client=redis_client
        )
        
        # Optimizer for meta-learner
        self.meta_optimizer = optim.Adam(
            self.meta_learner.parameters(),
            lr=meta_learning_rate
        )
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        
        # Default static weights (fallback)
        self.default_weights = torch.tensor([0.5, 0.2, 0.3])  # CLIP, BLIP, Fashion
        
        logger.info("AdaptiveFusionReranker initialized")
    
    def extract_context_features(self, context: RecommendationContext) -> torch.Tensor:
        """
        Extract numerical features from recommendation context
        
        Args:
            context: Recommendation context
            
        Returns:
            Context feature tensor [64 dimensions]
        """
        features = torch.zeros(64)
        
        # Time of day encoding (0-23 hours)
        if context.time_of_day:
            try:
                hour = int(context.time_of_day.split(':')[0])
                features[0] = hour / 23.0
                features[1] = np.sin(2 * np.pi * hour / 24)  # Cyclical encoding
                features[2] = np.cos(2 * np.pi * hour / 24)
            except:
                pass
        
        # Season encoding (one-hot)
        seasons = ['spring', 'summer', 'fall', 'winter']
        if context.season and context.season.lower() in seasons:
            season_idx = seasons.index(context.season.lower())
            features[3 + season_idx] = 1.0
        
        # Occasion encoding (one-hot)
        occasions = ['casual', 'formal', 'work', 'party', 'sport', 'travel']
        if context.occasion and context.occasion.lower() in occasions:
            occasion_idx = occasions.index(context.occasion.lower())
            features[7 + occasion_idx] = 1.0
        
        # Budget range encoding
        if context.budget_range:
            min_budget, max_budget = context.budget_range
            features[13] = min(min_budget / 1000.0, 1.0)  # Normalize to [0, 1]
            features[14] = min(max_budget / 1000.0, 1.0)
            features[15] = (max_budget - min_budget) / 1000.0  # Budget flexibility
        
        # Session history length
        features[16] = min(len(context.session_history) / 10.0, 1.0)
        
        # User style embedding (if available)
        if context.user_style_embedding is not None:
            # Use first 32 dimensions of style embedding
            style_dim = min(32, context.user_style_embedding.size(0))
            features[17:17+style_dim] = context.user_style_embedding[:style_dim]
        
        return features
    
    def compute_adaptive_weights(self, 
                               context: RecommendationContext,
                               model_confidences: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive weights for model fusion
        
        Args:
            context: Recommendation context
            model_confidences: Confidence scores from each model [num_models]
            
        Returns:
            Dictionary with weights and metadata
        """
        try:
            # Get user embedding
            user_embedding = self.user_manager.get_user_embedding(context.user_id)
            
            # Extract context features
            context_features = self.extract_context_features(context)
            
            # Prepare inputs for meta-learner (add batch dimension)
            user_emb_batch = user_embedding.unsqueeze(0)
            conf_batch = model_confidences.unsqueeze(0)
            context_batch = context_features.unsqueeze(0)
            
            # Predict adaptive weights
            with torch.no_grad():
                meta_output = self.meta_learner(
                    user_emb_batch, conf_batch, context_batch
                )
            
            weights = meta_output['weights'].squeeze(0)
            uncertainty = meta_output['uncertainty'].squeeze(0)
            
            # Apply uncertainty-based adjustment
            if uncertainty.item() > 0.7:  # High uncertainty
                # Fall back towards default weights
                weights = 0.7 * weights + 0.3 * self.default_weights
            
            return {
                'weights': weights,
                'uncertainty': uncertainty,
                'user_embedding': user_embedding,
                'context_features': context_features
            }
            
        except Exception as e:
            logger.warning(f"Failed to compute adaptive weights: {e}")
            # Fallback to default weights
            return {
                'weights': self.default_weights,
                'uncertainty': torch.tensor(1.0),
                'user_embedding': torch.zeros(self.embedding_dim),
                'context_features': torch.zeros(64)
            }
    
    def fuse_model_scores(self, 
                         clip_scores: torch.Tensor,
                         blip_scores: torch.Tensor,
                         fashion_scores: torch.Tensor,
                         weights: torch.Tensor) -> torch.Tensor:
        """
        Fuse model scores using adaptive weights
        
        Args:
            clip_scores: CLIP model scores [num_items]
            blip_scores: BLIP model scores [num_items]
            fashion_scores: Fashion encoder scores [num_items]
            weights: Adaptive weights [3]
            
        Returns:
            Fused scores [num_items]
        """
        # Normalize scores to [0, 1]
        clip_norm = torch.sigmoid(clip_scores)
        blip_norm = torch.sigmoid(blip_scores)
        fashion_norm = torch.sigmoid(fashion_scores)
        
        # Weighted fusion
        fused_scores = (
            weights[0] * clip_norm +
            weights[1] * blip_norm +
            weights[2] * fashion_norm
        )
        
        return fused_scores
    
    def rerank_recommendations(self, 
                             recommendations: List[Dict[str, Any]],
                             context: RecommendationContext) -> List[Dict[str, Any]]:
        """
        Rerank recommendations using adaptive fusion
        
        Args:
            recommendations: List of recommendation items with scores
            context: Recommendation context
            
        Returns:
            Reranked recommendations
        """
        if not recommendations:
            return recommendations
        
        # Extract model scores
        clip_scores = torch.tensor([item.get('clip_score', 0.0) for item in recommendations])
        blip_scores = torch.tensor([item.get('blip_score', 0.0) for item in recommendations])
        fashion_scores = torch.tensor([item.get('fashion_score', 0.0) for item in recommendations])
        
        # Calculate model confidences (based on score variance)
        model_confidences = torch.tensor([
            1.0 - torch.std(clip_scores).item(),
            1.0 - torch.std(blip_scores).item(),
            1.0 - torch.std(fashion_scores).item()
        ])
        model_confidences = torch.clamp(model_confidences, 0.1, 1.0)
        
        # Compute adaptive weights
        weight_info = self.compute_adaptive_weights(context, model_confidences)
        weights = weight_info['weights']
        
        # Fuse scores
        fused_scores = self.fuse_model_scores(
            clip_scores, blip_scores, fashion_scores, weights
        )
        
        # Add fused scores to recommendations
        for i, item in enumerate(recommendations):
            item['fused_score'] = fused_scores[i].item()
            item['adaptive_weights'] = weights.tolist()
            item['uncertainty'] = weight_info['uncertainty'].item()
        
        # Sort by fused score
        reranked = sorted(recommendations, key=lambda x: x['fused_score'], reverse=True)
        
        logger.debug(f"Reranked {len(recommendations)} items with weights {weights.tolist()}")
        
        return reranked
    
    def update_from_feedback(self, 
                           feedback: UserFeedback,
                           item_embedding: torch.Tensor,
                           predicted_weights: torch.Tensor) -> None:
        """
        Update model based on user feedback
        
        Args:
            feedback: User feedback data
            item_embedding: Embedding of the feedback item
            predicted_weights: Weights used for this recommendation
        """
        # Update user embedding
        self.user_manager.update_user_embedding(
            feedback.user_id, feedback, item_embedding
        )
        
        # Prepare training data for meta-learner
        if feedback.feedback_type in ['like', 'dislike', 'purchase']:
            # Create target weights based on feedback
            target_weights = predicted_weights.clone()
            
            if feedback.feedback_type in ['like', 'purchase']:
                # Increase weight of best performing model
                best_model_idx = torch.argmax(torch.tensor([
                    feedback.model_confidence.get('clip', 0.5),
                    feedback.model_confidence.get('blip', 0.5),
                    feedback.model_confidence.get('fashion', 0.5)
                ]))
                target_weights[best_model_idx] = min(target_weights[best_model_idx] + 0.1, 0.8)
            elif feedback.feedback_type == 'dislike':
                # Decrease weight of worst performing model
                worst_model_idx = torch.argmin(torch.tensor([
                    feedback.model_confidence.get('clip', 0.5),
                    feedback.model_confidence.get('blip', 0.5),
                    feedback.model_confidence.get('fashion', 0.5)
                ]))
                target_weights[worst_model_idx] = max(target_weights[worst_model_idx] - 0.1, 0.1)
            
            # Renormalize weights
            target_weights = target_weights / target_weights.sum()
            
            # Store for batch training
            self.performance_history[feedback.user_id].append({
                'predicted_weights': predicted_weights,
                'target_weights': target_weights,
                'feedback_value': feedback.feedback_value,
                'timestamp': feedback.timestamp
            })
        
        logger.debug(f"Updated model from feedback: {feedback.feedback_type} for user {feedback.user_id}")
    
    def train_meta_learner(self, batch_size: int = 32) -> float:
        """
        Train meta-learner on accumulated feedback
        
        Args:
            batch_size: Training batch size
            
        Returns:
            Training loss
        """
        # Collect training data from all users
        training_data = []
        for user_id, history in self.performance_history.items():
            if len(history) >= 5:  # Minimum feedback for training
                training_data.extend(history[-batch_size:])  # Recent feedback
        
        if len(training_data) < batch_size:
            return 0.0  # Not enough data
        
        # Sample batch
        batch_indices = np.random.choice(len(training_data), batch_size, replace=False)
        batch_data = [training_data[i] for i in batch_indices]
        
        # Prepare tensors
        predicted_weights = torch.stack([item['predicted_weights'] for item in batch_data])
        target_weights = torch.stack([item['target_weights'] for item in batch_data])
        
        # Calculate loss (MSE between predicted and target weights)
        loss = F.mse_loss(predicted_weights, target_weights)
        
        # Backward pass
        self.meta_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_learner.parameters(), max_norm=1.0)
        self.meta_optimizer.step()
        
        logger.debug(f"Meta-learner training loss: {loss.item():.4f}")
        
        return loss.item()

class RecommendationEvaluator:
    """
    Evaluator for recommendation performance metrics
    """
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        
        logger.info("RecommendationEvaluator initialized")
    
    def calculate_ndcg(self, 
                      predicted_scores: List[float],
                      true_relevance: List[float],
                      k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG)
        
        Args:
            predicted_scores: Predicted relevance scores
            true_relevance: True relevance scores
            k: Number of top items to consider
            
        Returns:
            NDCG@k score
        """
        if len(predicted_scores) != len(true_relevance):
            return 0.0
        
        # Convert to numpy arrays
        y_true = np.array([true_relevance])
        y_score = np.array([predicted_scores])
        
        try:
            ndcg = ndcg_score(y_true, y_score, k=k)
            return ndcg
        except:
            return 0.0
    
    def calculate_map(self, 
                     predicted_ranks: List[int],
                     true_relevance: List[float]) -> float:
        """
        Calculate Mean Average Precision (MAP)
        
        Args:
            predicted_ranks: Predicted item ranks
            true_relevance: True relevance scores
            
        Returns:
            MAP score
        """
        if not predicted_ranks or not true_relevance:
            return 0.0
        
        # Find relevant items (relevance > 0.5)
        relevant_items = [i for i, rel in enumerate(true_relevance) if rel > 0.5]
        
        if not relevant_items:
            return 0.0
        
        # Calculate average precision
        precision_sum = 0.0
        relevant_found = 0
        
        for rank, item_idx in enumerate(predicted_ranks):
            if item_idx in relevant_items:
                relevant_found += 1
                precision_at_k = relevant_found / (rank + 1)
                precision_sum += precision_at_k
        
        if relevant_found == 0:
            return 0.0
        
        map_score = precision_sum / len(relevant_items)
        return map_score
    
    def evaluate_recommendation_session(self, 
                                      recommendations: List[Dict[str, Any]],
                                      user_feedback: List[UserFeedback]) -> Dict[str, float]:
        """
        Evaluate a recommendation session
        
        Args:
            recommendations: List of recommended items
            user_feedback: List of user feedback for the session
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not recommendations or not user_feedback:
            return {}
        
        # Extract scores and create relevance labels
        predicted_scores = [item.get('fused_score', 0.0) for item in recommendations]
        
        # Create relevance labels based on feedback
        true_relevance = [0.0] * len(recommendations)
        feedback_map = {fb.item_id: fb for fb in user_feedback}
        
        for i, item in enumerate(recommendations):
            item_id = item.get('item_id', str(i))
            if item_id in feedback_map:
                feedback = feedback_map[item_id]
                if feedback.feedback_type == 'like':
                    true_relevance[i] = 1.0
                elif feedback.feedback_type == 'purchase':
                    true_relevance[i] = 1.0
                elif feedback.feedback_type == 'dislike':
                    true_relevance[i] = 0.0
                elif feedback.feedback_type == 'view_time':
                    # Normalize view time to [0, 1]
                    true_relevance[i] = min(feedback.feedback_value / 30.0, 1.0)
        
        # Calculate metrics
        ndcg_5 = self.calculate_ndcg(predicted_scores, true_relevance, k=5)
        ndcg_10 = self.calculate_ndcg(predicted_scores, true_relevance, k=10)
        
        predicted_ranks = list(range(len(recommendations)))
        map_score = self.calculate_map(predicted_ranks, true_relevance)
        
        # Calculate click-through rate
        clicks = sum(1 for rel in true_relevance if rel > 0.5)
        ctr = clicks / len(recommendations) if recommendations else 0.0
        
        metrics = {
            'ndcg_5': ndcg_5,
            'ndcg_10': ndcg_10,
            'map': map_score,
            'ctr': ctr,
            'total_recommendations': len(recommendations),
            'total_feedback': len(user_feedback)
        }
        
        # Store in history
        for metric, value in metrics.items():
            self.metrics_history[metric].append(value)
        
        return metrics


def create_sample_adaptive_reranker():
    """
    Create a sample adaptive fusion reranker for testing
    """
    # Initialize Redis client (optional)
    redis_client = None
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        redis_client.ping()
        logger.info("Connected to Redis for user embedding storage")
    except:
        logger.warning("Redis not available, using in-memory storage only")
    
    # Create adaptive reranker
    reranker = AdaptiveFusionReranker(
        embedding_dim=512,
        num_models=3,
        redis_client=redis_client
    )
    
    # Create evaluator
    evaluator = RecommendationEvaluator()
    
    logger.info("Sample adaptive fusion reranker created")
    logger.info(f"Meta-learner parameters: {sum(p.numel() for p in reranker.meta_learner.parameters()):,}")
    
    return reranker, evaluator


if __name__ == "__main__":
    # Create sample pipeline
    reranker, evaluator = create_sample_adaptive_reranker()
    
    logger.info("Adaptive Fusion Reranker ready for Phase 2")
    logger.info("Key features:")
    logger.info("- Meta-learner for adaptive weight computation")
    logger.info("- Dynamic weight vector [w_clip, w_blip, w_fashion]")
    logger.info("- Online learning with user feedback")
    logger.info("- Confidence-based weight adjustments")
    logger.info("- Performance tracking with NDCG, MAP metrics")
    logger.info("- Redis integration for user embedding persistence")