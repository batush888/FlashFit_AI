#!/usr/bin/env python3
"""
Personalization Layer for Phase 2

This module implements:
1. Per-user embeddings updated via feedback loop
2. Real-time adaptation of user preferences
3. Integration with Redis for fast retrieval
4. FAISS integration for similarity search
5. Online learning with incremental updates
6. Style preference modeling
7. Contextual personalization (time, season, occasion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle
import redis
import faiss
from dataclasses import dataclass, asdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """Comprehensive user profile structure"""
    user_id: str
    embedding: np.ndarray
    style_preferences: Dict[str, float]
    color_preferences: Dict[str, float]
    brand_preferences: Dict[str, float]
    size_preferences: Dict[str, str]
    budget_range: Tuple[float, float]
    occasion_preferences: Dict[str, float]
    seasonal_preferences: Dict[str, float]
    interaction_count: int
    last_updated: datetime
    feedback_history: List[Dict[str, Any]]
    cluster_id: Optional[int] = None
    confidence_score: float = 0.0

@dataclass
class PersonalizationContext:
    """Context for personalized recommendations"""
    user_id: str
    current_season: str
    time_of_day: str
    occasion: str
    budget_constraint: Optional[Tuple[float, float]]
    recent_purchases: List[str]
    browsing_session: List[str]
    location: Optional[str] = None
    weather: Optional[str] = None

class UserEmbeddingNetwork(nn.Module):
    """
    Neural network for learning user embeddings
    """
    
    def __init__(self, 
                 item_embedding_dim: int = 512,
                 user_embedding_dim: int = 256,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.item_embedding_dim = item_embedding_dim
        self.user_embedding_dim = user_embedding_dim
        
        # Item encoder
        self.item_encoder = nn.Sequential(
            nn.Linear(item_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # User preference encoder
        layers = []
        input_dim = hidden_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Final projection to user embedding space
        layers.append(nn.Linear(hidden_dim, user_embedding_dim))
        
        self.user_encoder = nn.Sequential(*layers)
        
        # Attention mechanism for item importance
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout_rate
        )
        
        # Context integration
        self.context_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim // 2),  # Context features
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(user_embedding_dim + hidden_dim // 2, user_embedding_dim),
            nn.LayerNorm(user_embedding_dim),
            nn.ReLU(),
            nn.Linear(user_embedding_dim, user_embedding_dim)
        )
        
        logger.info(f"UserEmbeddingNetwork initialized with {user_embedding_dim}D embeddings")
    
    def forward(self, 
                item_embeddings: torch.Tensor,
                feedback_weights: torch.Tensor,
                context_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to generate user embedding
        
        Args:
            item_embeddings: Item embeddings [num_items, item_embedding_dim]
            feedback_weights: Feedback weights [num_items]
            context_features: Context features [64] (optional)
            
        Returns:
            User embedding [user_embedding_dim]
        """
        # Encode items
        encoded_items = self.item_encoder(item_embeddings)  # [num_items, hidden_dim]
        
        # Apply attention with feedback weights as importance scores
        item_seq = encoded_items.unsqueeze(1)  # [num_items, 1, hidden_dim]
        
        # Self-attention to find important items
        attended_items, attention_weights = self.attention(
            query=item_seq, key=item_seq, value=item_seq
        )
        attended_items = attended_items.squeeze(1)  # [num_items, hidden_dim]
        
        # Weight by feedback importance
        weighted_items = attended_items * feedback_weights.unsqueeze(1)
        
        # Aggregate to single representation
        user_representation = torch.mean(weighted_items, dim=0)  # [hidden_dim]
        
        # Generate user embedding
        user_embedding = self.user_encoder(user_representation)
        
        # Integrate context if available
        if context_features is not None:
            context_encoded = self.context_encoder(context_features)
            combined = torch.cat([user_embedding, context_encoded], dim=0)
            user_embedding = self.fusion_layer(combined)
        
        return user_embedding

class PersonalizationEngine:
    """
    Main personalization engine with Redis and FAISS integration
    """
    
    def __init__(self, 
                 embedding_dim: int = 256,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 1,
                 faiss_index_path: str = 'data/user_embeddings.index',
                 learning_rate: float = 0.01,
                 decay_factor: float = 0.95,
                 min_interactions: int = 5):
        
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.min_interactions = min_interactions
        self.faiss_index_path = faiss_index_path
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                db=redis_db, 
                decode_responses=False
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
            self.redis_client = None
        
        # Initialize FAISS index
        self.faiss_index = None
        self.user_id_to_faiss_id = {}
        self.faiss_id_to_user_id = {}
        self._initialize_faiss_index()
        
        # User embedding network
        self.embedding_network = UserEmbeddingNetwork(
            user_embedding_dim=embedding_dim
        )
        
        # Optimizer for online learning
        self.optimizer = optim.Adam(
            self.embedding_network.parameters(),
            lr=learning_rate
        )
        
        # In-memory cache for frequently accessed users
        self.user_cache = {}
        self.cache_size = 1000
        
        # User clustering for cold start
        self.user_clusters = None
        self.cluster_centroids = None
        
        # Performance tracking
        self.personalization_stats = {
            'total_users': 0,
            'active_users': 0,
            'cold_start_users': 0,
            'embedding_updates': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Background update thread
        self.update_thread = None
        self.update_queue = deque(maxlen=10000)
        self.stop_background_updates = False
        
        logger.info("PersonalizationEngine initialized")
    
    def _initialize_faiss_index(self):
        """
        Initialize or load FAISS index for user embeddings
        """
        try:
            if Path(self.faiss_index_path).exists():
                # Load existing index
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                
                # Load user ID mappings
                mapping_path = self.faiss_index_path.replace('.index', '_mapping.json')
                if Path(mapping_path).exists():
                    with open(mapping_path, 'r') as f:
                        mapping_data = json.load(f)
                        self.user_id_to_faiss_id = mapping_data['user_to_faiss']
                        self.faiss_id_to_user_id = mapping_data['faiss_to_user']
                
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} user embeddings")
            else:
                # Create new index
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
                logger.info("Created new FAISS index")
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
    
    def _save_faiss_index(self):
        """
        Save FAISS index and mappings to disk
        """
        try:
            # Ensure directory exists
            Path(self.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save index
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            
            # Save mappings
            mapping_path = self.faiss_index_path.replace('.index', '_mapping.json')
            mapping_data = {
                'user_to_faiss': self.user_id_to_faiss_id,
                'faiss_to_user': self.faiss_id_to_user_id
            }
            
            with open(mapping_path, 'w') as f:
                json.dump(mapping_data, f)
            
            logger.debug("FAISS index and mappings saved")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def _get_user_profile_key(self, user_id: str) -> str:
        """Get Redis key for user profile"""
        return f"user_profile:{user_id}"
    
    def _get_user_embedding_key(self, user_id: str) -> str:
        """Get Redis key for user embedding"""
        return f"user_embedding:{user_id}"
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get user profile from cache, Redis, or create new one
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile or None if not found
        """
        # Check in-memory cache first
        if user_id in self.user_cache:
            self.personalization_stats['cache_hits'] += 1
            return self.user_cache[user_id]
        
        self.personalization_stats['cache_misses'] += 1
        
        # Try Redis
        if self.redis_client:
            try:
                profile_data = self.redis_client.get(self._get_user_profile_key(user_id))
                if profile_data:
                    profile_dict = pickle.loads(profile_data)
                    profile = UserProfile(**profile_dict)
                    
                    # Add to cache
                    self._add_to_cache(user_id, profile)
                    
                    return profile
            except Exception as e:
                logger.warning(f"Failed to load user profile from Redis: {e}")
        
        # Create new user profile
        return self._create_new_user_profile(user_id)
    
    def _create_new_user_profile(self, user_id: str) -> UserProfile:
        """
        Create a new user profile with default values
        
        Args:
            user_id: User identifier
            
        Returns:
            New user profile
        """
        # Initialize with small random embedding
        embedding = np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        profile = UserProfile(
            user_id=user_id,
            embedding=embedding,
            style_preferences={},
            color_preferences={},
            brand_preferences={},
            size_preferences={},
            budget_range=(0.0, 1000.0),
            occasion_preferences={},
            seasonal_preferences={},
            interaction_count=0,
            last_updated=datetime.now(),
            feedback_history=[],
            confidence_score=0.0
        )
        
        # Apply cold start strategy
        self._apply_cold_start_strategy(profile)
        
        self.personalization_stats['cold_start_users'] += 1
        self.personalization_stats['total_users'] += 1
        
        logger.debug(f"Created new user profile for {user_id}")
        
        return profile
    
    def _apply_cold_start_strategy(self, profile: UserProfile):
        """
        Apply cold start strategy using user clustering
        
        Args:
            profile: User profile to initialize
        """
        if self.user_clusters is not None and self.cluster_centroids is not None:
            # Assign to nearest cluster based on random initialization
            distances = np.linalg.norm(
                self.cluster_centroids - profile.embedding.reshape(1, -1), 
                axis=1
            )
            cluster_id = np.argmin(distances)
            
            # Update embedding towards cluster centroid
            profile.embedding = (
                0.7 * profile.embedding + 
                0.3 * self.cluster_centroids[cluster_id]
            )
            profile.embedding = profile.embedding / np.linalg.norm(profile.embedding)
            profile.cluster_id = int(cluster_id)
            
            logger.debug(f"Applied cold start strategy: assigned to cluster {cluster_id}")
    
    def _add_to_cache(self, user_id: str, profile: UserProfile):
        """
        Add user profile to in-memory cache
        
        Args:
            user_id: User identifier
            profile: User profile
        """
        # Remove oldest entry if cache is full
        if len(self.user_cache) >= self.cache_size:
            oldest_user = next(iter(self.user_cache))
            del self.user_cache[oldest_user]
        
        self.user_cache[user_id] = profile
    
    def save_user_profile(self, profile: UserProfile):
        """
        Save user profile to Redis and update FAISS index
        
        Args:
            profile: User profile to save
        """
        # Update cache
        self._add_to_cache(profile.user_id, profile)
        
        # Save to Redis
        if self.redis_client:
            try:
                profile_dict = asdict(profile)
                # Convert numpy array to list for JSON serialization
                profile_dict['embedding'] = profile.embedding.tolist()
                
                self.redis_client.set(
                    self._get_user_profile_key(profile.user_id),
                    pickle.dumps(profile_dict),
                    ex=86400 * 30  # 30 days expiration
                )
                
                # Also save just the embedding for faster access
                self.redis_client.set(
                    self._get_user_embedding_key(profile.user_id),
                    pickle.dumps(profile.embedding),
                    ex=86400 * 30
                )
                
            except Exception as e:
                logger.warning(f"Failed to save user profile to Redis: {e}")
        
        # Update FAISS index
        self._update_faiss_index(profile.user_id, profile.embedding)
        
        # Update stats
        if profile.interaction_count >= self.min_interactions:
            self.personalization_stats['active_users'] += 1
    
    def _update_faiss_index(self, user_id: str, embedding: np.ndarray):
        """
        Update FAISS index with user embedding
        
        Args:
            user_id: User identifier
            embedding: User embedding
        """
        try:
            embedding_normalized = embedding / np.linalg.norm(embedding)
            embedding_normalized = embedding_normalized.reshape(1, -1).astype(np.float32)
            
            if user_id in self.user_id_to_faiss_id:
                # Update existing embedding (remove and re-add)
                faiss_id = self.user_id_to_faiss_id[user_id]
                # FAISS doesn't support direct updates, so we'll rebuild periodically
                pass
            else:
                # Add new embedding
                faiss_id = self.faiss_index.ntotal
                self.faiss_index.add(embedding_normalized)
                
                self.user_id_to_faiss_id[user_id] = faiss_id
                self.faiss_id_to_user_id[str(faiss_id)] = user_id
            
            # Save index periodically
            if self.faiss_index.ntotal % 100 == 0:
                self._save_faiss_index()
                
        except Exception as e:
            logger.error(f"Failed to update FAISS index: {e}")
    
    def find_similar_users(self, user_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar users using FAISS
        
        Args:
            user_id: Target user ID
            k: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        profile = self.get_user_profile(user_id)
        if not profile:
            return []
        
        try:
            # Search in FAISS index
            embedding = profile.embedding.reshape(1, -1).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            similarities, indices = self.faiss_index.search(embedding, k + 1)  # +1 to exclude self
            
            similar_users = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                
                similar_user_id = self.faiss_id_to_user_id.get(str(idx))
                if similar_user_id and similar_user_id != user_id:
                    similar_users.append((similar_user_id, float(similarity)))
            
            return similar_users[:k]
            
        except Exception as e:
            logger.error(f"Failed to find similar users: {e}")
            return []
    
    def extract_context_features(self, context: PersonalizationContext) -> np.ndarray:
        """
        Extract numerical features from personalization context
        
        Args:
            context: Personalization context
            
        Returns:
            Context feature array [64 dimensions]
        """
        features = np.zeros(64)
        
        # Time of day (0-23 hours)
        if context.time_of_day:
            try:
                hour = int(context.time_of_day.split(':')[0])
                features[0] = hour / 23.0
                features[1] = np.sin(2 * np.pi * hour / 24)
                features[2] = np.cos(2 * np.pi * hour / 24)
            except:
                pass
        
        # Season encoding
        seasons = ['spring', 'summer', 'fall', 'winter']
        if context.current_season.lower() in seasons:
            season_idx = seasons.index(context.current_season.lower())
            features[3 + season_idx] = 1.0
        
        # Occasion encoding
        occasions = ['casual', 'formal', 'work', 'party', 'sport', 'travel']
        if context.occasion.lower() in occasions:
            occasion_idx = occasions.index(context.occasion.lower())
            features[7 + occasion_idx] = 1.0
        
        # Budget constraint
        if context.budget_constraint:
            min_budget, max_budget = context.budget_constraint
            features[13] = min(min_budget / 1000.0, 1.0)
            features[14] = min(max_budget / 1000.0, 1.0)
            features[15] = (max_budget - min_budget) / 1000.0
        
        # Recent activity
        features[16] = min(len(context.recent_purchases) / 10.0, 1.0)
        features[17] = min(len(context.browsing_session) / 20.0, 1.0)
        
        # Location and weather (if available)
        if context.location:
            # Simple hash-based encoding for location
            location_hash = hash(context.location) % 100
            features[18] = location_hash / 100.0
        
        if context.weather:
            weather_types = ['sunny', 'rainy', 'cloudy', 'snowy', 'windy']
            if context.weather.lower() in weather_types:
                weather_idx = weather_types.index(context.weather.lower())
                features[19 + weather_idx] = 1.0
        
        return features
    
    def update_user_from_feedback(self, 
                                user_id: str,
                                item_embedding: np.ndarray,
                                feedback_type: str,
                                feedback_value: float,
                                context: Optional[PersonalizationContext] = None,
                                item_metadata: Optional[Dict[str, Any]] = None) -> UserProfile:
        """
        Update user profile based on feedback
        
        Args:
            user_id: User identifier
            item_embedding: Embedding of the item that received feedback
            feedback_type: Type of feedback ('like', 'dislike', 'purchase', etc.)
            feedback_value: Numerical feedback value
            context: Personalization context
            item_metadata: Additional item information
            
        Returns:
            Updated user profile
        """
        profile = self.get_user_profile(user_id)
        
        # Update interaction count
        profile.interaction_count += 1
        profile.last_updated = datetime.now()
        
        # Calculate learning rate based on confidence and interaction count
        base_lr = self.learning_rate
        confidence_factor = max(0.1, 1.0 - profile.confidence_score)
        interaction_factor = min(1.0, profile.interaction_count / 50.0)
        adaptive_lr = base_lr * confidence_factor * interaction_factor
        
        # Update embedding based on feedback
        if feedback_type in ['like', 'purchase', 'add_to_cart']:
            # Move towards liked items
            direction = item_embedding - profile.embedding
            update_strength = feedback_value * adaptive_lr
        elif feedback_type == 'dislike':
            # Move away from disliked items
            direction = profile.embedding - item_embedding
            update_strength = adaptive_lr
        elif feedback_type == 'view_time':
            # Update based on engagement time
            normalized_time = min(feedback_value / 30.0, 1.0)
            direction = item_embedding - profile.embedding
            update_strength = normalized_time * adaptive_lr * 0.5
        else:
            direction = np.zeros_like(profile.embedding)
            update_strength = 0.0
        
        # Apply update with momentum
        profile.embedding = (
            profile.embedding * self.decay_factor + 
            direction * update_strength
        )
        
        # Normalize embedding
        profile.embedding = profile.embedding / np.linalg.norm(profile.embedding)
        
        # Update preferences based on item metadata
        if item_metadata:
            self._update_preferences(profile, item_metadata, feedback_type, feedback_value)
        
        # Update confidence score
        profile.confidence_score = min(
            1.0, 
            profile.confidence_score + 0.01 * feedback_value
        )
        
        # Add to feedback history
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'feedback_type': feedback_type,
            'feedback_value': feedback_value,
            'item_metadata': item_metadata,
            'context': asdict(context) if context else None
        }
        profile.feedback_history.append(feedback_entry)
        
        # Keep only recent feedback (last 100 interactions)
        if len(profile.feedback_history) > 100:
            profile.feedback_history = profile.feedback_history[-100:]
        
        # Save updated profile
        self.save_user_profile(profile)
        
        # Update stats
        self.personalization_stats['embedding_updates'] += 1
        
        logger.debug(f"Updated user {user_id} embedding from {feedback_type} feedback")
        
        return profile
    
    def _update_preferences(self, 
                          profile: UserProfile,
                          item_metadata: Dict[str, Any],
                          feedback_type: str,
                          feedback_value: float):
        """
        Update user preferences based on item metadata
        
        Args:
            profile: User profile to update
            item_metadata: Item metadata
            feedback_type: Type of feedback
            feedback_value: Feedback value
        """
        # Calculate preference update strength
        if feedback_type in ['like', 'purchase']:
            update_strength = feedback_value * 0.1
        elif feedback_type == 'dislike':
            update_strength = -0.1
        else:
            update_strength = feedback_value * 0.05
        
        # Update style preferences
        if 'style' in item_metadata:
            style = item_metadata['style']
            current_pref = profile.style_preferences.get(style, 0.0)
            profile.style_preferences[style] = np.clip(
                current_pref + update_strength, -1.0, 1.0
            )
        
        # Update color preferences
        if 'color' in item_metadata:
            color = item_metadata['color']
            current_pref = profile.color_preferences.get(color, 0.0)
            profile.color_preferences[color] = np.clip(
                current_pref + update_strength, -1.0, 1.0
            )
        
        # Update brand preferences
        if 'brand' in item_metadata:
            brand = item_metadata['brand']
            current_pref = profile.brand_preferences.get(brand, 0.0)
            profile.brand_preferences[brand] = np.clip(
                current_pref + update_strength, -1.0, 1.0
            )
        
        # Update budget range based on price
        if 'price' in item_metadata and feedback_type in ['like', 'purchase']:
            price = float(item_metadata['price'])
            current_min, current_max = profile.budget_range
            
            # Expand budget range towards liked items
            new_min = min(current_min, price * 0.8)
            new_max = max(current_max, price * 1.2)
            profile.budget_range = (new_min, new_max)
    
    def get_personalized_recommendations(self, 
                                       user_id: str,
                                       candidate_items: List[Dict[str, Any]],
                                       context: PersonalizationContext,
                                       top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations for a user
        
        Args:
            user_id: User identifier
            candidate_items: List of candidate items with embeddings
            context: Personalization context
            top_k: Number of recommendations to return
            
        Returns:
            List of personalized recommendations
        """
        profile = self.get_user_profile(user_id)
        
        if not candidate_items:
            return []
        
        # Extract context features
        context_features = self.extract_context_features(context)
        
        # Calculate personalization scores
        personalized_items = []
        
        for item in candidate_items:
            item_embedding = np.array(item.get('embedding', []))
            if item_embedding.size == 0:
                continue
            
            # Base similarity score
            similarity = np.dot(profile.embedding, item_embedding)
            
            # Apply preference modifiers
            preference_boost = 0.0
            
            # Style preference
            if 'style' in item and item['style'] in profile.style_preferences:
                preference_boost += profile.style_preferences[item['style']] * 0.2
            
            # Color preference
            if 'color' in item and item['color'] in profile.color_preferences:
                preference_boost += profile.color_preferences[item['color']] * 0.1
            
            # Brand preference
            if 'brand' in item and item['brand'] in profile.brand_preferences:
                preference_boost += profile.brand_preferences[item['brand']] * 0.15
            
            # Budget compatibility
            if 'price' in item:
                price = float(item['price'])
                min_budget, max_budget = profile.budget_range
                if min_budget <= price <= max_budget:
                    preference_boost += 0.1
                elif price > max_budget:
                    preference_boost -= 0.2
            
            # Context compatibility
            context_boost = 0.0
            
            # Seasonal compatibility
            if context.current_season in profile.seasonal_preferences:
                context_boost += profile.seasonal_preferences[context.current_season] * 0.1
            
            # Occasion compatibility
            if context.occasion in profile.occasion_preferences:
                context_boost += profile.occasion_preferences[context.occasion] * 0.1
            
            # Final personalized score
            personalized_score = similarity + preference_boost + context_boost
            
            # Apply confidence weighting
            confidence_weight = 0.5 + 0.5 * profile.confidence_score
            final_score = personalized_score * confidence_weight
            
            item_copy = item.copy()
            item_copy['personalized_score'] = float(final_score)
            item_copy['base_similarity'] = float(similarity)
            item_copy['preference_boost'] = float(preference_boost)
            item_copy['context_boost'] = float(context_boost)
            item_copy['confidence_weight'] = float(confidence_weight)
            
            personalized_items.append(item_copy)
        
        # Sort by personalized score
        personalized_items.sort(key=lambda x: x['personalized_score'], reverse=True)
        
        # Apply diversity filtering to avoid too similar items
        diverse_items = self._apply_diversity_filtering(personalized_items, top_k)
        
        logger.debug(f"Generated {len(diverse_items)} personalized recommendations for user {user_id}")
        
        return diverse_items[:top_k]
    
    def _apply_diversity_filtering(self, 
                                 items: List[Dict[str, Any]], 
                                 target_count: int) -> List[Dict[str, Any]]:
        """
        Apply diversity filtering to recommendations
        
        Args:
            items: Sorted list of items
            target_count: Target number of diverse items
            
        Returns:
            Filtered list with diversity
        """
        if len(items) <= target_count:
            return items
        
        diverse_items = [items[0]]  # Always include top item
        
        for item in items[1:]:
            if len(diverse_items) >= target_count:
                break
            
            # Check diversity with already selected items
            is_diverse = True
            item_embedding = np.array(item.get('embedding', []))
            
            for selected_item in diverse_items:
                selected_embedding = np.array(selected_item.get('embedding', []))
                
                if item_embedding.size > 0 and selected_embedding.size > 0:
                    similarity = np.dot(item_embedding, selected_embedding)
                    if similarity > 0.8:  # Too similar
                        is_diverse = False
                        break
            
            if is_diverse:
                diverse_items.append(item)
        
        return diverse_items
    
    def build_user_clusters(self, min_users: int = 100, n_clusters: int = 10):
        """
        Build user clusters for cold start recommendations
        
        Args:
            min_users: Minimum number of users needed for clustering
            n_clusters: Number of clusters to create
        """
        if len(self.user_cache) < min_users:
            logger.info(f"Not enough users for clustering ({len(self.user_cache)} < {min_users})")
            return
        
        # Collect user embeddings
        user_embeddings = []
        user_ids = []
        
        for user_id, profile in self.user_cache.items():
            if profile.interaction_count >= self.min_interactions:
                user_embeddings.append(profile.embedding)
                user_ids.append(user_id)
        
        if len(user_embeddings) < min_users:
            logger.info(f"Not enough active users for clustering ({len(user_embeddings)} < {min_users})")
            return
        
        # Perform clustering
        embeddings_array = np.array(user_embeddings)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
        # Update user profiles with cluster assignments
        for user_id, cluster_id in zip(user_ids, cluster_labels):
            profile = self.user_cache[user_id]
            profile.cluster_id = int(cluster_id)
            self.save_user_profile(profile)
        
        # Store cluster centroids
        self.cluster_centroids = kmeans.cluster_centers_
        self.user_clusters = cluster_labels
        
        logger.info(f"Built {n_clusters} user clusters from {len(user_embeddings)} active users")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get personalization engine statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = self.personalization_stats.copy()
        stats['cache_size'] = len(self.user_cache)
        stats['faiss_index_size'] = self.faiss_index.ntotal if self.faiss_index else 0
        stats['redis_connected'] = self.redis_client is not None
        
        return stats


def create_sample_personalization_engine():
    """
    Create a sample personalization engine for testing
    """
    engine = PersonalizationEngine(
        embedding_dim=256,
        redis_host='localhost',
        redis_port=6379,
        redis_db=1,
        faiss_index_path='data/user_embeddings.index'
    )
    
    logger.info("Sample personalization engine created")
    logger.info(f"Embedding dimension: {engine.embedding_dim}")
    logger.info(f"Redis connected: {engine.redis_client is not None}")
    logger.info(f"FAISS index initialized: {engine.faiss_index is not None}")
    
    return engine


if __name__ == "__main__":
    # Create sample engine
    engine = create_sample_personalization_engine()
    
    logger.info("Personalization Layer ready for Phase 2")
    logger.info("Key features:")
    logger.info("- Per-user embeddings with online learning")
    logger.info("- Real-time adaptation via feedback loops")
    logger.info("- Redis integration for fast retrieval")
    logger.info("- FAISS integration for similarity search")
    logger.info("- Style and preference modeling")
    logger.info("- Contextual personalization")
    logger.info("- Cold start handling with user clustering")
    logger.info("- Diversity filtering for recommendations")