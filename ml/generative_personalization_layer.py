#!/usr/bin/env python3
"""
Generative Personalization Layer with RLHF

This module implements:
1. RLHF-style feedback tuning for generative models
2. Per-user embedding adaptation for generative outputs
3. Reward modeling for subjective preferences
4. Policy optimization for personalized generation
5. Integration with embedding diffusion model
6. Multi-objective optimization (compatibility + preference)
7. Online learning with preference updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict, deque
import pickle
from dataclasses import dataclass, asdict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Import existing components
from personalization_layer import PersonalizationEngine, UserProfile, PersonalizationContext
from embedding_diffusion import EmbeddingDiffusionModel, DiffusionConfig
from generative_meta_learner import GenerativeMetaLearner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RLHFConfig:
    """Configuration for RLHF training"""
    reward_model_dim: int = 256
    policy_lr: float = 1e-4
    reward_lr: float = 1e-3
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 32
    buffer_size: int = 10000
    gamma: float = 0.99
    gae_lambda: float = 0.95

@dataclass
class PreferenceData:
    """Structure for preference feedback"""
    user_id: str
    query_embedding: np.ndarray
    generated_embeddings: List[np.ndarray]
    preferences: List[float]  # 0-1 scores for each generated item
    context: PersonalizationContext
    timestamp: datetime
    session_id: str
    feedback_type: str  # 'explicit', 'implicit', 'comparative'

class RewardModel(nn.Module):
    """
    Neural network that learns to predict user preferences
    """
    
    def __init__(self, 
                 embedding_dim: int = 512,
                 user_embedding_dim: int = 256,
                 context_dim: int = 64,
                 hidden_dim: int = 256,
                 num_layers: int = 3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.user_embedding_dim = user_embedding_dim
        self.context_dim = context_dim
        
        # Input processing
        self.query_encoder = nn.Linear(embedding_dim, hidden_dim)
        self.candidate_encoder = nn.Linear(embedding_dim, hidden_dim)
        self.user_encoder = nn.Linear(user_embedding_dim, hidden_dim)
        self.context_encoder = nn.Linear(context_dim, hidden_dim)
        
        # Attention mechanism for query-candidate interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Reward prediction layers
        layers = []
        input_dim = hidden_dim * 4  # query + candidate + user + context
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))  # Single reward score
        self.reward_head = nn.Sequential(*layers)
        
        # Value function for PPO
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, 
                query_embedding: torch.Tensor,
                candidate_embedding: torch.Tensor,
                user_embedding: torch.Tensor,
                context_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for reward prediction
        
        Args:
            query_embedding: [batch_size, embedding_dim]
            candidate_embedding: [batch_size, embedding_dim]
            user_embedding: [batch_size, user_embedding_dim]
            context_features: [batch_size, context_dim]
        
        Returns:
            reward: [batch_size, 1]
            value: [batch_size, 1]
        """
        batch_size = query_embedding.size(0)
        
        # Encode inputs
        query_enc = F.relu(self.query_encoder(query_embedding))  # [batch_size, hidden_dim]
        candidate_enc = F.relu(self.candidate_encoder(candidate_embedding))  # [batch_size, hidden_dim]
        user_enc = F.relu(self.user_encoder(user_embedding))  # [batch_size, hidden_dim]
        context_enc = F.relu(self.context_encoder(context_features))  # [batch_size, hidden_dim]
        
        # Attention between query and candidate
        query_expanded = query_enc.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        candidate_expanded = candidate_enc.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        attended_query, _ = self.attention(
            query_expanded, candidate_expanded, candidate_expanded
        )
        attended_query = attended_query.squeeze(1)  # [batch_size, hidden_dim]
        
        # Combine all features
        combined_features = torch.cat([
            attended_query, candidate_enc, user_enc, context_enc
        ], dim=1)  # [batch_size, hidden_dim * 4]
        
        # Predict reward and value
        reward = self.reward_head(combined_features)
        value = self.value_head(combined_features)
        
        return reward, value

class GenerativePersonalizationEngine:
    """
    Enhanced personalization engine with generative AI and RLHF capabilities
    """
    
    def __init__(self,
                 base_personalization_engine: PersonalizationEngine,
                 diffusion_model: EmbeddingDiffusionModel,
                 meta_learner: GenerativeMetaLearner,
                 rlhf_config: RLHFConfig = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.base_engine = base_personalization_engine
        self.diffusion_model = diffusion_model
        self.meta_learner = meta_learner
        self.device = device
        self.rlhf_config = rlhf_config or RLHFConfig()
        
        # Initialize reward model
        self.reward_model = RewardModel(
            embedding_dim=diffusion_model.config.embedding_dim,
            user_embedding_dim=base_personalization_engine.embedding_dim,
            context_dim=64,  # Context feature dimension
            hidden_dim=self.rlhf_config.reward_model_dim
        ).to(device)
        
        # Optimizers
        self.reward_optimizer = optim.Adam(
            self.reward_model.parameters(), 
            lr=self.rlhf_config.reward_lr
        )
        
        self.policy_optimizer = optim.Adam(
            list(self.diffusion_model.parameters()) + list(self.meta_learner.parameters()),
            lr=self.rlhf_config.policy_lr
        )
        
        # Experience buffer for RLHF
        self.preference_buffer = deque(maxlen=self.rlhf_config.buffer_size)
        self.experience_buffer = deque(maxlen=self.rlhf_config.buffer_size)
        
        # Training statistics
        self.training_stats = {
            'reward_losses': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'preference_accuracy': [],
            'generation_quality': []
        }
        
        logger.info(f"Initialized GenerativePersonalizationEngine on {device}")
    
    def generate_personalized_items(self,
                                  user_id: str,
                                  query_embedding: np.ndarray,
                                  context: PersonalizationContext,
                                  num_items: int = 5,
                                  temperature: float = 1.0) -> List[Dict[str, Any]]:
        """
        Generate personalized item embeddings using diffusion model
        
        Args:
            user_id: User identifier
            query_embedding: Query item embedding
            context: Personalization context
            num_items: Number of items to generate
            temperature: Generation temperature
        
        Returns:
            List of generated items with embeddings and metadata
        """
        # Get user profile and embedding
        user_profile = self.base_engine.get_user_profile(user_id)
        if user_profile is None:
            user_profile = self.base_engine._create_new_user_profile(user_id)
        
        # Extract context features
        context_features = self.base_engine.extract_context_features(context)
        
        # Prepare inputs for diffusion model
        query_tensor = torch.from_numpy(query_embedding).float().unsqueeze(0).to(self.device)
        user_tensor = torch.from_numpy(user_profile.embedding).float().unsqueeze(0).to(self.device)
        context_tensor = torch.from_numpy(context_features).float().unsqueeze(0).to(self.device)
        
        # Generate embeddings using diffusion model
        with torch.no_grad():
            generated_embeddings = []
            
            for _ in range(num_items):
                # Sample from diffusion model
                generated_embedding = self.diffusion_model.sample(
                    query_embedding=query_tensor,
                    context_embedding=torch.cat([user_tensor, context_tensor], dim=1),
                    num_inference_steps=20,
                    temperature=temperature
                )
                
                generated_embeddings.append(generated_embedding.cpu().numpy().squeeze())
        
        # Use meta-learner to refine and rank generations
        refined_items = self._refine_with_meta_learner(
            user_id, query_embedding, generated_embeddings, context
        )
        
        return refined_items
    
    def _refine_with_meta_learner(self,
                                user_id: str,
                                query_embedding: np.ndarray,
                                generated_embeddings: List[np.ndarray],
                                context: PersonalizationContext) -> List[Dict[str, Any]]:
        """
        Use meta-learner to refine and rank generated embeddings
        """
        user_profile = self.base_engine.get_user_profile(user_id)
        context_features = self.base_engine.extract_context_features(context)
        
        refined_items = []
        
        for i, embedding in enumerate(generated_embeddings):
            # Predict compatibility and personalization scores
            compatibility_score = self.meta_learner.predict_compatibility(
                query_embedding, embedding, context_features
            )
            
            personalization_score = self.meta_learner.predict_personalization(
                embedding, user_profile.embedding, context_features
            )
            
            # Predict reward using reward model
            with torch.no_grad():
                query_tensor = torch.from_numpy(query_embedding).float().unsqueeze(0).to(self.device)
                embedding_tensor = torch.from_numpy(embedding).float().unsqueeze(0).to(self.device)
                user_tensor = torch.from_numpy(user_profile.embedding).float().unsqueeze(0).to(self.device)
                context_tensor = torch.from_numpy(context_features).float().unsqueeze(0).to(self.device)
                
                reward, _ = self.reward_model(
                    query_tensor, embedding_tensor, user_tensor, context_tensor
                )
                reward_score = reward.item()
            
            # Combine scores
            final_score = (
                0.4 * compatibility_score +
                0.4 * personalization_score +
                0.2 * reward_score
            )
            
            refined_items.append({
                'embedding': embedding,
                'compatibility_score': compatibility_score,
                'personalization_score': personalization_score,
                'reward_score': reward_score,
                'final_score': final_score,
                'generated_id': f"gen_{user_id}_{i}_{int(time.time())}"
            })
        
        # Sort by final score
        refined_items.sort(key=lambda x: x['final_score'], reverse=True)
        
        return refined_items
    
    def collect_preference_feedback(self,
                                  user_id: str,
                                  query_embedding: np.ndarray,
                                  generated_items: List[Dict[str, Any]],
                                  preferences: List[float],
                                  context: PersonalizationContext,
                                  feedback_type: str = 'explicit'):
        """
        Collect preference feedback for RLHF training
        
        Args:
            user_id: User identifier
            query_embedding: Original query embedding
            generated_items: List of generated items
            preferences: Preference scores (0-1) for each item
            context: Personalization context
            feedback_type: Type of feedback ('explicit', 'implicit', 'comparative')
        """
        preference_data = PreferenceData(
            user_id=user_id,
            query_embedding=query_embedding,
            generated_embeddings=[item['embedding'] for item in generated_items],
            preferences=preferences,
            context=context,
            timestamp=datetime.now(),
            session_id=f"session_{user_id}_{int(time.time())}",
            feedback_type=feedback_type
        )
        
        self.preference_buffer.append(preference_data)
        
        # Update user profile with feedback
        for item, preference in zip(generated_items, preferences):
            feedback_value = preference * 2 - 1  # Convert 0-1 to -1-1
            self.base_engine.update_user_from_feedback(
                user_id=user_id,
                item_embedding=item['embedding'],
                feedback_type='preference',
                feedback_value=feedback_value,
                context=context
            )
        
        logger.info(f"Collected preference feedback for user {user_id}: {len(preferences)} items")
    
    def train_reward_model(self, num_epochs: int = 10, batch_size: int = 32):
        """
        Train reward model on collected preference data
        """
        if len(self.preference_buffer) < batch_size:
            logger.warning("Not enough preference data for training")
            return
        
        self.reward_model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            # Sample batches from preference buffer
            batch_data = list(self.preference_buffer)[-batch_size * 10:]  # Use recent data
            np.random.shuffle(batch_data)
            
            for i in range(0, len(batch_data) - batch_size + 1, batch_size):
                batch = batch_data[i:i + batch_size]
                
                # Prepare batch tensors
                query_embeddings = []
                candidate_embeddings = []
                user_embeddings = []
                context_features = []
                preference_scores = []
                
                for pref_data in batch:
                    user_profile = self.base_engine.get_user_profile(pref_data.user_id)
                    context_feat = self.base_engine.extract_context_features(pref_data.context)
                    
                    for embedding, preference in zip(pref_data.generated_embeddings, pref_data.preferences):
                        query_embeddings.append(pref_data.query_embedding)
                        candidate_embeddings.append(embedding)
                        user_embeddings.append(user_profile.embedding)
                        context_features.append(context_feat)
                        preference_scores.append(preference)
                
                if not query_embeddings:
                    continue
                
                # Convert to tensors
                query_tensor = torch.from_numpy(np.array(query_embeddings)).float().to(self.device)
                candidate_tensor = torch.from_numpy(np.array(candidate_embeddings)).float().to(self.device)
                user_tensor = torch.from_numpy(np.array(user_embeddings)).float().to(self.device)
                context_tensor = torch.from_numpy(np.array(context_features)).float().to(self.device)
                target_tensor = torch.from_numpy(np.array(preference_scores)).float().unsqueeze(1).to(self.device)
                
                # Forward pass
                predicted_rewards, _ = self.reward_model(
                    query_tensor, candidate_tensor, user_tensor, context_tensor
                )
                
                # Compute loss (MSE for preference prediction)
                loss = F.mse_loss(torch.sigmoid(predicted_rewards), target_tensor)
                
                # Backward pass
                self.reward_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.rlhf_config.max_grad_norm)
                self.reward_optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                
                # Accuracy (binary classification with 0.5 threshold)
                predictions = (torch.sigmoid(predicted_rewards) > 0.5).float()
                targets = (target_tensor > 0.5).float()
                epoch_correct += (predictions == targets).sum().item()
                epoch_total += targets.size(0)
            
            if epoch_total > 0:
                epoch_accuracy = epoch_correct / epoch_total
                logger.info(f"Reward model epoch {epoch + 1}/{num_epochs}: "
                          f"Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}")
                
                self.training_stats['reward_losses'].append(epoch_loss)
                self.training_stats['preference_accuracy'].append(epoch_accuracy)
        
        self.reward_model.eval()
        logger.info("Reward model training completed")
    
    def train_policy_with_ppo(self, num_epochs: int = 5):
        """
        Train generative policy using PPO with learned reward model
        """
        if len(self.preference_buffer) < self.rlhf_config.batch_size:
            logger.warning("Not enough data for PPO training")
            return
        
        self.diffusion_model.train()
        self.meta_learner.train()
        self.reward_model.eval()
        
        for epoch in range(num_epochs):
            # Sample experience from recent interactions
            recent_data = list(self.preference_buffer)[-100:]  # Use recent data
            
            policy_losses = []
            value_losses = []
            entropy_losses = []
            
            for pref_data in recent_data:
                # Generate new samples with current policy
                user_profile = self.base_engine.get_user_profile(pref_data.user_id)
                context_features = self.base_engine.extract_context_features(pref_data.context)
                
                # Sample from current policy
                query_tensor = torch.from_numpy(pref_data.query_embedding).float().unsqueeze(0).to(self.device)
                user_tensor = torch.from_numpy(user_profile.embedding).float().unsqueeze(0).to(self.device)
                context_tensor = torch.from_numpy(context_features).float().unsqueeze(0).to(self.device)
                
                # Generate with current policy
                with torch.no_grad():
                    generated_embedding = self.diffusion_model.sample(
                        query_embedding=query_tensor,
                        context_embedding=torch.cat([user_tensor, context_tensor], dim=1),
                        num_inference_steps=10  # Fewer steps for training efficiency
                    )
                
                # Compute reward
                with torch.no_grad():
                    reward, value = self.reward_model(
                        query_tensor, generated_embedding, user_tensor, context_tensor
                    )
                
                # Compute policy loss (simplified PPO)
                # In practice, you'd need to store old policy probabilities
                # and compute the full PPO objective
                
                # For now, use a simplified policy gradient approach
                log_prob = -self.diffusion_model.compute_loss(
                    generated_embedding, query_tensor, 
                    torch.cat([user_tensor, context_tensor], dim=1)
                )
                
                policy_loss = -(log_prob * reward.detach()).mean()
                value_loss = F.mse_loss(value, reward.detach())
                
                # Total loss
                total_loss = (
                    policy_loss + 
                    self.rlhf_config.value_loss_coef * value_loss
                )
                
                # Backward pass
                self.policy_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.diffusion_model.parameters()) + list(self.meta_learner.parameters()),
                    self.rlhf_config.max_grad_norm
                )
                self.policy_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
            
            if policy_losses:
                avg_policy_loss = np.mean(policy_losses)
                avg_value_loss = np.mean(value_losses)
                
                logger.info(f"PPO epoch {epoch + 1}/{num_epochs}: "
                          f"Policy Loss = {avg_policy_loss:.4f}, "
                          f"Value Loss = {avg_value_loss:.4f}")
                
                self.training_stats['policy_losses'].append(avg_policy_loss)
                self.training_stats['value_losses'].append(avg_value_loss)
        
        logger.info("PPO training completed")
    
    def evaluate_generation_quality(self, test_users: List[str], num_samples: int = 10) -> Dict[str, float]:
        """
        Evaluate quality of generated recommendations
        """
        self.diffusion_model.eval()
        self.meta_learner.eval()
        
        metrics = {
            'avg_reward': 0.0,
            'avg_compatibility': 0.0,
            'avg_personalization': 0.0,
            'diversity_score': 0.0
        }
        
        total_samples = 0
        all_embeddings = []
        
        with torch.no_grad():
            for user_id in test_users:
                user_profile = self.base_engine.get_user_profile(user_id)
                if user_profile is None:
                    continue
                
                # Create dummy context
                context = PersonalizationContext(
                    user_id=user_id,
                    current_season='spring',
                    time_of_day='afternoon',
                    occasion='casual',
                    budget_constraint=None,
                    recent_purchases=[],
                    browsing_session=[]
                )
                
                # Generate random query embedding for testing
                query_embedding = np.random.randn(512).astype(np.float32)
                
                # Generate items
                generated_items = self.generate_personalized_items(
                    user_id, query_embedding, context, num_items=num_samples
                )
                
                for item in generated_items:
                    metrics['avg_reward'] += item['reward_score']
                    metrics['avg_compatibility'] += item['compatibility_score']
                    metrics['avg_personalization'] += item['personalization_score']
                    all_embeddings.append(item['embedding'])
                    total_samples += 1
        
        if total_samples > 0:
            metrics['avg_reward'] /= total_samples
            metrics['avg_compatibility'] /= total_samples
            metrics['avg_personalization'] /= total_samples
            
            # Compute diversity (average pairwise distance)
            if len(all_embeddings) > 1:
                embeddings_array = np.array(all_embeddings)
                pairwise_distances = []
                for i in range(len(embeddings_array)):
                    for j in range(i + 1, len(embeddings_array)):
                        dist = np.linalg.norm(embeddings_array[i] - embeddings_array[j])
                        pairwise_distances.append(dist)
                metrics['diversity_score'] = np.mean(pairwise_distances)
        
        logger.info(f"Generation quality evaluation: {metrics}")
        return metrics
    
    def save_model(self, save_path: str):
        """
        Save the generative personalization model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save reward model
        torch.save({
            'reward_model_state_dict': self.reward_model.state_dict(),
            'reward_optimizer_state_dict': self.reward_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'rlhf_config': asdict(self.rlhf_config),
            'training_stats': self.training_stats
        }, save_path / 'generative_personalization.pt')
        
        # Save preference buffer
        with open(save_path / 'preference_buffer.pkl', 'wb') as f:
            pickle.dump(list(self.preference_buffer), f)
        
        logger.info(f"Generative personalization model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """
        Load the generative personalization model
        """
        load_path = Path(load_path)
        
        # Load model state
        checkpoint = torch.load(load_path / 'generative_personalization.pt', map_location=self.device)
        
        self.reward_model.load_state_dict(checkpoint['reward_model_state_dict'])
        self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        
        # Load preference buffer
        if (load_path / 'preference_buffer.pkl').exists():
            with open(load_path / 'preference_buffer.pkl', 'rb') as f:
                preference_data = pickle.load(f)
                self.preference_buffer.extend(preference_data)
        
        logger.info(f"Generative personalization model loaded from {load_path}")

def create_generative_personalization_engine(
    base_engine: PersonalizationEngine,
    diffusion_model: EmbeddingDiffusionModel,
    meta_learner: GenerativeMetaLearner,
    rlhf_config: RLHFConfig = None
) -> GenerativePersonalizationEngine:
    """
    Factory function to create generative personalization engine
    """
    return GenerativePersonalizationEngine(
        base_personalization_engine=base_engine,
        diffusion_model=diffusion_model,
        meta_learner=meta_learner,
        rlhf_config=rlhf_config
    )

if __name__ == "__main__":
    # Example usage
    from personalization_layer import create_sample_personalization_engine
    
    # Create components
    base_engine = create_sample_personalization_engine()
    
    # Create diffusion config and model
    diffusion_config = DiffusionConfig(
        embedding_dim=512,
        context_dim=128,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        timesteps=1000
    )
    
    diffusion_model = EmbeddingDiffusionModel(diffusion_config)
    meta_learner = GenerativeMetaLearner(embedding_dim=512, context_dim=128)
    
    # Create generative personalization engine
    gen_engine = create_generative_personalization_engine(
        base_engine=base_engine,
        diffusion_model=diffusion_model,
        meta_learner=meta_learner
    )
    
    logger.info("Generative Personalization Layer with RLHF ready!")
    logger.info("Key features:")
    logger.info("- RLHF-style feedback tuning for generative models")
    logger.info("- Per-user embedding adaptation for generative outputs")
    logger.info("- Reward modeling for subjective preferences")
    logger.info("- Policy optimization for personalized generation")
    logger.info("- Integration with embedding diffusion model")
    logger.info("- Multi-objective optimization (compatibility + preference)")
    logger.info("- Online learning with preference updates")