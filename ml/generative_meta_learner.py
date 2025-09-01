#!/usr/bin/env python3
"""
Generative Meta-Learner for FlashFit AI

This module implements:
1. Enhanced meta-learner that combines generative and traditional scores
2. Integration with embedding diffusion model outputs
3. Multi-modal fusion (CLIP, BLIP, Fashion Encoder, Generative scores)
4. Adaptive weighting based on context and user feedback
5. Training pipeline for preference learning
6. Real-time inference optimization
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
from datetime import datetime
from dataclasses import dataclass
import pickle
from sklearn.metrics import ndcg_score, precision_score
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MetaLearnerConfig:
    """Configuration for generative meta-learner"""
    input_dim: int = 2048  # Combined features from all models
    hidden_dims: List[int] = None
    output_dim: int = 1  # Final compatibility score
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    temperature: float = 0.1  # For softmax scaling
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [1024, 512, 256]

class FeatureExtractor(nn.Module):
    """Extract and normalize features from different models"""
    
    def __init__(self, config: MetaLearnerConfig):
        super().__init__()
        self.config = config
        
        # Feature dimensions from different models
        self.clip_dim = 512
        self.blip_dim = 512
        self.fashion_dim = 512
        self.generative_dim = 512
        
        # Feature projectors to normalize dimensions
        self.clip_projector = nn.Sequential(
            nn.Linear(self.clip_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        self.blip_projector = nn.Sequential(
            nn.Linear(self.blip_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        self.fashion_projector = nn.Sequential(
            nn.Linear(self.fashion_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        self.generative_projector = nn.Sequential(
            nn.Linear(self.generative_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(128, 256),  # Context features
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # User preference encoder
        self.user_encoder = nn.Sequential(
            nn.Linear(256, 256),  # User embedding
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Interaction features
        self.interaction_encoder = nn.Sequential(
            nn.Linear(512, 256),  # Pairwise interaction features
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract and combine features from all models
        
        Args:
            features: Dictionary containing:
                - clip_similarity: CLIP similarity scores [batch_size, 1]
                - blip_similarity: BLIP similarity scores [batch_size, 1]
                - fashion_similarity: Fashion encoder scores [batch_size, 1]
                - generative_score: Generative model scores [batch_size, 1]
                - query_embedding: Query item embedding [batch_size, 512]
                - candidate_embedding: Candidate item embedding [batch_size, 512]
                - context_features: Context features [batch_size, 128]
                - user_embedding: User preference embedding [batch_size, 256]
                
        Returns:
            Combined feature vector [batch_size, feature_dim]
        """
        batch_size = features['query_embedding'].size(0)
        device = features['query_embedding'].device
        
        # Project individual model outputs
        clip_feat = self.clip_projector(features['query_embedding'])  # Use embedding as proxy
        blip_feat = self.blip_projector(features['candidate_embedding'])  # Use embedding as proxy
        fashion_feat = self.fashion_projector(features['query_embedding'])  # Use embedding as proxy
        generative_feat = self.generative_projector(features.get('generative_embedding', 
                                                                torch.zeros(batch_size, 512, device=device)))
        
        # Encode context and user preferences
        context_feat = self.context_encoder(features.get('context_features', 
                                                        torch.zeros(batch_size, 128, device=device)))
        user_feat = self.user_encoder(features.get('user_embedding', 
                                                  torch.zeros(batch_size, 256, device=device)))
        
        # Compute interaction features
        query_emb = features['query_embedding']
        candidate_emb = features['candidate_embedding']
        
        # Pairwise interactions
        element_wise_product = query_emb * candidate_emb
        cosine_similarity = F.cosine_similarity(query_emb, candidate_emb, dim=1, keepdim=True)
        euclidean_distance = torch.norm(query_emb - candidate_emb, dim=1, keepdim=True)
        
        interaction_features = torch.cat([
            element_wise_product,
            cosine_similarity.expand(-1, 256),  # Broadcast to match dimension
            euclidean_distance.expand(-1, 255)   # Broadcast to match dimension
        ], dim=1)
        
        interaction_feat = self.interaction_encoder(interaction_features)
        
        # Combine all features
        combined_features = torch.cat([
            clip_feat,
            blip_feat, 
            fashion_feat,
            generative_feat,
            context_feat,
            user_feat,
            interaction_feat,
            # Add raw similarity scores
            features.get('clip_similarity', torch.zeros(batch_size, 1, device=device)),
            features.get('blip_similarity', torch.zeros(batch_size, 1, device=device)),
            features.get('fashion_similarity', torch.zeros(batch_size, 1, device=device)),
            features.get('generative_score', torch.zeros(batch_size, 1, device=device))
        ], dim=1)
        
        return combined_features

class AttentionFusion(nn.Module):
    """Attention-based fusion of different model outputs"""
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention to feature representations
        
        Args:
            features: Input features [batch_size, feature_dim]
            
        Returns:
            Attention-weighted features [batch_size, feature_dim]
        """
        # Add sequence dimension for attention
        x = features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # Self-attention
        attn_out, attn_weights = self.attention(x, x, x)
        
        # Residual connection and normalization
        x = self.norm(x + attn_out)
        
        # Remove sequence dimension
        return x.squeeze(1)

class GenerativeMetaLearner(nn.Module):
    """Enhanced meta-learner for generative fashion AI"""
    
    def __init__(self, config: MetaLearnerConfig):
        super().__init__()
        self.config = config
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor(config)
        
        # Calculate actual input dimension after feature extraction
        # 7 * 256 (projected features) + 4 (raw scores) = 1796
        actual_input_dim = 7 * 256 + 4
        
        # Attention fusion
        self.attention_fusion = AttentionFusion(actual_input_dim)
        
        # Main neural network
        layers = []
        prev_dim = actual_input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Confidence estimation branch
        self.confidence_branch = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Confidence between 0 and 1
        )
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.tensor(config.temperature))
        
    def forward(self, features: Dict[str, torch.Tensor], return_confidence: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through meta-learner
        
        Args:
            features: Dictionary of input features
            return_confidence: Whether to return confidence scores
            
        Returns:
            Compatibility scores [batch_size, 1] or (scores, confidence)
        """
        # Extract and combine features
        combined_features = self.feature_extractor(features)
        
        # Apply attention fusion
        fused_features = self.attention_fusion(combined_features)
        
        # Get intermediate representation before final layer
        intermediate = fused_features
        for layer in self.network[:-1]:
            intermediate = layer(intermediate)
        
        # Final compatibility score
        compatibility_score = self.network[-1](intermediate)
        
        # Apply temperature scaling
        compatibility_score = compatibility_score / self.temperature
        
        if return_confidence:
            confidence = self.confidence_branch(intermediate)
            return compatibility_score, confidence
        
        return compatibility_score
    
    def predict_ranking(self, 
                       query_features: Dict[str, torch.Tensor],
                       candidate_features_list: List[Dict[str, torch.Tensor]],
                       top_k: int = 10) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Rank candidates for a query item
        
        Args:
            query_features: Query item features
            candidate_features_list: List of candidate item features
            top_k: Number of top candidates to return
            
        Returns:
            Tuple of (top_k_indices, scores, confidences)
        """
        self.eval()
        
        scores = []
        confidences = []
        
        with torch.no_grad():
            for candidate_features in candidate_features_list:
                # Combine query and candidate features
                combined_features = {
                    **query_features,
                    **candidate_features
                }
                
                # Get score and confidence
                score, confidence = self.forward(combined_features, return_confidence=True)
                scores.append(score.item())
                confidences.append(confidence.item())
        
        scores = torch.tensor(scores)
        confidences = torch.tensor(confidences)
        
        # Get top-k indices
        top_k_indices = torch.topk(scores, min(top_k, len(scores))).indices.tolist()
        
        return top_k_indices, scores, confidences

class PreferenceDataset(Dataset):
    """Dataset for training meta-learner on user preferences"""
    
    def __init__(self, 
                 preference_data: List[Dict[str, Any]]):
        """
        Args:
            preference_data: List of preference examples, each containing:
                - query_features: Query item features
                - candidate_features: Candidate item features  
                - preference_score: User preference score (0-1)
                - context: Context information
        """
        self.preference_data = preference_data
        
    def __len__(self) -> int:
        return len(self.preference_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.preference_data[idx]

class MetaLearnerTrainer:
    """Training pipeline for generative meta-learner"""
    
    def __init__(self, 
                 model: GenerativeMetaLearner,
                 device: torch.device,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        
    def compute_loss(self, 
                    predictions: torch.Tensor, 
                    targets: torch.Tensor,
                    confidences: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute training loss with optional confidence weighting
        
        Args:
            predictions: Model predictions [batch_size, 1]
            targets: Target preference scores [batch_size, 1]
            confidences: Confidence scores [batch_size, 1]
            
        Returns:
            Combined loss
        """
        # Main prediction loss (MSE)
        mse_loss = F.mse_loss(predictions, targets)
        
        # Ranking loss (pairwise)
        batch_size = predictions.size(0)
        if batch_size > 1:
            # Create pairwise comparisons
            pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # [batch_size, batch_size]
            target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # [batch_size, batch_size]
            
            # Ranking loss: encourage correct ordering
            ranking_loss = F.relu(1.0 - pred_diff * torch.sign(target_diff)).mean()
        else:
            ranking_loss = torch.tensor(0.0, device=predictions.device)
        
        # Confidence regularization
        if confidences is not None:
            # Encourage high confidence for correct predictions
            prediction_error = torch.abs(predictions - targets)
            confidence_loss = F.mse_loss(confidences.squeeze(), 1.0 - prediction_error.squeeze())
        else:
            confidence_loss = torch.tensor(0.0, device=predictions.device)
        
        # Combine losses
        total_loss = mse_loss + 0.1 * ranking_loss + 0.05 * confidence_loss
        
        return total_loss
    
    def train_step(self, batch: Dict[str, Any]) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Prepare features
        features = {
            'query_embedding': batch['query_embedding'].to(self.device),
            'candidate_embedding': batch['candidate_embedding'].to(self.device),
            'clip_similarity': batch.get('clip_similarity', torch.zeros(batch['query_embedding'].size(0), 1)).to(self.device),
            'blip_similarity': batch.get('blip_similarity', torch.zeros(batch['query_embedding'].size(0), 1)).to(self.device),
            'fashion_similarity': batch.get('fashion_similarity', torch.zeros(batch['query_embedding'].size(0), 1)).to(self.device),
            'generative_score': batch.get('generative_score', torch.zeros(batch['query_embedding'].size(0), 1)).to(self.device),
            'context_features': batch.get('context_features', torch.zeros(batch['query_embedding'].size(0), 128)).to(self.device),
            'user_embedding': batch.get('user_embedding', torch.zeros(batch['query_embedding'].size(0), 256)).to(self.device)
        }
        
        targets = batch['preference_score'].to(self.device)
        
        # Forward pass
        predictions, confidences = self.model(features, return_confidence=True)
        
        # Compute loss
        loss = self.compute_loss(predictions, targets, confidences)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validation step with metrics"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Prepare features
                features = {
                    'query_embedding': batch['query_embedding'].to(self.device),
                    'candidate_embedding': batch['candidate_embedding'].to(self.device),
                    'clip_similarity': batch.get('clip_similarity', torch.zeros(batch['query_embedding'].size(0), 1)).to(self.device),
                    'blip_similarity': batch.get('blip_similarity', torch.zeros(batch['query_embedding'].size(0), 1)).to(self.device),
                    'fashion_similarity': batch.get('fashion_similarity', torch.zeros(batch['query_embedding'].size(0), 1)).to(self.device),
                    'generative_score': batch.get('generative_score', torch.zeros(batch['query_embedding'].size(0), 1)).to(self.device),
                    'context_features': batch.get('context_features', torch.zeros(batch['query_embedding'].size(0), 128)).to(self.device),
                    'user_embedding': batch.get('user_embedding', torch.zeros(batch['query_embedding'].size(0), 256)).to(self.device)
                }
                
                targets = batch['preference_score'].to(self.device)
                
                # Forward pass
                predictions, confidences = self.model(features, return_confidence=True)
                
                # Compute loss
                loss = self.compute_loss(predictions, targets, confidences)
                total_loss += loss.item()
                
                # Collect predictions for metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_confidences = np.array(all_confidences)
        
        # Correlation
        correlation = np.corrcoef(all_predictions.flatten(), all_targets.flatten())[0, 1]
        
        # MSE
        mse = np.mean((all_predictions - all_targets) ** 2)
        
        # NDCG (treat as ranking problem)
        try:
            ndcg = ndcg_score([all_targets], [all_predictions])
        except:
            ndcg = 0.0
        
        metrics = {
            'correlation': correlation if not np.isnan(correlation) else 0.0,
            'mse': mse,
            'ndcg': ndcg,
            'mean_confidence': np.mean(all_confidences)
        }
        
        return avg_loss, metrics
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              save_path: str = "models/generative_meta_learner.pth") -> Dict[str, Any]:
        """Full training loop"""
        logger.info(f"Starting meta-learner training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            epoch_train_loss = 0.0
            num_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                loss = self.train_step(batch)
                epoch_train_loss += loss
                num_batches += 1
            
            avg_train_loss = epoch_train_loss / num_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation
            val_loss, metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.metrics_history.append(metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
            logger.info(f"Metrics: Correlation = {metrics['correlation']:.4f}, NDCG = {metrics['ndcg']:.4f}, Confidence = {metrics['mean_confidence']:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'metrics': metrics,
                    'config': self.model.config
                }, save_path)
                
                logger.info(f"New best model saved with val_loss = {val_loss:.6f}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }

def create_sample_meta_learner() -> Tuple[GenerativeMetaLearner, MetaLearnerTrainer]:
    """Create sample meta-learner and trainer for testing"""
    config = MetaLearnerConfig(
        input_dim=2048,
        hidden_dims=[1024, 512, 256],
        output_dim=1,
        dropout_rate=0.2,
        learning_rate=1e-3
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenerativeMetaLearner(config)
    trainer = MetaLearnerTrainer(model, device)
    
    return model, trainer

if __name__ == "__main__":
    # Create sample model
    model, trainer = create_sample_meta_learner()
    
    logger.info("Generative Meta-Learner ready for FlashFit AI")
    logger.info("Key features:")
    logger.info("- Multi-modal fusion (CLIP, BLIP, Fashion, Generative)")
    logger.info("- Attention-based feature combination")
    logger.info("- Confidence estimation")
    logger.info("- Preference learning with ranking loss")
    logger.info("- Temperature scaling for calibration")
    logger.info("- Real-time ranking and recommendation")
    
    # Test forward pass
    batch_size = 4
    device = next(model.parameters()).device
    
    # Sample features
    features = {
        'query_embedding': torch.randn(batch_size, 512, device=device),
        'candidate_embedding': torch.randn(batch_size, 512, device=device),
        'clip_similarity': torch.rand(batch_size, 1, device=device),
        'blip_similarity': torch.rand(batch_size, 1, device=device),
        'fashion_similarity': torch.rand(batch_size, 1, device=device),
        'generative_score': torch.rand(batch_size, 1, device=device),
        'context_features': torch.randn(batch_size, 128, device=device),
        'user_embedding': torch.randn(batch_size, 256, device=device)
    }
    
    # Forward pass
    with torch.no_grad():
        scores, confidences = model(features, return_confidence=True)
        logger.info(f"Forward pass successful: Scores {scores.shape}, Confidences {confidences.shape}")
        logger.info(f"Sample scores: {scores.flatten().tolist()}")
        logger.info(f"Sample confidences: {confidences.flatten().tolist()}")