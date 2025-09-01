#!/usr/bin/env python3
"""
Embedding Diffusion Model for Generative Fashion AI

This module implements:
1. Latent diffusion model in embedding space for outfit generation
2. Conditional generation based on query item + context (occasion, season, user)
3. Noise scheduling and denoising process
4. Integration with existing CLIP/BLIP/Fashion encoders
5. Training pipeline for compatibility learning
6. Evaluation metrics for generative quality
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
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DiffusionConfig:
    """Configuration for embedding diffusion model"""
    embedding_dim: int = 512
    hidden_dim: int = 1024
    num_layers: int = 8
    num_heads: int = 16
    dropout_rate: float = 0.1
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    
class NoiseScheduler:
    """Noise scheduling for diffusion process"""
    
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to embeddings according to diffusion schedule"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training"""
        return torch.randint(0, self.timesteps, (batch_size,), device=device)

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ContextEncoder(nn.Module):
    """Encode contextual information (occasion, season, user preferences)"""
    
    def __init__(self, context_dim: int = 256, embedding_dim: int = 512):
        super().__init__()
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
        
        # Occasion embeddings
        self.occasion_vocab = {
            'casual': 0, 'work': 1, 'formal': 2, 'party': 3, 'sport': 4, 
            'date': 5, 'travel': 6, 'home': 7, 'outdoor': 8, 'special': 9
        }
        self.occasion_embedding = nn.Embedding(len(self.occasion_vocab), context_dim // 4)
        
        # Season embeddings
        self.season_vocab = {'spring': 0, 'summer': 1, 'autumn': 2, 'winter': 3}
        self.season_embedding = nn.Embedding(len(self.season_vocab), context_dim // 4)
        
        # Style embeddings
        self.style_vocab = {
            'minimalist': 0, 'bohemian': 1, 'classic': 2, 'trendy': 3, 'edgy': 4,
            'romantic': 5, 'sporty': 6, 'vintage': 7, 'preppy': 8, 'artsy': 9
        }
        self.style_embedding = nn.Embedding(len(self.style_vocab), context_dim // 4)
        
        # User preference encoder
        self.user_encoder = nn.Sequential(
            nn.Linear(256, context_dim // 4),  # Assuming 256-dim user embedding
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(context_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, context: Dict[str, Any]) -> torch.Tensor:
        """Encode context into embedding space"""
        batch_size = context['batch_size']
        device = next(self.parameters()).device
        
        # Encode occasion
        occasion_ids = torch.tensor([
            self.occasion_vocab.get(occ, 0) for occ in context.get('occasion', ['casual'] * batch_size)
        ], device=device)
        occasion_emb = self.occasion_embedding(occasion_ids)
        
        # Encode season
        season_ids = torch.tensor([
            self.season_vocab.get(season, 0) for season in context.get('season', ['spring'] * batch_size)
        ], device=device)
        season_emb = self.season_embedding(season_ids)
        
        # Encode style
        style_ids = torch.tensor([
            self.style_vocab.get(style, 0) for style in context.get('style', ['casual'] * batch_size)
        ], device=device)
        style_emb = self.style_embedding(style_ids)
        
        # Encode user preferences
        user_embeddings = context.get('user_embedding', torch.zeros(batch_size, 256, device=device))
        user_emb = self.user_encoder(user_embeddings)
        
        # Concatenate all context embeddings
        context_emb = torch.cat([occasion_emb, season_emb, style_emb, user_emb], dim=-1)
        
        # Fuse context
        return self.context_fusion(context_emb)

class TransformerBlock(nn.Module):
    """Transformer block for diffusion model"""
    
    def __init__(self, embedding_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with optional cross-attention to context
        if context is not None:
            attn_out, _ = self.attention(x, context, context)
        else:
            attn_out, _ = self.attention(x, x, x)
        
        x = self.norm1(x + attn_out)
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x

class EmbeddingDiffusionModel(nn.Module):
    """Main embedding diffusion model for generative fashion AI"""
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(config.embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )
        
        # Context encoder
        self.context_encoder = ContextEncoder(embedding_dim=config.embedding_dim)
        
        # Input projection
        self.input_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(config.embedding_dim, config.num_heads, config.dropout_rate)
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(config.timesteps, config.beta_start, config.beta_end)
    
    def forward(self, 
                noisy_embeddings: torch.Tensor, 
                timesteps: torch.Tensor, 
                query_embedding: torch.Tensor,
                context: Dict[str, Any]) -> torch.Tensor:
        """
        Predict noise to be removed from noisy embeddings
        
        Args:
            noisy_embeddings: Noisy target item embeddings [batch_size, embedding_dim]
            timesteps: Diffusion timesteps [batch_size]
            query_embedding: Query item embedding [batch_size, embedding_dim]
            context: Context information (occasion, season, user preferences)
            
        Returns:
            Predicted noise [batch_size, embedding_dim]
        """
        batch_size = noisy_embeddings.size(0)
        
        # Time embedding
        time_emb = self.time_embedding(timesteps)
        time_emb = self.time_mlp(time_emb)
        
        # Context embedding
        context['batch_size'] = batch_size
        context_emb = self.context_encoder(context)
        
        # Combine query, noisy embeddings, and time
        x = self.input_projection(noisy_embeddings)
        x = x + time_emb + query_embedding
        
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        context_emb = context_emb.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # Apply transformer layers with cross-attention to context
        for i, layer in enumerate(self.transformer_layers):
            if i % 2 == 0:  # Self-attention layers
                x = layer(x)
            else:  # Cross-attention to context
                x = layer(x, context_emb)
        
        # Remove sequence dimension and project to output
        x = x.squeeze(1)  # [batch_size, embedding_dim]
        predicted_noise = self.output_projection(x)
        
        return predicted_noise
    
    def sample(self, 
               query_embedding: torch.Tensor, 
               context: Dict[str, Any], 
               num_samples: int = 1,
               guidance_scale: float = 1.0) -> torch.Tensor:
        """
        Generate compatible item embeddings using DDPM sampling
        
        Args:
            query_embedding: Query item embedding [1, embedding_dim]
            context: Context information
            num_samples: Number of samples to generate
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Generated compatible embeddings [num_samples, embedding_dim]
        """
        device = query_embedding.device
        
        # Expand query embedding for batch sampling
        query_batch = query_embedding.repeat(num_samples, 1)
        
        # Start from pure noise
        x = torch.randn(num_samples, self.config.embedding_dim, device=device)
        
        # Reverse diffusion process
        for t in reversed(range(self.noise_scheduler.timesteps)):
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                predicted_noise = self.forward(x, timesteps, query_batch, context)
                
                # Apply classifier-free guidance if enabled
                if guidance_scale > 1.0:
                    # Unconditional prediction (empty context)
                    empty_context = {k: [''] * num_samples if isinstance(v, list) else torch.zeros_like(v) 
                                   for k, v in context.items()}
                    empty_context['batch_size'] = num_samples
                    
                    uncond_noise = self.forward(x, timesteps, query_batch, empty_context)
                    predicted_noise = uncond_noise + guidance_scale * (predicted_noise - uncond_noise)
            
            # Compute previous sample
            alpha_t = self.noise_scheduler.alphas[t]
            alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]
            beta_t = self.noise_scheduler.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            # DDPM sampling formula
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)
            
            if t > 0:
                x = x + torch.sqrt(self.noise_scheduler.posterior_variance[t]) * noise
        
        return x

class CompatibilityDataset(Dataset):
    """Dataset for training embedding diffusion on outfit compatibility"""
    
    def __init__(self, 
                 outfit_pairs: List[Tuple[np.ndarray, np.ndarray]], 
                 contexts: List[Dict[str, Any]]):
        self.outfit_pairs = outfit_pairs
        self.contexts = contexts
        
    def __len__(self) -> int:
        return len(self.outfit_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        query_emb, target_emb = self.outfit_pairs[idx]
        context = self.contexts[idx]
        
        return {
            'query_embedding': torch.tensor(query_emb, dtype=torch.float32),
            'target_embedding': torch.tensor(target_emb, dtype=torch.float32),
            'context': context
        }

class EmbeddingDiffusionTrainer:
    """Training pipeline for embedding diffusion model"""
    
    def __init__(self, 
                 model: EmbeddingDiffusionModel, 
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-6):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        self.train_losses = []
        self.val_losses = []
        
    def train_step(self, batch: Dict[str, Any]) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        query_embeddings = batch['query_embedding'].to(self.device)
        target_embeddings = batch['target_embedding'].to(self.device)
        contexts = batch['context']
        
        batch_size = query_embeddings.size(0)
        
        # Sample noise and timesteps
        noise = torch.randn_like(target_embeddings)
        timesteps = self.model.noise_scheduler.sample_timesteps(batch_size, self.device)
        
        # Add noise to target embeddings
        noisy_embeddings = self.model.noise_scheduler.add_noise(target_embeddings, noise, timesteps)
        
        # Predict noise
        predicted_noise = self.model(noisy_embeddings, timesteps, query_embeddings, contexts)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validation step"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                query_embeddings = batch['query_embedding'].to(self.device)
                target_embeddings = batch['target_embedding'].to(self.device)
                contexts = batch['context']
                
                batch_size = query_embeddings.size(0)
                
                # Sample noise and timesteps
                noise = torch.randn_like(target_embeddings)
                timesteps = self.model.noise_scheduler.sample_timesteps(batch_size, self.device)
                
                # Add noise to target embeddings
                noisy_embeddings = self.model.noise_scheduler.add_noise(target_embeddings, noise, timesteps)
                
                # Predict noise
                predicted_noise = self.model(noisy_embeddings, timesteps, query_embeddings, contexts)
                
                # Compute loss
                loss = F.mse_loss(predicted_noise, noise)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              num_epochs: int = 100,
              save_path: str = "models/embedding_diffusion.pth") -> Dict[str, List[float]]:
        """Full training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            epoch_train_loss = 0.0
            num_train_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                loss = self.train_step(batch)
                epoch_train_loss += loss
                num_train_batches += 1
            
            avg_train_loss = epoch_train_loss / num_train_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
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
            'val_losses': self.val_losses
        }

def create_sample_diffusion_model() -> Tuple[EmbeddingDiffusionModel, EmbeddingDiffusionTrainer]:
    """Create sample diffusion model and trainer for testing"""
    config = DiffusionConfig(
        embedding_dim=512,
        hidden_dim=1024,
        num_layers=8,
        num_heads=16,
        timesteps=1000
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingDiffusionModel(config)
    trainer = EmbeddingDiffusionTrainer(model, device)
    
    return model, trainer

if __name__ == "__main__":
    # Create sample model
    model, trainer = create_sample_diffusion_model()
    
    logger.info("Embedding Diffusion Model ready for Generative Fashion AI")
    logger.info("Key features:")
    logger.info("- Latent diffusion in embedding space")
    logger.info("- Conditional generation with context (occasion, season, user)")
    logger.info("- DDPM sampling with classifier-free guidance")
    logger.info("- Integration with existing CLIP/BLIP/Fashion encoders")
    logger.info("- Training pipeline for outfit compatibility")
    logger.info("- Noise scheduling and denoising process")
    
    # Test forward pass
    batch_size = 4
    device = next(model.parameters()).device
    
    # Sample inputs
    noisy_embeddings = torch.randn(batch_size, 512, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    query_embedding = torch.randn(batch_size, 512, device=device)
    context = {
        'occasion': ['casual', 'work', 'party', 'formal'],
        'season': ['spring', 'summer', 'autumn', 'winter'],
        'style': ['minimalist', 'bohemian', 'classic', 'trendy'],
        'user_embedding': torch.randn(batch_size, 256, device=device)
    }
    
    # Forward pass
    with torch.no_grad():
        predicted_noise = model(noisy_embeddings, timesteps, query_embedding, context)
        logger.info(f"Forward pass successful: {predicted_noise.shape}")
        
        # Test sampling
        generated_embeddings = model.sample(
            query_embedding[:1], 
            {k: [v[0]] if isinstance(v, list) else v[:1] for k, v in context.items()},
            num_samples=3
        )
        logger.info(f"Sampling successful: {generated_embeddings.shape}")