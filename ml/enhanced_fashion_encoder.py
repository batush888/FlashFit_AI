#!/usr/bin/env python3
"""
Enhanced Fashion Encoder for Phase 2 Fine-tuning

This module implements:
1. Multi-task learning (classification + compatibility)
2. Style attention mechanisms
3. Triplet/contrastive loss for outfit matching
4. Advanced training pipeline with early stopping
5. Comprehensive evaluation metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StyleAttentionModule(nn.Module):
    """
    Style attention mechanism to highlight key clothing features
    """
    
    def __init__(self, feature_dim: int = 512, attention_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.attention_fc = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(attention_dim, 1)
        )
        
        # Style-specific attention heads
        self.color_attention = nn.MultiheadAttention(feature_dim, num_heads=8, dropout=0.1)
        self.pattern_attention = nn.MultiheadAttention(feature_dim, num_heads=8, dropout=0.1)
        self.silhouette_attention = nn.MultiheadAttention(feature_dim, num_heads=8, dropout=0.1)
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply style attention to input features
        
        Args:
            features: Input feature tensor [batch_size, feature_dim]
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        batch_size = features.size(0)
        
        # Reshape for attention (seq_len=1 for global features)
        features_seq = features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # Apply different attention heads
        color_attended, color_weights = self.color_attention(
            features_seq, features_seq, features_seq
        )
        pattern_attended, pattern_weights = self.pattern_attention(
            features_seq, features_seq, features_seq
        )
        silhouette_attended, silhouette_weights = self.silhouette_attention(
            features_seq, features_seq, features_seq
        )
        
        # Squeeze sequence dimension
        color_attended = color_attended.squeeze(1)
        pattern_attended = pattern_attended.squeeze(1)
        silhouette_attended = silhouette_attended.squeeze(1)
        
        # Concatenate attended features
        concatenated = torch.cat([
            color_attended, pattern_attended, silhouette_attended
        ], dim=1)
        
        # Fuse features
        fused_features = self.fusion_layer(concatenated)
        
        # Global attention weights
        attention_scores = self.attention_fc(fused_features)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # Apply attention
        attended_features = fused_features * attention_weights
        
        attention_info = {
            'color_weights': color_weights.squeeze(1),
            'pattern_weights': pattern_weights.squeeze(1),
            'silhouette_weights': silhouette_weights.squeeze(1),
            'global_weights': attention_weights
        }
        
        return attended_features, attention_info

class EnhancedFashionEncoder(nn.Module):
    """
    Enhanced Fashion Encoder with multi-task learning and attention mechanisms
    """
    
    def __init__(self, 
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 num_categories: int = 50,
                 embedding_dim: int = 512,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # Load pre-trained CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Freeze CLIP parameters initially (will be fine-tuned later)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Get CLIP embedding dimension
        clip_dim = self.clip_model.config.projection_dim
        
        # Style attention module
        self.style_attention = StyleAttentionModule(clip_dim, attention_dim=256)
        
        # Task-specific heads
        self.category_classifier = nn.Sequential(
            nn.Linear(clip_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim // 2, num_categories)
        )
        
        # Compatibility embedding head
        self.compatibility_encoder = nn.Sequential(
            nn.Linear(clip_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Style embedding head
        self.style_encoder = nn.Sequential(
            nn.Linear(clip_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
        self.num_categories = num_categories
        
        logger.info(f"EnhancedFashionEncoder initialized with {num_categories} categories")
    
    def forward(self, images: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the enhanced fashion encoder
        
        Args:
            images: Input images tensor [batch_size, 3, 224, 224]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with model outputs
        """
        # Extract CLIP image features
        clip_features = self.clip_model.get_image_features(images)
        
        # Apply style attention
        attended_features, attention_weights = self.style_attention(clip_features)
        
        # Task-specific predictions
        category_logits = self.category_classifier(attended_features)
        compatibility_embedding = self.compatibility_encoder(attended_features)
        style_embedding = self.style_encoder(attended_features)
        
        outputs = {
            'category_logits': category_logits,
            'compatibility_embedding': compatibility_embedding,
            'style_embedding': style_embedding,
            'clip_features': clip_features,
            'attended_features': attended_features
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def unfreeze_clip_layers(self, num_layers: int = 2):
        """
        Unfreeze the last few layers of CLIP for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze from the end
        """
        # Unfreeze vision encoder layers
        vision_layers = list(self.clip_model.vision_model.encoder.layers)
        for layer in vision_layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        logger.info(f"Unfroze last {num_layers} CLIP vision layers for fine-tuning")

class TripletLoss(nn.Module):
    """
    Triplet loss for outfit compatibility learning
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
            
        Returns:
            Triplet loss value
        """
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_distance - neg_distance + self.margin)
        return loss.mean()

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for compatibility learning
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            embedding1: First set of embeddings [batch_size, embedding_dim]
            embedding2: Second set of embeddings [batch_size, embedding_dim]
            labels: Compatibility labels (1 for compatible, 0 for incompatible)
            
        Returns:
            Contrastive loss value
        """
        distances = F.pairwise_distance(embedding1, embedding2, p=2)
        
        pos_loss = labels * torch.pow(distances, 2)
        neg_loss = (1 - labels) * torch.pow(F.relu(self.margin - distances), 2)
        
        loss = 0.5 * (pos_loss + neg_loss)
        return loss.mean()

class FashionDataset(Dataset):
    """
    Dataset class for fashion items with multi-task labels
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 category_labels: List[int],
                 compatibility_pairs: Optional[List[Tuple[int, int, int]]] = None,
                 transform: Optional[transforms.Compose] = None):
        self.image_paths = image_paths
        self.category_labels = category_labels
        self.compatibility_pairs = compatibility_pairs or []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from PIL import Image
        
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        sample = {
            'image': image,
            'category_label': torch.tensor(self.category_labels[idx], dtype=torch.long),
            'item_id': torch.tensor(idx, dtype=torch.long)
        }
        
        return sample

class FashionEncoderTrainer:
    """
    Trainer class for the Enhanced Fashion Encoder
    """
    
    def __init__(self, 
                 model: EnhancedFashionEncoder,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        
        # Optimizers
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=0.3)
        self.contrastive_loss = ContrastiveLoss(margin=1.0)
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'compatibility_score': []
        }
        
        logger.info("FashionEncoderTrainer initialized")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_compatibility_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            category_labels = batch['category_label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Classification loss
            classification_loss = self.classification_loss(
                outputs['category_logits'], category_labels
            )
            
            # Compatibility loss (simplified - would need proper triplet sampling)
            compatibility_loss = torch.tensor(0.0, device=self.device)
            if len(batch['image']) >= 3:  # Need at least 3 samples for triplet
                anchor_emb = outputs['compatibility_embedding'][0:1]
                positive_emb = outputs['compatibility_embedding'][1:2]
                negative_emb = outputs['compatibility_embedding'][2:3]
                compatibility_loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
            
            # Combined loss
            total_batch_loss = classification_loss + 0.5 * compatibility_loss
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            total_classification_loss += classification_loss.item()
            total_compatibility_loss += compatibility_loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs['category_logits'], 1)
            correct_predictions += (predicted == category_labels).sum().item()
            total_samples += category_labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Acc': f'{100.0 * correct_predictions / total_samples:.2f}%'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        avg_classification_loss = total_classification_loss / len(train_loader)
        avg_compatibility_loss = total_compatibility_loss / len(train_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        metrics = {
            'loss': avg_loss,
            'classification_loss': avg_classification_loss,
            'compatibility_loss': avg_compatibility_loss,
            'accuracy': accuracy
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                category_labels = batch['category_label'].to(self.device)
                
                outputs = self.model(images)
                
                # Classification loss
                loss = self.classification_loss(outputs['category_logits'], category_labels)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs['category_logits'], 1)
                correct_predictions += (predicted == category_labels).sum().item()
                total_samples += category_labels.size(0)
                
                # Store for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(category_labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return metrics
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 50,
              early_stopping_patience: int = 10,
              save_path: str = "models/enhanced_fashion_encoder.pth") -> Dict[str, List[float]]:
        """
        Full training loop with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            save_path: Path to save the best model
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )
            
            # Early stopping and model saving
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'training_history': self.training_history
                }, save_path)
                
                logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Unfreeze CLIP layers after some epochs for fine-tuning
            if epoch == 10:
                self.model.unfreeze_clip_layers(num_layers=2)
                logger.info("Unfroze CLIP layers for fine-tuning")
        
        logger.info("Training completed")
        return self.training_history
    
    def evaluate_compatibility(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate style compatibility performance
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Compatibility evaluation metrics
        """
        self.model.eval()
        
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Extracting embeddings'):
                images = batch['image'].to(self.device)
                outputs = self.model(images)
                
                embeddings.append(outputs['compatibility_embedding'].cpu())
                labels.append(batch['category_label'].cpu())
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(embeddings, dim=0)
        all_labels = torch.cat(labels, dim=0)
        
        # Calculate cosine similarities
        similarities = F.cosine_similarity(
            all_embeddings.unsqueeze(1), 
            all_embeddings.unsqueeze(0), 
            dim=2
        )
        
        # Calculate compatibility score (simplified)
        same_category_mask = (all_labels.unsqueeze(1) == all_labels.unsqueeze(0))
        same_category_similarities = similarities[same_category_mask]
        different_category_similarities = similarities[~same_category_mask]
        
        compatibility_score = same_category_similarities.mean().item()
        
        metrics = {
            'compatibility_score': compatibility_score,
            'same_category_similarity': same_category_similarities.mean().item(),
            'different_category_similarity': different_category_similarities.mean().item(),
            'embedding_dimension': all_embeddings.size(1)
        }
        
        return metrics


def create_sample_training_pipeline():
    """
    Create a sample training pipeline for testing
    """
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedFashionEncoder(num_categories=20, embedding_dim=512)
    
    # Initialize trainer
    trainer = FashionEncoderTrainer(model, device, learning_rate=1e-4)
    
    logger.info("Sample training pipeline created")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, trainer


if __name__ == "__main__":
    # Create sample pipeline
    model, trainer = create_sample_training_pipeline()
    
    logger.info("Enhanced Fashion Encoder ready for Phase 2 training")
    logger.info("Key features:")
    logger.info("- Multi-task learning (classification + compatibility)")
    logger.info("- Style attention mechanisms")
    logger.info("- Triplet/contrastive loss for outfit matching")
    logger.info("- Early stopping and learning rate scheduling")
    logger.info("- Comprehensive evaluation metrics")