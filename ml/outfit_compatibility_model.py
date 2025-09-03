import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import List, Tuple, Optional

class OutfitCompatibilityModel(nn.Module):
    """Deep learning model for outfit compatibility prediction"""
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 num_items: int = 4,
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 dropout_rate: float = 0.3,
                 pretrained: bool = True):
        """
        Initialize the outfit compatibility model
        
        Args:
            backbone: CNN backbone architecture
            num_items: Maximum number of items in an outfit
            feature_dim: Dimension of item features
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout rate for regularization
            pretrained: Use pretrained weights
        """
        super(OutfitCompatibilityModel, self).__init__()
        
        self.num_items = num_items
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor for individual items
        self.feature_extractor = self._build_feature_extractor(backbone, pretrained)
        
        # Compatibility network
        self.compatibility_net = self._build_compatibility_network()
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def _build_feature_extractor(self, backbone: str, pretrained: bool) -> nn.Module:
        """Build feature extractor based on backbone architecture"""
        if backbone == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            # Remove the final classification layer
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            # Add adaptive pooling and projection
            feature_extractor.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, 1)))
            feature_extractor.add_module('flatten', nn.Flatten())
            feature_extractor.add_module('projection', nn.Linear(2048, self.feature_dim))
            
        elif backbone == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            feature_extractor.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, 1)))
            feature_extractor.add_module('flatten', nn.Flatten())
            feature_extractor.add_module('projection', nn.Linear(512, self.feature_dim))
            
        elif backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            feature_extractor.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, 1)))
            feature_extractor.add_module('flatten', nn.Flatten())
            feature_extractor.add_module('projection', nn.Linear(1280, self.feature_dim))
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        return feature_extractor
    
    def _build_compatibility_network(self) -> nn.Module:
        """Build compatibility network using attention mechanism"""
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        return nn.ModuleDict({
            'attention': self.attention_layer,
            'fusion': self.fusion_layers
        })
    
    def extract_item_features(self, items: torch.Tensor) -> torch.Tensor:
        """Extract features from individual clothing items"""
        batch_size, num_items, channels, height, width = items.shape
        
        # Reshape to process all items at once
        items_flat = items.view(batch_size * num_items, channels, height, width)
        
        # Extract features
        features = self.feature_extractor(items_flat)
        
        # Reshape back to (batch_size, num_items, feature_dim)
        features = features.view(batch_size, num_items, self.feature_dim)
        
        return features
    
    def compute_compatibility(self, item_features: torch.Tensor) -> torch.Tensor:
        """Compute compatibility between items using attention"""
        batch_size, num_items, feature_dim = item_features.shape
        
        # Apply multi-head attention
        attended_features, attention_weights = self.attention_layer(
            item_features, item_features, item_features
        )
        
        # Global average pooling over items
        pooled_features = torch.mean(attended_features, dim=1)
        
        # Apply compatibility fusion layers
        compatibility_features = self.fusion_layers(pooled_features)
        
        return compatibility_features
    
    def forward(self, items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Extract item features
        item_features = self.extract_item_features(items)
        
        # Compute compatibility
        compatibility_features = self.compute_compatibility(item_features)
        
        # Final classification
        compatibility_score = self.classifier(compatibility_features)
        
        return compatibility_score.squeeze(-1), item_features

class OutfitCompatibilityLoss(nn.Module):
    """Custom loss function for outfit compatibility"""
    
    def __init__(self, 
                 bce_weight: float = 1.0,
                 triplet_weight: float = 0.1,
                 margin: float = 0.5):
        """
        Initialize the loss function
        
        Args:
            bce_weight: Weight for binary cross-entropy loss
            triplet_weight: Weight for triplet loss
            margin: Margin for triplet loss
        """
        super(OutfitCompatibilityLoss, self).__init__()
        self.bce_weight = bce_weight
        self.triplet_weight = triplet_weight
        self.margin = margin
        
        self.bce_loss = nn.BCELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                item_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute combined loss"""
        # Binary cross-entropy loss
        bce_loss = self.bce_loss(predictions, targets)
        total_loss = self.bce_weight * bce_loss
        
        # Optional triplet loss for feature learning
        if item_features is not None and self.triplet_weight > 0:
            # Create triplets from compatible/incompatible outfits
            batch_size = item_features.shape[0]
            
            # Simple triplet creation (can be improved)
            if batch_size >= 3:
                anchor_idx = torch.arange(0, batch_size, 3)
                positive_idx = torch.arange(1, batch_size, 3)
                negative_idx = torch.arange(2, batch_size, 3)
                
                # Ensure indices are valid
                max_idx = min(len(anchor_idx), len(positive_idx), len(negative_idx))
                if max_idx > 0:
                    anchor_idx = anchor_idx[:max_idx]
                    positive_idx = positive_idx[:max_idx]
                    negative_idx = negative_idx[:max_idx]
                    
                    # Get pooled features for triplet loss
                    pooled_features = torch.mean(item_features, dim=1)
                    
                    anchor_features = pooled_features[anchor_idx]
                    positive_features = pooled_features[positive_idx]
                    negative_features = pooled_features[negative_idx]
                    
                    triplet_loss = self.triplet_loss(
                        anchor_features, positive_features, negative_features
                    )
                    total_loss += self.triplet_weight * triplet_loss
        
        return total_loss

class OutfitRecommendationModel(nn.Module):
    """Model for recommending compatible items for an existing outfit"""
    
    def __init__(self, 
                 compatibility_model: OutfitCompatibilityModel,
                 item_database: Optional[torch.Tensor] = None):
        """
        Initialize recommendation model
        
        Args:
            compatibility_model: Trained compatibility model
            item_database: Database of item features for recommendation
        """
        super(OutfitRecommendationModel, self).__init__()
        self.compatibility_model = compatibility_model
        self.item_database = item_database
        
    def recommend_items(self, 
                       partial_outfit: torch.Tensor,
                       candidate_items: torch.Tensor,
                       top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend compatible items for a partial outfit"""
        batch_size = partial_outfit.shape[0]
        num_candidates = candidate_items.shape[0]
        
        recommendations = []
        scores = []
        
        for i in range(batch_size):
            outfit = partial_outfit[i:i+1]  # Keep batch dimension
            outfit_scores = []
            
            for j in range(num_candidates):
                candidate = candidate_items[j:j+1]
                
                # Create complete outfit by combining partial outfit with candidate
                complete_outfit = torch.cat([outfit, candidate.unsqueeze(0)], dim=1)
                
                # Pad to required number of items
                while complete_outfit.shape[1] < self.compatibility_model.num_items:
                    complete_outfit = torch.cat([
                        complete_outfit, 
                        torch.zeros_like(candidate.unsqueeze(0))
                    ], dim=1)
                
                # Get compatibility score
                with torch.no_grad():
                    score, _ = self.compatibility_model(complete_outfit)
                    outfit_scores.append(score.item())
            
            # Get top-k recommendations
            outfit_scores = torch.tensor(outfit_scores)
            top_scores, top_indices = torch.topk(outfit_scores, min(top_k, len(outfit_scores)))
            
            recommendations.append(top_indices)
            scores.append(top_scores)
        
        return torch.stack(recommendations), torch.stack(scores)

def create_model(config: dict) -> OutfitCompatibilityModel:
    """Factory function to create model from config"""
    return OutfitCompatibilityModel(
        backbone=config.get('backbone', 'resnet50'),
        num_items=config.get('num_items', 4),
        feature_dim=config.get('feature_dim', 512),
        hidden_dim=config.get('hidden_dim', 256),
        dropout_rate=config.get('dropout_rate', 0.3),
        pretrained=config.get('pretrained', True)
    )

if __name__ == "__main__":
    # Example usage
    model = OutfitCompatibilityModel()
    
    # Test with dummy data
    batch_size = 8
    num_items = 4
    dummy_items = torch.randn(batch_size, num_items, 3, 224, 224)
    
    with torch.no_grad():
        compatibility_scores, item_features = model(dummy_items)
        print(f"Compatibility scores shape: {compatibility_scores.shape}")
        print(f"Item features shape: {item_features.shape}")
        print(f"Sample compatibility scores: {compatibility_scores[:5]}")