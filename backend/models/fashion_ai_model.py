import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import Dict, List, Tuple, Optional

class AttentionBlock(nn.Module):
    """Attention mechanism for feature enhancement"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super(AttentionBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction for different clothing details"""
    def __init__(self, in_channels: int, out_channels: int):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class FashionAIModel(nn.Module):
    """Advanced Fashion AI Model for classification and feature learning"""
    
    def __init__(self, num_classes: int = 20, pretrained: bool = True):
        super(FashionAIModel, self).__init__()
        
        # Define fashion categories based on dataset structure
        self.categories = {
            'MEN': ['Denim', 'Jackets_Vests', 'Pants', 'Shirts_Polos', 'Shorts', 
                   'Sweaters', 'Tees_Tanks'],
            'WOMEN': ['Blouses_Shirts', 'Cardigans', 'Denim', 'Dresses', 
                     'Graphic_Tees', 'Jackets_Coats', 'Leggings', 'Pants', 
                     'Shorts', 'Skirts', 'Sweaters', 'Tees_Tanks']
        }
        
        # Backbone: ResNet50 with modifications
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Multi-scale feature extraction
        self.multi_scale = MultiScaleFeatureExtractor(2048, 512)
        
        # Attention mechanism
        self.attention = AttentionBlock(512)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension reduction
        self.feature_reducer = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classification heads
        self.gender_classifier = nn.Linear(256, 2)  # Male/Female
        self.category_classifier = nn.Linear(256, num_classes)
        
        # Texture and shape feature extractors
        self.texture_features = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )
        
        self.shape_features = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Dictionary containing predictions and features
        """
        # Extract backbone features
        features = self.backbone(x)  # (batch_size, 2048, H, W)
        
        # Multi-scale feature extraction
        multi_scale_features = self.multi_scale(features)  # (batch_size, 512, H, W)
        
        # Apply attention
        attended_features = self.attention(multi_scale_features)
        
        # Global pooling
        pooled_features = self.global_pool(attended_features).flatten(1)  # (batch_size, 512)
        
        # Feature reduction
        reduced_features = self.feature_reducer(pooled_features)  # (batch_size, 256)
        
        # Classifications
        gender_logits = self.gender_classifier(reduced_features)
        category_logits = self.category_classifier(reduced_features)
        
        # Extract specialized features
        texture_feat = self.texture_features(reduced_features)
        shape_feat = self.shape_features(reduced_features)
        
        return {
            'gender_logits': gender_logits,
            'category_logits': category_logits,
            'features': reduced_features,
            'texture_features': texture_feat,
            'shape_features': shape_feat,
            'raw_features': pooled_features
        }
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for similarity computation"""
        with torch.no_grad():
            output = self.forward(x)
            return output['features']
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            
            gender_probs = F.softmax(output['gender_logits'], dim=1)
            category_probs = F.softmax(output['category_logits'], dim=1)
            
            return {
                'gender_predictions': torch.argmax(gender_probs, dim=1),
                'category_predictions': torch.argmax(category_probs, dim=1),
                'gender_probabilities': gender_probs,
                'category_probabilities': category_probs,
                'features': output['features']
            }

class FashionGAN(nn.Module):
    """Generative Adversarial Network for fashion image generation"""
    
    def __init__(self, latent_dim: int = 100, num_classes: int = 20):
        super(FashionGAN, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Generator
        self.generator = self._build_generator()
        
        # Discriminator
        self.discriminator = self._build_discriminator()
    
    def _build_generator(self):
        """Build generator network"""
        return nn.Sequential(
            # Input: latent_dim + num_classes
            nn.Linear(self.latent_dim + self.num_classes, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU(inplace=True),
            
            # Reshape to (batch_size, 256, 8, 8)
            nn.Unflatten(1, (256, 8, 8)),
            
            # Upsample to 16x16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample to 32x32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample to 64x64
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Upsample to 128x128
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final layer to 256x256
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def _build_discriminator(self):
        """Build discriminator network"""
        return nn.Sequential(
            # Input: 3x256x256
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x128x128
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x64x64
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x32x32
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x16x16
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256x8x8
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid()
        )
    
    def generate(self, batch_size: int, class_labels: torch.Tensor, device: str = 'cpu'):
        """Generate fashion images"""
        # Sample random noise
        noise = torch.randn(batch_size, self.latent_dim, device=device)
        
        # One-hot encode class labels
        class_one_hot = F.one_hot(class_labels, self.num_classes).float()
        
        # Concatenate noise and class labels
        gen_input = torch.cat([noise, class_one_hot], dim=1)
        
        # Generate images
        generated_images = self.generator(gen_input)
        
        return generated_images

class FashionAISystem:
    """Complete Fashion AI System integrating classification and generation"""
    
    def __init__(self, num_classes: int = 20, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        
        # Initialize models
        self.classifier = FashionAIModel(num_classes=num_classes).to(device)
        self.gan = FashionGAN(num_classes=num_classes).to(device)
        
        # Category mappings
        self.category_to_idx = self._build_category_mapping()
        self.idx_to_category = {v: k for k, v in self.category_to_idx.items()}
    
    def _build_category_mapping(self) -> Dict[str, int]:
        """Build category to index mapping"""
        categories = []
        
        # Men's categories
        men_categories = ['MEN-Denim', 'MEN-Jackets_Vests', 'MEN-Pants', 
                         'MEN-Shirts_Polos', 'MEN-Shorts', 'MEN-Sweaters', 'MEN-Tees_Tanks']
        
        # Women's categories
        women_categories = ['WOMEN-Blouses_Shirts', 'WOMEN-Cardigans', 'WOMEN-Denim', 
                           'WOMEN-Dresses', 'WOMEN-Graphic_Tees', 'WOMEN-Jackets_Coats', 
                           'WOMEN-Leggings', 'WOMEN-Pants', 'WOMEN-Shorts', 'WOMEN-Skirts', 
                           'WOMEN-Sweaters', 'WOMEN-Tees_Tanks']
        
        categories.extend(men_categories)
        categories.extend(women_categories)
        
        return {cat: idx for idx, cat in enumerate(categories[:self.num_classes])}
    
    def classify_image(self, image: torch.Tensor) -> Dict[str, any]:
        """Classify a fashion image"""
        predictions = self.classifier.predict(image.unsqueeze(0).to(self.device))
        
        category_idx = int(predictions['category_predictions'][0].item())
        gender_idx = int(predictions['gender_predictions'][0].item())
        
        return {
            'category': self.idx_to_category.get(category_idx, 'Unknown'),
            'gender': 'Male' if gender_idx == 0 else 'Female',
            'category_confidence': float(predictions['category_probabilities'][0][category_idx].item()),
            'gender_confidence': float(predictions['gender_probabilities'][0][gender_idx].item()),
            'features': predictions['features'][0]
        }
    
    def generate_similar_image(self, reference_image: torch.Tensor) -> torch.Tensor:
        """Generate an image similar to the reference"""
        # Classify reference image
        classification = self.classify_image(reference_image)
        
        # Get category index
        category_name = classification['category']
        category_idx = self.category_to_idx.get(category_name, 0)
        
        # Generate similar image
        class_labels = torch.tensor([category_idx], device=self.device)
        generated_image = self.gan.generate(1, class_labels, self.device)
        
        return generated_image[0]
    
    def save_models(self, classifier_path: str, gan_path: str):
        """Save trained models"""
        torch.save({
            'model_state_dict': self.classifier.state_dict(),
            'category_mapping': self.category_to_idx
        }, classifier_path)
        
        torch.save({
            'generator_state_dict': self.gan.generator.state_dict(),
            'discriminator_state_dict': self.gan.discriminator.state_dict()
        }, gan_path)
    
    def load_models(self, classifier_path: str, gan_path: str):
        """Load trained models"""
        # Load classifier
        classifier_checkpoint = torch.load(classifier_path, map_location=self.device)
        self.classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
        self.category_to_idx = classifier_checkpoint['category_mapping']
        self.idx_to_category = {v: k for k, v in self.category_to_idx.items()}
        
        # Load GAN
        gan_checkpoint = torch.load(gan_path, map_location=self.device)
        self.gan.generator.load_state_dict(gan_checkpoint['generator_state_dict'])
        self.gan.discriminator.load_state_dict(gan_checkpoint['discriminator_state_dict'])

# Model configuration
MODEL_CONFIG = {
    'num_classes': 19,  # Based on dataset analysis
    'input_size': (256, 256),
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'num_epochs': 100,
    'patience': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}