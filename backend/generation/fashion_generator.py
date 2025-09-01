import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Import our models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.fashion_ai_model import FashionGAN, FashionAISystem
from inference.fashion_predictor import FashionPredictor

class FashionGenerator:
    """Fashion image generator using GANs"""
    
    def __init__(self,
                 generator_path: str = None,
                 predictor: FashionPredictor = None,
                 latent_dim: int = 128,
                 device: str = 'auto'):
        """
        Args:
            generator_path: Path to trained generator model
            predictor: Fashion predictor for guidance
            latent_dim: Latent dimension for noise vector
            device: Device to use
        """
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.latent_dim = latent_dim
        self.predictor = predictor
        
        # Initialize or load generator
        if generator_path and os.path.exists(generator_path):
            self.generator = self._load_generator(generator_path)
        else:
            # Create new generator
            self.gan = FashionGAN(latent_dim=latent_dim)
            self.generator = self.gan.generator
        
        self.generator = self.generator.to(self.device)
        self.generator.eval()
        
        # Setup transforms
        self.denormalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        
        print(f"Fashion generator initialized on {self.device}")
    
    def _load_generator(self, generator_path: str) -> nn.Module:
        """Load trained generator"""
        checkpoint = torch.load(generator_path, map_location=self.device)
        
        # Create generator
        gan = FashionGAN(latent_dim=self.latent_dim)
        
        # Load state dict
        if 'generator_state_dict' in checkpoint:
            gan.generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            gan.generator.load_state_dict(checkpoint)
        
        return gan.generator
    
    def generate_random(self, num_images: int = 1, 
                       category_condition: str = None,
                       save_path: str = None) -> List[torch.Tensor]:
        """Generate random fashion images"""
        with torch.no_grad():
            # Generate random noise
            noise = torch.randn(num_images, self.latent_dim, device=self.device)
            
            # Add category conditioning if available
            if category_condition and hasattr(self.generator, 'num_classes'):
                # This would require a conditional GAN
                pass
            
            # Generate images
            generated_images = self.generator(noise)
            
            # Denormalize for visualization
            images_denorm = []
            for i in range(num_images):
                img = self.denormalize(generated_images[i])
                img = torch.clamp(img, 0, 1)
                images_denorm.append(img)
            
            # Save images if path provided
            if save_path:
                self._save_generated_images(images_denorm, save_path)
            
            return images_denorm
    
    def generate_similar(self, reference_image: Union[str, Image.Image, np.ndarray],
                        num_variations: int = 5,
                        similarity_strength: float = 0.7,
                        save_path: str = None) -> List[torch.Tensor]:
        """Generate images similar to a reference image"""
        if self.predictor is None:
            raise ValueError("Predictor required for similarity-based generation")
        
        # Analyze reference image
        analysis = self.predictor.analyze_image_features(reference_image)
        reference_features = analysis['features']['backbone_features']
        
        # Generate multiple candidates and select most similar
        candidates = []
        similarities = []
        
        # Generate more candidates than needed
        num_candidates = num_variations * 3
        
        with torch.no_grad():
            for _ in range(num_candidates):
                # Generate random noise
                noise = torch.randn(1, self.latent_dim, device=self.device)
                
                # Generate image
                generated_img = self.generator(noise)[0]
                
                # Convert to PIL for analysis
                img_pil = self._tensor_to_pil(generated_img)
                
                # Analyze generated image
                try:
                    gen_analysis = self.predictor.analyze_image_features(img_pil)
                    gen_features = gen_analysis['features']['backbone_features']
                    
                    # Calculate similarity
                    ref_flat = reference_features.flatten()
                    gen_flat = gen_features.flatten()
                    similarity = np.dot(ref_flat, gen_flat) / (np.linalg.norm(ref_flat) * np.linalg.norm(gen_flat))
                    
                    candidates.append(generated_img)
                    similarities.append(similarity)
                    
                except Exception as e:
                    print(f"Error analyzing generated image: {e}")
                    continue
        
        # Select top similar images
        if similarities:
            sorted_indices = np.argsort(similarities)[-num_variations:]
            selected_images = [candidates[i] for i in sorted_indices]
            
            # Denormalize
            images_denorm = []
            for img in selected_images:
                img_denorm = self.denormalize(img)
                img_denorm = torch.clamp(img_denorm, 0, 1)
                images_denorm.append(img_denorm)
            
            # Save images if path provided
            if save_path:
                self._save_generated_images(images_denorm, save_path, prefix='similar_')
            
            return images_denorm
        else:
            print("No valid candidates generated")
            return self.generate_random(num_variations, save_path=save_path)
    
    def interpolate_between(self, image1: Union[str, Image.Image, np.ndarray],
                           image2: Union[str, Image.Image, np.ndarray],
                           num_steps: int = 5,
                           save_path: str = None) -> List[torch.Tensor]:
        """Generate interpolation between two images"""
        # This is a simplified version - in practice, you'd need to invert images to latent space
        # For now, we'll generate a sequence that transitions between similar styles
        
        if self.predictor is None:
            raise ValueError("Predictor required for interpolation")
        
        # Analyze both images
        analysis1 = self.predictor.analyze_image_features(image1)
        analysis2 = self.predictor.analyze_image_features(image2)
        
        # Generate images that gradually transition
        interpolated_images = []
        
        with torch.no_grad():
            for i in range(num_steps):
                # Linear interpolation factor
                alpha = i / (num_steps - 1)
                
                # Generate with varying noise (simplified approach)
                noise1 = torch.randn(1, self.latent_dim, device=self.device)
                noise2 = torch.randn(1, self.latent_dim, device=self.device)
                
                # Interpolate noise
                interpolated_noise = (1 - alpha) * noise1 + alpha * noise2
                
                # Generate image
                generated_img = self.generator(interpolated_noise)[0]
                
                # Denormalize
                img_denorm = self.denormalize(generated_img)
                img_denorm = torch.clamp(img_denorm, 0, 1)
                interpolated_images.append(img_denorm)
        
        # Save images if path provided
        if save_path:
            self._save_generated_images(interpolated_images, save_path, prefix='interp_')
        
        return interpolated_images
    
    def generate_trend_prediction(self, recent_images: List[Union[str, Image.Image, np.ndarray]],
                                 num_predictions: int = 3,
                                 save_path: str = None) -> Dict[str, Any]:
        """Generate images representing predicted future trends"""
        if self.predictor is None:
            raise ValueError("Predictor required for trend prediction")
        
        # Analyze recent trends
        from inference.fashion_predictor import FashionPatternLearner
        pattern_learner = FashionPatternLearner(self.predictor)
        
        # Learn from recent images
        patterns = pattern_learner.learn_from_images(recent_images)
        
        # Predict next trend
        trend_prediction = pattern_learner.predict_next_trend(recent_images)
        
        # Generate images based on trend prediction
        predicted_images = []
        
        with torch.no_grad():
            for _ in range(num_predictions):
                # Generate with trend-influenced noise
                noise = torch.randn(1, self.latent_dim, device=self.device)
                
                # Add some trend bias (simplified)
                if trend_prediction['trend_direction'] == 'evolving':
                    # Add more variation for evolving trends
                    noise = noise * 1.2
                
                generated_img = self.generator(noise)[0]
                
                # Denormalize
                img_denorm = self.denormalize(generated_img)
                img_denorm = torch.clamp(img_denorm, 0, 1)
                predicted_images.append(img_denorm)
        
        # Save images if path provided
        if save_path:
            self._save_generated_images(predicted_images, save_path, prefix='trend_')
        
        return {
            'predicted_images': predicted_images,
            'trend_analysis': trend_prediction,
            'learned_patterns': patterns
        }
    
    def generate_style_transfer(self, content_image: Union[str, Image.Image, np.ndarray],
                               style_category: str,
                               save_path: str = None) -> torch.Tensor:
        """Generate image with content from one image and style from a category"""
        # This is a simplified version - real style transfer would require additional training
        
        # For now, generate an image and analyze if it matches the desired style
        best_image = None
        best_score = -1
        
        with torch.no_grad():
            for _ in range(20):  # Try multiple generations
                noise = torch.randn(1, self.latent_dim, device=self.device)
                generated_img = self.generator(noise)[0]
                
                # Convert to PIL for analysis
                img_pil = self._tensor_to_pil(generated_img)
                
                if self.predictor:
                    try:
                        prediction = self.predictor.predict_single(img_pil)
                        
                        # Check if it matches desired style category
                        if style_category.lower() in prediction['category']['predicted'].lower():
                            score = prediction['category']['confidence']
                            if score > best_score:
                                best_score = score
                                best_image = generated_img
                    except Exception:
                        continue
        
        if best_image is not None:
            # Denormalize
            img_denorm = self.denormalize(best_image)
            img_denorm = torch.clamp(img_denorm, 0, 1)
            
            # Save if path provided
            if save_path:
                self._save_generated_images([img_denorm], save_path, prefix='style_')
            
            return img_denorm
        else:
            # Fallback to random generation
            return self.generate_random(1, save_path=save_path)[0]
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        # Denormalize
        img = self.denormalize(tensor)
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        
        return Image.fromarray(img_np)
    
    def _save_generated_images(self, images: List[torch.Tensor], 
                              save_path: str, 
                              prefix: str = 'generated_'):
        """Save generated images"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for i, img in enumerate(images):
            filename = f"{prefix}{timestamp}_{i:03d}.png"
            filepath = save_dir / filename
            save_image(img, filepath)
        
        # Also save a grid
        if len(images) > 1:
            grid = make_grid(torch.stack(images), nrow=min(len(images), 5), padding=2)
            grid_filename = f"{prefix}grid_{timestamp}.png"
            save_image(grid, save_dir / grid_filename)
        
        print(f"Saved {len(images)} images to {save_dir}")
    
    def create_fashion_collection(self, theme: str, 
                                 num_items: int = 10,
                                 save_path: str = None) -> Dict[str, Any]:
        """Create a cohesive fashion collection based on a theme"""
        collection_images = []
        collection_analysis = []
        
        # Generate base style
        base_noise = torch.randn(1, self.latent_dim, device=self.device)
        
        with torch.no_grad():
            for i in range(num_items):
                # Add variation to base style
                variation = torch.randn(1, self.latent_dim, device=self.device) * 0.3
                noise = base_noise + variation
                
                # Generate image
                generated_img = self.generator(noise)[0]
                
                # Denormalize
                img_denorm = self.denormalize(generated_img)
                img_denorm = torch.clamp(img_denorm, 0, 1)
                collection_images.append(img_denorm)
                
                # Analyze if predictor available
                if self.predictor:
                    try:
                        img_pil = self._tensor_to_pil(generated_img)
                        analysis = self.predictor.analyze_image_features(img_pil)
                        collection_analysis.append(analysis)
                    except Exception:
                        collection_analysis.append(None)
        
        # Save collection
        if save_path:
            collection_dir = Path(save_path) / f"collection_{theme}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._save_generated_images(collection_images, collection_dir, prefix=f'{theme}_')
            
            # Save collection metadata
            metadata = {
                'theme': theme,
                'num_items': num_items,
                'generation_timestamp': datetime.now().isoformat(),
                'analysis': collection_analysis
            }
            
            with open(collection_dir / 'collection_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        return {
            'theme': theme,
            'images': collection_images,
            'analysis': collection_analysis,
            'cohesion_score': self._calculate_collection_cohesion(collection_analysis)
        }
    
    def _calculate_collection_cohesion(self, analyses: List[Dict]) -> float:
        """Calculate how cohesive a collection is"""
        if not analyses or len(analyses) < 2:
            return 0.0
        
        valid_analyses = [a for a in analyses if a is not None]
        if len(valid_analyses) < 2:
            return 0.0
        
        # Calculate average similarity between items
        similarities = []
        
        for i in range(len(valid_analyses)):
            for j in range(i + 1, len(valid_analyses)):
                try:
                    feat1 = valid_analyses[i]['features']['backbone_features'].flatten()
                    feat2 = valid_analyses[j]['features']['backbone_features'].flatten()
                    
                    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
                    similarities.append(similarity)
                except Exception:
                    continue
        
        return float(np.mean(similarities)) if similarities else 0.0

class FashionGANTrainer:
    """Trainer for Fashion GAN models"""
    
    def __init__(self, 
                 data_loader,
                 latent_dim: int = 128,
                 device: str = 'auto'):
        """
        Args:
            data_loader: DataLoader for training data
            latent_dim: Latent dimension
            device: Device to use
        """
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.data_loader = data_loader
        self.latent_dim = latent_dim
        
        # Initialize GAN
        self.gan = FashionGAN(latent_dim=latent_dim)
        self.generator = self.gan.generator.to(self.device)
        self.discriminator = self.gan.discriminator.to(self.device)
        
        # Setup optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        print(f"GAN trainer initialized on {self.device}")
    
    def train(self, epochs: int = 100, save_interval: int = 10, save_dir: str = 'gan_checkpoints'):
        """Train the GAN"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            for i, batch in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch+1}")):
                real_images = batch['image'].to(self.device)
                batch_size = real_images.size(0)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                # Real images
                real_labels = torch.ones(batch_size, 1, device=self.device)
                real_output = self.discriminator(real_images)
                d_loss_real = self.criterion(real_output, real_labels)
                
                # Fake images
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_images = self.generator(noise)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                fake_output = self.discriminator(fake_images.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                
                fake_output = self.discriminator(fake_images)
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
            
            print(f"Epoch [{epoch+1}/{epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(save_path / f'gan_epoch_{epoch+1}.pth', epoch + 1)
                
                # Generate sample images
                self.generate_samples(save_path / f'samples_epoch_{epoch+1}.png')
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
        }, path)
    
    def generate_samples(self, save_path: str, num_samples: int = 16):
        """Generate sample images"""
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, device=self.device)
            fake_images = self.generator(noise)
            
            # Denormalize
            denormalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
            
            fake_images = torch.stack([denormalize(img) for img in fake_images])
            fake_images = torch.clamp(fake_images, 0, 1)
            
            # Save grid
            grid = make_grid(fake_images, nrow=4, padding=2)
            save_image(grid, save_path)

# Utility functions
def create_generator(generator_path: str = None, 
                   predictor: FashionPredictor = None,
                   device: str = 'auto') -> FashionGenerator:
    """Create a fashion generator"""
    return FashionGenerator(
        generator_path=generator_path,
        predictor=predictor,
        device=device
    )

def train_fashion_gan(data_loader, 
                     epochs: int = 100,
                     save_dir: str = 'gan_checkpoints',
                     device: str = 'auto') -> FashionGANTrainer:
    """Train a fashion GAN"""
    trainer = FashionGANTrainer(data_loader, device=device)
    trainer.train(epochs=epochs, save_dir=save_dir)
    return trainer