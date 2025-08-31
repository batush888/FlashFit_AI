#!/usr/bin/env python3
"""
BLIP + CLIP Integration Enhancement for Phase 2

This module implements:
1. Expanded BLIP vocabulary with fashion terminology (>95% coverage)
2. CLIP text encoder re-encoding for stronger alignment
3. Fusion layer combining BLIP + CLIP text embeddings
4. Training on outfit dataset pairs for caption-to-image similarity
5. Advanced evaluation metrics for text-image retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPModel, CLIPProcessor, CLIPTokenizer
)
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
from datetime import datetime
from PIL import Image
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FashionVocabularyExpander:
    """
    Expands BLIP vocabulary with comprehensive fashion terminology
    """
    
    def __init__(self):
        self.fashion_vocabulary = self._create_comprehensive_fashion_vocabulary()
        self.vocabulary_coverage = 0.0
        
        logger.info(f"Fashion vocabulary initialized with {len(self.fashion_vocabulary)} terms")
    
    def _create_comprehensive_fashion_vocabulary(self) -> Dict[str, List[str]]:
        """Create comprehensive fashion vocabulary for >95% coverage"""
        return {
            "clothing_types": [
                # Tops
                "shirt", "blouse", "t-shirt", "tank top", "camisole", "sweater", "cardigan",
                "hoodie", "sweatshirt", "pullover", "turtleneck", "crop top", "tube top",
                "halter top", "off-shoulder top", "bodysuit", "vest", "waistcoat",
                
                # Bottoms
                "pants", "trousers", "jeans", "leggings", "shorts", "skirt", "mini skirt",
                "maxi skirt", "midi skirt", "pencil skirt", "a-line skirt", "pleated skirt",
                "culottes", "palazzo pants", "wide-leg pants", "skinny jeans", "bootcut jeans",
                
                # Dresses
                "dress", "maxi dress", "mini dress", "midi dress", "cocktail dress",
                "evening gown", "sundress", "shift dress", "wrap dress", "bodycon dress",
                "a-line dress", "fit and flare dress", "shirtdress", "slip dress",
                
                # Outerwear
                "jacket", "blazer", "coat", "trench coat", "pea coat", "puffer jacket",
                "bomber jacket", "denim jacket", "leather jacket", "windbreaker",
                "parka", "overcoat", "cape", "poncho", "shawl", "wrap",
                
                # Footwear
                "shoes", "sneakers", "boots", "sandals", "heels", "flats", "loafers",
                "oxfords", "pumps", "stilettos", "wedges", "ankle boots", "knee-high boots",
                "combat boots", "chelsea boots", "moccasins", "espadrilles", "clogs",
                
                # Accessories
                "bag", "purse", "handbag", "clutch", "tote bag", "crossbody bag",
                "backpack", "satchel", "messenger bag", "wallet", "belt", "scarf",
                "hat", "cap", "beanie", "beret", "fedora", "sunglasses", "jewelry",
                "necklace", "bracelet", "earrings", "ring", "watch", "brooch"
            ],
            
            "colors": [
                # Basic colors
                "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
                "black", "white", "gray", "grey", "beige", "tan", "cream", "ivory",
                
                # Extended colors
                "navy", "navy blue", "royal blue", "sky blue", "turquoise", "teal",
                "emerald", "forest green", "olive", "lime", "mint", "sage",
                "burgundy", "wine", "maroon", "crimson", "scarlet", "coral",
                "salmon", "peach", "apricot", "gold", "silver", "bronze",
                "copper", "rose gold", "champagne", "taupe", "khaki", "camel",
                "chocolate", "espresso", "charcoal", "slate", "pearl", "platinum",
                
                # Color combinations
                "multicolor", "colorblock", "ombre", "gradient", "two-tone",
                "monochrome", "neutral", "pastel", "neon", "metallic"
            ],
            
            "materials": [
                # Natural fibers
                "cotton", "silk", "wool", "linen", "hemp", "bamboo", "cashmere",
                "mohair", "alpaca", "angora", "merino wool", "organic cotton",
                
                # Synthetic fibers
                "polyester", "nylon", "spandex", "elastane", "lycra", "acrylic",
                "rayon", "viscose", "modal", "tencel", "microfiber",
                
                # Specialty materials
                "denim", "leather", "suede", "patent leather", "faux leather",
                "velvet", "corduroy", "tweed", "flannel", "jersey", "knit",
                "mesh", "lace", "chiffon", "satin", "taffeta", "organza",
                "tulle", "sequins", "beads", "embroidery", "applique",
                
                # Textures
                "smooth", "textured", "ribbed", "cable knit", "waffle knit",
                "fleece", "terry cloth", "canvas", "twill", "poplin"
            ],
            
            "patterns": [
                "solid", "striped", "polka dot", "floral", "geometric", "abstract",
                "plaid", "checkered", "gingham", "houndstooth", "herringbone",
                "paisley", "animal print", "leopard print", "zebra print",
                "snake print", "camouflage", "tie-dye", "ombre", "gradient",
                "embroidered", "printed", "painted", "beaded", "sequined",
                "metallic", "glittery", "holographic", "iridescent"
            ],
            
            "styles": [
                # General styles
                "casual", "formal", "business", "professional", "elegant", "chic",
                "sophisticated", "classy", "trendy", "fashionable", "stylish",
                "modern", "contemporary", "classic", "timeless", "vintage",
                "retro", "antique", "traditional", "ethnic", "cultural",
                
                # Specific aesthetics
                "minimalist", "maximalist", "bohemian", "boho", "hippie",
                "romantic", "feminine", "masculine", "androgynous", "edgy",
                "punk", "gothic", "grunge", "preppy", "sporty", "athletic",
                "streetwear", "urban", "hip-hop", "skater", "surfer",
                "country", "western", "cowboy", "rustic", "outdoor",
                
                # Fashion movements
                "art deco", "mod", "psychedelic", "disco", "new wave",
                "power dressing", "normcore", "cottagecore", "dark academia"
            ],
            
            "occasions": [
                "everyday", "casual", "work", "office", "business", "meeting",
                "formal", "black tie", "cocktail", "party", "evening", "date",
                "wedding", "graduation", "prom", "homecoming", "gala",
                "vacation", "travel", "beach", "summer", "winter", "spring",
                "fall", "autumn", "holiday", "christmas", "new year",
                "workout", "gym", "yoga", "running", "sports", "outdoor",
                "hiking", "camping", "festival", "concert", "club", "brunch"
            ],
            
            "fits": [
                "tight", "fitted", "slim", "skinny", "tailored", "structured",
                "loose", "relaxed", "oversized", "baggy", "flowy", "draped",
                "regular fit", "straight fit", "bootcut", "wide leg", "cropped",
                "high-waisted", "low-waisted", "mid-rise", "empire waist",
                "a-line", "fit and flare", "bodycon", "wrap", "asymmetric"
            ],
            
            "details": [
                "buttons", "zipper", "pockets", "belt", "tie", "bow", "ruffle",
                "pleats", "gathering", "smocking", "shirring", "elastic",
                "drawstring", "lace-up", "buckle", "snap", "velcro", "hook",
                "collar", "lapel", "hood", "sleeves", "sleeveless", "cap sleeves",
                "long sleeves", "short sleeves", "three-quarter sleeves",
                "bell sleeves", "puff sleeves", "balloon sleeves", "cuffs",
                "hem", "fringe", "tassels", "trim", "piping", "contrast"
            ]
        }
    
    def calculate_vocabulary_coverage(self, text_corpus: List[str]) -> float:
        """
        Calculate vocabulary coverage for a given text corpus
        
        Args:
            text_corpus: List of text descriptions
            
        Returns:
            Coverage percentage (0-100)
        """
        all_fashion_terms = set()
        for category_terms in self.fashion_vocabulary.values():
            all_fashion_terms.update([term.lower() for term in category_terms])
        
        # Count matches in corpus
        corpus_text = " ".join(text_corpus).lower()
        matched_terms = sum(1 for term in all_fashion_terms if term in corpus_text)
        
        coverage = (matched_terms / len(all_fashion_terms)) * 100
        self.vocabulary_coverage = coverage
        
        logger.info(f"Vocabulary coverage: {coverage:.2f}%")
        return coverage
    
    def enhance_caption_with_vocabulary(self, caption: str) -> str:
        """
        Enhance a caption with additional fashion vocabulary
        
        Args:
            caption: Original caption
            
        Returns:
            Enhanced caption with fashion terms
        """
        enhanced_caption = caption.lower()
        
        # Add relevant fashion terms based on detected keywords
        for category, terms in self.fashion_vocabulary.items():
            for term in terms:
                if term.lower() in enhanced_caption:
                    # Add related terms from the same category
                    related_terms = [t for t in terms if t != term][:2]
                    if related_terms:
                        enhanced_caption += f" {' '.join(related_terms)}"
                    break
        
        return enhanced_caption

class BLIPCLIPFusionModel(nn.Module):
    """
    Fusion model combining BLIP and CLIP for enhanced text-image alignment
    """
    
    def __init__(self, 
                 blip_model_name: str = "Salesforce/blip-image-captioning-base",
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 fusion_dim: int = 512,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # Load pre-trained models
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name)
        
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        # Get embedding dimensions
        self.blip_dim = self.blip_model.config.text_config.hidden_size
        self.clip_dim = self.clip_model.config.projection_dim
        
        # Fusion layers
        self.blip_projection = nn.Sequential(
            nn.Linear(self.blip_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.clip_projection = nn.Sequential(
            nn.Linear(self.clip_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention-based fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout_rate
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.fusion_dim = fusion_dim
        
        logger.info(f"BLIPCLIPFusionModel initialized with fusion_dim={fusion_dim}")
    
    def encode_image_blip(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using BLIP
        
        Args:
            images: Input images tensor
            
        Returns:
            BLIP image embeddings
        """
        # Generate captions with BLIP
        with torch.no_grad():
            generated_ids = self.blip_model.generate(
                pixel_values=images,
                max_length=50,
                num_beams=4,
                early_stopping=True
            )
        
        # Get text embeddings from generated captions
        text_embeddings = self.blip_model.get_text_features(
            input_ids=generated_ids
        )
        
        return text_embeddings
    
    def encode_text_clip(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts using CLIP
        
        Args:
            texts: List of text descriptions
            
        Returns:
            CLIP text embeddings
        """
        # Tokenize texts
        inputs = self.clip_tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Get text embeddings
        text_embeddings = self.clip_model.get_text_features(**inputs)
        
        return text_embeddings
    
    def encode_image_clip(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using CLIP
        
        Args:
            images: Input images tensor
            
        Returns:
            CLIP image embeddings
        """
        image_embeddings = self.clip_model.get_image_features(images)
        return image_embeddings
    
    def fuse_embeddings(self, blip_embeddings: torch.Tensor, 
                       clip_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Fuse BLIP and CLIP embeddings using attention mechanism
        
        Args:
            blip_embeddings: BLIP text embeddings
            clip_embeddings: CLIP text embeddings
            
        Returns:
            Fused embeddings
        """
        # Project to fusion dimension
        blip_proj = self.blip_projection(blip_embeddings)
        clip_proj = self.clip_projection(clip_embeddings)
        
        # Prepare for attention (add sequence dimension)
        blip_seq = blip_proj.unsqueeze(1)  # [batch, 1, fusion_dim]
        clip_seq = clip_proj.unsqueeze(1)  # [batch, 1, fusion_dim]
        
        # Cross-attention between BLIP and CLIP
        attended_blip, _ = self.fusion_attention(
            query=blip_seq, key=clip_seq, value=clip_seq
        )
        attended_clip, _ = self.fusion_attention(
            query=clip_seq, key=blip_seq, value=blip_seq
        )
        
        # Remove sequence dimension
        attended_blip = attended_blip.squeeze(1)
        attended_clip = attended_clip.squeeze(1)
        
        # Concatenate and fuse
        concatenated = torch.cat([attended_blip, attended_clip], dim=1)
        fused_embeddings = self.fusion_layer(concatenated)
        
        return fused_embeddings
    
    def forward(self, images: torch.Tensor, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the fusion model
        
        Args:
            images: Input images tensor
            texts: List of text descriptions
            
        Returns:
            Dictionary with model outputs
        """
        # Get individual embeddings
        blip_text_embeddings = self.encode_image_blip(images)
        clip_text_embeddings = self.encode_text_clip(texts)
        clip_image_embeddings = self.encode_image_clip(images)
        
        # Fuse text embeddings
        fused_text_embeddings = self.fuse_embeddings(
            blip_text_embeddings, clip_text_embeddings
        )
        
        # Normalize embeddings
        fused_text_embeddings = F.normalize(fused_text_embeddings, dim=-1)
        clip_image_embeddings = F.normalize(clip_image_embeddings, dim=-1)
        
        outputs = {
            'fused_text_embeddings': fused_text_embeddings,
            'image_embeddings': clip_image_embeddings,
            'blip_text_embeddings': blip_text_embeddings,
            'clip_text_embeddings': clip_text_embeddings,
            'temperature': self.temperature
        }
        
        return outputs

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for text-image alignment
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, text_embeddings: torch.Tensor, 
                image_embeddings: torch.Tensor, 
                temperature: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between text and image embeddings
        
        Args:
            text_embeddings: Text embeddings [batch_size, embedding_dim]
            image_embeddings: Image embeddings [batch_size, embedding_dim]
            temperature: Temperature parameter for scaling
            
        Returns:
            Contrastive loss value
        """
        # Calculate similarity matrix
        logits = torch.matmul(text_embeddings, image_embeddings.T) * temperature.exp()
        
        # Create labels (diagonal should be positive pairs)
        batch_size = text_embeddings.size(0)
        labels = torch.arange(batch_size, device=text_embeddings.device)
        
        # Calculate cross-entropy loss for both directions
        loss_text_to_image = F.cross_entropy(logits, labels)
        loss_image_to_text = F.cross_entropy(logits.T, labels)
        
        # Average the losses
        total_loss = (loss_text_to_image + loss_image_to_text) / 2
        
        return total_loss

class FashionCaptionDataset(Dataset):
    """
    Dataset for fashion images with captions
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 captions: List[str],
                 vocabulary_expander: FashionVocabularyExpander,
                 transform: Optional[torch.nn.Module] = None,
                 enhance_captions: bool = True):
        self.image_paths = image_paths
        self.captions = captions
        self.vocabulary_expander = vocabulary_expander
        self.enhance_captions = enhance_captions
        self.transform = transform
        
        # Enhance captions if requested
        if self.enhance_captions:
            self.enhanced_captions = [
                self.vocabulary_expander.enhance_caption_with_vocabulary(caption)
                for caption in captions
            ]
        else:
            self.enhanced_captions = captions
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'caption': self.enhanced_captions[idx],
            'original_caption': self.captions[idx],
            'image_path': self.image_paths[idx]
        }

class BLIPCLIPTrainer:
    """
    Trainer for BLIP+CLIP fusion model
    """
    
    def __init__(self, 
                 model: BLIPCLIPFusionModel,
                 device: torch.device,
                 learning_rate: float = 1e-5,
                 weight_decay: float = 1e-6):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-7
        )
        
        # Loss function
        self.contrastive_loss = ContrastiveLoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'retrieval_accuracy': []
        }
        
        logger.info("BLIPCLIPTrainer initialized")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            captions = batch['caption']
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images, captions)
            
            # Calculate contrastive loss
            loss = self.contrastive_loss(
                outputs['fused_text_embeddings'],
                outputs['image_embeddings'],
                outputs['temperature']
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
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
        
        all_text_embeddings = []
        all_image_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                captions = batch['caption']
                
                outputs = self.model(images, captions)
                
                # Calculate loss
                loss = self.contrastive_loss(
                    outputs['fused_text_embeddings'],
                    outputs['image_embeddings'],
                    outputs['temperature']
                )
                total_loss += loss.item()
                
                # Store embeddings for retrieval evaluation
                all_text_embeddings.append(outputs['fused_text_embeddings'].cpu())
                all_image_embeddings.append(outputs['image_embeddings'].cpu())
        
        # Calculate retrieval accuracy
        text_embeddings = torch.cat(all_text_embeddings, dim=0)
        image_embeddings = torch.cat(all_image_embeddings, dim=0)
        
        retrieval_accuracy = self._calculate_retrieval_accuracy(
            text_embeddings, image_embeddings
        )
        
        avg_loss = total_loss / len(val_loader)
        
        metrics = {
            'loss': avg_loss,
            'retrieval_accuracy': retrieval_accuracy,
            'text_to_image_accuracy': retrieval_accuracy['text_to_image'],
            'image_to_text_accuracy': retrieval_accuracy['image_to_text']
        }
        
        return metrics
    
    def _calculate_retrieval_accuracy(self, text_embeddings: torch.Tensor, 
                                    image_embeddings: torch.Tensor, 
                                    k: int = 5) -> Dict[str, float]:
        """
        Calculate text-to-image and image-to-text retrieval accuracy
        
        Args:
            text_embeddings: Text embeddings
            image_embeddings: Image embeddings
            k: Top-k accuracy
            
        Returns:
            Dictionary with retrieval accuracies
        """
        # Calculate similarity matrix
        similarities = torch.matmul(text_embeddings, image_embeddings.T)
        
        # Text-to-image retrieval
        text_to_image_ranks = torch.argsort(similarities, dim=1, descending=True)
        text_to_image_correct = sum(
            i in text_to_image_ranks[i, :k] for i in range(len(text_embeddings))
        )
        text_to_image_accuracy = text_to_image_correct / len(text_embeddings)
        
        # Image-to-text retrieval
        image_to_text_ranks = torch.argsort(similarities.T, dim=1, descending=True)
        image_to_text_correct = sum(
            i in image_to_text_ranks[i, :k] for i in range(len(image_embeddings))
        )
        image_to_text_accuracy = image_to_text_correct / len(image_embeddings)
        
        return {
            'text_to_image': text_to_image_accuracy,
            'image_to_text': image_to_text_accuracy,
            'average': (text_to_image_accuracy + image_to_text_accuracy) / 2
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 50,
              save_path: str = "models/blip_clip_fusion.pth") -> Dict[str, List[float]]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            save_path: Path to save the best model
            
        Returns:
            Training history
        """
        best_retrieval_accuracy = 0.0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['retrieval_accuracy'].append(
                val_metrics['retrieval_accuracy']['average']
            )
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Retrieval Acc: {val_metrics['retrieval_accuracy']['average']:.4f}"
            )
            
            # Save best model
            if val_metrics['retrieval_accuracy']['average'] > best_retrieval_accuracy:
                best_retrieval_accuracy = val_metrics['retrieval_accuracy']['average']
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'retrieval_accuracy': best_retrieval_accuracy,
                    'training_history': self.training_history
                }, save_path)
                
                logger.info(f"New best model saved with retrieval accuracy: {best_retrieval_accuracy:.4f}")
        
        logger.info("Training completed")
        return self.training_history


def create_sample_fusion_pipeline():
    """
    Create a sample BLIP+CLIP fusion pipeline for testing
    """
    # Initialize components
    vocabulary_expander = FashionVocabularyExpander()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fusion_model = BLIPCLIPFusionModel(fusion_dim=512)
    
    trainer = BLIPCLIPTrainer(fusion_model, device, learning_rate=1e-5)
    
    logger.info("Sample BLIP+CLIP fusion pipeline created")
    logger.info(f"Fashion vocabulary size: {sum(len(terms) for terms in vocabulary_expander.fashion_vocabulary.values())}")
    logger.info(f"Model parameters: {sum(p.numel() for p in fusion_model.parameters()):,}")
    
    return vocabulary_expander, fusion_model, trainer


if __name__ == "__main__":
    # Create sample pipeline
    vocab_expander, model, trainer = create_sample_fusion_pipeline()
    
    logger.info("BLIP+CLIP Integration Enhancement ready for Phase 2")
    logger.info("Key features:")
    logger.info("- Expanded fashion vocabulary (>95% coverage target)")
    logger.info("- CLIP text encoder re-encoding for stronger alignment")
    logger.info("- Attention-based fusion layer")
    logger.info("- Contrastive learning for text-image similarity")
    logger.info("- Comprehensive retrieval evaluation metrics")