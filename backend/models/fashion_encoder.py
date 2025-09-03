from .clip_encoder import CLIPEncoder
from typing import Optional, List, Union, Dict, Any
import numpy as np
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
import nltk
import jieba
import re
from collections import Counter
import json
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class FashionEncoder(CLIPEncoder):
    """
    Enhanced Fashion-specific encoder that extends CLIPEncoder with advanced NLP capabilities.
    
    Features:
    - CLIP-based visual encoding
    - Sentence transformers for semantic text understanding
    - Multi-language support (English/Chinese) with NLTK and jieba
    - Fashion-specific text processing and attribute extraction
    - Advanced similarity computation with multiple modalities
    """
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", 
                 device: Optional[str] = None, sentence_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize enhanced fashion-specific encoder
        
        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights to use
            device: Device to run model on
            sentence_model: Sentence transformer model for text encoding
        """
        # Initialize base CLIP encoder
        super().__init__(model_name=model_name, pretrained=pretrained, device=device)
        
        # Initialize sentence transformer for advanced text processing
        try:
            self.sentence_model = SentenceTransformer(sentence_model)
            print(f"Sentence transformer '{sentence_model}' loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load sentence transformer: {e}")
            self.sentence_model = None
        
        # Initialize stopwords for text processing
        try:
            self.english_stopwords = set(stopwords.words('english'))
        except:
            self.english_stopwords = set()
            
        # Fashion-specific vocabulary and attributes
        self.fashion_attributes = {
            'colors': ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 
                      'brown', 'gray', 'grey', 'orange', 'navy', 'beige', 'khaki', 'maroon'],
            'materials': ['cotton', 'silk', 'wool', 'leather', 'denim', 'polyester', 'linen',
                         'cashmere', 'velvet', 'satin', 'chiffon', 'lace', 'suede', 'canvas'],
            'styles': ['casual', 'formal', 'vintage', 'modern', 'classic', 'trendy', 'bohemian',
                      'minimalist', 'elegant', 'sporty', 'chic', 'edgy', 'romantic', 'preppy'],
            'occasions': ['work', 'party', 'wedding', 'casual', 'formal', 'date', 'vacation',
                         'business', 'evening', 'daytime', 'weekend', 'holiday', 'summer', 'winter']
        }
        
        print(f"Enhanced FashionEncoder initialized:")
        print(f"  - CLIP model: {model_name} with {pretrained} weights")
        print(f"  - Sentence transformer: {sentence_model if self.sentence_model else 'Not available'}")
        print(f"  - Multi-language NLP support: {'Enabled' if self.english_stopwords else 'Limited'}")
    
    def embed_fashion_image(self, img_path: str) -> np.ndarray:
        """
        Generate fashion-specific image embedding
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Normalized embedding vector
        """
        # For now, this is the same as regular image embedding
        # TODO: Add fashion-specific preprocessing or post-processing
        return self.embed_image(img_path)
    
    def embed_fashion_text(self, text: str) -> np.ndarray:
        """
        Generate enhanced fashion-specific text embedding
        
        Args:
            text: Text description to embed
            
        Returns:
            Normalized embedding vector
        """
        # Preprocess text for fashion context
        processed_text = self.preprocess_fashion_text(text)
        
        # Use sentence transformer if available for better semantic understanding
        if self.sentence_model:
            try:
                embedding = self.sentence_model.encode(processed_text)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    return embedding / norm
                return embedding
            except Exception as e:
                print(f"Warning: Sentence transformer failed, falling back to CLIP: {e}")
        
        # Fallback to CLIP text embedding
        return self.embed_text(processed_text)
    
    def preprocess_fashion_text(self, text: str) -> str:
        """
        Preprocess text for fashion-specific understanding
        
        Args:
            text: Raw text description
            
        Returns:
            Processed text optimized for fashion understanding
        """
        if not text:
            return text
            
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Tokenize based on language detection
        if self._contains_chinese(text):
            # Use jieba for Chinese text
            try:
                tokens = list(jieba.cut(text))
            except:
                tokens = text.split()
        else:
            # Use NLTK for English text
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
        
        # Remove stopwords and short tokens
        filtered_tokens = []
        for token in tokens:
            if len(token) > 1 and token not in self.english_stopwords:
                filtered_tokens.append(token)
        
        return ' '.join(filtered_tokens)
    
    def _contains_chinese(self, text: str) -> bool:
        """
        Check if text contains Chinese characters
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains Chinese characters
        """
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    
    def extract_fashion_attributes(self, text: str) -> Dict[str, List[str]]:
        """
        Extract fashion attributes from text description
        
        Args:
            text: Fashion item description
            
        Returns:
            Dictionary of extracted attributes by category
        """
        text_lower = text.lower()
        extracted = {category: [] for category in self.fashion_attributes.keys()}
        
        for category, attributes in self.fashion_attributes.items():
            for attr in attributes:
                if attr in text_lower:
                    extracted[category].append(attr)
        
        return extracted
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two fashion descriptions
        
        Args:
            text1: First text description
            text2: Second text description
            
        Returns:
            Similarity score between 0 and 1
        """
        if self.sentence_model:
            try:
                embeddings = self.sentence_model.encode([text1, text2])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                return float(similarity)
            except Exception as e:
                print(f"Warning: Semantic similarity computation failed: {e}")
        
        # Fallback to simple token overlap
        tokens1 = set(self.preprocess_fashion_text(text1).split())
        tokens2 = set(self.preprocess_fashion_text(text2).split())
        
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def compute_fashion_similarity(self, img_path1: str, img_path2: str) -> float:
        """
        Compute fashion-specific similarity between two garments
        
        Args:
            img_path1: Path to first garment image
            img_path2: Path to second garment image
            
        Returns:
            Similarity score between 0 and 1
        """
        emb1 = self.embed_fashion_image(img_path1)
        emb2 = self.embed_fashion_image(img_path2)
        return self.compute_similarity(emb1, emb2)
    
    def find_similar_fashion_items(self, query_img_path: str, 
                                 candidate_paths: List[str], 
                                 top_k: int = 10) -> List[tuple]:
        """
        Find most similar fashion items to a query garment
        
        Args:
            query_img_path: Path to query garment image
            candidate_paths: List of candidate garment image paths
            top_k: Number of top matches to return
            
        Returns:
            List of (image_path, similarity_score) tuples
        """
        query_emb = self.embed_fashion_image(query_img_path)
        
        similarities = []
        for candidate_path in candidate_paths:
            try:
                candidate_emb = self.embed_fashion_image(candidate_path)
                similarity = self.compute_similarity(query_emb, candidate_emb)
                similarities.append((candidate_path, similarity))
            except Exception as e:
                print(f"Error processing {candidate_path}: {e}")
                continue
        
        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def batch_embed_fashion_images(self, img_paths: List[Union[str, Image.Image]]) -> np.ndarray:
        """
        Generate fashion embeddings for multiple images in batch
        
        Args:
            img_paths: List of image file paths or PIL Images
            
        Returns:
            Array of embeddings with shape (len(img_paths), embedding_dim)
        """
        # For now, use the parent class batch method
        # TODO: Add fashion-specific batch processing optimizations
        return self.batch_embed_images(img_paths)
    
    def embed_batch_fashion_images(self, image_paths: List[Union[str, Image.Image]]) -> np.ndarray:
        """Batch embed multiple fashion images"""
        return self.batch_embed_images(image_paths)
    
    def analyze_fashion_attributes(self, img_path: str) -> dict:
        """
        Analyze fashion attributes of a garment image
        
        Args:
            img_path: Path to garment image
            
        Returns:
            Dictionary with fashion attributes
        """
        # This is a placeholder for future fashion-specific analysis
        # TODO: Implement fashion attribute extraction (color, style, category, etc.)
        
        embedding = self.embed_fashion_image(img_path)
        
        # For now, return basic info
        return {
            "embedding_norm": float(np.linalg.norm(embedding)),
            "embedding_dim": len(embedding),
            "image_path": img_path,
            "analysis_method": "clip_based",
            "note": "Fashion-specific attribute analysis not yet implemented"
        }
    
    def get_fashion_categories(self) -> List[str]:
        """
        Get list of supported fashion categories
        
        Returns:
            List of fashion category names
        """
        # TODO: Load from configuration or trained model
        return [
            "dress", "shirt", "pants", "skirt", "jacket", "coat", 
            "sweater", "t-shirt", "jeans", "shorts", "blouse", 
            "blazer", "cardigan", "hoodie", "tank_top", "suit",
            "shoes", "boots", "sneakers", "sandals", "heels",
            "bag", "purse", "backpack", "hat", "scarf", "belt"
        ]
    
    def classify_garment_type(self, img_path: str, 
                            categories: Optional[List[str]] = None) -> dict:
        """
        Classify the type of garment in the image
        
        Args:
            img_path: Path to garment image
            categories: List of categories to consider (uses default if None)
            
        Returns:
            Dictionary with classification results
        """
        if categories is None:
            categories = self.get_fashion_categories()
        
        # Use text-image similarity for classification
        img_emb = self.embed_fashion_image(img_path)
        
        scores = {}
        for category in categories:
            # Create text prompts for each category
            text_prompts = [
                f"a {category}",
                f"a photo of a {category}",
                f"fashion {category}"
            ]
            
            category_scores = []
            for prompt in text_prompts:
                text_emb = self.embed_fashion_text(prompt)
                similarity = self.compute_similarity(img_emb, text_emb)
                category_scores.append(similarity)
            
            # Use average score for this category
            scores[category] = np.mean(category_scores)
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "top_category": sorted_scores[0][0],
            "top_score": float(sorted_scores[0][1]),
            "all_scores": {k: float(v) for k, v in scores.items()},
            "top_3": [(k, float(v)) for k, v in sorted_scores[:3]]
        }

# Global instance
_fashion_encoder = None

def get_fashion_encoder(model_name: str = "ViT-B-32", 
                       pretrained: str = "openai") -> FashionEncoder:
    """
    Get global fashion encoder instance
    
    Args:
        model_name: CLIP model architecture
        pretrained: Pretrained weights to use
        
    Returns:
        FashionEncoder instance
    """
    global _fashion_encoder
    if _fashion_encoder is None:
        _fashion_encoder = FashionEncoder(model_name=model_name, pretrained=pretrained)
    return _fashion_encoder