from .clip_encoder import CLIPEncoder
from typing import Optional, List, Union
import numpy as np
from PIL import Image
import torch

class FashionEncoder(CLIPEncoder):
    """
    Fashion-specific encoder that extends CLIPEncoder.
    
    For MVP, this uses the same CLIP model as the base encoder.
    In the future, this can be replaced with a fashion-tuned model
    or a specialized fashion encoder trained on fashion datasets.
    """
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", 
                 device: Optional[str] = None):
        """
        Initialize fashion-specific encoder
        
        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights to use
            device: Device to run model on
        """
        # For now, use the same CLIP model
        # TODO: Replace with fashion-tuned model when available
        super().__init__(model_name=model_name, pretrained=pretrained, device=device)
        
        print(f"FashionEncoder initialized (using {model_name} with {pretrained} weights)")
        print("Note: Currently using general CLIP model. Consider fine-tuning on fashion data.")
    
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
        Generate fashion-specific text embedding
        
        Args:
            text: Fashion-related text (e.g., "red dress", "casual outfit")
            
        Returns:
            Normalized embedding vector
        """
        # For now, this is the same as regular text embedding
        # TODO: Add fashion-specific text preprocessing
        return self.embed_text(text)
    
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