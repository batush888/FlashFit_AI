import torch
import open_clip
from PIL import Image
import numpy as np
from typing import Union, List

class CLIPEncoder:
    """CLIP encoder for robust visual semantics"""
    
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device=None):
        """
        Initialize CLIP encoder
        
        Args:
            model_name: CLIP model name
            pretrained: Pretrained weights source
            device: Device to run on (auto-detect if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        
        print(f"CLIPEncoder initialized with {model_name} on {self.device}")
    
    def embed_image(self, img_path: Union[str, Image.Image]) -> np.ndarray:
        """
        Encode image to normalized embedding vector
        
        Args:
            img_path: Path to image file or PIL Image object
            
        Returns:
            Normalized embedding vector as numpy array
        """
        # Handle different input types
        if isinstance(img_path, str):
            img = Image.open(img_path).convert("RGB")
        elif isinstance(img_path, Image.Image):
            img = img_path.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img_path)}")
        
        # Preprocess and encode
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            z = self.model.encode_image(img_tensor)
            z = z / z.norm(dim=-1, keepdim=True)  # L2 normalize
        
        return z.cpu().numpy()
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text to normalized embedding vector
        
        Args:
            text: Text string or list of text strings
            
        Returns:
            Normalized embedding vector(s) as numpy array
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize and encode
        text_tokens = self.tokenizer(text).to(self.device)
        
        with torch.no_grad():
            z = self.model.encode_text(text_tokens)
            z = z / z.norm(dim=-1, keepdim=True)  # L2 normalize
        
        return z.cpu().numpy()
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Ensure embeddings are 2D
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2.T)
        return float(similarity[0, 0])
    
    def batch_embed_images(self, image_paths: List[Union[str, Image.Image]]) -> np.ndarray:
        """
        Batch encode multiple images for efficiency
        
        Args:
            image_paths: List of image paths or PIL Image objects
            
        Returns:
            Batch of normalized embedding vectors
        """
        embeddings = []
        
        for img_path in image_paths:
            embedding = self.embed_image(img_path)
            embeddings.append(embedding)
        
        return np.vstack(embeddings)
    
    def find_top_matches(self, query_embedding: np.ndarray, 
                        candidate_embeddings: np.ndarray, 
                        top_k: int = 5) -> List[tuple]:
        """
        Find top-k most similar embeddings to query
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            top_k: Number of top matches to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute similarities
        similarities = np.dot(query_embedding, candidate_embeddings.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

# Global instance for reuse
_clip_encoder = None

def get_clip_encoder() -> CLIPEncoder:
    """Get global CLIP encoder instance"""
    global _clip_encoder
    if _clip_encoder is None:
        _clip_encoder = CLIPEncoder()
    return _clip_encoder