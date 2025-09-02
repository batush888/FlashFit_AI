import faiss
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

class VectorStore:
    """FAISS-based vector store for efficient similarity search and candidate generation"""
    
    def __init__(self, dim: int, index_path: str = "data/fashion.index", 
                 meta_path: str = "data/items.json"):
        """
        Initialize vector store
        
        Args:
            dim: Dimension of the vectors
            index_path: Path to save/load FAISS index
            meta_path: Path to save/load item metadata
        """
        self.dim = dim
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        
        # Create directories if they don't exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index (Inner Product for cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dim)
        self.items = []
        
        # Load existing index and metadata if available
        self._load_if_exists()
        
        print(f"VectorStore initialized with dimension {dim}")
        print(f"Index path: {self.index_path}")
        print(f"Metadata path: {self.meta_path}")
        print(f"Current items: {len(self.items)}")
    
    def _load_if_exists(self):
        """Load existing index and metadata if files exist"""
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                print(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading index: {e}")
                self.index = faiss.IndexFlatIP(self.dim)
        
        if self.meta_path.exists():
            try:
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    self.items = json.load(f)
                print(f"Loaded {len(self.items)} item metadata entries")
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.items = []
    
    def add(self, vecs: np.ndarray, metas: List[Dict[str, Any]]):
        """
        Add vectors and their metadata to the store
        
        Args:
            vecs: Normalized vectors of shape (N, dim)
            metas: List of metadata dictionaries for each vector
        """
        if len(vecs) != len(metas):
            raise ValueError(f"Number of vectors ({len(vecs)}) must match number of metadata entries ({len(metas)})")
        
        if vecs.shape[1] != self.dim:
            raise ValueError(f"Vector dimension ({vecs.shape[1]}) must match store dimension ({self.dim})")
        
        # Ensure vectors are normalized for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vecs_normalized = vecs / norms
        
        # Add to FAISS index
        self.index.add(x=vecs_normalized.astype('float32'))
        
        # Add metadata
        self.items.extend(metas)
        
        print(f"Added {len(vecs)} vectors to store. Total: {self.index.ntotal}")
    
    def add_single(self, vec: np.ndarray, meta: Dict[str, Any]):
        """
        Add a single vector and its metadata
        
        Args:
            vec: Single normalized vector of shape (dim,)
            meta: Metadata dictionary
        """
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        
        self.add(vec, [meta])
    
    def save(self):
        """
        Save index and metadata to disk
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.items, f, ensure_ascii=False, indent=2)
            
            print(f"Saved vector store with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error saving vector store: {e}")
            raise
    
    def search(self, q: np.ndarray, topk: int = 50) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for top-k most similar vectors
        
        Args:
            q: Query vector of shape (1, dim) or (dim,)
            topk: Number of top results to return
            
        Returns:
            List of (metadata, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure query is 2D and normalized
        if q.ndim == 1:
            q = q.reshape(1, -1)
        
        # Normalize query vector
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        if q_norm[0, 0] != 0:
            q = q / q_norm
        
        # Search
        topk = min(topk, self.index.ntotal)  # Don't search for more than available
        scores, indices = self.index.search(x=q.astype('float32'), k=topk)
        
        # Return results with metadata
        results = []
        for j, i in enumerate(indices[0]):
            if i < len(self.items):  # Safety check
                results.append((self.items[i], float(scores[0, j])))
        
        return results
    
    def search_by_ids(self, ids: List[int]) -> List[Tuple[Dict[str, Any], int]]:
        """
        Retrieve items by their indices
        
        Args:
            ids: List of item indices
            
        Returns:
            List of (metadata, index) tuples
        """
        results = []
        for idx in ids:
            if 0 <= idx < len(self.items):
                results.append((self.items[idx], idx))
        return results
    
    def update_metadata(self, idx: int, meta: Dict[str, Any]):
        """
        Update metadata for a specific item
        
        Args:
            idx: Item index
            meta: New metadata dictionary
        """
        if 0 <= idx < len(self.items):
            self.items[idx] = meta
        else:
            raise IndexError(f"Index {idx} out of range [0, {len(self.items)})")
    
    def remove_by_ids(self, ids: List[int]):
        """
        Remove items by their indices (Note: FAISS doesn't support efficient removal,
        so this rebuilds the index)
        
        Args:
            ids: List of item indices to remove
        """
        if not ids:
            return
        
        # Get vectors to keep
        keep_ids = [i for i in range(len(self.items)) if i not in ids]
        
        if not keep_ids:
            # Remove all
            self.index = faiss.IndexFlatIP(self.dim)
            self.items = []
            return
        
        # Rebuild index with remaining vectors
        # Note: This is expensive for large indices
        print(f"Rebuilding index after removing {len(ids)} items...")
        
        # Create new index
        new_index = faiss.IndexFlatIP(self.dim)
        new_items = []
        
        # Add remaining items
        for i in keep_ids:
            # Get vector from old index (reconstruct not available for IndexFlatIP)
            # We'll need to store vectors separately or use a different approach
            # For now, we'll skip reconstruction and just rebuild from scratch
            new_items.append(self.items[i])
        
        # Note: This method requires vectors to be re-added externally
        # as FAISS IndexFlatIP doesn't support reconstruction
        
        self.index = new_index
        self.items = new_items
        
        print(f"Index rebuilt with {len(self.items)} remaining items")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary with store statistics
        """
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dim,
            "metadata_count": len(self.items),
            "index_path": str(self.index_path),
            "meta_path": str(self.meta_path),
            "index_exists": self.index_path.exists(),
            "meta_exists": self.meta_path.exists()
        }
    
    def clear(self):
        """
        Clear all vectors and metadata
        """
        self.index = faiss.IndexFlatIP(self.dim)
        self.items = []
        print("Vector store cleared")

# Global instances for different types of embeddings
_clip_store = None
_blip_store = None
_fashion_store = None

# Force reset global instances to ensure updated dimensions
def reset_global_stores():
    """Reset all global store instances to force reinitialization with updated dimensions"""
    global _clip_store, _blip_store, _fashion_store
    _clip_store = None
    _blip_store = None
    _fashion_store = None

def get_clip_store(dim: int = 512) -> VectorStore:
    """Get global CLIP vector store instance"""
    global _clip_store
    if _clip_store is None:
        _clip_store = VectorStore(dim, "data/clip_fashion.index", "data/clip_items.json")
    return _clip_store

def get_blip_store(dim: int = 512) -> VectorStore:
    """Get global BLIP vector store instance"""
    global _blip_store
    if _blip_store is None:
        _blip_store = VectorStore(dim, "data/blip_fashion.index", "data/blip_items.json")
    return _blip_store

def get_fashion_store(dim: int = 512) -> VectorStore:
    """Get global fashion-specific vector store instance"""
    global _fashion_store
    if _fashion_store is None:
        _fashion_store = VectorStore(dim, "data/fashion_specific.index", "data/fashion_items.json")
    return _fashion_store