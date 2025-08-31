import numpy as np
from sklearn.linear_model import SGDClassifier
from typing import List, Dict, Any, Tuple
import pickle
import os
from pathlib import Path

class SimpleFusionReranker:
    """
    Simplified fusion reranker with learnable weights using SGDClassifier
    Based on user's suggested architecture for online learning
    """
    
    def __init__(self, model_path: str = "data/simple_fusion_model.pkl"):
        """
        Initialize the simple fusion reranker
        
        Args:
            model_path: Path to save/load the trained model
        """
        # Online logistic regression; features = [s_clip, s_blip, s_fashion]
        self.clf = SGDClassifier(loss="log_loss", learning_rate="optimal", random_state=42)
        self.is_fit = False
        self.default_w = np.array([0.5, 0.2, 0.3])  # heuristic start
        self.b = 0.0
        self.model_path = Path(model_path)
        
        # Ensure data directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing model
        self._load_model()
        
        print(f"SimpleFusionReranker initialized (fitted: {self.is_fit})")
    
    def score(self, s_clip: float, s_blip: float, s_fashion: float) -> float:
        """
        Compute fusion score for given component scores
        
        Args:
            s_clip: CLIP similarity score
            s_blip: BLIP similarity score
            s_fashion: Fashion encoder similarity score
            
        Returns:
            Final fusion score
        """
        if not self.is_fit:
            # Use default heuristic weights
            return float(np.dot(self.default_w, [s_clip, s_blip, s_fashion]) + self.b)
        
        # Use trained classifier decision function
        X = np.array([[s_clip, s_blip, s_fashion]])
        # decision_function ~ logit score
        return float(self.clf.decision_function(X)[0])
    
    def partial_learn(self, features: List[float], label: int):
        """
        Online learning from user feedback
        
        Args:
            features: [s_clip, s_blip, s_fashion] scores
            label: 1 for liked, 0 for disliked
        """
        X = np.array([features])
        y = np.array([label])
        
        if not self.is_fit:
            # First time fitting - need to specify classes
            self.clf.partial_fit(X, y, classes=np.array([0, 1]))
            self.is_fit = True
        else:
            # Continue learning
            self.clf.partial_fit(X, y)
        
        # Save model after learning
        self._save_model()
    
    def batch_learn(self, features_list: List[List[float]], labels: List[int]):
        """
        Batch learning from multiple feedback samples
        
        Args:
            features_list: List of [s_clip, s_blip, s_fashion] scores
            labels: List of labels (1 for liked, 0 for disliked)
        """
        X = np.array(features_list)
        y = np.array(labels)
        
        if not self.is_fit:
            # First time fitting
            self.clf.partial_fit(X, y, classes=np.array([0, 1]))
            self.is_fit = True
        else:
            # Continue learning
            self.clf.partial_fit(X, y)
        
        # Save model after learning
        self._save_model()
    
    def rerank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank candidates using fusion scores
        
        Args:
            candidates: List of candidate items with scores
            
        Returns:
            List of (candidate, fusion_score) tuples sorted by score
        """
        scored_candidates = []
        
        for candidate in candidates:
            # Extract component scores
            s_clip = candidate.get("clip_score", 0.0)
            s_blip = candidate.get("blip_score", 0.0)
            s_fashion = candidate.get("fashion_score", 0.0)
            
            # Compute fusion score
            fusion_score = self.score(s_clip, s_blip, s_fashion)
            
            scored_candidates.append((candidate, fusion_score))
        
        # Sort by fusion score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model state
        
        Returns:
            Dictionary with model information
        """
        info = {
            "is_fitted": self.is_fit,
            "default_weights": self.default_w.tolist(),
            "default_bias": self.b,
            "model_path": str(self.model_path)
        }
        
        if self.is_fit:
            info.update({
                "n_features": self.clf.n_features_in_,
                "classes": self.clf.classes_.tolist(),
                "n_iter": getattr(self.clf, 'n_iter_', 0)
            })
        
        return info
    
    def reset_model(self):
        """
        Reset the model to initial state
        """
        self.clf = SGDClassifier(loss="log_loss", learning_rate="optimal", random_state=42)
        self.is_fit = False
        
        # Remove saved model file
        if self.model_path.exists():
            self.model_path.unlink()
        
        print("Model reset to initial state")
    
    def _save_model(self):
        """
        Save the current model to disk
        """
        try:
            model_data = {
                "clf": self.clf,
                "is_fit": self.is_fit,
                "default_w": self.default_w,
                "b": self.b
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
    
    def _load_model(self):
        """
        Load existing model from disk
        """
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.clf = model_data["clf"]
                self.is_fit = model_data["is_fit"]
                self.default_w = model_data["default_w"]
                self.b = model_data["b"]
                
                print(f"Loaded existing model from {self.model_path}")
                
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            # Continue with default initialization

# Global instance
_simple_fusion_reranker = None

def get_simple_fusion_reranker() -> SimpleFusionReranker:
    """
    Get or create the global simple fusion reranker instance
    
    Returns:
        SimpleFusionReranker instance
    """
    global _simple_fusion_reranker
    if _simple_fusion_reranker is None:
        _simple_fusion_reranker = SimpleFusionReranker()
    return _simple_fusion_reranker