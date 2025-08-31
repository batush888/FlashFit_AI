import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import json
import os
from pathlib import Path
import pickle
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ScoringWeights:
    """Weights for different scoring components"""
    clip: float = 0.5
    blip: float = 0.2
    fashion: float = 0.3
    bias: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "clip": self.clip,
            "blip": self.blip,
            "fashion": self.fashion,
            "bias": self.bias
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'ScoringWeights':
        return cls(
            clip=data.get("clip", 0.5),
            blip=data.get("blip", 0.2),
            fashion=data.get("fashion", 0.3),
            bias=data.get("bias", 0.0)
        )

@dataclass
class FusionScore:
    """Individual component scores and final fusion score"""
    clip_score: float
    blip_score: float
    fashion_score: float
    final_score: float
    item_id: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_score": self.clip_score,
            "blip_score": self.blip_score,
            "fashion_score": self.fashion_score,
            "final_score": self.final_score,
            "item_id": self.item_id,
            "metadata": self.metadata
        }

class FusionReranker:
    """
    Multi-model fusion reranker that combines CLIP, BLIP, and Fashion encoder scores
    with online learning capabilities for weight optimization.
    """
    
    def __init__(self, weights_path: str = "data/fusion_weights.json",
                 model_path: str = "data/fusion_model.pkl",
                 enable_online_learning: bool = True):
        """
        Initialize fusion reranker
        
        Args:
            weights_path: Path to save/load fusion weights
            model_path: Path to save/load online learning model
            enable_online_learning: Whether to enable online weight learning
        """
        self.weights_path = Path(weights_path)
        self.model_path = Path(model_path)
        self.enable_online_learning = enable_online_learning
        
        # Create directories
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize weights
        self.weights = self._load_weights()
        
        # Initialize online learning components
        self.online_model = None
        self.scaler = None
        self.feedback_history = []
        
        if self.enable_online_learning:
            self._initialize_online_learning()
        
        print(f"FusionReranker initialized")
        print(f"Weights: {self.weights.to_dict()}")
        print(f"Online learning: {self.enable_online_learning}")
    
    def _load_weights(self) -> ScoringWeights:
        """Load fusion weights from file or use defaults"""
        if self.weights_path.exists():
            try:
                with open(self.weights_path, 'r') as f:
                    data = json.load(f)
                return ScoringWeights.from_dict(data)
            except Exception as e:
                print(f"Error loading weights: {e}. Using defaults.")
        
        return ScoringWeights()  # Default weights
    
    def _save_weights(self):
        """Save current weights to file"""
        try:
            with open(self.weights_path, 'w') as f:
                json.dump(self.weights.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Error saving weights: {e}")
    
    def _initialize_online_learning(self):
        """Initialize online learning components"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.online_model = data['model']
                    self.scaler = data['scaler']
                    self.feedback_history = data.get('feedback_history', [])
                print(f"Loaded online learning model with {len(self.feedback_history)} feedback entries")
            except Exception as e:
                print(f"Error loading online model: {e}. Initializing new model.")
        
        if self.online_model is None:
            # Initialize SGD regressor for online learning
            self.online_model = SGDRegressor(
                learning_rate='adaptive',
                eta0=0.01,
                random_state=42,
                warm_start=True
            )
            self.scaler = StandardScaler()
    
    def _save_online_model(self):
        """Save online learning model"""
        if self.online_model is not None:
            try:
                data = {
                    'model': self.online_model,
                    'scaler': self.scaler,
                    'feedback_history': self.feedback_history[-1000:]  # Keep last 1000 entries
                }
                with open(self.model_path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"Error saving online model: {e}")
    
    def compute_fusion_score(self, clip_score: float, blip_score: float, 
                           fashion_score: float) -> float:
        """
        Compute final fusion score from individual component scores
        
        Args:
            clip_score: CLIP similarity score
            blip_score: BLIP text similarity score
            fashion_score: Fashion encoder similarity score
            
        Returns:
            Final weighted fusion score
        """
        # Basic weighted combination
        fusion_score = (
            self.weights.clip * clip_score +
            self.weights.blip * blip_score +
            self.weights.fashion * fashion_score +
            self.weights.bias
        )
        
        # Apply online learning adjustment if available
        if self.enable_online_learning and self.online_model is not None:
            try:
                # Create feature vector
                features = np.array([[clip_score, blip_score, fashion_score]])
                
                # Scale features if scaler is fitted
                if hasattr(self.scaler, 'mean_'):
                    features_scaled = self.scaler.transform(features)
                    
                    # Get prediction from online model
                    online_adjustment = self.online_model.predict(features_scaled)[0]
                    
                    # Combine with base fusion score
                    fusion_score = 0.7 * fusion_score + 0.3 * online_adjustment
            except Exception as e:
                print(f"Error in online learning prediction: {e}")
        
        return float(fusion_score)
    
    def rerank_candidates(self, candidates: List[Dict[str, Any]]) -> List[FusionScore]:
        """
        Rerank candidates using fusion scoring
        
        Args:
            candidates: List of candidate items with scores
                       Each item should have: clip_score, blip_score, fashion_score, item_id, metadata
            
        Returns:
            List of FusionScore objects sorted by final score (descending)
        """
        fusion_scores = []
        
        for candidate in candidates:
            # Extract individual scores
            clip_score = candidate.get('clip_score', 0.0)
            blip_score = candidate.get('blip_score', 0.0)
            fashion_score = candidate.get('fashion_score', 0.0)
            
            # Compute fusion score
            final_score = self.compute_fusion_score(clip_score, blip_score, fashion_score)
            
            # Create FusionScore object
            fusion_score = FusionScore(
                clip_score=clip_score,
                blip_score=blip_score,
                fashion_score=fashion_score,
                final_score=final_score,
                item_id=candidate.get('item_id', ''),
                metadata=candidate.get('metadata', {})
            )
            
            fusion_scores.append(fusion_score)
        
        # Sort by final score (descending)
        fusion_scores.sort(key=lambda x: x.final_score, reverse=True)
        
        return fusion_scores
    
    def add_feedback(self, item_id: str, clip_score: float, blip_score: float,
                    fashion_score: float, user_rating: float, 
                    feedback_type: str = "rating"):
        """
        Add user feedback for online learning
        
        Args:
            item_id: ID of the item that received feedback
            clip_score: CLIP score for this item
            blip_score: BLIP score for this item
            fashion_score: Fashion score for this item
            user_rating: User rating/feedback (0.0 to 1.0)
            feedback_type: Type of feedback ("rating", "like", "dislike")
        """
        if not self.enable_online_learning:
            return
        
        # Convert feedback to numerical score
        if feedback_type == "like":
            target_score = 1.0
        elif feedback_type == "dislike":
            target_score = 0.0
        else:
            target_score = float(user_rating)
        
        # Store feedback
        feedback_entry = {
            "item_id": item_id,
            "clip_score": clip_score,
            "blip_score": blip_score,
            "fashion_score": fashion_score,
            "target_score": target_score,
            "feedback_type": feedback_type,
            "timestamp": datetime.now().isoformat()
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Update online model
        self._update_online_model(clip_score, blip_score, fashion_score, target_score)
        
        print(f"Added feedback for item {item_id}: {feedback_type} -> {target_score}")
    
    def _update_online_model(self, clip_score: float, blip_score: float,
                           fashion_score: float, target_score: float):
        """Update online learning model with new feedback"""
        try:
            # Prepare features and target
            features = np.array([[clip_score, blip_score, fashion_score]])
            target = np.array([target_score])
            
            # Fit scaler if not already fitted
            if not hasattr(self.scaler, 'mean_'):
                # Initialize with some dummy data to avoid issues
                dummy_features = np.array([
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.5, 0.5, 0.5]
                ])
                self.scaler.fit(dummy_features)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Update online model
            if not hasattr(self.online_model, 'coef_'):
                # First fit
                self.online_model.fit(features_scaled, target)
            else:
                # Partial fit for online learning
                self.online_model.partial_fit(features_scaled, target)
            
            # Periodically save model
            if len(self.feedback_history) % 10 == 0:
                self._save_online_model()
        
        except Exception as e:
            print(f"Error updating online model: {e}")
    
    def update_weights(self, new_weights: ScoringWeights):
        """
        Update fusion weights
        
        Args:
            new_weights: New ScoringWeights object
        """
        self.weights = new_weights
        self._save_weights()
        print(f"Updated weights: {self.weights.to_dict()}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            "current_weights": self.weights.to_dict(),
            "feedback_count": len(self.feedback_history),
            "online_learning_enabled": self.enable_online_learning,
            "model_trained": hasattr(self.online_model, 'coef_') if self.online_model else False
        }
        
        if self.feedback_history:
            # Calculate feedback statistics
            ratings = [f['target_score'] for f in self.feedback_history[-100:]]  # Last 100
            stats.update({
                "avg_rating": np.mean(ratings),
                "rating_std": np.std(ratings),
                "recent_feedback_count": len(ratings)
            })
        
        return stats
    
    def reset_online_learning(self):
        """Reset online learning model and feedback history"""
        if self.enable_online_learning:
            self.online_model = SGDRegressor(
                learning_rate='adaptive',
                eta0=0.01,
                random_state=42,
                warm_start=True
            )
            self.scaler = StandardScaler()
            self.feedback_history = []
            
            # Remove saved model
            if self.model_path.exists():
                self.model_path.unlink()
            
            print("Online learning model reset")

# Global instance
_fusion_reranker = None

def get_fusion_reranker(enable_online_learning: bool = True) -> FusionReranker:
    """
    Get global fusion reranker instance
    
    Args:
        enable_online_learning: Whether to enable online learning
        
    Returns:
        FusionReranker instance
    """
    global _fusion_reranker
    if _fusion_reranker is None:
        _fusion_reranker = FusionReranker(enable_online_learning=enable_online_learning)
    return _fusion_reranker