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
    clip: float = 0.4  # Reduced CLIP weight for better balance
    blip: float = 0.3  # Increased BLIP weight for better text understanding
    fashion: float = 0.3  # Keep fashion weight same
    bias: float = 0.0
    diversity_penalty: float = 0.1  # New: penalty for similar items
    novelty_boost: float = 0.05  # New: boost for novel recommendations
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "clip": self.clip,
            "blip": self.blip,
            "fashion": self.fashion,
            "bias": self.bias,
            "diversity_penalty": self.diversity_penalty,
            "novelty_boost": self.novelty_boost
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'ScoringWeights':
        return cls(
            clip=data.get("clip", 0.4),
            blip=data.get("blip", 0.3),
            fashion=data.get("fashion", 0.3),
            bias=data.get("bias", 0.0),
            diversity_penalty=data.get("diversity_penalty", 0.1),
            novelty_boost=data.get("novelty_boost", 0.05)
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
                if self.scaler is not None and hasattr(self.scaler, 'mean_'):
                    features_scaled = self.scaler.transform(features)
                    
                    # Get prediction from online model
                    if hasattr(self.online_model, 'predict'):
                        online_adjustment = self.online_model.predict(features_scaled)[0]
                        
                        # Combine with base fusion score
                        fusion_score = 0.7 * fusion_score + 0.3 * online_adjustment
            except Exception as e:
                print(f"Error in online learning prediction: {e}")
        
        return float(fusion_score)
    
    def rerank_candidates(self, candidates: List[Dict[str, Any]]) -> List[FusionScore]:
        """
        Rerank candidates using enhanced fusion scoring with diversity and novelty
        
        Args:
            candidates: List of candidate items with scores
                       Each item should have: clip_score, blip_score, fashion_score, item_id, metadata
            
        Returns:
            List of FusionScore objects sorted by final score (descending)
        """
        fusion_scores = []
        
        # Calculate novelty scores based on item frequency in feedback history
        item_frequencies = {}
        for feedback in self.feedback_history[-100:]:  # Last 100 feedback entries
            item_id = feedback.get('item_id', '')
            item_frequencies[item_id] = item_frequencies.get(item_id, 0) + 1
        
        for candidate in candidates:
            # Extract individual scores
            clip_score = candidate.get('clip_score', 0.0)
            blip_score = candidate.get('blip_score', 0.0)
            fashion_score = candidate.get('fashion_score', 0.0)
            item_id = candidate.get('item_id', '')
            
            # Compute base fusion score
            base_score = self.compute_fusion_score(clip_score, blip_score, fashion_score)
            
            # Calculate novelty boost (items seen less frequently get higher boost)
            frequency = item_frequencies.get(item_id, 0)
            novelty_score = max(0, 1.0 - (frequency / 10.0))  # Normalize frequency
            novelty_adjustment = self.weights.novelty_boost * novelty_score
            
            # Apply novelty boost
            final_score = base_score + novelty_adjustment
            
            # Create FusionScore object with enhanced metadata
            fusion_score = FusionScore(
                clip_score=clip_score,
                blip_score=blip_score,
                fashion_score=fashion_score,
                final_score=final_score,
                item_id=item_id,
                metadata={
                    **candidate.get('metadata', {}),
                    'base_score': base_score,
                    'novelty_score': novelty_score,
                    'novelty_adjustment': novelty_adjustment,
                    'frequency': frequency
                }
            )
            
            fusion_scores.append(fusion_score)
        
        # Sort by final score (descending)
        fusion_scores.sort(key=lambda x: x.final_score, reverse=True)
        
        # Apply diversity filtering to top results
        diverse_scores = self._apply_diversity_filtering(fusion_scores)
        
        return diverse_scores
    
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
            if self.scaler is not None and not hasattr(self.scaler, 'mean_'):
                # Initialize with some dummy data to avoid issues
                dummy_features = np.array([
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.5, 0.5, 0.5]
                ])
                self.scaler.fit(dummy_features)
            
            # Scale features
            if self.scaler is not None and hasattr(self.scaler, 'transform'):
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Update online model
            if self.online_model is not None:
                if not hasattr(self.online_model, 'coef_'):
                    # First fit
                    self.online_model.fit(features_scaled, target)
                elif hasattr(self.online_model, 'partial_fit'):
                    # Partial fit for online learning
                    self.online_model.partial_fit(features_scaled, target)
            
            # Periodically save model
            if len(self.feedback_history) % 10 == 0:
                self._save_online_model()
        
        except Exception as e:
            print(f"Error updating online model: {e}")
    
    def _apply_diversity_filtering(self, fusion_scores: List[FusionScore]) -> List[FusionScore]:
        """
        Apply diversity filtering to avoid too similar recommendations
        
        Args:
            fusion_scores: List of FusionScore objects sorted by score
            
        Returns:
            Filtered list with diversity penalty applied
        """
        if len(fusion_scores) <= 1:
            return fusion_scores
        
        diverse_scores = [fusion_scores[0]]  # Always include top item
        
        for candidate in fusion_scores[1:]:
            # Calculate similarity penalty with already selected items
            max_similarity = 0.0
            
            for selected in diverse_scores:
                # Simple similarity based on metadata categories
                similarity = 0.0
                
                candidate_meta = candidate.metadata
                selected_meta = selected.metadata
                
                # Category similarity
                if (candidate_meta.get('category') == selected_meta.get('category') and 
                    candidate_meta.get('category') is not None):
                    similarity += 0.3
                
                # Color similarity
                if (candidate_meta.get('color') == selected_meta.get('color') and 
                    candidate_meta.get('color') is not None):
                    similarity += 0.2
                
                # Brand similarity
                if (candidate_meta.get('brand') == selected_meta.get('brand') and 
                    candidate_meta.get('brand') is not None):
                    similarity += 0.1
                
                max_similarity = max(max_similarity, similarity)
            
            # Apply diversity penalty
            diversity_penalty = self.weights.diversity_penalty * max_similarity
            adjusted_score = candidate.final_score - diversity_penalty
            
            # Update the candidate's final score
            candidate.final_score = adjusted_score
            candidate.metadata['diversity_penalty'] = diversity_penalty
            candidate.metadata['max_similarity'] = max_similarity
            
            diverse_scores.append(candidate)
        
        # Re-sort after applying diversity penalties
        diverse_scores.sort(key=lambda x: x.final_score, reverse=True)
        
        return diverse_scores
    
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