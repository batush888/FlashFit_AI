#!/usr/bin/env python3
"""
Generative Feedback Service
Handles user feedback collection and integration with generative model fine-tuning.
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass, asdict

# Import ML components
import sys
ml_path = Path(__file__).parent.parent.parent / "ml"
sys.path.append(str(ml_path))

try:
    from ml.embedding_mlp_generator import EmbeddingMLPGenerator, GeneratorConfig
except ImportError as e:
    logging.warning(f"Could not import ML modules: {e}")
    EmbeddingMLPGenerator = None
    GeneratorConfig = None

@dataclass
class GenerativeFeedback:
    """Structure for generative model feedback."""
    feedback_id: str
    user_id: str
    suggestion_id: str
    query_embedding: List[float]
    generated_embedding: List[float]
    user_rating: float  # 0.0 to 1.0
    feedback_type: str  # 'like', 'dislike', 'rating'
    context: Dict[str, Any]  # occasion, style preferences, etc.
    timestamp: str
    model_version: str

class GenerativeFeedbackService:
    """Service for handling generative model feedback and fine-tuning."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feedback_file = Path("data/generative_feedback.json")
        self.training_queue_file = Path("data/generative_training_queue.json")
        
        # Initialize data files
        self._init_data_files()
        
        # Training configuration
        self.min_feedback_for_training = 50
        self.training_batch_size = 32
        self.fine_tuning_lr = 1e-5
        
        self.logger.info("Generative feedback service initialized")
    
    def _init_data_files(self):
        """Initialize feedback data files."""
        for file_path in [self.feedback_file, self.training_queue_file]:
            if not file_path.exists():
                file_path.parent.mkdir(exist_ok=True)
                with open(file_path, 'w') as f:
                    json.dump([], f)
    
    def _load_feedback(self) -> List[Dict[str, Any]]:
        """Load feedback data from file."""
        try:
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_feedback(self, feedback_list: List[Dict[str, Any]]):
        """Save feedback data to file."""
        with open(self.feedback_file, 'w') as f:
            json.dump(feedback_list, f, indent=2)
    
    def _load_training_queue(self) -> List[Dict[str, Any]]:
        """Load training queue from file."""
        try:
            with open(self.training_queue_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_training_queue(self, queue: List[Dict[str, Any]]):
        """Save training queue to file."""
        with open(self.training_queue_file, 'w') as f:
            json.dump(queue, f, indent=2)
    
    async def add_feedback(
        self,
        user_id: str,
        suggestion_id: str,
        query_embedding: List[float],
        generated_embedding: List[float],
        user_rating: float,
        feedback_type: str = "rating",
        context: Optional[Dict[str, Any]] = None,
        model_version: str = "v1.0"
    ) -> Dict[str, Any]:
        """Add user feedback for generative recommendations."""
        
        feedback = GenerativeFeedback(
            feedback_id=f"gen_fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id[:8]}",
            user_id=user_id,
            suggestion_id=suggestion_id,
            query_embedding=query_embedding,
            generated_embedding=generated_embedding,
            user_rating=user_rating,
            feedback_type=feedback_type,
            context=context or {},
            timestamp=datetime.now().isoformat(),
            model_version=model_version
        )
        
        # Save feedback
        feedback_list = self._load_feedback()
        feedback_list.append(asdict(feedback))
        self._save_feedback(feedback_list)
        
        # Add to training queue if rating is significant
        if user_rating <= 0.3 or user_rating >= 0.7:  # Strong negative or positive feedback
            await self._add_to_training_queue(feedback)
        
        # Check if we should trigger training
        await self._check_training_trigger()
        
        self.logger.info(f"Added generative feedback: {feedback.feedback_id}")
        
        return {
            "status": "success",
            "feedback_id": feedback.feedback_id,
            "queued_for_training": user_rating <= 0.3 or user_rating >= 0.7
        }
    
    async def _add_to_training_queue(self, feedback: GenerativeFeedback):
        """Add feedback to training queue."""
        queue = self._load_training_queue()
        
        training_sample = {
            "query_embedding": feedback.query_embedding,
            "target_embedding": feedback.generated_embedding,
            "weight": feedback.user_rating,  # Use rating as training weight
            "feedback_id": feedback.feedback_id,
            "timestamp": feedback.timestamp
        }
        
        queue.append(training_sample)
        self._save_training_queue(queue)
    
    async def _check_training_trigger(self):
        """Check if we should trigger model fine-tuning."""
        queue = self._load_training_queue()
        
        if len(queue) >= self.min_feedback_for_training:
            self.logger.info(f"Training queue has {len(queue)} samples, triggering fine-tuning")
            # Schedule training in background
            asyncio.create_task(self._fine_tune_model())
    
    async def _fine_tune_model(self):
        """Fine-tune the generative model with collected feedback."""
        try:
            if EmbeddingMLPGenerator is None:
                self.logger.warning("EmbeddingMLPGenerator not available for fine-tuning")
                return
            
            queue = self._load_training_queue()
            if len(queue) < self.min_feedback_for_training:
                return
            
            self.logger.info(f"Starting fine-tuning with {len(queue)} samples")
            
            # Prepare training data
            query_embeddings = []
            target_embeddings = []
            weights = []
            
            for sample in queue:
                query_embeddings.append(sample["query_embedding"])
                target_embeddings.append(sample["target_embedding"])
                weights.append(sample["weight"])
            
            # Convert to tensors
            query_tensor = torch.tensor(query_embeddings, dtype=torch.float32)
            target_tensor = torch.tensor(target_embeddings, dtype=torch.float32)
            weight_tensor = torch.tensor(weights, dtype=torch.float32)
            
            # Load existing model
            config = GeneratorConfig(
                embedding_dim=512,
                hidden_dims=[1024, 1024, 512],
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            generator = EmbeddingMLPGenerator(config)
            model_path = ml_path / "best_embedding_generator.pth"
            
            if model_path.exists():
                generator.load_model(str(model_path))
            
            # Fine-tune model
            generator.fine_tune(
                query_embeddings=query_tensor,
                target_embeddings=target_tensor,
                sample_weights=weight_tensor,
                learning_rate=self.fine_tuning_lr,
                epochs=10,
                batch_size=self.training_batch_size
            )
            
            # Save updated model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fine_tuned_path = ml_path / f"fine_tuned_generator_{timestamp}.pth"
            generator.save_model(str(fine_tuned_path))
            
            # Update best model if validation improves
            generator.save_model(str(model_path))
            
            # Clear training queue
            self._save_training_queue([])
            
            self.logger.info(f"Fine-tuning completed, model saved to {fine_tuned_path}")
            
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {e}")
    
    async def get_feedback_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get feedback statistics."""
        feedback_list = self._load_feedback()
        
        if user_id:
            feedback_list = [f for f in feedback_list if f["user_id"] == user_id]
        
        if not feedback_list:
            return {
                "total_feedback": 0,
                "average_rating": 0.0,
                "feedback_distribution": {},
                "recent_feedback_count": 0
            }
        
        # Calculate statistics
        ratings = [f["user_rating"] for f in feedback_list]
        feedback_types = [f["feedback_type"] for f in feedback_list]
        
        # Recent feedback (last 7 days)
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_feedback = [
            f for f in feedback_list 
            if datetime.fromisoformat(f["timestamp"]) > recent_cutoff
        ]
        
        # Feedback distribution
        distribution = {}
        for fb_type in feedback_types:
            distribution[fb_type] = distribution.get(fb_type, 0) + 1
        
        return {
            "total_feedback": len(feedback_list),
            "average_rating": np.mean(ratings),
            "rating_std": np.std(ratings),
            "feedback_distribution": distribution,
            "recent_feedback_count": len(recent_feedback),
            "training_queue_size": len(self._load_training_queue())
        }
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Analyze user preferences from feedback history."""
        feedback_list = self._load_feedback()
        user_feedback = [f for f in feedback_list if f["user_id"] == user_id]
        
        if not user_feedback:
            return {"preferences": {}, "confidence": 0.0}
        
        # Analyze context patterns for high-rated feedback
        high_rated = [f for f in user_feedback if f["user_rating"] >= 0.7]
        low_rated = [f for f in user_feedback if f["user_rating"] <= 0.3]
        
        preferences = {}
        
        # Extract preferred contexts
        for feedback in high_rated:
            context = feedback.get("context", {})
            for key, value in context.items():
                if key not in preferences:
                    preferences[key] = {"preferred": [], "avoided": []}
                if value not in preferences[key]["preferred"]:
                    preferences[key]["preferred"].append(value)
        
        # Extract avoided contexts
        for feedback in low_rated:
            context = feedback.get("context", {})
            for key, value in context.items():
                if key not in preferences:
                    preferences[key] = {"preferred": [], "avoided": []}
                if value not in preferences[key]["avoided"]:
                    preferences[key]["avoided"].append(value)
        
        confidence = min(len(user_feedback) / 20.0, 1.0)  # Max confidence at 20+ feedback
        
        return {
            "preferences": preferences,
            "confidence": confidence,
            "total_feedback": len(user_feedback),
            "average_rating": np.mean([f["user_rating"] for f in user_feedback])
        }

# Global service instance
_generative_feedback_service = None

def get_generative_feedback_service() -> GenerativeFeedbackService:
    """Get or create the global generative feedback service instance."""
    global _generative_feedback_service
    if _generative_feedback_service is None:
        _generative_feedback_service = GenerativeFeedbackService()
    return _generative_feedback_service