from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Dict, Any, Optional
import tempfile
import shutil
import os
import json
from pathlib import Path

# Import our multi-model components
from services.recommend_service import get_recommendation_service
from models.fusion_reranker import get_fusion_reranker

class FusionMatchHandler:
    """
    Simplified multi-model fusion recommendation handler based on user's architecture
    """
    
    def __init__(self):
        self.recommendation_service = get_recommendation_service()
        print("FusionMatchHandler initialized with multi-model fusion")
    
    async def match_with_fusion(self, file: UploadFile, target_count: int = 3) -> Dict[str, Any]:
        """
        Generate recommendations using multi-model fusion approach
        
        Args:
            file: Uploaded image file
            target_count: Number of recommendations to return
            
        Returns:
            Dictionary containing query caption and suggestions
        """
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".jpg")[1]) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
            
            try:
                # Generate recommendations using the multi-model service
                recommendations = await self.recommendation_service.generate_recommendations(
                    query_image_path=tmp_path,
                    top_k=target_count
                )
                
                # Extract the key information for the simplified response
                query_caption = recommendations.get("query_analysis", {}).get("blip_caption", "")
                
                # Format suggestions in the expected format
                suggestions = []
                for rec in recommendations.get("recommendations", []):
                    suggestion = {
                        "id": rec.get("item_id", f"item_{len(suggestions)}"),
                        "img_url": rec.get("image_path", "/static/default.png"),
                        "tags": rec.get("metadata", {}).get("tags", ["时尚", "推荐"]),
                        "scores": {
                            "clip": rec.get("clip_score", 0.0),
                            "blip": rec.get("blip_score", 0.0),
                            "fashion": rec.get("fashion_score", 0.0),
                            "final": rec.get("final_score", 0.0)
                        },
                        "metadata": rec.get("metadata", {})
                    }
                    suggestions.append(suggestion)
                
                # Add Chinese advice if available
                chinese_advice = recommendations.get("chinese_advice", {})
                
                result = {
                    "query_caption": query_caption,
                    "suggestions": suggestions,
                    "chinese_advice": chinese_advice,
                    "fusion_stats": recommendations.get("fusion_stats", {})
                }
                
                return result
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing recommendation: {str(e)}")
    
    async def add_feedback(self, suggestion_id: str, liked: bool, 
                          clip_score: float, blip_score: float, fashion_score: float) -> Dict[str, Any]:
        """
        Add user feedback for online learning
        
        Args:
            suggestion_id: ID of the suggestion
            liked: Whether user liked the suggestion
            clip_score: CLIP similarity score
            blip_score: BLIP similarity score  
            fashion_score: Fashion encoder score
            
        Returns:
            Feedback processing result
        """
        try:
            # Convert like/dislike to rating
            user_rating = 1.0 if liked else 0.0
            
            # Add feedback to the recommendation service
            self.recommendation_service.add_user_feedback(
                item_id=suggestion_id,
                clip_score=clip_score,
                blip_score=blip_score,
                fashion_score=fashion_score,
                user_rating=user_rating,
                feedback_type="like_dislike"
            )
            
            return {
                "status": "success",
                "message": "Feedback added successfully",
                "suggestion_id": suggestion_id,
                "rating": user_rating
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get recommendation service statistics
        
        Returns:
            Service statistics
        """
        try:
            return self.recommendation_service.get_service_stats()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# Global instance
_fusion_match_handler = None

def get_fusion_match_handler() -> FusionMatchHandler:
    """
    Get or create the global fusion match handler instance
    
    Returns:
        FusionMatchHandler instance
    """
    global _fusion_match_handler
    if _fusion_match_handler is None:
        _fusion_match_handler = FusionMatchHandler()
    return _fusion_match_handler