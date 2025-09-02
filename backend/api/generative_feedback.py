#!/usr/bin/env python3
"""
Generative Feedback API
Provides endpoints for collecting and managing user feedback on generative recommendations.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from services.generative_feedback_service import get_generative_feedback_service
# Note: get_current_user is defined in app.py, we'll use a dependency injection approach
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from api.auth import AuthHandler

security = HTTPBearer()
auth_handler = AuthHandler()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token."""
    try:
        user_id = auth_handler.verify_token(credentials.credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

router = APIRouter(prefix="/api/generative/feedback", tags=["generative-feedback"])
logger = logging.getLogger(__name__)

class GenerativeFeedbackRequest(BaseModel):
    """Request model for generative feedback submission."""
    suggestion_id: str = Field(..., description="ID of the suggestion being rated")
    query_embedding: List[float] = Field(..., description="Original query embedding")
    generated_embedding: List[float] = Field(..., description="Generated embedding that was rated")
    rating: float = Field(..., ge=0.0, le=1.0, description="User rating from 0.0 to 1.0")
    feedback_type: str = Field(default="rating", description="Type of feedback")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    model_version: str = Field(default="v1.0", description="Model version used")

class GenerativeFeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    status: str
    feedback_id: str
    queued_for_training: bool
    message: Optional[str] = None

class FeedbackStatsResponse(BaseModel):
    """Response model for feedback statistics."""
    total_feedback: int
    average_rating: float
    rating_std: float
    feedback_distribution: Dict[str, int]
    recent_feedback_count: int
    training_queue_size: int

class UserPreferencesResponse(BaseModel):
    """Response model for user preferences."""
    preferences: Dict[str, Dict[str, List[str]]]
    confidence: float
    total_feedback: int
    average_rating: float

@router.post("/submit", response_model=GenerativeFeedbackResponse)
async def submit_generative_feedback(
    request: GenerativeFeedbackRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Submit feedback for a generative recommendation.
    
    This endpoint allows users to provide feedback on generative recommendations,
    which is used to improve the model through fine-tuning.
    """
    try:
        feedback_service = get_generative_feedback_service()
        
        result = await feedback_service.add_feedback(
            user_id=user_id,
            suggestion_id=request.suggestion_id,
            query_embedding=request.query_embedding,
            generated_embedding=request.generated_embedding,
            user_rating=request.rating,
            feedback_type=request.feedback_type,
            context=request.context,
            model_version=request.model_version
        )
        
        return GenerativeFeedbackResponse(
            status=result["status"],
            feedback_id=result["feedback_id"],
            queued_for_training=result["queued_for_training"],
            message="Feedback submitted successfully"
        )
        
    except Exception as e:
        logger.error(f"Error submitting generative feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(
    user_id: str = Depends(get_current_user)
):
    """
    Get feedback statistics for the current user.
    
    Returns comprehensive statistics about the user's feedback history
    and the current state of the training queue.
    """
    try:
        feedback_service = get_generative_feedback_service()
        stats = await feedback_service.get_feedback_stats(user_id=user_id)
        
        return FeedbackStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.get("/stats/global", response_model=FeedbackStatsResponse)
async def get_global_feedback_stats(
    user_id: str = Depends(get_current_user)  # Still require auth but get global stats
):
    """
    Get global feedback statistics across all users.
    
    Provides insights into overall system performance and user satisfaction.
    """
    try:
        feedback_service = get_generative_feedback_service()
        stats = await feedback_service.get_feedback_stats()  # No user_id for global stats
        
        return FeedbackStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting global feedback stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get global stats: {str(e)}")

@router.get("/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(
    user_id: str = Depends(get_current_user)
):
    """
    Get learned user preferences from feedback history.
    
    Analyzes the user's feedback patterns to extract style preferences,
    preferred occasions, and other contextual preferences.
    """
    try:
        feedback_service = get_generative_feedback_service()
        preferences = await feedback_service.get_user_preferences(user_id)
        
        return UserPreferencesResponse(**preferences)
        
    except Exception as e:
        logger.error(f"Error getting user preferences: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get preferences: {str(e)}")

@router.post("/like/{suggestion_id}")
async def like_suggestion(
    suggestion_id: str,
    query_embedding: List[float],
    generated_embedding: List[float],
    context: Optional[Dict[str, Any]] = None,
    user_id: str = Depends(get_current_user)
):
    """
    Quick endpoint to like a suggestion.
    
    Convenience endpoint that automatically sets rating to 0.8 for liked suggestions.
    """
    try:
        feedback_service = get_generative_feedback_service()
        
        result = await feedback_service.add_feedback(
            user_id=user_id,
            suggestion_id=suggestion_id,
            query_embedding=query_embedding,
            generated_embedding=generated_embedding,
            user_rating=0.8,  # High rating for liked items
            feedback_type="like",
            context=context or {}
        )
        
        return {
            "status": "success",
            "message": "Suggestion liked successfully",
            "feedback_id": result["feedback_id"]
        }
        
    except Exception as e:
        logger.error(f"Error liking suggestion: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to like suggestion: {str(e)}")

@router.post("/dislike/{suggestion_id}")
async def dislike_suggestion(
    suggestion_id: str,
    query_embedding: List[float],
    generated_embedding: List[float],
    context: Optional[Dict[str, Any]] = None,
    user_id: str = Depends(get_current_user)
):
    """
    Quick endpoint to dislike a suggestion.
    
    Convenience endpoint that automatically sets rating to 0.2 for disliked suggestions.
    """
    try:
        feedback_service = get_generative_feedback_service()
        
        result = await feedback_service.add_feedback(
            user_id=user_id,
            suggestion_id=suggestion_id,
            query_embedding=query_embedding,
            generated_embedding=generated_embedding,
            user_rating=0.2,  # Low rating for disliked items
            feedback_type="dislike",
            context=context or {}
        )
        
        return {
            "status": "success",
            "message": "Suggestion disliked successfully",
            "feedback_id": result["feedback_id"]
        }
        
    except Exception as e:
        logger.error(f"Error disliking suggestion: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to dislike suggestion: {str(e)}")

@router.get("/training/status")
async def get_training_status(
    user_id: str = Depends(get_current_user)
):
    """
    Get the current status of model training.
    
    Returns information about the training queue and recent training activities.
    """
    try:
        feedback_service = get_generative_feedback_service()
        stats = await feedback_service.get_feedback_stats()
        
        return {
            "training_queue_size": stats["training_queue_size"],
            "total_feedback": stats["total_feedback"],
            "recent_feedback": stats["recent_feedback_count"],
            "min_feedback_for_training": feedback_service.min_feedback_for_training,
            "ready_for_training": stats["training_queue_size"] >= feedback_service.min_feedback_for_training
        }
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")