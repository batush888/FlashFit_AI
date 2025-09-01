from fastapi import UploadFile, File, HTTPException
from typing import List, Dict, Any, Optional
import tempfile
import shutil
import os
import json
from pathlib import Path

# Import our ultimate AI service
from services.ultimate_ai_service import get_ultimate_ai_service

class UltimateAIHandler:
    """
    Ultimate AI Handler that provides comprehensive fashion recommendations
    using all available AI models in the system
    """
    
    def __init__(self):
        self.ultimate_ai_service = get_ultimate_ai_service()
        print("UltimateAIHandler initialized with comprehensive AI models")
    
    async def generate_ultimate_recommendations(self, 
                                              file: UploadFile,
                                              user_preferences: Optional[Dict[str, Any]] = None,
                                              occasion: Optional[str] = None,
                                              season: Optional[str] = None,
                                              style_preference: Optional[str] = None,
                                              target_count: int = 10) -> Dict[str, Any]:
        """
        Generate ultimate fashion recommendations using all AI models
        
        Args:
            file: Uploaded image file
            user_preferences: User's style preferences
            occasion: Occasion type (casual, formal, party, etc.)
            season: Season (spring, summer, fall, winter)
            style_preference: Style preference (trendy, classic, etc.)
            target_count: Number of recommendations to return
            
        Returns:
            Comprehensive recommendation results
        """
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".jpg")[1]) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
            
            try:
                # Generate ultimate recommendations with proper None handling
                recommendations = await self.ultimate_ai_service.generate_ultimate_recommendations(
                    query_image_path=tmp_path,
                    user_preferences=user_preferences or {},
                    occasion=occasion or "casual",
                    season=season or "spring",
                    style_preference=style_preference or "trendy",
                    top_k=target_count
                )
                
                # Format response for API
                formatted_response = self._format_api_response(recommendations)
                
                return {
                    "status": "success",
                    "message": "Ultimate AI recommendations generated successfully",
                    "data": formatted_response
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating ultimate recommendations: {str(e)}")
    
    def _format_api_response(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the ultimate AI response for API consumption
        """
        # Extract query analysis
        query_analysis = recommendations.get('query_analysis', {})
        
        # Format recommendations
        formatted_recommendations = []
        for rec in recommendations.get('recommendations', []):
            formatted_rec = {
                "item_id": rec.get('item_id'),
                "name": rec.get('name', 'Fashion Item'),
                "category": rec.get('category', 'unknown'),
                "image_url": rec.get('image_path', '/static/default.png'),
                "tags": rec.get('tags', []),
                "description": rec.get('description', ''),
                "scores": {
                    "ultimate_score": rec.get('ultimate_score', 0.0),
                    "model_scores": rec.get('model_scores', {})
                },
                "confidence": self._calculate_confidence(rec.get('ultimate_score', 0.0)),
                "source_models": rec.get('source_models', []),
                "metadata": rec.get('metadata', {})
            }
            formatted_recommendations.append(formatted_rec)
        
        # Format AI advice
        ai_advice = recommendations.get('ai_advice', {})
        
        # Format outfit suggestions
        outfit_suggestions = recommendations.get('outfit_suggestions', [])
        
        return {
            "query_analysis": {
                "detected_style": query_analysis.get('blip_caption', 'Unknown style'),
                "ai_classification": self._extract_ai_classification(query_analysis),
                "confidence_level": ai_advice.get('confidence_level', 'medium'),
                "processing_models": list(recommendations.get('model_contributions', {}).get('models_used', {}).keys())
            },
            "recommendations": formatted_recommendations,
            "ai_insights": {
                "style_analysis": ai_advice.get('style_analysis', ''),
                "category_insight": ai_advice.get('category_insight', ''),
                "model_insights": ai_advice.get('model_insights', ''),
                "context_advice": ai_advice.get('context_advice', []),
                "recommendation_summary": f"Found {len(formatted_recommendations)} high-quality matches using {len(recommendations.get('model_contributions', {}).get('models_used', {}))} AI models"
            },
            "outfit_combinations": outfit_suggestions,
            "model_performance": {
                "models_used": recommendations.get('model_contributions', {}).get('models_used', {}),
                "fusion_weights": recommendations.get('model_contributions', {}).get('fusion_weights', {}),
                "processing_pipeline": recommendations.get('model_contributions', {}).get('processing_pipeline', [])
            },
            "processing_stats": recommendations.get('processing_stats', {})
        }
    
    def _extract_ai_classification(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract AI classification information
        """
        classification = {}
        
        # Fashion AI System classification
        if 'fashion_ai_analysis' in query_analysis:
            ai_analysis = query_analysis['fashion_ai_analysis']
            classification['fashion_ai'] = {
                'category': ai_analysis.get('category_prediction'),
                'gender': ai_analysis.get('gender_prediction'),
                'category_confidence': ai_analysis.get('category_confidence', 0.0),
                'gender_confidence': ai_analysis.get('gender_confidence', 0.0)
            }
        
        # Fashion Predictor classification
        if 'predictor_analysis' in query_analysis:
            predictor_analysis = query_analysis['predictor_analysis']
            if 'prediction' in predictor_analysis:
                pred = predictor_analysis['prediction']
                classification['fashion_predictor'] = {
                    'category': pred.get('category', {}).get('predicted'),
                    'confidence': pred.get('category', {}).get('confidence', 0.0)
                }
        
        return classification
    
    def _calculate_confidence(self, ultimate_score: float) -> str:
        """
        Calculate confidence level based on ultimate score
        """
        if ultimate_score >= 0.8:
            return "very_high"
        elif ultimate_score >= 0.6:
            return "high"
        elif ultimate_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    async def add_ultimate_feedback(self, 
                                   recommendation_id: str,
                                   feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add user feedback to improve ultimate AI recommendations
        
        Args:
            recommendation_id: ID of the recommendation
            feedback_data: Feedback information including rating, liked/disliked, etc.
            
        Returns:
            Feedback processing result
        """
        try:
            result = await self.ultimate_ai_service.add_user_feedback(
                recommendation_id, feedback_data
            )
            
            return {
                "status": "success",
                "message": "Feedback processed successfully",
                "data": result
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")
    
    def get_ultimate_ai_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the Ultimate AI system
        
        Returns:
            System statistics and model information
        """
        try:
            stats = self.ultimate_ai_service.get_service_stats()
            
            return {
                "status": "success",
                "message": "Ultimate AI statistics retrieved successfully",
                "data": {
                    "system_info": {
                        "service_name": stats.get('service_name'),
                        "total_models": len([m for m in stats.get('models_loaded', {}).values() if m]),
                        "initialization_time": stats.get('initialization_timestamp')
                    },
                    "model_status": stats.get('models_loaded', {}),
                    "vector_stores": stats.get('vector_stores', {}),
                    "fusion_configuration": {
                        "model_weights": stats.get('model_weights', {}),
                        "total_weight": sum(stats.get('model_weights', {}).values())
                    },
                    "capabilities": {
                        "multi_model_analysis": True,
                        "fusion_reranking": True,
                        "outfit_generation": stats.get('models_loaded', {}).get('fashion_generator', False),
                        "advanced_prediction": stats.get('models_loaded', {}).get('fashion_predictor', False),
                        "comprehensive_advice": True
                    }
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")
    
    async def analyze_fashion_item(self, file: UploadFile) -> Dict[str, Any]:
        """
        Comprehensive analysis of a fashion item using all AI models
        
        Args:
            file: Uploaded image file
            
        Returns:
            Detailed analysis results
        """
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".jpg")[1]) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
            
            try:
                # Generate comprehensive analysis
                recommendations = await self.ultimate_ai_service.generate_ultimate_recommendations(
                    query_image_path=tmp_path,
                    top_k=1  # We only need analysis, not recommendations
                )
                
                query_analysis = recommendations.get('query_analysis', {})
                
                return {
                    "status": "success",
                    "message": "Fashion item analysis completed",
                    "data": {
                        "style_description": query_analysis.get('blip_caption', 'Unknown style'),
                        "ai_classification": self._extract_ai_classification(query_analysis),
                        "feature_analysis": {
                            "clip_features": len(query_analysis.get('clip_embedding', [])),
                            "blip_features": len(query_analysis.get('blip_embedding', [])),
                            "fashion_features": len(query_analysis.get('fashion_embedding', [])),
                            "ai_features": len(query_analysis.get('fashion_ai_analysis', {}).get('features', []))
                        },
                        "model_insights": {
                            "models_analyzed": list(query_analysis.keys()),
                            "confidence_scores": self._extract_confidence_scores(query_analysis)
                        },
                        "processing_timestamp": query_analysis.get('timestamp')
                    }
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing fashion item: {str(e)}")
    
    def _extract_confidence_scores(self, query_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract confidence scores from different models
        """
        confidence_scores = {}
        
        # Fashion AI confidence
        if 'fashion_ai_analysis' in query_analysis:
            ai_analysis = query_analysis['fashion_ai_analysis']
            confidence_scores['fashion_ai_category'] = ai_analysis.get('category_confidence', 0.0)
            confidence_scores['fashion_ai_gender'] = ai_analysis.get('gender_confidence', 0.0)
        
        # Fashion Predictor confidence
        if 'predictor_analysis' in query_analysis:
            predictor_analysis = query_analysis['predictor_analysis']
            confidence_scores['fashion_predictor'] = predictor_analysis.get('confidence', 0.0)
        
        return confidence_scores

# Singleton instance
_ultimate_ai_handler = None

def get_ultimate_ai_handler() -> UltimateAIHandler:
    """Get singleton instance of Ultimate AI Handler"""
    global _ultimate_ai_handler
    if _ultimate_ai_handler is None:
        _ultimate_ai_handler = UltimateAIHandler()
    return _ultimate_ai_handler