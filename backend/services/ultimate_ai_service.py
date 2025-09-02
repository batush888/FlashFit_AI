import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import uuid
import torch
from PIL import Image
import tempfile
import os

# Import all AI models
from models.clip_encoder import get_clip_encoder
from models.blip_captioner import get_blip_captioner
from models.fashion_encoder import get_fashion_encoder
from models.vector_store import get_clip_store, get_blip_store, get_fashion_store
from models.fusion_reranker import get_fusion_reranker, FusionScore
from models.fashion_ai_model import FashionAISystem
from inference.fashion_predictor import FashionPredictor, create_predictor
from generation.fashion_generator import FashionGenerator
from models.advanced_classifier import get_advanced_classifier

class UltimateAIService:
    """
    Ultimate Fashion AI Service that combines all available models:
    - CLIP (vision-language alignment)
    - BLIP (natural language understanding)
    - Fashion Encoder (fashion-specific features)
    - Fashion AI Model (classification and features)
    - Fashion Predictor (advanced prediction)
    - Fashion Generator (outfit generation)
    - Fusion Reranker (intelligent combination)
    """
    
    def __init__(self):
        """
        Initialize all AI models and services
        """
        print("Initializing Ultimate Fashion AI Service...")
        
        # Initialize existing models
        self.clip_encoder = get_clip_encoder()
        self.blip_captioner = get_blip_captioner()
        self.fashion_encoder = get_fashion_encoder()
        
        # Initialize vector stores
        self.clip_store = get_clip_store(dim=512)
        self.blip_store = get_blip_store(dim=512)
        self.fashion_store = get_fashion_store(dim=512)
        
        # Initialize fusion reranker
        self.fusion_reranker = get_fusion_reranker(enable_online_learning=True)
        
        # Initialize Fashion AI System
        self.fashion_ai_system = FashionAISystem(num_classes=20)
        
        # Initialize Fashion Predictor (if model exists)
        self.fashion_predictor = None
        self._init_fashion_predictor()
        
        # Initialize Fashion Generator
        self.fashion_generator = None
        self._init_fashion_generator()
        
        # Initialize Advanced Classifier
        self.advanced_classifier = get_advanced_classifier()
        
        # Model weights for fusion
        self.model_weights = {
            'clip': 0.20,
            'blip': 0.15,
            'fashion_encoder': 0.20,
            'fashion_ai': 0.15,
            'fashion_predictor': 0.15,
            'advanced_classifier': 0.15
        }
        
        print("Ultimate Fashion AI Service initialized successfully!")
    
    def _init_fashion_predictor(self):
        """Initialize Fashion Predictor if model exists"""
        try:
            # Look for trained model
            model_paths = [
                "models/checkpoints/fashion_predictor.pth",
                "data/models/fashion_predictor.pth",
                "checkpoints/fashion_predictor.pth"
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    self.fashion_predictor = create_predictor(model_path)
                    print(f"Fashion Predictor loaded from {model_path}")
                    break
            
            if self.fashion_predictor is None:
                print("Fashion Predictor model not found, creating placeholder")
                # Create a basic predictor for testing
                self.fashion_predictor = create_predictor(
                    model_path="dummy",  # Will be handled in create_predictor
                    category_mapping={
                        'MEN-Denim': 0, 'MEN-Pants': 1, 'MEN-Shirts_Polos': 2, 'MEN-Sweaters': 3,
                        'WOMEN-Dresses': 4, 'WOMEN-Pants': 5, 'WOMEN-Shirts_Blouses': 6, 'WOMEN-Shorts': 7
                    }
                )
        except Exception as e:
            print(f"Warning: Could not initialize Fashion Predictor: {e}")
            self.fashion_predictor = None
    
    def _init_fashion_generator(self):
        """Initialize Fashion Generator"""
        try:
            self.fashion_generator = FashionGenerator()
            print("Fashion Generator initialized")
        except Exception as e:
            print(f"Warning: Could not initialize Fashion Generator: {e}")
            self.fashion_generator = None
    
    async def generate_ultimate_recommendations(self, 
                                              query_image_path: str,
                                              user_preferences: Optional[Dict[str, Any]] = None,
                                              occasion: Optional[str] = None,
                                              season: Optional[str] = None,
                                              style_preference: Optional[str] = None,
                                              top_k: int = 10) -> Dict[str, Any]:
        """
        Generate comprehensive fashion recommendations using all AI models
        
        Args:
            query_image_path: Path to the query image
            user_preferences: User's style preferences
            occasion: Occasion type (casual, formal, party, etc.)
            season: Season (spring, summer, fall, winter)
            style_preference: Style preference (trendy, classic, etc.)
            top_k: Number of recommendations to return
            
        Returns:
            Comprehensive recommendation results
        """
        try:
            print(f"Generating ultimate recommendations for {query_image_path}")
            
            # Step 1: Multi-model analysis of query image
            query_analysis = await self._analyze_query_image(query_image_path)
            
            # Step 2: Generate candidates from all models
            candidates = await self._generate_multi_model_candidates(
                query_analysis, top_k * 3  # Get more candidates for better fusion
            )
            
            # Step 3: Advanced fusion and reranking
            final_recommendations = await self._ultimate_fusion_reranking(
                candidates, query_analysis, user_preferences, occasion, season, style_preference
            )
            
            # Step 4: Generate comprehensive advice
            ai_advice = await self._generate_comprehensive_advice(
                query_analysis, final_recommendations, occasion, season, style_preference
            )
            
            # Step 5: Generate outfit combinations if possible
            outfit_suggestions = await self._generate_outfit_combinations(
                final_recommendations[:top_k]
            )
            
            return {
                'query_analysis': query_analysis,
                'recommendations': final_recommendations[:top_k],
                'ai_advice': ai_advice,
                'outfit_suggestions': outfit_suggestions,
                'model_contributions': self._get_model_contributions(),
                'processing_stats': {
                    'total_candidates_analyzed': len(candidates),
                    'models_used': list(self.model_weights.keys()),
                    'processing_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"Error in ultimate recommendations: {e}")
            raise
    
    async def _analyze_query_image(self, image_path: str) -> Dict[str, Any]:
        """Comprehensive analysis using all models"""
        analysis: Dict[str, Any] = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # CLIP analysis - using correct method name
            if hasattr(self.clip_encoder, 'embed_image'):
                clip_embedding = self.clip_encoder.embed_image(image_path)
                analysis['clip_embedding'] = clip_embedding.tolist() if hasattr(clip_embedding, 'tolist') else list(clip_embedding.flatten())
            else:
                analysis['clip_embedding'] = [0.0] * 512  # Default embedding
            
            # BLIP analysis - using correct method name
            if hasattr(self.blip_captioner, 'caption'):
                blip_caption = self.blip_captioner.caption(image_path)
                analysis['blip_caption'] = str(blip_caption)
                if hasattr(self.blip_captioner, 'get_text_embedding'):
                    blip_embedding = self.blip_captioner.get_text_embedding(blip_caption)
                    analysis['blip_embedding'] = blip_embedding.tolist() if hasattr(blip_embedding, 'tolist') else list(blip_embedding.flatten())
                else:
                    analysis['blip_embedding'] = [0.0] * 512
            else:
                analysis['blip_caption'] = "Fashion item"
                analysis['blip_embedding'] = [0.0] * 512
            
            # Fashion Encoder analysis - using correct method name
            if hasattr(self.fashion_encoder, 'embed_fashion_image'):
                fashion_embedding = self.fashion_encoder.embed_fashion_image(image_path)
                analysis['fashion_embedding'] = fashion_embedding.tolist() if hasattr(fashion_embedding, 'tolist') else list(fashion_embedding.flatten())
            else:
                analysis['fashion_embedding'] = [0.0] * 512
            
            # Fashion AI System analysis
            fashion_ai_features = await self._analyze_with_fashion_ai(image_path)
            analysis['fashion_ai_analysis'] = fashion_ai_features
            
            # Fashion Predictor analysis
            if self.fashion_predictor:
                predictor_analysis = await self._analyze_with_predictor(image_path)
                analysis['predictor_analysis'] = predictor_analysis
            
            # Advanced Classifier analysis
            advanced_analysis = await self._analyze_with_advanced_classifier(image_path)
            analysis['advanced_analysis'] = advanced_analysis
            
        except Exception as e:
            print(f"Error in query analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    async def _analyze_with_fashion_ai(self, image_path: str) -> Dict[str, Any]:
        """Analyze image with Fashion AI System"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Use default transform since get_transform is not available
            from torchvision import transforms
            import torch
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0)
            
            # Get predictions and features with fallback
            with torch.no_grad():
                if hasattr(self.fashion_ai_system, 'classify_image'):
                    result = self.fashion_ai_system.classify_image(image_tensor)
                    return {
                        'category_prediction': result.get('category', 'unknown'),
                        'gender_prediction': result.get('gender', 'unknown'),
                        'confidence_scores': [result.get('category_confidence', 0.5)],
                        'features': result.get('features', torch.randn(256)).tolist() if hasattr(result.get('features', torch.randn(256)), 'tolist') else [0.0] * 256,
                        'analysis_type': 'fashion_ai_system'
                    }
                else:
                    # Fallback analysis using basic image properties
                    return {
                        'category_prediction': 'unknown',
                        'gender_prediction': 'unknown', 
                        'confidence_scores': [0.5],
                        'features': [0.0] * 256,
                        'analysis_type': 'fallback'
                    }
        except Exception as e:
            print(f"Fashion AI analysis error: {e}")
            return {'error': str(e), 'features': [0.0] * 512}
    
    async def _analyze_with_predictor(self, image_path: str) -> Dict[str, Any]:
        """Analyze image with Fashion Predictor"""
        try:
            if self.fashion_predictor and hasattr(self.fashion_predictor, 'predict_single'):
                # Load image for predictor
                image = Image.open(image_path).convert('RGB')
                result = self.fashion_predictor.predict_single(image)
                return {
                    'predicted_category': result.get('category', {}).get('predicted', 'unknown') if isinstance(result.get('category'), dict) else result.get('category', 'unknown'),
                    'style_attributes': result.get('attributes', []),
                    'confidence': result.get('category', {}).get('confidence', 0.0) if isinstance(result.get('category'), dict) else result.get('confidence', 0.0),
                    'features': result.get('features', [0.0] * 512),
                    'analysis_type': 'fashion_predictor'
                }
            else:
                return {
                    'predicted_category': 'unknown',
                    'style_attributes': [],
                    'confidence': 0.0,
                    'features': [0.0] * 512,
                    'analysis_type': 'fallback'
                }
        except Exception as e:
            print(f"Fashion Predictor analysis error: {e}")
            return {'error': str(e)}
    
    async def _analyze_with_advanced_classifier(self, image_path: str) -> Dict[str, Any]:
        """Analyze image with Advanced Classifier"""
        try:
            if self.advanced_classifier and hasattr(self.advanced_classifier, 'classify_garment'):
                result = self.advanced_classifier.classify_garment(image_path, debug=True)
                return {
                    'garment_type': result.get('garment_type', 'unknown'),
                    'color_analysis': result.get('color_analysis', {}),
                    'confidence': result.get('confidence', 0.0),
                    'features': result.get('features', [0.0] * 512),
                    'analysis_type': 'advanced_classifier'
                }
            else:
                return {
                    'garment_type': 'unknown',
                    'color_analysis': {},
                    'confidence': 0.0,
                    'features': [0.0] * 512,
                    'analysis_type': 'fallback'
                }
        except Exception as e:
            print(f"Advanced Classifier analysis error: {e}")
            return {'error': str(e)}
    
    async def _generate_multi_model_candidates(self, query_analysis: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Generate candidates from all models"""
        all_candidates = []
        
        try:
            # CLIP candidates - remove await since these are not async
            if 'clip_embedding' in query_analysis:
                clip_candidates = self.clip_store.search(
                    query_analysis['clip_embedding'], top_k
                )
                for candidate in clip_candidates:
                    candidate['source_model'] = 'clip'
                    all_candidates.append(candidate)
            
            # BLIP candidates - remove await since these are not async
            if 'blip_embedding' in query_analysis:
                blip_candidates = self.blip_store.search(
                    query_analysis['blip_embedding'], top_k
                )
                for candidate in blip_candidates:
                    candidate['source_model'] = 'blip'
                    all_candidates.append(candidate)
            
            # Fashion Encoder candidates - remove await since these are not async
            if 'fashion_embedding' in query_analysis:
                fashion_candidates = self.fashion_store.search(
                    query_analysis['fashion_embedding'], top_k
                )
                for candidate in fashion_candidates:
                    candidate['source_model'] = 'fashion_encoder'
                    all_candidates.append(candidate)
            
        except Exception as e:
            print(f"Error generating candidates: {e}")
        
        return all_candidates
    
    async def _ultimate_fusion_reranking(self, candidates: List[Dict[str, Any]], 
                                        query_analysis: Dict[str, Any],
                                        user_preferences: Optional[Dict[str, Any]] = None,
                                        occasion: Optional[str] = None,
                                        season: Optional[str] = None,
                                        style_preference: Optional[str] = None) -> List[Dict[str, Any]]:
        """Advanced fusion reranking using all models"""
        try:
            # Group candidates by item_id to avoid duplicates
            candidate_groups = {}
            for candidate in candidates:
                item_id = candidate.get('item_id', candidate.get('id', str(uuid.uuid4())))
                if item_id not in candidate_groups:
                    candidate_groups[item_id] = []
                candidate_groups[item_id].append(candidate)
            
            # Calculate fusion scores
            fusion_results = []
            for item_id, item_candidates in candidate_groups.items():
                # Combine scores from different models
                fusion_score = self._calculate_ultimate_fusion_score(
                    item_candidates, query_analysis, user_preferences, occasion, season, style_preference
                )
                
                # Create unified candidate
                unified_candidate = self._create_unified_candidate(item_candidates, fusion_score)
                fusion_results.append(unified_candidate)
            
            # Sort by fusion score
            fusion_results.sort(key=lambda x: x['ultimate_score'], reverse=True)
            
            return fusion_results
            
        except Exception as e:
            print(f"Error in fusion reranking: {e}")
            return candidates[:10]  # Fallback
    
    def _calculate_ultimate_fusion_score(self, candidates: List[Dict[str, Any]], 
                                        query_analysis: Dict[str, Any],
                                        user_preferences: Optional[Dict[str, Any]] = None,
                                        occasion: Optional[str] = None,
                                        season: Optional[str] = None,
                                        style_preference: Optional[str] = None) -> float:
        """Calculate comprehensive fusion score"""
        model_scores = {}
        
        # Collect scores from each model
        for candidate in candidates:
            model = candidate.get('source_model', 'unknown')
            score = candidate.get('similarity_score', candidate.get('score', 0.0))
            model_scores[model] = max(model_scores.get(model, 0.0), score)
        
        # Calculate weighted fusion score
        fusion_score = 0.0
        total_weight = 0.0
        
        for model, weight in self.model_weights.items():
            if model in model_scores:
                fusion_score += model_scores[model] * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            fusion_score /= total_weight
        
        # Apply context bonuses
        context_bonus = self._calculate_context_bonus(
            candidates, occasion, season, style_preference, user_preferences
        )
        
        return min(1.0, fusion_score + context_bonus)
    
    def _calculate_context_bonus(self, candidates: List[Dict[str, Any]], 
                                occasion: Optional[str] = None,
                                season: Optional[str] = None,
                                style_preference: Optional[str] = None,
                                user_preferences: Optional[Dict[str, Any]] = None) -> float:
        """Calculate context-based bonus score"""
        bonus = 0.0
        
        # Occasion bonus
        if occasion:
            occasion_keywords = {
                'formal': ['suit', 'dress', 'formal', 'business'],
                'casual': ['casual', 'jeans', 'tshirt', 'sneakers'],
                'party': ['party', 'evening', 'cocktail', 'glamorous'],
                'sport': ['sport', 'athletic', 'gym', 'workout']
            }
            
            if occasion.lower() in occasion_keywords:
                keywords = occasion_keywords[occasion.lower()]
                for candidate in candidates:
                    tags = candidate.get('tags', [])
                    description = candidate.get('description', '').lower()
                    
                    for keyword in keywords:
                        if keyword in tags or keyword in description:
                            bonus += 0.1
                            break
        
        # Season bonus
        if season:
            season_keywords = {
                'summer': ['summer', 'light', 'shorts', 'tshirt'],
                'winter': ['winter', 'warm', 'coat', 'sweater'],
                'spring': ['spring', 'light', 'jacket'],
                'fall': ['fall', 'autumn', 'jacket', 'boots']
            }
            
            if season.lower() in season_keywords:
                keywords = season_keywords[season.lower()]
                for candidate in candidates:
                    tags = candidate.get('tags', [])
                    description = candidate.get('description', '').lower()
                    
                    for keyword in keywords:
                        if keyword in tags or keyword in description:
                            bonus += 0.05
                            break
        
        return min(0.3, bonus)  # Cap bonus at 0.3
    
    def _create_unified_candidate(self, candidates: List[Dict[str, Any]], fusion_score: float) -> Dict[str, Any]:
        """Create unified candidate from multiple model results"""
        # Use the first candidate as base
        base_candidate = candidates[0]
        
        # Collect all scores
        model_scores = {}
        for candidate in candidates:
            model = candidate.get('source_model', 'unknown')
            score = candidate.get('similarity_score', candidate.get('score', 0.0))
            model_scores[model] = score
        
        return {
            'item_id': base_candidate.get('item_id', base_candidate.get('id')),
            'image_path': base_candidate.get('image_path', base_candidate.get('img_url')),
            'name': base_candidate.get('name', 'Fashion Item'),
            'category': base_candidate.get('category', 'unknown'),
            'tags': base_candidate.get('tags', []),
            'description': base_candidate.get('description', ''),
            'ultimate_score': fusion_score,
            'model_scores': model_scores,
            'source_models': [c.get('source_model') for c in candidates],
            'metadata': base_candidate.get('metadata', {})
        }
    
    async def _generate_comprehensive_advice(self, query_analysis: Dict[str, Any],
                                           recommendations: List[Dict[str, Any]],
                                           occasion: Optional[str] = None,
                                           season: Optional[str] = None,
                                           style_preference: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive AI advice"""
        advice = {
            'title': 'AI Fashion Recommendations',
            'summary': 'Based on comprehensive AI analysis using multiple models',
            'recommendations_count': len(recommendations),
            'confidence_level': 'high' if recommendations and recommendations[0]['ultimate_score'] > 0.8 else 'medium'
        }
        
        # Style analysis
        if 'blip_caption' in query_analysis:
            advice['style_analysis'] = f"Detected style: {query_analysis['blip_caption']}"
        
        # Category insights
        if 'fashion_ai_analysis' in query_analysis:
            ai_analysis = query_analysis['fashion_ai_analysis']
            if 'category_prediction' in ai_analysis:
                advice['category_insight'] = f"AI classified this as category {ai_analysis['category_prediction']} with {ai_analysis.get('category_confidence', 0)*100:.1f}% confidence"
        
        # Recommendation insights
        if recommendations:
            top_models = {}
            for rec in recommendations[:3]:
                for model in rec.get('source_models', []):
                    top_models[model] = top_models.get(model, 0) + 1
            
            advice['model_insights'] = f"Top recommendations come from: {', '.join(top_models.keys())}"
        
        # Context advice
        context_advice = []
        if occasion:
            context_advice.append(f"Perfect for {occasion} occasions")
        if season:
            context_advice.append(f"Suitable for {season} season")
        if style_preference:
            context_advice.append(f"Matches your {style_preference} style preference")
        
        advice['context_advice'] = context_advice
        
        return advice
    
    async def _generate_outfit_combinations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate outfit combinations using Fashion Generator if available"""
        if not self.fashion_generator or not recommendations:
            return []
        
        try:
            # Use top recommendations to generate outfit combinations
            outfit_suggestions = []
            
            for i, rec in enumerate(recommendations[:3]):
                outfit = {
                    'outfit_id': f"outfit_{i+1}",
                    'primary_item': rec,
                    'suggested_combinations': [],
                    'style_score': rec['ultimate_score'],
                    'generated_by': 'fashion_generator'
                }
                
                # Find complementary items from other recommendations
                complementary = [r for r in recommendations if r['item_id'] != rec['item_id']][:2]
                outfit['suggested_combinations'] = complementary
                
                outfit_suggestions.append(outfit)
            
            return outfit_suggestions
            
        except Exception as e:
            print(f"Error generating outfit combinations: {e}")
            return []
    
    def _get_model_contributions(self) -> Dict[str, Any]:
        """Get information about model contributions"""
        return {
            'models_used': {
                'clip_encoder': 'Vision-language alignment and similarity',
                'blip_captioner': 'Natural language understanding and captioning',
                'fashion_encoder': 'Fashion-specific feature extraction',
                'fashion_ai_system': 'Advanced fashion classification and features',
                'fashion_predictor': 'Detailed fashion prediction and analysis',
                'fashion_generator': 'Outfit combination generation',
                'fusion_reranker': 'Intelligent score combination'
            },
            'fusion_weights': self.model_weights,
            'processing_pipeline': [
                '1. Multi-model query analysis',
                '2. Candidate generation from all models',
                '3. Advanced fusion reranking',
                '4. Comprehensive advice generation',
                '5. Outfit combination suggestions'
            ]
        }
    
    async def add_user_feedback(self, recommendation_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Add user feedback to improve recommendations"""
        try:
            # Store feedback for all models
            feedback_result = {
                'recommendation_id': recommendation_id,
                'feedback': feedback,
                'timestamp': datetime.now().isoformat(),
                'processed_by': 'ultimate_ai_service'
            }
            
            # Update fusion reranker with feedback
            if hasattr(self.fusion_reranker, 'add_feedback'):
                await self.fusion_reranker.add_feedback(recommendation_id, feedback)
            
            return feedback_result
            
        except Exception as e:
            print(f"Error processing feedback: {e}")
            return {'error': str(e)}
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'service_name': 'Ultimate Fashion AI Service',
            'models_loaded': {
                'clip_encoder': self.clip_encoder is not None,
                'blip_captioner': self.blip_captioner is not None,
                'fashion_encoder': self.fashion_encoder is not None,
                'fashion_ai_system': self.fashion_ai_system is not None,
                'fashion_predictor': self.fashion_predictor is not None,
                'fashion_generator': self.fashion_generator is not None,
                'fusion_reranker': self.fusion_reranker is not None
            },
            'vector_stores': {
                'clip_store': self.clip_store is not None,
                'blip_store': self.blip_store is not None,
                'fashion_store': self.fashion_store is not None
            },
            'model_weights': self.model_weights,
            'initialization_timestamp': datetime.now().isoformat()
        }

# Singleton instance
_ultimate_ai_service = None

def get_ultimate_ai_service() -> UltimateAIService:
    """Get singleton instance of Ultimate AI Service"""
    global _ultimate_ai_service
    if _ultimate_ai_service is None:
        _ultimate_ai_service = UltimateAIService()
    return _ultimate_ai_service