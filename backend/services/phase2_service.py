#!/usr/bin/env python3
"""
Phase 2 Integration Service

This service integrates all Phase 2 enhancements:
1. Enhanced Fashion Encoder with fine-tuning
2. BLIP+CLIP fusion with expanded vocabulary
3. Adaptive Fusion Reranker with meta-learning
4. Personalization Layer with user embeddings
5. Real-time feedback integration
6. Context-aware recommendations
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import uuid
import logging
import sys
import os

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add ml directory to path for Phase 2 models
ml_path = os.path.join(os.path.dirname(__file__), '../../ml')
if ml_path not in sys.path:
    sys.path.append(ml_path)

# Initialize Phase 2 model classes as None (will be imported dynamically)
EnhancedFashionEncoder = None
BLIPCLIPFusion = None
AdaptiveFusionReranker = None
PersonalizationEngine = None
PersonalizationContext = None

# Try to import Phase 2 models dynamically
try:
    from enhanced_fashion_encoder import EnhancedFashionEncoder  # type: ignore
except ImportError:
    pass

try:
    from blip_clip_fusion import BLIPCLIPFusion  # type: ignore
except ImportError:
    pass

try:
    from adaptive_fusion_reranker import AdaptiveFusionReranker  # type: ignore
except ImportError:
    pass

try:
    from personalization_layer import PersonalizationEngine, PersonalizationContext  # type: ignore
except ImportError:
    pass

logger.info(f"Phase 2 models availability:")
logger.info(f"  EnhancedFashionEncoder: {EnhancedFashionEncoder is not None}")
logger.info(f"  BLIPCLIPFusion: {BLIPCLIPFusion is not None}")
logger.info(f"  AdaptiveFusionReranker: {AdaptiveFusionReranker is not None}")
logger.info(f"  PersonalizationEngine: {PersonalizationEngine is not None}")

# Import existing models
from models.clip_encoder import get_clip_encoder
from models.blip_captioner import get_blip_captioner
from models.fashion_encoder import get_fashion_encoder
from models.vector_store import get_clip_store, get_blip_store, get_fashion_store
from models.fusion_reranker import get_fusion_reranker

# Logger already set up above

class Phase2RecommendationService:
    """
    Advanced recommendation service with Phase 2 enhancements
    """
    
    def __init__(self, 
                 enable_enhanced_encoder: bool = True,
                 enable_blip_clip_fusion: bool = True,
                 enable_adaptive_reranker: bool = True,
                 enable_personalization: bool = True):
        """
        Initialize Phase 2 recommendation service
        
        Args:
            enable_enhanced_encoder: Enable enhanced fashion encoder
            enable_blip_clip_fusion: Enable BLIP+CLIP fusion
            enable_adaptive_reranker: Enable adaptive fusion reranker
            enable_personalization: Enable personalization layer
        """
        self.enable_enhanced_encoder = enable_enhanced_encoder
        self.enable_blip_clip_fusion = enable_blip_clip_fusion
        self.enable_adaptive_reranker = enable_adaptive_reranker
        self.enable_personalization = enable_personalization
        
        # Initialize base models (fallback)
        self.clip_encoder = get_clip_encoder()
        self.blip_captioner = get_blip_captioner()
        self.fashion_encoder = get_fashion_encoder()
        
        # Initialize vector stores
        self.clip_store = get_clip_store(dim=512)
        self.blip_store = get_blip_store(dim=768)
        self.fashion_store = get_fashion_store(dim=512)
        
        # Initialize base fusion reranker
        self.base_fusion_reranker = get_fusion_reranker(enable_online_learning=True)
        
        # Initialize Phase 2 models
        self._initialize_phase2_models()
        
        # Performance tracking
        self.phase2_stats = {
            'total_requests': 0,
            'enhanced_encoder_requests': 0,
            'fusion_requests': 0,
            'adaptive_reranker_requests': 0,
            'personalized_requests': 0,
            'fallback_requests': 0,
            'average_response_time': 0.0,
            'user_feedback_count': 0
        }
        
        logger.info("Phase2RecommendationService initialized")
        logger.info(f"Enhanced Encoder: {self.enhanced_encoder is not None}")
        logger.info(f"BLIP+CLIP Fusion: {self.blip_clip_fusion is not None}")
        logger.info(f"Adaptive Reranker: {self.adaptive_reranker is not None}")
        logger.info(f"Personalization: {self.personalization_engine is not None}")
    
    def _initialize_phase2_models(self):
        """
        Initialize Phase 2 models with error handling
        """
        # Enhanced Fashion Encoder
        self.enhanced_encoder = None
        if self.enable_enhanced_encoder and EnhancedFashionEncoder:
            try:
                self.enhanced_encoder = EnhancedFashionEncoder(
                    base_model='clip',
                    embedding_dim=512,
                    num_categories=50,
                    style_attention_heads=8
                )
                logger.info("Enhanced Fashion Encoder initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced Fashion Encoder: {e}")
        
        # BLIP+CLIP Fusion
        self.blip_clip_fusion = None
        if self.enable_blip_clip_fusion and BLIPCLIPFusion:
            try:
                self.blip_clip_fusion = BLIPCLIPFusion(
                    clip_model_name='openai/clip-vit-base-patch32',
                    blip_model_name='Salesforce/blip-image-captioning-base',
                    fusion_dim=512
                )
                logger.info("BLIP+CLIP Fusion initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize BLIP+CLIP Fusion: {e}")
        
        # Adaptive Fusion Reranker
        self.adaptive_reranker = None
        if self.enable_adaptive_reranker and AdaptiveFusionReranker:
            try:
                self.adaptive_reranker = AdaptiveFusionReranker(
                    embedding_dim=512,
                    meta_learning_rate=0.01,
                    confidence_threshold=0.7
                )
                logger.info("Adaptive Fusion Reranker initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Adaptive Fusion Reranker: {e}")
        
        # Personalization Engine
        self.personalization_engine = None
        if self.enable_personalization and PersonalizationEngine:
            try:
                self.personalization_engine = PersonalizationEngine(
                    embedding_dim=256,
                    redis_host='localhost',
                    redis_port=6379,
                    redis_db=1,
                    faiss_index_path='data/user_embeddings.index'
                )
                logger.info("Personalization Engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Personalization Engine: {e}")
    
    async def generate_recommendations(self,
                                    query_image_path: str,
                                    user_id: Optional[str] = None,
                                    context: Optional[Dict[str, Any]] = None,
                                    top_k: int = 10) -> Dict[str, Any]:
        """
        Generate Phase 2 enhanced recommendations
        
        Args:
            query_image_path: Path to query image
            user_id: User identifier for personalization
            context: Recommendation context (season, occasion, etc.)
            top_k: Number of recommendations to return
            
        Returns:
            Enhanced recommendations with Phase 2 features
        """
        start_time = datetime.now()
        self.phase2_stats['total_requests'] += 1
        
        try:
            # Step 1: Enhanced image analysis
            analysis_results = await self._analyze_query_image(query_image_path)
            
            # Step 2: Generate candidates using multiple approaches
            candidates = await self._generate_candidates(
                analysis_results, 
                top_k=min(100, top_k * 10)  # Get more candidates for reranking
            )
            
            if not candidates:
                return self._generate_fallback_response(query_image_path, analysis_results)
            
            # Step 3: Apply Phase 2 reranking
            reranked_candidates = await self._apply_phase2_reranking(
                candidates, 
                analysis_results, 
                user_id, 
                context
            )
            
            # Step 4: Apply personalization if available
            if user_id and self.personalization_engine:
                personalized_candidates = await self._apply_personalization(
                    reranked_candidates,
                    user_id,
                    context,
                    top_k
                )
                self.phase2_stats['personalized_requests'] += 1
            else:
                personalized_candidates = reranked_candidates[:top_k]
            
            # Step 5: Generate enhanced response
            response = self._generate_enhanced_response(
                analysis_results,
                personalized_candidates,
                user_id,
                context
            )
            
            # Update performance stats
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self._update_performance_stats(response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in Phase 2 recommendations: {e}")
            self.phase2_stats['fallback_requests'] += 1
            return self._generate_fallback_response(query_image_path, {})
    
    async def _analyze_query_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze query image using Phase 2 enhanced models
        
        Args:
            image_path: Path to image
            
        Returns:
            Analysis results from all models
        """
        results = {}
        
        # Base CLIP analysis
        try:
            results['clip_embedding'] = self.clip_encoder.embed_image(image_path)
        except Exception as e:
            logger.warning(f"CLIP analysis failed: {e}")
            results['clip_embedding'] = np.zeros(512)
        
        # Enhanced Fashion Encoder analysis
        if self.enhanced_encoder:
            try:
                enhanced_results = self.enhanced_encoder.analyze_image(image_path)
                results['enhanced_embedding'] = enhanced_results['embedding']
                results['category_predictions'] = enhanced_results['category_predictions']
                results['style_attention'] = enhanced_results['style_attention']
                results['compatibility_features'] = enhanced_results['compatibility_features']
                self.phase2_stats['enhanced_encoder_requests'] += 1
            except Exception as e:
                logger.warning(f"Enhanced Fashion Encoder analysis failed: {e}")
                results['enhanced_embedding'] = results['clip_embedding']
        else:
            results['enhanced_embedding'] = results['clip_embedding']
        
        # BLIP+CLIP Fusion analysis
        if self.blip_clip_fusion:
            try:
                fusion_results = self.blip_clip_fusion.analyze_image(image_path)
                results['blip_caption'] = fusion_results['caption']
                results['enhanced_caption'] = fusion_results['enhanced_caption']
                results['fusion_embedding'] = fusion_results['fusion_embedding']
                results['fashion_attributes'] = fusion_results['fashion_attributes']
                self.phase2_stats['fusion_requests'] += 1
            except Exception as e:
                logger.warning(f"BLIP+CLIP Fusion analysis failed: {e}")
                # Fallback to base BLIP
                results['blip_caption'] = self.blip_captioner.caption(image_path)
                results['fusion_embedding'] = results['clip_embedding']
        else:
            # Fallback to base models
            results['blip_caption'] = self.blip_captioner.caption(image_path)
            results['fusion_embedding'] = results['clip_embedding']
        
        # Base fashion encoder analysis
        try:
            fashion_results = self.fashion_encoder.analyze_fashion_attributes(image_path)
            results['fashion_attributes_base'] = fashion_results
            results['garment_classification'] = self.fashion_encoder.classify_garment_type(image_path)
        except Exception as e:
            logger.warning(f"Base fashion encoder analysis failed: {e}")
            results['fashion_attributes_base'] = {}
            results['garment_classification'] = {'top_category': 'unknown', 'top_score': 0.0}
        
        return results
    
    async def _generate_candidates(self, 
                                analysis_results: Dict[str, Any], 
                                top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Generate candidate items using multiple embedding approaches
        
        Args:
            analysis_results: Image analysis results
            top_k: Number of candidates to generate
            
        Returns:
            List of candidate items
        """
        all_candidates = {}
        
        # CLIP-based candidates
        try:
            clip_embedding = analysis_results.get('clip_embedding')
            if clip_embedding is not None:
                clip_candidates = self.clip_store.search(
                    clip_embedding.reshape(1, -1), 
                    topk=top_k // 3
                )
                
                for candidate_meta, score in clip_candidates:
                    item_id = candidate_meta.get('item_id', '')
                    if item_id not in all_candidates:
                        all_candidates[item_id] = {
                            'metadata': candidate_meta,
                            'clip_score': float(score),
                            'blip_score': 0.0,
                            'fashion_score': 0.0,
                            'enhanced_score': 0.0
                        }
        except Exception as e:
            logger.warning(f"CLIP candidate generation failed: {e}")
        
        # Enhanced embedding candidates
        try:
            enhanced_embedding = analysis_results.get('enhanced_embedding')
            if enhanced_embedding is not None and self.enhanced_encoder:
                enhanced_candidates = self.fashion_store.search(
                    enhanced_embedding.reshape(1, -1),
                    topk=top_k // 3
                )
                
                for candidate_meta, score in enhanced_candidates:
                    item_id = candidate_meta.get('item_id', '')
                    if item_id in all_candidates:
                        all_candidates[item_id]['enhanced_score'] = float(score)
                    else:
                        all_candidates[item_id] = {
                            'metadata': candidate_meta,
                            'clip_score': 0.0,
                            'blip_score': 0.0,
                            'fashion_score': 0.0,
                            'enhanced_score': float(score)
                        }
        except Exception as e:
            logger.warning(f"Enhanced embedding candidate generation failed: {e}")
        
        # Fusion embedding candidates
        try:
            fusion_embedding = analysis_results.get('fusion_embedding')
            if fusion_embedding is not None:
                fusion_candidates = self.blip_store.search(
                    fusion_embedding.reshape(1, -1),
                    topk=top_k // 3
                )
                
                for candidate_meta, score in fusion_candidates:
                    item_id = candidate_meta.get('item_id', '')
                    if item_id in all_candidates:
                        all_candidates[item_id]['blip_score'] = float(score)
                    else:
                        all_candidates[item_id] = {
                            'metadata': candidate_meta,
                            'clip_score': 0.0,
                            'blip_score': float(score),
                            'fashion_score': 0.0,
                            'enhanced_score': 0.0
                        }
        except Exception as e:
            logger.warning(f"Fusion embedding candidate generation failed: {e}")
        
        # Convert to list and sort by combined score
        candidates_list = []
        for item_id, scores in all_candidates.items():
            # Calculate combined score
            combined_score = (
                0.3 * scores['clip_score'] +
                0.25 * scores['blip_score'] +
                0.25 * scores['fashion_score'] +
                0.2 * scores['enhanced_score']
            )
            
            candidate = {
                'item_id': item_id,
                'combined_score': combined_score,
                **scores
            }
            candidates_list.append(candidate)
        
        # Sort by combined score
        candidates_list.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return candidates_list[:top_k]
    
    async def _apply_phase2_reranking(self,
                                    candidates: List[Dict[str, Any]],
                                    analysis_results: Dict[str, Any],
                                    user_id: Optional[str] = None,
                                    context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Apply Phase 2 adaptive reranking
        
        Args:
            candidates: List of candidate items
            analysis_results: Query analysis results
            user_id: User identifier
            context: Recommendation context
            
        Returns:
            Reranked candidates
        """
        if not self.adaptive_reranker or not candidates:
            return candidates
        
        try:
            # Prepare reranking input
            candidate_embeddings = []
            candidate_scores = []
            
            for candidate in candidates:
                # Use enhanced embedding if available, otherwise clip
                embedding = analysis_results.get('enhanced_embedding', 
                                                analysis_results.get('clip_embedding'))
                candidate_embeddings.append(embedding)
                
                scores = {
                    'clip': candidate.get('clip_score', 0.0),
                    'blip': candidate.get('blip_score', 0.0),
                    'fashion': candidate.get('fashion_score', 0.0),
                    'enhanced': candidate.get('enhanced_score', 0.0)
                }
                candidate_scores.append(scores)
            
            # Apply adaptive reranking
            reranked_scores = self.adaptive_reranker.rerank_candidates(
                query_embedding=analysis_results.get('enhanced_embedding', 
                                                    analysis_results.get('clip_embedding')),
                candidate_embeddings=np.array(candidate_embeddings),
                candidate_scores=candidate_scores,
                user_id=user_id,
                context=context
            )
            
            # Update candidates with new scores
            for i, candidate in enumerate(candidates):
                if i < len(reranked_scores):
                    candidate['adaptive_score'] = float(reranked_scores[i])
                    candidate['final_score'] = float(reranked_scores[i])
                else:
                    candidate['adaptive_score'] = candidate['combined_score']
                    candidate['final_score'] = candidate['combined_score']
            
            # Sort by adaptive score
            candidates.sort(key=lambda x: x['final_score'], reverse=True)
            
            self.phase2_stats['adaptive_reranker_requests'] += 1
            
        except Exception as e:
            logger.warning(f"Adaptive reranking failed: {e}")
            # Use combined scores as fallback
            for candidate in candidates:
                candidate['final_score'] = candidate['combined_score']
        
        return candidates
    
    async def _apply_personalization(self,
                                   candidates: List[Dict[str, Any]],
                                   user_id: str,
                                   context: Optional[Dict[str, Any]] = None,
                                   top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Apply personalization to candidates
        
        Args:
            candidates: List of candidates
            user_id: User identifier
            context: Recommendation context
            top_k: Number of personalized recommendations
            
        Returns:
            Personalized recommendations
        """
        if not self.personalization_engine:
            return candidates[:top_k]
        
        try:
            # Create personalization context
            if PersonalizationContext and context:
                personalization_context = PersonalizationContext(
                    user_id=user_id,
                    current_season=context.get('season', 'spring'),
                    time_of_day=context.get('time_of_day', '12:00'),
                    occasion=context.get('occasion', 'casual'),
                    budget_constraint=context.get('budget_constraint'),
                    recent_purchases=context.get('recent_purchases', []),
                    browsing_session=context.get('browsing_session', []),
                    location=context.get('location'),
                    weather=context.get('weather')
                )
            else:
                # Create minimal context
                personalization_context = None
            
            # Prepare candidate items for personalization
            candidate_items = []
            for candidate in candidates:
                item = {
                    'item_id': candidate['item_id'],
                    'embedding': candidate.get('metadata', {}).get('embedding', np.zeros(512)),
                    'style': candidate.get('metadata', {}).get('style', 'casual'),
                    'color': candidate.get('metadata', {}).get('color', 'neutral'),
                    'brand': candidate.get('metadata', {}).get('brand', 'unknown'),
                    'price': candidate.get('metadata', {}).get('price', 50.0),
                    'final_score': candidate['final_score']
                }
                candidate_items.append(item)
            
            # Get personalized recommendations
            if personalization_context:
                personalized_items = self.personalization_engine.get_personalized_recommendations(
                    user_id=user_id,
                    candidate_items=candidate_items,
                    context=personalization_context,
                    top_k=top_k
                )
            else:
                # Fallback without context
                personalized_items = candidate_items[:top_k]
            
            # Map back to original candidate format
            personalized_candidates = []
            for item in personalized_items:
                # Find original candidate
                original_candidate = next(
                    (c for c in candidates if c['item_id'] == item['item_id']), 
                    None
                )
                
                if original_candidate:
                    personalized_candidate = original_candidate.copy()
                    personalized_candidate['personalized_score'] = item.get('personalized_score', item['final_score'])
                    personalized_candidate['preference_boost'] = item.get('preference_boost', 0.0)
                    personalized_candidate['context_boost'] = item.get('context_boost', 0.0)
                    personalized_candidate['confidence_weight'] = item.get('confidence_weight', 1.0)
                    personalized_candidates.append(personalized_candidate)
            
            return personalized_candidates
            
        except Exception as e:
            logger.warning(f"Personalization failed: {e}")
            return candidates[:top_k]
    
    def _generate_enhanced_response(self,
                                  analysis_results: Dict[str, Any],
                                  recommendations: List[Dict[str, Any]],
                                  user_id: Optional[str] = None,
                                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate enhanced response with Phase 2 features
        
        Args:
            analysis_results: Query analysis results
            recommendations: Final recommendations
            user_id: User identifier
            context: Recommendation context
            
        Returns:
            Enhanced response dictionary
        """
        response = {
            'query_analysis': {
                'blip_caption': analysis_results.get('blip_caption', ''),
                'enhanced_caption': analysis_results.get('enhanced_caption', ''),
                'garment_type': analysis_results.get('garment_classification', {}).get('top_category', 'unknown'),
                'confidence': analysis_results.get('garment_classification', {}).get('top_score', 0.0),
                'fashion_attributes': analysis_results.get('fashion_attributes', {}),
                'style_attention': analysis_results.get('style_attention', {})
            },
            'recommendations': [],
            'phase2_features': {
                'enhanced_encoder_used': self.enhanced_encoder is not None,
                'blip_clip_fusion_used': self.blip_clip_fusion is not None,
                'adaptive_reranking_used': self.adaptive_reranker is not None,
                'personalization_used': user_id is not None and self.personalization_engine is not None
            },
            'performance_stats': self.get_performance_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Format recommendations
        for rec in recommendations:
            recommendation = {
                'item_id': rec['item_id'],
                'metadata': rec.get('metadata', {}),
                'scores': {
                    'clip': rec.get('clip_score', 0.0),
                    'blip': rec.get('blip_score', 0.0),
                    'fashion': rec.get('fashion_score', 0.0),
                    'enhanced': rec.get('enhanced_score', 0.0),
                    'combined': rec.get('combined_score', 0.0),
                    'adaptive': rec.get('adaptive_score', 0.0),
                    'personalized': rec.get('personalized_score', 0.0),
                    'final': rec.get('final_score', rec.get('personalized_score', rec.get('adaptive_score', rec.get('combined_score', 0.0))))
                },
                'personalization': {
                    'preference_boost': rec.get('preference_boost', 0.0),
                    'context_boost': rec.get('context_boost', 0.0),
                    'confidence_weight': rec.get('confidence_weight', 1.0)
                } if user_id else None
            }
            response['recommendations'].append(recommendation)
        
        return response
    
    def _generate_fallback_response(self, 
                                  image_path: str, 
                                  analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate fallback response when Phase 2 models fail
        
        Args:
            image_path: Query image path
            analysis_results: Available analysis results
            
        Returns:
            Fallback response
        """
        return {
            'query_analysis': {
                'blip_caption': analysis_results.get('blip_caption', 'Fashion item'),
                'garment_type': 'unknown',
                'confidence': 0.0,
                'error': 'Phase 2 models unavailable, using fallback'
            },
            'recommendations': [],
            'phase2_features': {
                'enhanced_encoder_used': False,
                'blip_clip_fusion_used': False,
                'adaptive_reranking_used': False,
                'personalization_used': False
            },
            'performance_stats': self.get_performance_stats(),
            'timestamp': datetime.now().isoformat()
        }
    
    async def add_user_feedback(self,
                              user_id: str,
                              item_id: str,
                              feedback_type: str,
                              feedback_value: float,
                              item_embedding: Optional[np.ndarray] = None,
                              context: Optional[Dict[str, Any]] = None,
                              item_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add user feedback for Phase 2 learning
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            feedback_type: Type of feedback ('like', 'dislike', 'purchase', etc.)
            feedback_value: Numerical feedback value
            item_embedding: Item embedding (optional)
            context: Feedback context
            item_metadata: Item metadata
            
        Returns:
            Feedback processing result
        """
        self.phase2_stats['user_feedback_count'] += 1
        
        results = {
            'status': 'success',
            'feedback_processed': [],
            'errors': []
        }
        
        # Update adaptive reranker
        if self.adaptive_reranker:
            try:
                self.adaptive_reranker.update_from_feedback(
                    user_id=user_id,
                    item_id=item_id,
                    feedback_type=feedback_type,
                    feedback_value=feedback_value,
                    context=context
                )
                results['feedback_processed'].append('adaptive_reranker')
            except Exception as e:
                results['errors'].append(f"Adaptive reranker update failed: {e}")
        
        # Update personalization engine
        if self.personalization_engine and item_embedding is not None:
            try:
                # Create personalization context if available
                personalization_context = None
                if PersonalizationContext and context:
                    personalization_context = PersonalizationContext(
                        user_id=user_id,
                        current_season=context.get('season', 'spring'),
                        time_of_day=context.get('time_of_day', '12:00'),
                        occasion=context.get('occasion', 'casual'),
                        budget_constraint=context.get('budget_constraint'),
                        recent_purchases=context.get('recent_purchases', []),
                        browsing_session=context.get('browsing_session', [])
                    )
                
                self.personalization_engine.update_user_from_feedback(
                    user_id=user_id,
                    item_embedding=item_embedding,
                    feedback_type=feedback_type,
                    feedback_value=feedback_value,
                    context=personalization_context,
                    item_metadata=item_metadata
                )
                results['feedback_processed'].append('personalization_engine')
            except Exception as e:
                results['errors'].append(f"Personalization update failed: {e}")
        
        # Update base fusion reranker
        try:
            self.base_fusion_reranker.add_feedback(
                item_id=item_id,
                clip_score=0.0,  # Will be filled by the reranker
                blip_score=0.0,
                fashion_score=0.0,
                user_rating=feedback_value,
                feedback_type=feedback_type
            )
            results['feedback_processed'].append('base_fusion_reranker')
        except Exception as e:
            results['errors'].append(f"Base fusion reranker update failed: {e}")
        
        return results
    
    def _update_performance_stats(self, response_time: float):
        """
        Update performance statistics
        
        Args:
            response_time: Response time in seconds
        """
        # Update average response time
        current_avg = self.phase2_stats['average_response_time']
        total_requests = self.phase2_stats['total_requests']
        
        new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
        self.phase2_stats['average_response_time'] = new_avg
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Performance statistics dictionary
        """
        stats = self.phase2_stats.copy()
        
        # Add model availability
        stats['model_availability'] = {
            'enhanced_encoder': self.enhanced_encoder is not None,
            'blip_clip_fusion': self.blip_clip_fusion is not None,
            'adaptive_reranker': self.adaptive_reranker is not None,
            'personalization_engine': self.personalization_engine is not None
        }
        
        # Add personalization stats if available
        if self.personalization_engine:
            try:
                personalization_stats = self.personalization_engine.get_stats()
                stats['personalization_stats'] = personalization_stats
            except Exception as e:
                logger.warning(f"Failed to get personalization stats: {e}")
        
        return stats


# Global instance
_phase2_service = None

def get_phase2_service() -> Phase2RecommendationService:
    """
    Get or create the global Phase 2 service instance
    
    Returns:
        Phase2RecommendationService instance
    """
    global _phase2_service
    if _phase2_service is None:
        _phase2_service = Phase2RecommendationService()
    return _phase2_service


if __name__ == "__main__":
    # Test Phase 2 service initialization
    service = get_phase2_service()
    
    logger.info("Phase 2 Service Test")
    logger.info(f"Service initialized: {service is not None}")
    logger.info(f"Performance stats: {service.get_performance_stats()}")