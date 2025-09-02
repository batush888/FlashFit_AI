import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import uuid

# Import our multi-model components
from models.clip_encoder import get_clip_encoder
from models.blip_captioner import get_blip_captioner
from models.fashion_encoder import get_fashion_encoder
from models.vector_store import get_clip_store, get_blip_store, get_fashion_store
from models.fusion_reranker import get_fusion_reranker, FusionScore

class RecommendationService:
    """
    Multi-model fusion recommendation service that combines CLIP, BLIP, and Fashion encoders
    for comprehensive fashion recommendations with Chinese advice generation.
    """
    
    def __init__(self):
        """
        Initialize the recommendation service with all required models
        """
        # Initialize encoders
        self.clip_encoder = get_clip_encoder()
        self.blip_captioner = get_blip_captioner()
        self.fashion_encoder = get_fashion_encoder()
        
        # Initialize vector stores
        self.clip_store = get_clip_store(dim=512)  # CLIP embedding dimension
        self.blip_store = get_blip_store(dim=512)  # BLIP text embedding dimension (updated to match projection)
        self.fashion_store = get_fashion_store(dim=512)  # Fashion encoder dimension
        
        # Initialize fusion reranker
        self.fusion_reranker = get_fusion_reranker(enable_online_learning=True)
        
        print("RecommendationService initialized with multi-model fusion")
    
    async def generate_recommendations(self, 
                                    query_image_path: str,
                                    user_wardrobe: Optional[List[Dict[str, Any]]] = None,
                                    occasion: Optional[str] = None,
                                    top_k: int = 10,
                                    user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive fashion recommendations using multi-model fusion
        
        Args:
            query_image_path: Path to the query garment image
            user_wardrobe: User's wardrobe items (optional)
            occasion: Target occasion (optional)
            top_k: Number of recommendations to return
            user_preferences: User style preferences (optional)
            
        Returns:
            Dictionary with recommendations and Chinese advice
        """
        try:
            # Step A: Per-item understanding
            print(f"Analyzing query image: {query_image_path}")
            
            # 1. CLIP image embedding (fast, general)
            clip_embedding = self.clip_encoder.embed_image(query_image_path)
            
            # 2. BLIP caption & attributes (text summary)
            blip_caption = self.blip_captioner.caption(query_image_path)
            blip_description = self.blip_captioner.generate_fashion_description(query_image_path)
            blip_text_embedding = self.blip_captioner.get_text_embedding(blip_caption)
            
            # 3. Fashion-specific embedding
            fashion_embedding = self.fashion_encoder.embed_fashion_image(query_image_path)
            fashion_attributes = self.fashion_encoder.analyze_fashion_attributes(query_image_path)
            garment_classification = self.fashion_encoder.classify_garment_type(query_image_path)
            
            print(f"Query analysis complete:")
            print(f"- BLIP caption: {blip_caption}")
            print(f"- Garment type: {garment_classification['top_category']} (confidence: {garment_classification['top_score']:.2f})")
            
            # Step B: Candidate generation using CLIP embedding
            print("Generating candidates from vector stores...")
            
            clip_candidates = self.clip_store.search(clip_embedding.reshape(1, -1), topk=50)
            
            if not clip_candidates:
                print("No candidates found in vector store. Using fallback recommendations.")
                return self._generate_fallback_recommendations(query_image_path, blip_caption, garment_classification)
            
            # Step C: Cross-model re-ranking (fusion)
            print("Computing fusion scores for candidates...")
            
            fusion_candidates = []
            
            for candidate_meta, clip_score in clip_candidates:
                try:
                    candidate_id = candidate_meta.get('item_id', '')
                    candidate_image_path = candidate_meta.get('image_path', '')
                    
                    if not candidate_image_path or not Path(candidate_image_path).exists():
                        continue
                    
                    # Compute BLIP similarity
                    candidate_caption = self.blip_captioner.caption(candidate_image_path)
                    candidate_blip_embedding = self.blip_captioner.get_text_embedding(candidate_caption)
                    blip_score = np.dot(blip_text_embedding.flatten(), candidate_blip_embedding.flatten()) / (np.linalg.norm(blip_text_embedding) * np.linalg.norm(candidate_blip_embedding))
                    
                    # Compute Fashion similarity
                    candidate_fashion_embedding = self.fashion_encoder.embed_fashion_image(candidate_image_path)
                    fashion_score = self.fashion_encoder.compute_similarity(fashion_embedding, candidate_fashion_embedding)
                    
                    # Create candidate for fusion scoring
                    fusion_candidate = {
                        'item_id': candidate_id,
                        'clip_score': float(clip_score),
                        'blip_score': float(blip_score),
                        'fashion_score': float(fashion_score),
                        'metadata': {
                            **candidate_meta,
                            'candidate_caption': candidate_caption,
                            'image_path': candidate_image_path
                        }
                    }
                    
                    fusion_candidates.append(fusion_candidate)
                    
                except Exception as e:
                    print(f"Error processing candidate {candidate_meta.get('item_id', 'unknown')}: {e}")
                    continue
            
            # Rerank using fusion scores
            fusion_scores = self.fusion_reranker.rerank_candidates(fusion_candidates)
            
            # Take top-k results
            top_recommendations = fusion_scores[:top_k]
            
            # Step D: Output & Chinese advice
            print("Generating Chinese advice and final recommendations...")
            
            recommendations = []
            for i, fusion_score in enumerate(top_recommendations):
                recommendation = {
                    "id": f"rec_{uuid.uuid4().hex[:8]}",
                    "rank": i + 1,
                    "item_id": fusion_score.item_id,
                    "image_path": fusion_score.metadata.get('image_path', ''),
                    "similarity_score": fusion_score.final_score,
                    "component_scores": {
                        "clip": fusion_score.clip_score,
                        "blip": fusion_score.blip_score,
                        "fashion": fusion_score.fashion_score
                    },
                    "candidate_description": fusion_score.metadata.get('candidate_caption', ''),
                    "metadata": fusion_score.metadata
                }
                recommendations.append(recommendation)
            
            # Generate Chinese advice
            chinese_advice = self._generate_chinese_advice(
                query_caption=blip_caption,
                query_classification=garment_classification,
                recommendations=recommendations,
                occasion=occasion,
                user_preferences=user_preferences
            )
            
            # Compile final result
            result = {
                "query_analysis": {
                    "image_path": query_image_path,
                    "blip_caption": blip_caption,
                    "blip_description": blip_description,
                    "garment_type": garment_classification['top_category'],
                    "garment_confidence": garment_classification['top_score'],
                    "fashion_attributes": fashion_attributes
                },
                "recommendations": recommendations,
                "chinese_advice": chinese_advice,
                "fusion_stats": {
                    "total_candidates": len(clip_candidates),
                    "processed_candidates": len(fusion_candidates),
                    "final_recommendations": len(recommendations),
                    "fusion_weights": self.fusion_reranker.weights.to_dict()
                },
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"Generated {len(recommendations)} recommendations successfully")
            return result
            
        except Exception as e:
            print(f"Error in generate_recommendations: {e}")
            return self._generate_fallback_recommendations(query_image_path, "", {})
    
    def _generate_chinese_advice(self, 
                               query_caption: str,
                               query_classification: Dict[str, Any],
                               recommendations: List[Dict[str, Any]],
                               occasion: Optional[str] = None,
                               user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate Chinese fashion advice based on analysis results
        
        Args:
            query_caption: BLIP caption of query item
            query_classification: Garment classification results
            recommendations: List of recommendations
            occasion: Target occasion
            user_preferences: User preferences
            
        Returns:
            Dictionary with Chinese advice
        """
        garment_type = query_classification.get('top_category', '服装')
        
        # Base advice templates
        advice_templates = {
            'shirt': {
                'title': '衬衫搭配建议',
                'tips': [
                    '选择合身的剪裁，避免过于宽松或紧身',
                    '可以搭配西装裤营造正式感，或配牛仔裤展现休闲风',
                    '注意颜色协调，白色和浅蓝色是经典选择',
                    '可以通过配饰如领带、胸针来增加亮点'
                ]
            },
            'dress': {
                'title': '连衣裙搭配建议',
                'tips': [
                    '根据场合选择合适的长度和款式',
                    '可以用腰带强调腰线，展现身材比例',
                    '选择合适的鞋子，高跟鞋显优雅，平底鞋更舒适',
                    '配饰要与裙子风格协调，避免过于繁复'
                ]
            },
            'pants': {
                'title': '裤装搭配建议',
                'tips': [
                    '选择合适的腰线高度，高腰显腿长',
                    '注意裤长，避免拖地或过短',
                    '可以搭配不同风格的上衣创造层次感',
                    '鞋子的选择很重要，影响整体比例'
                ]
            }
        }
        
        # Get advice template or use default
        advice = advice_templates.get(garment_type, {
            'title': '时尚搭配建议',
            'tips': [
                '注重整体色彩协调，避免过多撞色',
                '选择合身的剪裁，突出身材优势',
                '根据场合选择合适的风格',
                '适当使用配饰提升整体造型'
            ]
        })
        
        # Customize advice based on occasion
        occasion_advice = {
            'work': '工作场合建议选择简洁大方的款式，颜色以中性色为主，避免过于花哨的设计。',
            'casual': '休闲场合可以更加随性，尝试不同的色彩搭配和有趣的图案。',
            'formal': '正式场合需要注重细节，选择质感好的面料，整体造型要端庄得体。',
            'party': '聚会场合可以大胆一些，选择有设计感的单品，适当加入亮色或金属元素。'
        }
        
        # Generate style recommendations based on top matches
        style_suggestions = []
        if recommendations:
            top_3 = recommendations[:3]
            for i, rec in enumerate(top_3):
                suggestion = f"推荐搭配{i+1}：{rec.get('candidate_description', '时尚单品')}，相似度{rec['similarity_score']:.1%}"
                style_suggestions.append(suggestion)
        
        return {
            'title_cn': advice['title'],
            'tips_cn': advice['tips'],
            'occasion_advice': occasion_advice.get(occasion, '根据个人喜好和场合需求进行搭配。'),
            'style_suggestions': style_suggestions,
            'color_advice': self._generate_color_advice(query_caption),
            'season_advice': self._generate_season_advice(),
            'confidence_note': f"基于{garment_type}的AI分析，置信度{query_classification.get('top_score', 0.8):.1%}"
        }
    
    def _generate_color_advice(self, caption: str) -> str:
        """
        Generate color matching advice based on item description
        """
        color_keywords = {
            'white': '白色是百搭色，可以与任何颜色搭配',
            'black': '黑色经典优雅，适合正式场合',
            'blue': '蓝色清新自然，适合日常穿搭',
            'red': '红色热情醒目，适合作为点缀色使用',
            'green': '绿色清新活力，适合春夏季节',
            'yellow': '黄色明亮温暖，可以提升整体活力'
        }
        
        if caption:
            for color, advice in color_keywords.items():
                if color in caption.lower():
                    return advice
        
        return '建议选择与肤色相配的颜色，营造和谐的整体效果。'
    
    def _generate_season_advice(self) -> str:
        """
        Generate seasonal advice
        """
        import datetime
        month = datetime.datetime.now().month
        
        if month in [12, 1, 2]:
            return '冬季建议选择保暖性好的面料，可以通过层次搭配增加温度和时尚感。'
        elif month in [3, 4, 5]:
            return '春季适合清新的色彩，可以选择轻薄的面料，展现春天的活力。'
        elif month in [6, 7, 8]:
            return '夏季建议选择透气性好的面料，浅色系更适合炎热天气。'
        else:
            return '秋季适合温暖的色调，可以尝试叠穿搭配，展现层次感。'
    
    def _generate_fallback_recommendations(self, 
                                        query_image_path: str,
                                        caption: str,
                                        classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate fallback recommendations when vector store is empty
        """
        garment_type = classification.get('top_category', '服装')
        
        fallback_recommendations = [
            {
                "id": f"fallback_{i}",
                "rank": i + 1,
                "item_id": f"template_{i}",
                "image_path": "",
                "similarity_score": 0.7 - i * 0.1,
                "component_scores": {
                    "clip": 0.7 - i * 0.1,
                    "blip": 0.6 - i * 0.1,
                    "fashion": 0.8 - i * 0.1
                },
                "candidate_description": f"推荐的{garment_type}搭配方案{i+1}",
                "metadata": {
                    "type": "fallback",
                    "category": garment_type
                }
            }
            for i in range(3)
        ]
        
        chinese_advice = {
            'title_cn': f'{garment_type}搭配建议',
            'tips_cn': [
                '选择合适的尺寸和剪裁',
                '注意颜色搭配的和谐性',
                '根据场合选择合适的风格',
                '适当使用配饰提升造型'
            ],
            'occasion_advice': '建议根据具体场合和个人喜好进行调整。',
            'style_suggestions': ['暂无具体搭配数据，建议上传更多衣物建立个人衣橱。'],
            'color_advice': '建议选择与个人肤色相配的颜色。',
            'season_advice': self._generate_season_advice(),
            'confidence_note': '当前为基础推荐，建议完善衣橱数据以获得更精准的建议。'
        }
        
        return {
            "query_analysis": {
                "image_path": query_image_path,
                "blip_caption": caption,
                "garment_type": garment_type,
                "garment_confidence": classification.get('top_score', 0.5)
            },
            "recommendations": fallback_recommendations,
            "chinese_advice": chinese_advice,
            "fusion_stats": {
                "total_candidates": 0,
                "processed_candidates": 0,
                "final_recommendations": len(fallback_recommendations),
                "note": "Fallback recommendations - vector store empty"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def add_item_to_stores(self, 
                               item_id: str,
                               image_path: str,
                               metadata: Dict[str, Any]) -> bool:
        """
        Add a new item to all vector stores
        
        Args:
            item_id: Unique item identifier
            image_path: Path to item image
            metadata: Item metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embeddings
            clip_embedding = self.clip_encoder.embed_image(image_path)
            fashion_embedding = self.fashion_encoder.embed_fashion_image(image_path)
            
            # Generate caption and text embedding
            caption = self.blip_captioner.caption(image_path)
            blip_embedding = self.blip_captioner.get_text_embedding(caption)
            
            # Prepare metadata
            item_metadata = {
                **metadata,
                'item_id': item_id,
                'image_path': image_path,
                'caption': caption,
                'added_at': datetime.now().isoformat()
            }
            
            # Add to vector stores
            self.clip_store.add_single(clip_embedding, item_metadata)
            self.blip_store.add_single(blip_embedding, item_metadata)
            self.fashion_store.add_single(fashion_embedding, item_metadata)
            
            # Save stores
            self.clip_store.save()
            self.blip_store.save()
            self.fashion_store.save()
            
            print(f"Added item {item_id} to all vector stores")
            return True
            
        except Exception as e:
            print(f"Error adding item to stores: {e}")
            return False
    
    def add_user_feedback(self, 
                         item_id: str,
                         clip_score: float,
                         blip_score: float,
                         fashion_score: float,
                         user_rating: float,
                         feedback_type: str = "rating"):
        """
        Add user feedback for online learning
        
        Args:
            item_id: ID of the item that received feedback
            clip_score: CLIP score for this item
            blip_score: BLIP score for this item
            fashion_score: Fashion score for this item
            user_rating: User rating (0.0 to 1.0)
            feedback_type: Type of feedback
        """
        self.fusion_reranker.add_feedback(
            item_id=item_id,
            clip_score=clip_score,
            blip_score=blip_score,
            fashion_score=fashion_score,
            user_rating=user_rating,
            feedback_type=feedback_type
        )
    
    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive service statistics
        
        Returns:
            Dictionary with service statistics
        """
        return {
            "vector_stores": {
                "clip": self.clip_store.get_stats(),
                "blip": self.blip_store.get_stats(),
                "fashion": self.fashion_store.get_stats()
            },
            "fusion_reranker": self.fusion_reranker.get_performance_stats(),
            "service_status": "active",
            "timestamp": datetime.now().isoformat()
        }

# Global instance
_recommendation_service = None

def get_recommendation_service() -> RecommendationService:
    """
    Get global recommendation service instance
    
    Returns:
        RecommendationService instance
    """
    global _recommendation_service
    if _recommendation_service is None:
        _recommendation_service = RecommendationService()
    return _recommendation_service