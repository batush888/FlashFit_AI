#!/usr/bin/env python3
"""
Phase 2 Integration Tests

Comprehensive test suite for Phase 2 FlashFit AI implementation:
1. Enhanced Fashion Encoder integration
2. BLIP+CLIP fusion functionality
3. Adaptive Fusion Reranker with meta-learning
4. Personalization Layer with user embeddings
5. End-to-end API integration
6. Performance and reliability testing
"""

import pytest
import asyncio
import numpy as np
import tempfile
import os
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from PIL import Image
import io

# Import the application and services
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app import app, get_current_user
from services.phase2_service import Phase2RecommendationService, get_phase2_service

class TestPhase2Integration:
    """
    Integration tests for Phase 2 FlashFit AI system
    """
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user_token(self):
        """Mock JWT token for authentication"""
        return "Bearer mock_jwt_token_for_testing"
    
    @pytest.fixture
    def sample_image_file(self):
        """Create a sample image file for testing"""
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    @pytest.fixture
    def phase2_service(self):
        """Phase 2 service instance for testing"""
        return Phase2RecommendationService(
            enable_enhanced_encoder=True,
            enable_blip_clip_fusion=True,
            enable_adaptive_reranker=True,
            enable_personalization=True
        )
    
    def test_phase2_service_initialization(self, phase2_service):
        """Test Phase 2 service initializes correctly"""
        assert phase2_service is not None
        assert hasattr(phase2_service, 'clip_encoder')
        assert hasattr(phase2_service, 'blip_captioner')
        assert hasattr(phase2_service, 'fashion_encoder')
        assert hasattr(phase2_service, 'base_fusion_reranker')
        
        # Check performance stats initialization
        stats = phase2_service.get_performance_stats()
        assert 'total_requests' in stats
        assert 'model_availability' in stats
        assert stats['total_requests'] == 0
    
    @pytest.mark.asyncio
    async def test_image_analysis_pipeline(self, phase2_service):
        """Test the complete image analysis pipeline"""
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            img = Image.new('RGB', (224, 224), color='blue')
            img.save(tmp_file.name, 'JPEG')
            
            try:
                # Test image analysis
                analysis_results = await phase2_service._analyze_query_image(tmp_file.name)
                
                # Verify analysis results structure
                assert 'clip_embedding' in analysis_results
                assert 'enhanced_embedding' in analysis_results
                assert 'blip_caption' in analysis_results
                assert 'garment_classification' in analysis_results
                
                # Verify embedding dimensions
                clip_embedding = analysis_results['clip_embedding']
                assert isinstance(clip_embedding, np.ndarray)
                # CLIP embedding can be (1, 512) or (512,) depending on implementation
                assert clip_embedding.shape in [(512,), (1, 512)]
                
                # Verify caption is generated
                caption = analysis_results['blip_caption']
                assert isinstance(caption, str)
                assert len(caption) > 0
                
            finally:
                os.unlink(tmp_file.name)
    
    @pytest.mark.asyncio
    async def test_candidate_generation(self, phase2_service):
        """Test candidate generation from multiple models"""
        # Mock analysis results
        analysis_results = {
            'clip_embedding': np.random.rand(512),
            'enhanced_embedding': np.random.rand(512),
            'fusion_embedding': np.random.rand(512),
            'blip_caption': 'A blue shirt',
            'garment_classification': {'top_category': 'shirt', 'top_score': 0.9}
        }
        
        # Mock vector store search results
        mock_candidates = [
            ({'item_id': f'item_{i}', 'category': 'shirt'}, 0.8 - i * 0.1)
            for i in range(5)
        ]
        
        with patch.object(phase2_service.clip_store, 'search', return_value=mock_candidates):
            with patch.object(phase2_service.fashion_store, 'search', return_value=mock_candidates):
                with patch.object(phase2_service.blip_store, 'search', return_value=mock_candidates):
                    
                    candidates = await phase2_service._generate_candidates(analysis_results, top_k=10)
                    
                    # Verify candidates structure
                    assert isinstance(candidates, list)
                    assert len(candidates) > 0
                    
                    for candidate in candidates:
                        assert 'item_id' in candidate
                        assert 'combined_score' in candidate
                        assert 'clip_score' in candidate
                        assert 'blip_score' in candidate
                        assert 'fashion_score' in candidate
                        assert 'enhanced_score' in candidate
    
    @pytest.mark.asyncio
    async def test_adaptive_reranking(self, phase2_service):
        """Test adaptive reranking functionality"""
        # Mock candidates
        candidates = [
            {
                'item_id': f'item_{i}',
                'combined_score': 0.8 - i * 0.1,
                'clip_score': 0.7,
                'blip_score': 0.6,
                'fashion_score': 0.8,
                'enhanced_score': 0.9
            }
            for i in range(5)
        ]
        
        analysis_results = {
            'enhanced_embedding': np.random.rand(512),
            'clip_embedding': np.random.rand(512)
        }
        
        # Test reranking
        reranked = await phase2_service._apply_phase2_reranking(
            candidates, analysis_results, user_id='test_user'
        )
        
        # Verify reranking results
        assert isinstance(reranked, list)
        assert len(reranked) == len(candidates)
        
        for candidate in reranked:
            # Should have either final_score or combined_score depending on reranker availability
            assert 'final_score' in candidate or 'combined_score' in candidate
            # Score should be non-negative
            score = candidate.get('final_score', candidate.get('combined_score', 0))
            assert score >= 0
    
    @pytest.mark.asyncio
    async def test_personalization_application(self, phase2_service):
        """Test personalization layer application"""
        # Mock candidates with scores
        candidates = [
            {
                'item_id': f'item_{i}',
                'final_score': 0.8 - i * 0.1,
                'metadata': {
                    'embedding': np.random.rand(512),
                    'style': 'casual',
                    'color': 'blue',
                    'brand': 'test_brand',
                    'price': 50.0
                }
            }
            for i in range(5)
        ]
        
        context = {
            'season': 'spring',
            'occasion': 'casual',
            'time_of_day': '14:00'
        }
        
        # Test personalization
        personalized = await phase2_service._apply_personalization(
            candidates, user_id='test_user', context=context, top_k=3
        )
        
        # Verify personalization results
        assert isinstance(personalized, list)
        assert len(personalized) <= 3  # Should respect top_k limit
        
        # Should return candidates even if personalization engine is not available
        assert len(personalized) > 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_recommendations(self, phase2_service):
        """Test complete end-to-end recommendation pipeline"""
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            img = Image.new('RGB', (224, 224), color='green')
            img.save(tmp_file.name, 'JPEG')
            
            try:
                # Mock vector store searches to return some results
                mock_candidates = [
                    ({'item_id': f'item_{i}', 'category': 'shirt', 'embedding': np.random.rand(512)}, 0.8 - i * 0.1)
                    for i in range(3)
                ]
                
                with patch.object(phase2_service.clip_store, 'search', return_value=mock_candidates):
                    with patch.object(phase2_service.fashion_store, 'search', return_value=mock_candidates):
                        with patch.object(phase2_service.blip_store, 'search', return_value=mock_candidates):
                            
                            # Generate recommendations
                            result = await phase2_service.generate_recommendations(
                                query_image_path=tmp_file.name,
                                user_id='test_user',
                                context={'season': 'spring', 'occasion': 'casual'},
                                top_k=5
                            )
                            
                            # Verify response structure
                            assert isinstance(result, dict)
                            assert 'query_analysis' in result
                            assert 'recommendations' in result
                            assert 'phase2_features' in result
                            assert 'performance_stats' in result
                            assert 'timestamp' in result
                            
                            # Verify query analysis
                            query_analysis = result['query_analysis']
                            assert 'blip_caption' in query_analysis
                            assert 'garment_type' in query_analysis
                            assert 'confidence' in query_analysis
                            
                            # Verify Phase 2 features tracking
                            phase2_features = result['phase2_features']
                            assert 'enhanced_encoder_used' in phase2_features
                            assert 'blip_clip_fusion_used' in phase2_features
                            assert 'adaptive_reranking_used' in phase2_features
                            assert 'personalization_used' in phase2_features
                            
            finally:
                os.unlink(tmp_file.name)
    
    @pytest.mark.asyncio
    async def test_feedback_processing(self, phase2_service):
        """Test user feedback processing for learning systems"""
        # Test feedback submission
        result = await phase2_service.add_user_feedback(
            user_id='test_user',
            item_id='test_item_123',
            feedback_type='like',
            feedback_value=0.9,
            item_embedding=np.random.rand(512),
            context={'season': 'spring', 'occasion': 'casual'},
            item_metadata={'category': 'shirt', 'color': 'blue'}
        )
        
        # Verify feedback processing result
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'feedback_processed' in result
        assert 'errors' in result
        
        # Should process base fusion reranker feedback at minimum
        assert 'base_fusion_reranker' in result['feedback_processed']
        
        # Update performance stats
        stats = phase2_service.get_performance_stats()
        assert stats['user_feedback_count'] > 0
    
    def test_performance_stats_tracking(self, phase2_service):
        """Test performance statistics tracking"""
        # Get initial stats
        initial_stats = phase2_service.get_performance_stats()
        
        # Verify stats structure
        expected_keys = [
            'total_requests', 'enhanced_encoder_requests', 'fusion_requests',
            'adaptive_reranker_requests', 'personalized_requests', 'fallback_requests',
            'average_response_time', 'user_feedback_count', 'model_availability'
        ]
        
        for key in expected_keys:
            assert key in initial_stats
        
        # Verify model availability tracking
        model_availability = initial_stats['model_availability']
        assert 'enhanced_encoder' in model_availability
        assert 'blip_clip_fusion' in model_availability
        assert 'adaptive_reranker' in model_availability
        assert 'personalization_engine' in model_availability
    
    def test_phase2_api_health_endpoint(self, client):
        """Test Phase 2 health check API endpoint"""
        response = client.get('/api/phase2/health')
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'status' in data
        assert 'models_available' in data
        assert 'total_requests' in data
        assert 'average_response_time' in data
        assert 'timestamp' in data
    
    def test_phase2_api_stats_endpoint(self, client):
        """Test Phase 2 statistics API endpoint"""
        # Override the dependency
        app.dependency_overrides[get_current_user] = lambda: "test_user"
        try:
            response = client.get(
                '/api/phase2/stats',
                headers={'Authorization': 'Bearer test_token'}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Should return performance stats
            assert 'total_requests' in data
            assert 'model_availability' in data
        finally:
            app.dependency_overrides.clear()
    
    def test_phase2_api_match_endpoint(self, client, sample_image_file):
        """Test Phase 2 match API endpoint"""
        # Override the dependency
        app.dependency_overrides[get_current_user] = lambda: "test_user"
        try:
            # Mock the phase2_service to avoid actual model calls
            with patch('backend.app.phase2_service') as mock_service:
                mock_service.generate_recommendations = AsyncMock(return_value={
                    'query_analysis': {'blip_caption': 'test shirt', 'garment_type': 'shirt'},
                    'recommendations': [],
                    'phase2_features': {'enhanced_encoder_used': True},
                    'performance_stats': {'total_requests': 1},
                    'timestamp': '2024-01-01T00:00:00'
                })
                
                response = client.post(
                    '/api/phase2/match',
                    files={'file': ('test.jpg', sample_image_file, 'image/jpeg')},
                    data={'top_k': '5'},
                    headers={'Authorization': 'Bearer test_token'}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert 'query_analysis' in data
                assert 'recommendations' in data
                assert 'phase2_features' in data
        finally:
            app.dependency_overrides.clear()
    
    def test_phase2_api_feedback_endpoint(self, client):
        """Test Phase 2 feedback API endpoint"""
        # Override the dependency
        app.dependency_overrides[get_current_user] = lambda: "test_user"
        try:
            # Mock the phase2_service feedback processing
            with patch('backend.app.phase2_service') as mock_service:
                mock_service.add_user_feedback = AsyncMock(return_value={
                    'status': 'success',
                    'feedback_processed': ['base_fusion_reranker'],
                    'errors': []
                })
                
                feedback_data = {
                    'item_id': 'test_item_123',
                    'feedback_type': 'like',
                    'feedback_value': 0.9,
                    'context': {'season': 'spring'},
                    'item_metadata': {'category': 'shirt'}
                }
                
                response = client.post(
                    '/api/phase2/feedback',
                    json=feedback_data,
                    headers={'Authorization': 'Bearer test_token'}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert data['status'] == 'success'
                assert 'feedback_processed' in data
                assert 'errors' in data
        finally:
            app.dependency_overrides.clear()
    
    def test_fallback_behavior(self, phase2_service):
        """Test system behavior when Phase 2 models are unavailable"""
        # Force disable all Phase 2 models
        phase2_service.enhanced_encoder = None
        phase2_service.blip_clip_fusion = None
        phase2_service.adaptive_reranker = None
        phase2_service.personalization_engine = None
        
        # Test fallback response generation
        fallback_response = phase2_service._generate_fallback_response(
            'test_image.jpg', 
            {'blip_caption': 'test caption'}
        )
        
        assert isinstance(fallback_response, dict)
        assert 'query_analysis' in fallback_response
        assert 'recommendations' in fallback_response
        assert 'phase2_features' in fallback_response
        
        # Verify all Phase 2 features are marked as unused
        phase2_features = fallback_response['phase2_features']
        assert not phase2_features['enhanced_encoder_used']
        assert not phase2_features['blip_clip_fusion_used']
        assert not phase2_features['adaptive_reranking_used']
        assert not phase2_features['personalization_used']
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, phase2_service):
        """Test system behavior under concurrent requests"""
        # Create multiple concurrent recommendation requests
        async def make_request(request_id):
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                img = Image.new('RGB', (224, 224), color='red')
                img.save(tmp_file.name, 'JPEG')
                
                try:
                    # Mock vector stores to avoid actual searches
                    mock_candidates = [
                        ({'item_id': f'item_{request_id}_{i}', 'embedding': np.random.rand(512)}, 0.8)
                        for i in range(2)
                    ]
                    
                    with patch.object(phase2_service.clip_store, 'search', return_value=mock_candidates):
                        with patch.object(phase2_service.fashion_store, 'search', return_value=mock_candidates):
                            with patch.object(phase2_service.blip_store, 'search', return_value=mock_candidates):
                                
                                result = await phase2_service.generate_recommendations(
                                    query_image_path=tmp_file.name,
                                    user_id=f'user_{request_id}',
                                    top_k=3
                                )
                                
                                return result
                finally:
                    os.unlink(tmp_file.name)
        
        # Run 5 concurrent requests
        tasks = [make_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all requests completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 5
        
        # Verify stats were updated correctly
        stats = phase2_service.get_performance_stats()
        assert stats['total_requests'] >= 5


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])