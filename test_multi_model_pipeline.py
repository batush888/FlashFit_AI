#!/usr/bin/env python3
"""
Test script for the multi-model recommendation pipeline.
This script tests the integration of CLIP, BLIP, and Fashion encoders with the fusion reranker.
"""

import sys
import os
from pathlib import Path
import asyncio
import json
from datetime import datetime

# Add backend to path
backend_path = str(Path(__file__).parent / 'backend')
sys.path.insert(0, backend_path)

try:
    import services.recommend_service as recommend_service_module
    import models.clip_encoder as clip_encoder_module
    import models.blip_captioner as blip_captioner_module
    import models.fashion_encoder as fashion_encoder_module
    import models.vector_store as vector_store_module
    import models.fusion_reranker as fusion_reranker_module
    
    get_recommendation_service = recommend_service_module.get_recommendation_service
    get_clip_encoder = clip_encoder_module.get_clip_encoder
    get_blip_captioner = blip_captioner_module.get_blip_captioner
    get_fashion_encoder = fashion_encoder_module.get_fashion_encoder
    get_clip_store = vector_store_module.get_clip_store
    get_blip_store = vector_store_module.get_blip_store
    get_fashion_store = vector_store_module.get_fashion_store
    get_fusion_reranker = fusion_reranker_module.get_fusion_reranker
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed and the backend path is correct.")
    print(f"Backend path: {backend_path}")
    sys.exit(1)

def test_individual_components():
    """
    Test individual components to ensure they initialize correctly
    """
    print("\n=== Testing Individual Components ===")
    
    try:
        print("1. Testing CLIP Encoder...")
        clip_encoder = get_clip_encoder()
        print(f"   ✓ CLIP Encoder initialized on device: {clip_encoder.device}")
        
        print("2. Testing BLIP Captioner...")
        blip_captioner = get_blip_captioner()
        print(f"   ✓ BLIP Captioner initialized on device: {blip_captioner.device}")
        
        print("3. Testing Fashion Encoder...")
        fashion_encoder = get_fashion_encoder()
        print(f"   ✓ Fashion Encoder initialized")
        
        print("4. Testing Vector Stores...")
        clip_store = get_clip_store(dim=512)
        blip_store = get_blip_store(dim=512)
        fashion_store = get_fashion_store(dim=512)
        print(f"   ✓ Vector stores initialized")
        print(f"     - CLIP store: {clip_store.get_stats()['total_vectors']} vectors")
        print(f"     - BLIP store: {blip_store.get_stats()['total_vectors']} vectors")
        print(f"     - Fashion store: {fashion_store.get_stats()['total_vectors']} vectors")
        
        print("5. Testing Fusion Reranker...")
        fusion_reranker = get_fusion_reranker()
        print(f"   ✓ Fusion Reranker initialized")
        print(f"     - Weights: {fusion_reranker.weights.to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Component test failed: {e}")
        return False

def test_recommendation_service():
    """
    Test the recommendation service initialization
    """
    print("\n=== Testing Recommendation Service ===")
    
    try:
        print("Initializing RecommendationService...")
        service = get_recommendation_service()
        print("   ✓ RecommendationService initialized successfully")
        
        # Get service stats
        stats = service.get_service_stats()
        print("   ✓ Service stats retrieved:")
        print(f"     - Service status: {stats['service_status']}")
        print(f"     - Vector stores: {len(stats['vector_stores'])} stores")
        
        return True, service
        
    except Exception as e:
        print(f"   ✗ RecommendationService test failed: {e}")
        return False, None

def create_test_image():
    """
    Create a simple test image for testing purposes
    """
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple pattern
        draw.rectangle([50, 50, 174, 174], fill='darkblue', outline='navy', width=3)
        draw.text((80, 100), "TEST\nSHIRT", fill='white')
        
        # Save test image
        test_image_path = Path("test_garment.jpg")
        img.save(test_image_path)
        
        print(f"   ✓ Test image created: {test_image_path}")
        return str(test_image_path)
        
    except Exception as e:
        print(f"   ✗ Failed to create test image: {e}")
        return None

async def test_recommendation_pipeline(service, test_image_path):
    """
    Test the complete recommendation pipeline
    """
    print("\n=== Testing Recommendation Pipeline ===")
    
    try:
        print(f"Testing with image: {test_image_path}")
        
        # Test recommendation generation
        print("Generating recommendations...")
        result = await service.generate_recommendations(
            query_image_path=test_image_path,
            occasion="casual",
            top_k=5
        )
        
        print("   ✓ Recommendations generated successfully")
        
        # Analyze results
        query_analysis = result.get('query_analysis', {})
        recommendations = result.get('recommendations', [])
        chinese_advice = result.get('chinese_advice', {})
        fusion_stats = result.get('fusion_stats', {})
        
        print(f"\n   Query Analysis:")
        print(f"     - BLIP caption: {query_analysis.get('blip_caption', 'N/A')}")
        print(f"     - Garment type: {query_analysis.get('garment_type', 'N/A')}")
        print(f"     - Confidence: {query_analysis.get('garment_confidence', 0):.2f}")
        
        print(f"\n   Recommendations: {len(recommendations)} items")
        for i, rec in enumerate(recommendations[:3]):
            print(f"     {i+1}. Score: {rec['similarity_score']:.3f}")
            print(f"        CLIP: {rec['component_scores']['clip']:.3f}, "
                  f"BLIP: {rec['component_scores']['blip']:.3f}, "
                  f"Fashion: {rec['component_scores']['fashion']:.3f}")
        
        print(f"\n   Chinese Advice:")
        print(f"     - Title: {chinese_advice.get('title_cn', 'N/A')}")
        print(f"     - Tips: {len(chinese_advice.get('tips_cn', []))} tips")
        print(f"     - Occasion advice: {chinese_advice.get('occasion_advice', 'N/A')[:50]}...")
        
        print(f"\n   Fusion Stats:")
        print(f"     - Candidates processed: {fusion_stats.get('processed_candidates', 0)}")
        print(f"     - Final recommendations: {fusion_stats.get('final_recommendations', 0)}")
        
        return True, result
        
    except Exception as e:
        print(f"   ✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_feedback_system(service):
    """
    Test the feedback system for online learning
    """
    print("\n=== Testing Feedback System ===")
    
    try:
        print("Adding test feedback...")
        
        # Add some test feedback
        service.add_user_feedback(
            item_id="test_item_1",
            clip_score=0.8,
            blip_score=0.6,
            fashion_score=0.7,
            user_rating=0.9,
            feedback_type="like"
        )
        
        service.add_user_feedback(
            item_id="test_item_2",
            clip_score=0.5,
            blip_score=0.4,
            fashion_score=0.6,
            user_rating=0.2,
            feedback_type="dislike"
        )
        
        print("   ✓ Feedback added successfully")
        
        # Get updated stats
        stats = service.fusion_reranker.get_performance_stats()
        print(f"   ✓ Performance stats: {stats.get('total_feedback', 0)} feedback items")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Feedback test failed: {e}")
        return False

def cleanup_test_files():
    """
    Clean up test files
    """
    test_files = ["test_garment.jpg"]
    
    for file_path in test_files:
        try:
            if Path(file_path).exists():
                Path(file_path).unlink()
                print(f"   ✓ Cleaned up: {file_path}")
        except Exception as e:
            print(f"   ✗ Failed to clean up {file_path}: {e}")

async def main():
    """
    Main test function
    """
    print("Multi-Model Recommendation Pipeline Test")
    print("=" * 50)
    
    start_time = datetime.now()
    
    # Test 1: Individual components
    components_ok = test_individual_components()
    if not components_ok:
        print("\n❌ Component tests failed. Stopping.")
        return
    
    # Test 2: Recommendation service
    service_ok, service = test_recommendation_service()
    if not service_ok:
        print("\n❌ Service tests failed. Stopping.")
        return
    
    # Test 3: Create test image
    print("\n=== Creating Test Image ===")
    test_image_path = create_test_image()
    if not test_image_path:
        print("\n❌ Test image creation failed. Stopping.")
        return
    
    # Test 4: Recommendation pipeline
    pipeline_ok, result = await test_recommendation_pipeline(service, test_image_path)
    if not pipeline_ok:
        print("\n❌ Pipeline tests failed.")
    
    # Test 5: Feedback system
    feedback_ok = test_feedback_system(service)
    if not feedback_ok:
        print("\n❌ Feedback tests failed.")
    
    # Cleanup
    print("\n=== Cleanup ===")
    cleanup_test_files()
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"✓ Components: {'PASS' if components_ok else 'FAIL'}")
    print(f"✓ Service: {'PASS' if service_ok else 'FAIL'}")
    print(f"✓ Pipeline: {'PASS' if pipeline_ok else 'FAIL'}")
    print(f"✓ Feedback: {'PASS' if feedback_ok else 'FAIL'}")
    print(f"\nTotal test time: {duration:.2f} seconds")
    
    all_passed = components_ok and service_ok and pipeline_ok and feedback_ok
    
    if all_passed:
        print("\n🎉 All tests PASSED! Multi-model pipeline is working correctly.")
    else:
        print("\n⚠️  Some tests FAILED. Check the output above for details.")
    
    # Save test results
    if pipeline_ok and result:
        try:
            with open("test_results.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n📄 Test results saved to: test_results.json")
        except Exception as e:
            print(f"\n⚠️  Failed to save test results: {e}")

if __name__ == "__main__":
    asyncio.run(main())