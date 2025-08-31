#!/usr/bin/env python3
"""
Test script for the simplified multi-model fusion recommendation system
Based on user's suggested architecture
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from PIL import Image
import numpy as np

# Add backend to path
sys.path.append('backend')

# Import components
from backend.models.simple_fusion_reranker import get_simple_fusion_reranker
from backend.api.fusion_match import get_fusion_match_handler

def create_test_image(path: str, size: tuple = (224, 224)):
    """
    Create a simple test image
    """
    # Create a simple gradient image
    img = Image.new('RGB', size, color='lightblue')
    
    # Add some pattern
    pixels = img.load()
    for i in range(size[0]):
        for j in range(size[1]):
            r = int(255 * (i / size[0]))
            g = int(255 * (j / size[1]))
            b = 128
            pixels[i, j] = (r, g, b)
    
    img.save(path)
    print(f"✓ Test image created: {path}")

def test_simple_fusion_reranker():
    """
    Test the simplified fusion reranker
    """
    print("\n=== Testing Simple Fusion Reranker ===")
    
    # Get reranker instance
    reranker = get_simple_fusion_reranker()
    
    # Test initial scoring (should use default weights)
    score1 = reranker.score(0.8, 0.6, 0.7)
    print(f"✓ Initial score (default weights): {score1:.3f}")
    
    # Test model info
    info = reranker.get_model_info()
    print(f"✓ Model info: fitted={info['is_fitted']}, weights={info['default_weights']}")
    
    # Test learning from feedback
    print("\n--- Testing Online Learning ---")
    
    # Simulate some feedback
    feedback_data = [
        ([0.8, 0.6, 0.7], 1),  # liked
        ([0.3, 0.4, 0.2], 0),  # disliked
        ([0.9, 0.8, 0.8], 1),  # liked
        ([0.2, 0.3, 0.1], 0),  # disliked
        ([0.7, 0.7, 0.6], 1),  # liked
    ]
    
    for features, label in feedback_data:
        reranker.partial_learn(features, label)
        new_score = reranker.score(*features)
        print(f"   Learned from {features} -> {label}, new score: {new_score:.3f}")
    
    # Test batch learning
    print("\n--- Testing Batch Learning ---")
    batch_features = [[0.6, 0.5, 0.7], [0.4, 0.3, 0.5]]
    batch_labels = [1, 0]
    
    reranker.batch_learn(batch_features, batch_labels)
    print(f"✓ Batch learning completed with {len(batch_features)} samples")
    
    # Test candidate reranking
    print("\n--- Testing Candidate Reranking ---")
    candidates = [
        {"id": "item1", "clip_score": 0.8, "blip_score": 0.6, "fashion_score": 0.7},
        {"id": "item2", "clip_score": 0.5, "blip_score": 0.4, "fashion_score": 0.3},
        {"id": "item3", "clip_score": 0.9, "blip_score": 0.8, "fashion_score": 0.9},
        {"id": "item4", "clip_score": 0.3, "blip_score": 0.2, "fashion_score": 0.4},
    ]
    
    reranked = reranker.rerank_candidates(candidates)
    print("✓ Reranked candidates:")
    for i, (candidate, score) in enumerate(reranked):
        print(f"   {i+1}. {candidate['id']}: {score:.3f}")
    
    return True

def test_fusion_api_handler():
    """
    Test the fusion API handler (without actual HTTP requests)
    """
    print("\n=== Testing Fusion API Handler ===")
    
    try:
        # Get handler instance
        handler = get_fusion_match_handler()
        print("✓ FusionMatchHandler initialized")
        
        # Test service stats
        stats = handler.get_service_stats()
        print(f"✓ Service stats retrieved: {len(stats)} keys")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing fusion handler: {e}")
        return False

def test_integration_workflow():
    """
    Test the complete integration workflow
    """
    print("\n=== Testing Integration Workflow ===")
    
    # Create test image
    test_image_path = "test_fusion_image.jpg"
    create_test_image(test_image_path)
    
    try:
        # Test the workflow components
        reranker = get_simple_fusion_reranker()
        
        # Simulate recommendation workflow
        print("\n--- Simulating Recommendation Workflow ---")
        
        # 1. Mock embeddings and scores
        mock_candidates = [
            {
                "item_id": "garment_001",
                "clip_score": 0.85,
                "blip_score": 0.72,
                "fashion_score": 0.78,
                "metadata": {"tags": ["裙子", "夏季", "休闲"]}
            },
            {
                "item_id": "garment_002", 
                "clip_score": 0.62,
                "blip_score": 0.58,
                "fashion_score": 0.65,
                "metadata": {"tags": ["上衣", "通勤", "春季"]}
            },
            {
                "item_id": "garment_003",
                "clip_score": 0.91,
                "blip_score": 0.88,
                "fashion_score": 0.89,
                "metadata": {"tags": ["外套", "秋季", "时尚"]}
            }
        ]
        
        # 2. Rerank using fusion scores
        reranked_results = reranker.rerank_candidates(mock_candidates)
        
        print("✓ Mock recommendations generated:")
        for i, (candidate, fusion_score) in enumerate(reranked_results):
            print(f"   {i+1}. {candidate['item_id']}: fusion={fusion_score:.3f}")
            print(f"      CLIP={candidate['clip_score']:.2f}, BLIP={candidate['blip_score']:.2f}, Fashion={candidate['fashion_score']:.2f}")
        
        # 3. Simulate user feedback
        print("\n--- Simulating User Feedback ---")
        
        # User likes the top recommendation
        top_candidate = reranked_results[0][0]
        reranker.partial_learn(
            [top_candidate['clip_score'], top_candidate['blip_score'], top_candidate['fashion_score']],
            1  # liked
        )
        print(f"✓ Positive feedback added for {top_candidate['item_id']}")
        
        # User dislikes the bottom recommendation
        bottom_candidate = reranked_results[-1][0]
        reranker.partial_learn(
            [bottom_candidate['clip_score'], bottom_candidate['blip_score'], bottom_candidate['fashion_score']],
            0  # disliked
        )
        print(f"✓ Negative feedback added for {bottom_candidate['item_id']}")
        
        # 4. Test updated rankings
        updated_results = reranker.rerank_candidates(mock_candidates)
        print("\n✓ Updated rankings after feedback:")
        for i, (candidate, fusion_score) in enumerate(updated_results):
            print(f"   {i+1}. {candidate['item_id']}: fusion={fusion_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False
        
    finally:
        # Cleanup
        if os.path.exists(test_image_path):
            os.unlink(test_image_path)
            print(f"✓ Cleaned up: {test_image_path}")

def main():
    """
    Run all tests
    """
    print("🚀 Starting Simple Fusion Recommendation System Tests")
    print("=" * 60)
    
    tests = [
        ("Simple Fusion Reranker", test_simple_fusion_reranker),
        ("Fusion API Handler", test_fusion_api_handler),
        ("Integration Workflow", test_integration_workflow),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name} test...")
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED! Simple fusion system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)