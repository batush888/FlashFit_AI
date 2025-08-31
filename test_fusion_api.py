#!/usr/bin/env python3
"""
Test script for multi-model fusion API endpoints
"""

import requests
import json
import os
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8080"

def test_fusion_stats():
    """Test the fusion stats endpoint"""
    print("\n📊 Testing Fusion Stats Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/api/fusion/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"   ✓ Stats retrieved successfully:")
            print(f"     - Service status: {stats.get('service_status')}")
            print(f"     - CLIP vectors: {stats.get('clip_vectors', 0)}")
            print(f"     - BLIP vectors: {stats.get('blip_vectors', 0)}")
            print(f"     - Fashion vectors: {stats.get('fashion_vectors', 0)}")
            print(f"     - Fusion reranker: {stats.get('fusion_reranker_status')}")
            return True
        else:
            print(f"   ✗ Stats endpoint failed: {response.status_code}")
            print(f"     Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ✗ Stats endpoint error: {e}")
        return False

def test_fusion_match():
    """Test the fusion match endpoint with image upload"""
    print("\n🔍 Testing Fusion Match Endpoint")
    print("=" * 50)
    
    # Test image path
    test_image = "backend/data/static/skirts/blue_a.png"
    
    if not os.path.exists(test_image):
        print(f"   ✗ Test image not found: {test_image}")
        return False
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': ('test_image.png', f, 'image/png')}
            data = {'target_count': 3}
            
            response = requests.post(
                f"{BASE_URL}/api/fusion/match",
                files=files,
                data=data
            )
            
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Match request successful")
            print(f"     - Query caption: {result.get('query_caption', 'N/A')}")
            
            suggestions = result.get('suggestions', [])
            print(f"     - Found {len(suggestions)} suggestions:")
            
            for i, suggestion in enumerate(suggestions[:3]):
                print(f"       {i+1}. ID: {suggestion.get('id', 'N/A')}")
                print(f"          Scores: CLIP={suggestion.get('clip_score', 0):.3f}, "
                      f"BLIP={suggestion.get('blip_score', 0):.3f}, "
                      f"Fashion={suggestion.get('fashion_score', 0):.3f}")
                print(f"          Final: {suggestion.get('final_score', 0):.3f}")
            
            return suggestions
        else:
            print(f"   ✗ Match endpoint failed: {response.status_code}")
            print(f"     Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ✗ Match endpoint error: {e}")
        return False

def test_fusion_feedback(suggestions):
    """Test the fusion feedback endpoint"""
    print("\n👍 Testing Fusion Feedback Endpoint")
    print("=" * 50)
    
    if not suggestions:
        print("   ⚠️  No suggestions available for feedback testing")
        return False
    
    try:
        # Test positive feedback
        suggestion = suggestions[0]
        feedback_data = {
            "suggestion_id": suggestion.get('id', 'test_item'),
            "liked": True,
            "clip_score": suggestion.get('clip_score', 0.8),
            "blip_score": suggestion.get('blip_score', 0.6),
            "fashion_score": suggestion.get('fashion_score', 0.7)
        }
        
        response = requests.post(
            f"{BASE_URL}/api/fusion/feedback",
            json=feedback_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Positive feedback submitted successfully")
            print(f"     - Status: {result.get('status')}")
            print(f"     - Message: {result.get('message')}")
            
            # Test negative feedback
            if len(suggestions) > 1:
                suggestion2 = suggestions[1]
                feedback_data2 = {
                    "suggestion_id": suggestion2.get('id', 'test_item_2'),
                    "liked": False,
                    "clip_score": suggestion2.get('clip_score', 0.5),
                    "blip_score": suggestion2.get('blip_score', 0.4),
                    "fashion_score": suggestion2.get('fashion_score', 0.3)
                }
                
                response2 = requests.post(
                    f"{BASE_URL}/api/fusion/feedback",
                    json=feedback_data2
                )
                
                if response2.status_code == 200:
                    result2 = response2.json()
                    print(f"   ✓ Negative feedback submitted successfully")
                    print(f"     - Status: {result2.get('status')}")
            
            return True
        else:
            print(f"   ✗ Feedback endpoint failed: {response.status_code}")
            print(f"     Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ✗ Feedback endpoint error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 Testing Multi-Model Fusion API Endpoints")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print(f"⚠️  Server health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        print("   Please make sure the FastAPI server is running on port 8080")
        return
    
    # Run tests
    stats_success = test_fusion_stats()
    suggestions = test_fusion_match()
    feedback_success = test_fusion_feedback(suggestions)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 API Testing Complete!")
    print("\n📋 Summary:")
    print(f"   - Stats endpoint: {'✓' if stats_success else '✗'}")
    print(f"   - Match endpoint: {'✓' if suggestions else '✗'}")
    print(f"   - Feedback endpoint: {'✓' if feedback_success else '✗'}")
    
    if stats_success and suggestions and feedback_success:
        print("\n🚀 All fusion API endpoints are working correctly!")
    else:
        print("\n⚠️  Some endpoints need attention.")

if __name__ == "__main__":
    main()