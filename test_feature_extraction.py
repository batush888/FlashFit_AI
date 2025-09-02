#!/usr/bin/env python3
"""
Feature Extraction Verification Test
Tests if AI models are properly converting images to embeddings
"""

import sys
import os
from pathlib import Path

# Add backend directory to Python path
backend_path = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_path))

import torch
from PIL import Image
import numpy as np

# Import our models
from models.clip_encoder import CLIPEncoder
from models.fashion_encoder import FashionEncoder
from models.blip_captioner import BLIPCaptioner

def test_feature_extraction():
    print("🔍 Testing Feature Extraction Pipeline...")
    print("=" * 50)
    
    # Find a test image
    test_image_paths = [
        "backend/data/uploads/user_3_1756552528/user_3_1756552528_d1177182926d4faba6a6da46846dc3b2.jpg",
        "backend/data/uploads/user_4_1756553127/user_4_1756553127_02c47560893a4401b05c1395262f24ec.jpg",
        "backend/data/uploads/user_5_1756715906/user_5_1756715906_2a6cd59f40194899913f407c52efb5bc.webp",
        "test_image.jpg"
    ]
    
    test_image = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image = path
            break
    
    if not test_image:
        print("❌ No test image found. Please ensure there are uploaded images.")
        return False
    
    print(f"📸 Using test image: {test_image}")
    
    try:
        # Load image
        img = Image.open(test_image)
        print(f"✅ Image loaded successfully: {img.size} pixels, mode: {img.mode}")
        
        # Test CLIP Encoder
        print("\n🧠 Testing CLIP Encoder...")
        try:
            clip_encoder = CLIPEncoder()
            clip_features = clip_encoder.embed_image(img)
            print(f"✅ CLIP features shape: {clip_features.shape}")
            print(f"✅ CLIP features type: {type(clip_features)}")
            print(f"✅ CLIP features sample: {clip_features[:5]}")
        except Exception as e:
            print(f"❌ CLIP Encoder failed: {e}")
            return False
        
        # Test Fashion Encoder
        print("\n👗 Testing Fashion Encoder...")
        try:
            fashion_encoder = FashionEncoder()
            fashion_features = fashion_encoder.embed_fashion_image(test_image)
            print(f"✅ Fashion features shape: {fashion_features.shape}")
            print(f"✅ Fashion features type: {type(fashion_features)}")
            print(f"✅ Fashion features sample: {fashion_features[:5]}")
        except Exception as e:
            print(f"❌ Fashion Encoder failed: {e}")
            return False
        
        # Test BLIP Captioner
        print("\n📝 Testing BLIP Captioner...")
        try:
            blip_captioner = BLIPCaptioner()
            caption = blip_captioner.caption(img)
            print(f"✅ BLIP caption: '{caption}'")
            print(f"✅ Caption type: {type(caption)}")
        except Exception as e:
            print(f"❌ BLIP Captioner failed: {e}")
            return False
        
        # Verify embeddings are valid
        print("\n🔬 Verifying Embedding Quality...")
        
        # Check if embeddings are not None or empty
        if clip_features is None or len(clip_features) == 0:
            print("❌ CLIP features are None or empty")
            return False
        
        if fashion_features is None or len(fashion_features) == 0:
            print("❌ Fashion features are None or empty")
            return False
        
        # Check if embeddings contain valid numbers
        if np.any(np.isnan(clip_features)) or np.any(np.isinf(clip_features)):
            print("❌ CLIP features contain NaN or Inf values")
            return False
        
        if np.any(np.isnan(fashion_features)) or np.any(np.isinf(fashion_features)):
            print("❌ Fashion features contain NaN or Inf values")
            return False
        
        # Check embedding dimensions
        expected_clip_dim = 512  # Standard CLIP dimension
        expected_fashion_dim = 512  # Standard fashion encoder dimension
        
        if len(clip_features) != expected_clip_dim:
            print(f"⚠️  CLIP features dimension {len(clip_features)} != expected {expected_clip_dim}")
        
        if len(fashion_features) != expected_fashion_dim:
            print(f"⚠️  Fashion features dimension {len(fashion_features)} != expected {expected_fashion_dim}")
        
        print("\n✅ All feature extraction tests passed!")
        print("🎉 AI models are properly converting images to embeddings")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_extraction()
    if success:
        print("\n🎯 Feature extraction pipeline is working correctly!")
        sys.exit(0)
    else:
        print("\n💥 Feature extraction pipeline has issues that need fixing.")
        sys.exit(1)