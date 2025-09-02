#!/usr/bin/env python3
"""
Detailed Embedding Shape Analysis
Analyzes the exact shapes and dimensions of AI model outputs
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

def analyze_embedding_shapes():
    print("🔍 Detailed Embedding Shape Analysis")
    print("=" * 50)
    
    # Find a test image
    test_image_paths = [
        "backend/data/uploads/user_3_1756552528/user_3_1756552528_d1177182926d4faba6a6da46846dc3b2.jpg",
        "backend/data/uploads/user_4_1756553127/user_4_1756553127_02c47560893a4401b05c1395262f24ec.jpg",
        "backend/data/uploads/user_5_1756715906/user_5_1756715906_2a6cd59f40194899913f407c52efb5bc.webp"
    ]
    
    test_image = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image = path
            break
    
    if not test_image:
        print("❌ No test image found")
        return False
    
    print(f"📸 Using: {test_image}")
    
    try:
        # Load image
        img = Image.open(test_image)
        print(f"✅ Image: {img.size} pixels, {img.mode} mode")
        
        # Test CLIP Encoder
        print("\n🧠 CLIP Encoder Analysis:")
        clip_encoder = CLIPEncoder()
        clip_features = clip_encoder.embed_image(img)
        
        print(f"  📊 Raw shape: {clip_features.shape}")
        print(f"  📊 Data type: {type(clip_features)}")
        print(f"  📊 Element type: {clip_features.dtype}")
        print(f"  📊 Number of dimensions: {clip_features.ndim}")
        print(f"  📊 Total elements: {clip_features.size}")
        
        if clip_features.ndim == 2:
            print(f"  📊 Batch size: {clip_features.shape[0]}")
            print(f"  📊 Feature dimension: {clip_features.shape[1]}")
            # Flatten to 1D if needed
            clip_flat = clip_features.flatten()
            print(f"  📊 Flattened shape: {clip_flat.shape}")
        else:
            clip_flat = clip_features
        
        print(f"  📊 First 5 values: {clip_flat[:5]}")
        print(f"  📊 Value range: [{float(np.min(clip_flat)):.6f}, {float(np.max(clip_flat)):.6f}]")
        print(f"  📊 Mean: {float(np.mean(clip_flat)):.6f}")
        print(f"  📊 Std: {float(np.std(clip_flat)):.6f}")
        
        # Test Fashion Encoder
        print("\n👗 Fashion Encoder Analysis:")
        fashion_encoder = FashionEncoder()
        fashion_features = fashion_encoder.embed_fashion_image(test_image)
        
        print(f"  📊 Raw shape: {fashion_features.shape}")
        print(f"  📊 Data type: {type(fashion_features)}")
        print(f"  📊 Element type: {fashion_features.dtype}")
        print(f"  📊 Number of dimensions: {fashion_features.ndim}")
        print(f"  📊 Total elements: {fashion_features.size}")
        
        if fashion_features.ndim == 2:
            print(f"  📊 Batch size: {fashion_features.shape[0]}")
            print(f"  📊 Feature dimension: {fashion_features.shape[1]}")
            # Flatten to 1D if needed
            fashion_flat = fashion_features.flatten()
            print(f"  📊 Flattened shape: {fashion_flat.shape}")
        else:
            fashion_flat = fashion_features
        
        print(f"  📊 First 5 values: {fashion_flat[:5]}")
        print(f"  📊 Value range: [{float(np.min(fashion_flat)):.6f}, {float(np.max(fashion_flat)):.6f}]")
        print(f"  📊 Mean: {float(np.mean(fashion_flat)):.6f}")
        print(f"  📊 Std: {float(np.std(fashion_flat)):.6f}")
        
        # Test BLIP Captioner
        print("\n📝 BLIP Captioner Analysis:")
        blip_captioner = BLIPCaptioner()
        caption = blip_captioner.caption(img)
        
        print(f"  📊 Caption: '{caption}'")
        print(f"  📊 Caption length: {len(caption)} characters")
        print(f"  📊 Word count: {len(caption.split())} words")
        
        # Verify embeddings are valid
        print("\n🔬 Validation Results:")
        
        # Check for valid numerical values
        clip_valid = not (np.any(np.isnan(clip_flat)) or np.any(np.isinf(clip_flat)))
        fashion_valid = not (np.any(np.isnan(fashion_flat)) or np.any(np.isinf(fashion_flat)))
        caption_valid = isinstance(caption, str) and len(caption.strip()) > 0
        
        print(f"  ✅ CLIP embeddings valid: {clip_valid}")
        print(f"  ✅ Fashion embeddings valid: {fashion_valid}")
        print(f"  ✅ BLIP caption valid: {caption_valid}")
        
        # Check if embeddings are normalized
        clip_norm = np.linalg.norm(clip_flat)
        fashion_norm = np.linalg.norm(fashion_flat)
        
        print(f"  📊 CLIP L2 norm: {clip_norm:.6f}")
        print(f"  📊 Fashion L2 norm: {fashion_norm:.6f}")
        
        # Check if they're unit normalized (norm ≈ 1.0)
        clip_normalized = abs(clip_norm - 1.0) < 0.1
        fashion_normalized = abs(fashion_norm - 1.0) < 0.1
        
        print(f"  📊 CLIP normalized: {clip_normalized}")
        print(f"  📊 Fashion normalized: {fashion_normalized}")
        
        print("\n🎯 Summary:")
        print(f"  • CLIP produces {clip_flat.shape[0]}-dimensional embeddings")
        print(f"  • Fashion produces {fashion_flat.shape[0]}-dimensional embeddings")
        print(f"  • BLIP produces text captions")
        print(f"  • All outputs are valid: {clip_valid and fashion_valid and caption_valid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = analyze_embedding_shapes()
    if success:
        print("\n✅ Embedding analysis completed successfully!")
    else:
        print("\n❌ Embedding analysis failed.")