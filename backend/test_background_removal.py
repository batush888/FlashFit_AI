#!/usr/bin/env python3
"""
Test background removal functionality
"""

import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np

# Add backend to path
sys.path.append('/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend')

from models.classifier import GarmentClassifier

def test_background_removal(image_path: str):
    """Test background removal on an image"""
    print(f"\n🔍 Testing background removal on: {os.path.basename(image_path)}")
    print("=" * 80)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    print(f"📏 Original image size: {original_image.size}")
    
    # Apply background removal (same logic as in upload.py)
    img_array = np.array(original_image)
    print(f"📊 Image array shape: {img_array.shape}")
    
    # Create mask for white/light background
    white_mask = np.all(img_array > 240, axis=2)
    white_pixels = np.sum(white_mask)
    total_pixels = img_array.shape[0] * img_array.shape[1]
    white_percentage = (white_pixels / total_pixels) * 100
    
    print(f"🎨 White pixels: {white_pixels:,} / {total_pixels:,} ({white_percentage:.1f}%)")
    
    if white_percentage > 30:
        print(f"✅ Background removal triggered (>{30}% white pixels)")
        # Apply background removal
        img_array[white_mask] = [128, 128, 128]
        processed_image = Image.fromarray(img_array)
        
        # Save processed image for inspection
        output_path = image_path.replace('.webp', '_processed.jpg').replace('.jpg', '_processed.jpg')
        processed_image.save(output_path)
        print(f"💾 Processed image saved to: {output_path}")
        
        # Test classification on both images
        classifier = GarmentClassifier()
        
        print(f"\n🤖 ORIGINAL IMAGE CLASSIFICATION:")
        print("-" * 40)
        original_result = classifier.classify_garment(original_image)
        print(f"   Category: {original_result['category']} ({original_result['category_cn']})")
        print(f"   Confidence: {original_result['confidence']:.2f}")
        print(f"   Top colors:")
        for i, color in enumerate(original_result['dominant_colors'][:3]):
            print(f"     {i+1}. {color['name']} ({color['name_cn']}): {color['percentage']}%")
        
        print(f"\n🚀 PROCESSED IMAGE CLASSIFICATION:")
        print("-" * 40)
        processed_result = classifier.classify_garment(processed_image)
        print(f"   Category: {processed_result['category']} ({processed_result['category_cn']})")
        print(f"   Confidence: {processed_result['confidence']:.2f}")
        print(f"   Top colors:")
        for i, color in enumerate(processed_result['dominant_colors'][:3]):
            print(f"     {i+1}. {color['name']} ({color['name_cn']}): {color['percentage']}%")
        
        # Compare results
        print(f"\n📊 COMPARISON:")
        print("-" * 40)
        if original_result['category'] != processed_result['category']:
            print(f"   ✅ Category changed: {original_result['category']} → {processed_result['category']}")
        else:
            print(f"   ➖ Category unchanged: {original_result['category']}")
        
        return processed_result
    else:
        print(f"❌ Background removal not triggered (<={30}% white pixels)")
        return None

def main():
    """Main function"""
    print("🔬 Background Removal Test Tool")
    print("=" * 80)
    
    # Test with the problematic image
    test_image = "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/uploads/user_3_1756552528/user_3_1756552528_83075d9fa4b146cab9b3aca09f8d1fa8.webp"
    
    result = test_background_removal(test_image)
    
    if result:
        print(f"\n✅ Background removal test completed successfully!")
    else:
        print(f"\n❌ Background removal was not applied or failed.")

if __name__ == "__main__":
    main()