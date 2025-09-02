#!/usr/bin/env python3
"""
Test script for the Advanced Classifier to verify it correctly identifies:
1. Black jacket (not white)
2. Beige shorts (not pink) 
3. Blue shirt (not pink)
4. Proper garment types (shorts/shirts vs jackets)
"""

import sys
import os
sys.path.append('backend')

from models.advanced_classifier import get_advanced_classifier
from PIL import Image
import numpy as np

def create_test_images():
    """
    Create synthetic test images that represent the problematic cases
    """
    test_images = {}
    
    # 1. Black jacket - should be identified as black jacket, not white
    black_jacket = np.zeros((300, 250, 3), dtype=np.uint8)  # Black image
    black_jacket[50:250, 50:200] = [20, 20, 20]  # Dark gray jacket shape
    # Add some structure/edges to make it look like a jacket
    black_jacket[50:60, 50:200] = [40, 40, 40]  # Collar
    black_jacket[100:110, 120:130] = [60, 60, 60]  # Button
    black_jacket[140:150, 120:130] = [60, 60, 60]  # Button
    test_images['black_jacket'] = Image.fromarray(black_jacket)
    
    # 2. Beige shorts - should be identified as beige shorts, not pink
    beige_shorts = np.full((200, 200, 3), [200, 180, 150], dtype=np.uint8)  # Beige background
    beige_shorts[80:180, 60:140] = [190, 170, 140]  # Beige shorts shape
    # Add some texture
    beige_shorts[90:95, 70:130] = [180, 160, 130]  # Waistband
    test_images['beige_shorts'] = Image.fromarray(beige_shorts)
    
    # 3. Blue shirt - should be identified as blue shirt, not pink
    blue_shirt = np.full((250, 200, 3), [100, 150, 200], dtype=np.uint8)  # Light blue
    blue_shirt[60:220, 50:150] = [80, 130, 180]  # Blue shirt shape
    # Add collar and buttons
    blue_shirt[60:80, 90:110] = [70, 120, 170]  # Collar
    blue_shirt[100:105, 95:105] = [60, 110, 160]  # Button
    blue_shirt[130:135, 95:105] = [60, 110, 160]  # Button
    test_images['blue_shirt'] = Image.fromarray(blue_shirt)
    
    return test_images

def test_advanced_classifier():
    """
    Test the advanced classifier with problematic cases
    """
    print("🧪 Testing Advanced Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = get_advanced_classifier()
    
    # Create test images
    test_images = create_test_images()
    
    # Test each image
    for image_name, image in test_images.items():
        print(f"\n📸 Testing: {image_name}")
        print("-" * 30)
        
        # Classify the image
        result = classifier.classify_garment(image, debug=True)
        
        # Display results
        print(f"\n✅ Results for {image_name}:")
        print(f"   Category: {result['category']} ({result['category_cn']})")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Primary Color: {result['primary_color']}")
        
        if result['colors']:
            print(f"   All Colors:")
            for i, color in enumerate(result['colors'][:3]):
                print(f"     {i+1}. {color['name']} ({color['percentage']:.1f}%) - {color['hex']}")
        
        # Validation
        print(f"\n🔍 Validation:")
        if image_name == 'black_jacket':
            color_correct = result['primary_color'] in ['black', 'gray']
            type_correct = result['category'] == 'jacket'
            print(f"   Color Detection: {'✅ PASS' if color_correct else '❌ FAIL'} (Expected: black/gray, Got: {result['primary_color']})")
            print(f"   Type Detection: {'✅ PASS' if type_correct else '❌ FAIL'} (Expected: jacket, Got: {result['category']})")
            
        elif image_name == 'beige_shorts':
            color_correct = result['primary_color'] in ['beige', 'brown', 'yellow']
            type_correct = result['category'] == 'shorts'
            print(f"   Color Detection: {'✅ PASS' if color_correct else '❌ FAIL'} (Expected: beige/brown, Got: {result['primary_color']})")
            print(f"   Type Detection: {'✅ PASS' if type_correct else '❌ FAIL'} (Expected: shorts, Got: {result['category']})")
            
        elif image_name == 'blue_shirt':
            color_correct = result['primary_color'] in ['blue', 'light_blue']
            type_correct = result['category'] in ['shirt', 't-shirt']
            print(f"   Color Detection: {'✅ PASS' if color_correct else '❌ FAIL'} (Expected: blue, Got: {result['primary_color']})")
            print(f"   Type Detection: {'✅ PASS' if type_correct else '❌ FAIL'} (Expected: shirt/t-shirt, Got: {result['category']})")

def test_with_real_images():
    """
    Test with real images if available
    """
    print("\n\n🖼️  Testing with Real Images (if available)")
    print("=" * 50)
    
    # Look for test images in common locations
    test_dirs = [
        'test_images',
        'data/test_images', 
        'backend/test_images',
        'uploads'
    ]
    
    classifier = get_advanced_classifier()
    found_images = False
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for filename in os.listdir(test_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    found_images = True
                    image_path = os.path.join(test_dir, filename)
                    print(f"\n📸 Testing: {filename}")
                    print("-" * 30)
                    
                    try:
                        result = classifier.classify_garment(image_path, debug=True)
                        print(f"   Category: {result['category']} ({result['category_cn']})")
                        print(f"   Primary Color: {result['primary_color']}")
                        if result['colors']:
                            print(f"   Top Colors: {', '.join([c['name'] for c in result['colors'][:3]])}")
                    except Exception as e:
                        print(f"   ❌ Error: {e}")
    
    if not found_images:
        print("   No real test images found. Place images in 'test_images/' directory to test.")

if __name__ == "__main__":
    try:
        test_advanced_classifier()
        test_with_real_images()
        
        print("\n\n🎉 Advanced Classifier Testing Complete!")
        print("\n💡 Tips for better results:")
        print("   1. Ensure good lighting and clear images")
        print("   2. Images should focus on the garment (minimal background)")
        print("   3. Higher resolution images generally work better")
        print("   4. The classifier learns from feedback - report issues for improvement")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()