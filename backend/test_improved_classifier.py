#!/usr/bin/env python3
"""
Test script to compare old vs improved classifier
"""

import sys
import os
from pathlib import Path
from PIL import Image

# Add backend to path
sys.path.append('/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend')

from models.classifier import GarmentClassifier
from models.improved_classifier import ImprovedGarmentClassifier

def compare_classifiers(image_path: str):
    """Compare old vs improved classifier"""
    print(f"\n🔍 Testing image: {os.path.basename(image_path)}")
    print("=" * 80)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"📏 Image size: {image.size}")
    
    # Test old classifier
    print(f"\n🤖 OLD CLASSIFIER:")
    print("-" * 40)
    old_classifier = GarmentClassifier()
    old_result = old_classifier.classify_garment(image)
    
    print(f"   Category: {old_result['category']} ({old_result['category_cn']})")
    print(f"   Confidence: {old_result['confidence']:.2f}")
    print(f"   Colors:")
    for i, color in enumerate(old_result['dominant_colors'][:3]):
        print(f"     {i+1}. {color['name']} ({color['name_cn']}): {color['percentage']}%")
    
    # Test improved classifier
    print(f"\n🚀 IMPROVED CLASSIFIER:")
    print("-" * 40)
    improved_classifier = ImprovedGarmentClassifier()
    improved_result = improved_classifier.classify_garment(image)
    
    print(f"   Category: {improved_result['category']} ({improved_result['category_cn']})")
    print(f"   Confidence: {improved_result['confidence']:.2f}")
    print(f"   Colors:")
    for i, color in enumerate(improved_result['dominant_colors'][:3]):
        print(f"     {i+1}. {color['name']} ({color['name_cn']}): {color['percentage']}%")
    
    # Compare results
    print(f"\n📊 COMPARISON:")
    print("-" * 40)
    category_changed = old_result['category'] != improved_result['category']
    colors_changed = (
        len(old_result['dominant_colors']) != len(improved_result['dominant_colors']) or
        any(old_result['dominant_colors'][i]['name'] != improved_result['dominant_colors'][i]['name'] 
            for i in range(min(len(old_result['dominant_colors']), len(improved_result['dominant_colors']))))
    )
    
    if category_changed:
        print(f"   ✅ Category: {old_result['category']} → {improved_result['category']}")
    else:
        print(f"   ➖ Category: No change ({old_result['category']})")
    
    if colors_changed:
        print(f"   ✅ Colors: Changed")
        print(f"      Old: {', '.join([c['name'] for c in old_result['dominant_colors'][:3]])}")
        print(f"      New: {', '.join([c['name'] for c in improved_result['dominant_colors'][:3]])}")
    else:
        print(f"   ➖ Colors: No significant change")
    
    return {
        'old': old_result,
        'improved': improved_result,
        'category_changed': category_changed,
        'colors_changed': colors_changed
    }

def main():
    """Main function to test improved classifier"""
    print("🔬 Fashion Classifier Comparison Tool")
    print("=" * 80)
    
    # Test with the problematic images
    test_images = [
        "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/uploads/user_2_1756506140/user_2_1756506140_e896348dfccd472f8cabd4a95e661bd3.jpg",
        "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/uploads/user_3_1756552528/user_3_1756552528_2904e329a2bb44f0ab8318e74b7d48c5.jpg"
    ]
    
    results = []
    for image_path in test_images:
        result = compare_classifiers(image_path)
        if result:
            results.append(result)
        print("\n" + "="*80)
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print("=" * 80)
    category_improvements = sum(1 for r in results if r['category_changed'])
    color_improvements = sum(1 for r in results if r['colors_changed'])
    
    print(f"   Total images tested: {len(results)}")
    print(f"   Category improvements: {category_improvements}")
    print(f"   Color detection improvements: {color_improvements}")
    
    if category_improvements > 0 or color_improvements > 0:
        print(f"   ✅ Improved classifier shows better results!")
    else:
        print(f"   ➖ No significant improvements detected")

if __name__ == "__main__":
    main()