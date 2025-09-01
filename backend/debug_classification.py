#!/usr/bin/env python3
"""
Debug script to analyze color and garment classification issues
"""

import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# Add backend to path
sys.path.append('/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend')

from models.classifier import GarmentClassifier

def analyze_image_colors(image_path: str):
    """Analyze colors in an image step by step"""
    print(f"\n🔍 Analyzing image: {image_path}")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"📏 Image size: {image.size}")
    
    # Convert to numpy array
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)
    print(f"🎨 Total pixels: {len(pixels)}")
    
    # Show pixel distribution
    print(f"\n📊 RGB Statistics:")
    print(f"   Red   - Min: {pixels[:, 0].min():3d}, Max: {pixels[:, 0].max():3d}, Mean: {pixels[:, 0].mean():.1f}")
    print(f"   Green - Min: {pixels[:, 1].min():3d}, Max: {pixels[:, 1].max():3d}, Mean: {pixels[:, 1].mean():.1f}")
    print(f"   Blue  - Min: {pixels[:, 2].min():3d}, Max: {pixels[:, 2].max():3d}, Mean: {pixels[:, 2].mean():.1f}")
    
    # K-means clustering
    print(f"\n🎯 K-means clustering (3 colors):")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    
    labels = kmeans.labels_
    for i, color in enumerate(kmeans.cluster_centers_):
        percentage = np.sum(labels == i) / len(labels)
        rgb = (int(color[0]), int(color[1]), int(color[2]))
        
        # Manual color classification
        r, g, b = rgb
        if r > 220 and g > 220 and b > 220:
            color_name = "white"
        elif r < 40 and g < 40 and b < 40:
            color_name = "black"
        elif abs(r - g) < 25 and abs(g - b) < 25 and abs(r - b) < 25:
            if r > 180:
                color_name = "white"
            elif r > 120:
                color_name = "gray"
            else:
                color_name = "black"
        elif r > 100 and g > 80 and b > 60 and max(r,g,b) - min(r,g,b) < 80:
            if r > 150:
                color_name = "beige"
            else:
                color_name = "brown"
        elif b > r + 30 and b > g + 30:
            color_name = "blue"
        else:
            color_name = "unknown"
            
        print(f"   Color {i+1}: RGB{rgb} -> {color_name} ({percentage*100:.1f}%)")
    
    # Test with classifier
    print(f"\n🤖 Classifier Results:")
    classifier = GarmentClassifier()
    result = classifier.classify_garment(image)
    
    print(f"   Category: {result['category']} ({result['category_cn']})")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Detected colors:")
    for color in result['dominant_colors']:
        print(f"     - {color['name']} ({color['name_cn']}): {color['percentage']}% - RGB{color['rgb']}")
    
    # Analyze features used for classification
    features = classifier._extract_features(image)
    print(f"\n📐 Classification Features:")
    print(f"   Aspect ratio: {features['aspect_ratio']:.2f}")
    print(f"   Edge density: {features['edge_density']:.3f}")
    print(f"   Color variance: {features['color_variance']:.1f}")
    print(f"   Size: {features['width']}x{features['height']}")
    
    # Show classification logic
    aspect_ratio = features["aspect_ratio"]
    edge_density = features["edge_density"]
    color_variance = features["color_variance"]
    
    print(f"\n🧠 Classification Logic Analysis:")
    if aspect_ratio < 0.8 and edge_density > 0.12:
        print(f"   ✓ Would classify as SHOES (aspect < 0.8 AND edge > 0.12)")
    elif aspect_ratio > 1.6 and aspect_ratio < 2.5 and edge_density < 0.15:
        print(f"   ✓ Would classify as PANTS (1.6 < aspect < 2.5 AND edge < 0.15)")
    elif aspect_ratio > 1.4 and (color_variance > 800 or edge_density > 0.18):
        print(f"   ✓ Would classify as DRESS (aspect > 1.4 AND (variance > 800 OR edge > 0.18))")
    elif aspect_ratio >= 0.9 and aspect_ratio <= 1.4 and edge_density > 0.16:
        print(f"   ✓ Would classify as JACKET (0.9 <= aspect <= 1.4 AND edge > 0.16)")
    elif aspect_ratio > 1.1 and aspect_ratio <= 1.6 and edge_density < 0.16:
        print(f"   ✓ Would classify as SKIRT (1.1 < aspect <= 1.6 AND edge < 0.16)")
    elif edge_density > 0.25 or color_variance > 1500:
        print(f"   ✓ Would classify as ACCESSORY (edge > 0.25 OR variance > 1500)")
    else:
        print(f"   ✓ Would classify as SHIRT (default)")

def main():
    """Main function to test classification"""
    print("🔬 Fashion Classification Debug Tool")
    print("=" * 60)
    
    # Test with the problematic images
    test_images = [
        "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/uploads/user_2_1756506140/user_2_1756506140_e896348dfccd472f8cabd4a95e661bd3.jpg",
        "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/uploads/user_3_1756552528/user_3_1756552528_2904e329a2bb44f0ab8318e74b7d48c5.jpg"
    ]
    
    for image_path in test_images:
        analyze_image_colors(image_path)
        print("\n" + "="*60)

if __name__ == "__main__":
    main()