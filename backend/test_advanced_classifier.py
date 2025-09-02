#!/usr/bin/env python3

import numpy as np
from PIL import Image
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.advanced_classifier import get_advanced_classifier

def create_test_image(color, size=(200, 300)):
    """
    Create a synthetic test image with the specified color
    """
    # Convert color name to RGB
    color_map = {
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'beige': (245, 245, 220),
        'blue': (70, 130, 180),
        'pink': (255, 192, 203)
    }
    
    rgb = color_map.get(color.lower(), (128, 128, 128))
    
    # Create image array
    img_array = np.full((size[1], size[0], 3), rgb, dtype=np.uint8)
    
    # Add some texture/noise to make it more realistic
    noise = np.random.randint(-10, 10, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def test_classifier():
    print("Testing Advanced Garment Classifier...")
    
    # Get the classifier
    classifier = get_advanced_classifier()
    
    # Test cases that were problematic
    test_cases = [
        {'color': 'black', 'size': (200, 250), 'expected_type': 'jacket', 'description': 'Black jacket'},
        {'color': 'beige', 'size': (200, 180), 'expected_type': 'shorts', 'description': 'Beige shorts'},
        {'color': 'blue', 'size': (200, 240), 'expected_type': 'shirt', 'description': 'Blue shirt'}
    ]
    
    print("\n=== Testing with synthetic images ===")
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['description']}")
        
        # Create test image
        test_image = create_test_image(test_case['color'], test_case['size'])
        
        # Classify
        result = classifier.classify_garment(test_image)
        
        print(f"Expected: {test_case['expected_type']}")
        print(f"Got: {result['category']}")
        print(f"Color detected: {result['primary_color']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        # Check if classification is correct
        type_correct = result['category'].lower() == test_case['expected_type'].lower()
        color_correct = result['primary_color'].lower() == test_case['color'].lower()
        
        print(f"Type correct: {type_correct}")
        print(f"Color correct: {color_correct}")
        
        if result.get('features'):
            features = result['features']
            print(f"Features - Aspect ratio: {features.get('aspect_ratio', 'N/A'):.2f}, "
                  f"Edge density: {features.get('edge_density', 'N/A'):.3f}, "
                  f"Texture: {features.get('texture_complexity', 'N/A'):.3f}")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_classifier()