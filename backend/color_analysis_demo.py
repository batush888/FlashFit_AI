#!/usr/bin/env python3
"""
Color Analysis Demo - Shows exactly how the AI analyzes colors in images
This demonstrates the complete color detection pipeline used by the classifier
"""

import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple

# Add backend to path
sys.path.append('/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend')

from models.classifier import GarmentClassifier

class ColorAnalysisDemo:
    """Demonstrates the complete color analysis process"""
    
    def __init__(self):
        """Initialize color names mapping"""
        self.color_names = {
            "red": "红色",
            "blue": "蓝色", 
            "green": "绿色",
            "yellow": "黄色",
            "orange": "橙色",
            "purple": "紫色",
            "pink": "粉色",
            "brown": "棕色",
            "black": "黑色",
            "white": "白色",
            "gray": "灰色",
            "beige": "米色"
        }
    
    def analyze_image_colors(self, image_path: str, n_colors: int = 5):
        """Complete color analysis of an image"""
        print(f"\n🎨 COLOR ANALYSIS DEMO")
        print("=" * 80)
        print(f"📸 Analyzing: {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        print(f"📏 Image size: {image.size} (W x H)")
        
        # Step 1: Convert to numpy array
        print(f"\n🔧 STEP 1: Convert image to pixel data")
        print("-" * 50)
        img_array = np.array(image)
        print(f"   Array shape: {img_array.shape}")
        print(f"   Total pixels: {img_array.shape[0] * img_array.shape[1]:,}")
        
        # Step 2: Reshape for clustering
        print(f"\n🔧 STEP 2: Prepare pixels for color clustering")
        print("-" * 50)
        pixels = img_array.reshape(-1, 3)
        print(f"   Reshaped to: {pixels.shape} (each row is one pixel's RGB)")
        print(f"   Sample pixels (first 5):")
        for i in range(min(5, len(pixels))):
            r, g, b = pixels[i]
            print(f"     Pixel {i+1}: RGB({r:3d}, {g:3d}, {b:3d}) = #{r:02x}{g:02x}{b:02x}")
        
        # Step 3: K-means clustering
        print(f"\n🔧 STEP 3: Find dominant colors using K-means clustering")
        print("-" * 50)
        print(f"   Clustering into {n_colors} color groups...")
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
        kmeans.fit(pixels)
        
        print(f"   ✅ Clustering complete!")
        print(f"   Found {len(kmeans.cluster_centers_)} color centers")
        
        # Step 4: Analyze each color
        print(f"\n🔧 STEP 4: Analyze each dominant color")
        print("-" * 50)
        
        colors = []
        labels = kmeans.labels_
        
        for i, color_center in enumerate(kmeans.cluster_centers_):
            # Calculate percentage
            total_pixels = len(labels) if labels is not None else 1
            percentage = np.sum(labels == i) / total_pixels
            
            # Convert to integer RGB
            rgb = (int(color_center[0]), int(color_center[1]), int(color_center[2]))
            r, g, b = rgb
            
            # Get color name using the same logic as the classifier
            color_name = self._get_color_name(rgb)
            
            print(f"\n   🎯 COLOR {i+1}:")
            print(f"      Raw RGB: ({color_center[0]:.1f}, {color_center[1]:.1f}, {color_center[2]:.1f})")
            print(f"      Integer RGB: ({r}, {g}, {b})")
            print(f"      Hex: #{r:02x}{g:02x}{b:02x}")
            print(f"      Percentage: {percentage*100:.1f}%")
            print(f"      Color name: {color_name} ({self.color_names.get(color_name, color_name)})")
            print(f"      Classification logic:")
            self._explain_color_classification(rgb)
            
            colors.append({
                "rgb": rgb,
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "name": color_name,
                "name_cn": self.color_names.get(color_name, color_name),
                "percentage": round(percentage * 100, 1)
            })
        
        # Step 5: Sort by percentage
        colors.sort(key=lambda x: x["percentage"], reverse=True)
        
        print(f"\n🏆 FINAL RESULTS (sorted by dominance):")
        print("-" * 50)
        for i, color in enumerate(colors):
            print(f"   {i+1}. {color['name']} ({color['name_cn']}): {color['percentage']}%")
            print(f"      RGB{color['rgb']} | {color['hex']}")
        
        # Step 6: Compare with classifier
        print(f"\n🤖 CLASSIFIER COMPARISON:")
        print("-" * 50)
        classifier = GarmentClassifier()
        result = classifier.extract_dominant_colors(image, 3)
        
        print(f"   Classifier's top 3 colors:")
        for i, color in enumerate(result):
            print(f"     {i+1}. {color['name']} ({color['name_cn']}): {color['percentage']}%")
        
        return colors
    
    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Same color classification logic as the main classifier"""
        r, g, b = rgb
        
        # 1. Black/White/Gray detection
        if r > 220 and g > 220 and b > 220:
            return "white"
        elif r < 40 and g < 40 and b < 40:
            return "black"
        elif abs(r - g) < 25 and abs(g - b) < 25 and abs(r - b) < 25:
            if r > 180:
                return "white"
            elif r > 120:
                return "gray"
            else:
                return "black"
        
        # 2. Main color detection
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        
        # Red family
        if r == max_val and r > g + 30 and r > b + 30:
            if r > 180 and g < 80 and b < 80:
                return "red"
            elif r > 150 and g > 80 and b < 80:
                return "orange"
            elif r > 120 and g > 60 and b > 60:
                return "pink"
            else:
                return "brown"
        
        # Green family
        elif g == max_val and g > r + 30 and g > b + 30:
            return "green"
        
        # Blue family
        elif b == max_val and b > r + 30 and b > g + 30:
            if b > 150 and r < 100 and g < 100:
                return "blue"
            elif b > 120 and r > 80 and g < 120:
                return "purple"
            else:
                return "blue"
        
        # Yellow
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        
        # Purple
        elif r > 120 and b > 120 and g < 100:
            return "purple"
        
        # Orange
        elif r > 180 and g > 100 and g < 150 and b < 100:
            return "orange"
        
        # Brown/Beige
        elif r > 100 and g > 80 and b > 60 and max_val - min_val < 80:
            if r > 150:
                return "beige"
            else:
                return "brown"
        
        # Default
        else:
            return "gray"
    
    def _explain_color_classification(self, rgb: Tuple[int, int, int]):
        """Explain why a color gets classified the way it does"""
        r, g, b = rgb
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        
        print(f"         → Testing white: r>220 ({r}>220={r>220}), g>220 ({g}>220={g>220}), b>220 ({b}>220={b>220})")
        if r > 220 and g > 220 and b > 220:
            print(f"         → ✅ Classified as WHITE (all values > 220)")
            return
        
        print(f"         → Testing black: r<40 ({r}<40={r<40}), g<40 ({g}<40={g<40}), b<40 ({b}<40={b<40})")
        if r < 40 and g < 40 and b < 40:
            print(f"         → ✅ Classified as BLACK (all values < 40)")
            return
        
        r_g_diff = abs(r - g)
        g_b_diff = abs(g - b)
        r_b_diff = abs(r - b)
        print(f"         → Testing gray: |r-g|<25 ({r_g_diff}<25={r_g_diff<25}), |g-b|<25 ({g_b_diff}<25={g_b_diff<25}), |r-b|<25 ({r_b_diff}<25={r_b_diff<25})")
        if abs(r - g) < 25 and abs(g - b) < 25 and abs(r - b) < 25:
            if r > 180:
                print(f"         → ✅ Classified as WHITE (gray-like but r>180: {r}>180)")
            elif r > 120:
                print(f"         → ✅ Classified as GRAY (gray-like, r>120: {r}>120)")
            else:
                print(f"         → ✅ Classified as BLACK (gray-like, r<=120: {r}<=120)")
            return
        
        print(f"         → Testing beige/brown: r>100 ({r}>100={r>100}), g>80 ({g}>80={g>80}), b>60 ({b}>60={b>60}), max-min<80 ({max_val}-{min_val}={max_val-min_val}<80={max_val-min_val<80})")
        if r > 100 and g > 80 and b > 60 and max_val - min_val < 80:
            if r > 150:
                print(f"         → ✅ Classified as BEIGE (brown-like but r>150: {r}>150)")
            else:
                print(f"         → ✅ Classified as BROWN (brown-like, r<=150: {r}<=150)")
            return
        
        print(f"         → ✅ Classified using other rules or default to GRAY")

def main():
    """Main function"""
    print("🔬 Color Analysis Demo - Understanding AI Color Detection")
    print("=" * 80)
    print("This tool shows exactly how the AI analyzes colors in your images.")
    print("You can use this code to understand and improve color detection.")
    
    # Analyze the trench coat image
    test_image = "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/uploads/user_3_1756552528/user_3_1756552528_cb9d721ca28a412eb9991a8ccc7b5f13.webp"
    
    demo = ColorAnalysisDemo()
    colors = demo.analyze_image_colors(test_image, n_colors=5)
    
    print(f"\n💡 INSIGHTS FOR IMPROVEMENT:")
    print("-" * 50)
    print(f"   1. The AI uses K-means clustering to find dominant colors")
    print(f"   2. Each pixel's RGB values are analyzed using rule-based logic")
    print(f"   3. Colors are classified based on RGB thresholds and relationships")
    print(f"   4. The current rules may be too strict for subtle color variations")
    print(f"   5. You can modify the _get_color_name() function to improve accuracy")
    
    print(f"\n🚀 TO IMPROVE COLOR DETECTION:")
    print("-" * 50)
    print(f"   • Adjust RGB thresholds in _get_color_name() method")
    print(f"   • Add more specific color categories (e.g., tan, khaki, cream)")
    print(f"   • Consider using color space conversions (HSV, LAB)")
    print(f"   • Implement machine learning-based color classification")

if __name__ == "__main__":
    main()