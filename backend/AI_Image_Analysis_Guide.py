#!/usr/bin/env python3
"""
🤖 AI Image Analysis Guide - Complete Code & Logic Explanation
================================================================================

This file contains the EXACT code and logic that the AI uses to analyze images
and determine:
1. What type of garment it is (jacket, dress, pants, etc.)
2. What colors are present in the image
3. Style keywords and matching logic

You can use this code to understand, modify, and improve the AI's analysis.

📋 TABLE OF CONTENTS:
1. Image Feature Extraction
2. Garment Type Classification Rules
3. Color Detection & Analysis
4. Style Keyword Generation
5. How to Improve the AI
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import os

class AIImageAnalysisGuide:
    """
    🔍 COMPLETE AI IMAGE ANALYSIS SYSTEM
    
    This class contains ALL the logic the AI uses to understand images.
    Each method is documented with the exact rules and thresholds.
    """
    
    def __init__(self):
        print("🤖 AI Image Analysis System Initialized")
        print("📚 This system analyzes: garment type, colors, and style")
    
    # ============================================================================
    # 1️⃣ IMAGE FEATURE EXTRACTION
    # ============================================================================
    
    def extract_image_features(self, image_path):
        """
        🔬 STEP 1: Extract numerical features from the image
        
        Features extracted:
        - Aspect ratio (height/width)
        - Edge density (how many edges detected)
        - Color variance (how varied the colors are)
        - Dominant colors
        """
        print(f"\n🔍 ANALYZING IMAGE: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        height, width = image.shape[:2]
        aspect_ratio = height / width
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection using Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Color variance calculation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image_rgb.reshape(-1, 3).astype(np.float32)
        color_variance = float(np.var(pixels))
        
        features = {
            'aspect_ratio': aspect_ratio,
            'edge_density': edge_density,
            'color_variance': color_variance,
            'width': width,
            'height': height
        }
        
        print(f"📊 EXTRACTED FEATURES:")
        print(f"   • Aspect Ratio: {aspect_ratio:.3f}")
        print(f"   • Edge Density: {edge_density:.4f}")
        print(f"   • Color Variance: {color_variance:.1f}")
        
        return features
    
    # ============================================================================
    # 2️⃣ GARMENT TYPE CLASSIFICATION RULES
    # ============================================================================
    
    def classify_garment_type(self, features):
        """
        👔 STEP 2: Determine garment type using rule-based logic
        
        EXACT RULES USED BY THE AI:
        
        🥿 SHOES: aspect_ratio <= 1.1 AND edge_density > 0.15
        👖 PANTS: aspect_ratio > 1.6 AND edge_density > 0.12
        👗 DRESS: aspect_ratio > 1.8 AND edge_density < 0.20
        🧥 JACKET: (aspect_ratio > 1.1 AND <= 1.6) AND 
                   (color_variance > 1500 AND edge_density < 0.1)
        👜 ACCESSORY: edge_density > 0.25
        👚 TOP: aspect_ratio <= 1.3 AND edge_density > 0.10
        🩱 SWIMWEAR: edge_density < 0.08
        👔 SHIRT: aspect_ratio <= 1.4 AND edge_density > 0.08
        👖 SHORTS: aspect_ratio <= 1.2 AND edge_density > 0.15
        👠 HEELS: aspect_ratio <= 1.0 AND edge_density > 0.20
        🩲 UNDERWEAR: edge_density < 0.06
        👘 SKIRT: (aspect_ratio > 1.1 AND <= 1.6) AND 
                  (color_variance <= 1500 OR edge_density >= 0.16)
        """
        
        aspect_ratio = features['aspect_ratio']
        edge_density = features['edge_density']
        color_variance = features['color_variance']
        
        print(f"\n🤖 GARMENT CLASSIFICATION LOGIC:")
        print(f"📏 Input: aspect_ratio={aspect_ratio:.3f}, edge_density={edge_density:.4f}, color_variance={color_variance:.1f}")
        
        # Test each category in order
        if aspect_ratio <= 1.1 and edge_density > 0.15:
            result = "SHOES"
            reason = f"aspect_ratio <= 1.1 ({aspect_ratio:.3f} <= 1.1) AND edge_density > 0.15 ({edge_density:.4f} > 0.15)"
        elif aspect_ratio > 1.6 and edge_density > 0.12:
            result = "PANTS"
            reason = f"aspect_ratio > 1.6 ({aspect_ratio:.3f} > 1.6) AND edge_density > 0.12 ({edge_density:.4f} > 0.12)"
        elif aspect_ratio > 1.8 and edge_density < 0.20:
            result = "DRESS"
            reason = f"aspect_ratio > 1.8 ({aspect_ratio:.3f} > 1.8) AND edge_density < 0.20 ({edge_density:.4f} < 0.20)"
        elif (aspect_ratio > 1.1 and aspect_ratio <= 1.6) and (color_variance > 1500 and edge_density < 0.1):
            result = "JACKET"
            reason = f"aspect_ratio in (1.1, 1.6] ({aspect_ratio:.3f}) AND color_variance > 1500 ({color_variance:.1f}) AND edge_density < 0.1 ({edge_density:.4f})"
        elif edge_density > 0.25:
            result = "ACCESSORY"
            reason = f"edge_density > 0.25 ({edge_density:.4f} > 0.25)"
        elif aspect_ratio <= 1.3 and edge_density > 0.10:
            result = "TOP"
            reason = f"aspect_ratio <= 1.3 ({aspect_ratio:.3f} <= 1.3) AND edge_density > 0.10 ({edge_density:.4f} > 0.10)"
        elif edge_density < 0.08:
            result = "SWIMWEAR"
            reason = f"edge_density < 0.08 ({edge_density:.4f} < 0.08)"
        elif aspect_ratio <= 1.4 and edge_density > 0.08:
            result = "SHIRT"
            reason = f"aspect_ratio <= 1.4 ({aspect_ratio:.3f} <= 1.4) AND edge_density > 0.08 ({edge_density:.4f} > 0.08)"
        elif aspect_ratio <= 1.2 and edge_density > 0.15:
            result = "SHORTS"
            reason = f"aspect_ratio <= 1.2 ({aspect_ratio:.3f} <= 1.2) AND edge_density > 0.15 ({edge_density:.4f} > 0.15)"
        elif aspect_ratio <= 1.0 and edge_density > 0.20:
            result = "HEELS"
            reason = f"aspect_ratio <= 1.0 ({aspect_ratio:.3f} <= 1.0) AND edge_density > 0.20 ({edge_density:.4f} > 0.20)"
        elif edge_density < 0.06:
            result = "UNDERWEAR"
            reason = f"edge_density < 0.06 ({edge_density:.4f} < 0.06)"
        elif (aspect_ratio > 1.1 and aspect_ratio <= 1.6) and (color_variance <= 1500 or edge_density >= 0.16):
            result = "SKIRT"
            reason = f"aspect_ratio in (1.1, 1.6] ({aspect_ratio:.3f}) AND (color_variance <= 1500 ({color_variance:.1f}) OR edge_density >= 0.16 ({edge_density:.4f}))"
        else:
            result = "UNKNOWN"
            reason = "No rules matched"
        
        print(f"✅ CLASSIFICATION: {result}")
        print(f"📝 REASON: {reason}")
        
        return result
    
    # ============================================================================
    # 3️⃣ COLOR DETECTION & ANALYSIS
    # ============================================================================
    
    def extract_dominant_colors(self, image_path, n_colors=5):
        """
        🎨 STEP 3: Extract dominant colors using K-means clustering
        
        Process:
        1. Load image and convert to RGB
        2. Use K-means to find dominant color clusters
        3. Calculate percentage of each color
        4. Classify each RGB value to a color name
        """
        print(f"\n🎨 COLOR ANALYSIS:")
        
        # Load and process image
        image = Image.open(image_path)
        image = image.convert('RGB')
        image_array = np.array(image)
        
        # Reshape for clustering
        pixels = image_array.reshape(-1, 3)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
        kmeans.fit(pixels)
        
        # Get colors and their percentages
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Calculate percentages
        color_percentages = []
        total_pixels = len(labels) if labels is not None else 1
        for i in range(n_colors):
            percentage = np.sum(labels == i) / total_pixels * 100
            rgb = colors[i].astype(int)
            color_name = self.get_color_name(rgb)
            
            color_info = {
                'rgb': rgb,
                'percentage': percentage,
                'color_name': color_name,
                'hex': f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            }
            color_percentages.append(color_info)
        
        # Sort by percentage
        color_percentages.sort(key=lambda x: x['percentage'], reverse=True)
        
        print(f"🏆 DOMINANT COLORS (sorted by dominance):")
        for i, color in enumerate(color_percentages, 1):
            print(f"   {i}. {color['color_name']}: {color['percentage']:.1f}%")
            print(f"      RGB{tuple(color['rgb'])} | {color['hex']}")
        
        return color_percentages
    
    def get_color_name(self, rgb):
        """
        🌈 STEP 4: Convert RGB values to color names
        
        EXACT COLOR CLASSIFICATION RULES:
        
        ⚪ WHITE: r > 220 AND g > 220 AND b > 220
        ⚫ BLACK: r < 40 AND g < 40 AND b < 40
        🔘 GRAY: |r-g| < 25 AND |g-b| < 25 AND |r-b| < 25
        🔴 RED: r > max(g,b) + 30
        🔵 BLUE: b > max(r,g) + 30
        🟢 GREEN: g > max(r,b) + 30
        🟡 YELLOW: r > 180 AND g > 180 AND b < 100
        🟠 ORANGE: r > 200 AND g > 100 AND g < 200 AND b < 100
        🟣 PURPLE: r > 100 AND b > 100 AND g < min(r,b)
        🟤 BROWN/BEIGE: r > 100 AND g > 80 AND b > 60 AND (max-min) < 80
        🩷 PINK: r > 150 AND (r > g + 20) AND (r > b + 20)
        """
        r, g, b = rgb
        
        print(f"\n🔍 COLOR CLASSIFICATION for RGB({r}, {g}, {b}):")
        
        # Test each color rule
        if r > 220 and g > 220 and b > 220:
            result = "white"
            reason = f"r>220 ({r}>220), g>220 ({g}>220), b>220 ({b}>220)"
        elif r < 40 and g < 40 and b < 40:
            result = "black"
            reason = f"r<40 ({r}<40), g<40 ({g}<40), b<40 ({b}<40)"
        elif abs(r-g) < 25 and abs(g-b) < 25 and abs(r-b) < 25:
            result = "gray"
            reason = f"|r-g|<25 ({abs(r-g)}<25), |g-b|<25 ({abs(g-b)}<25), |r-b|<25 ({abs(r-b)}<25)"
        elif r > max(g, b) + 30:
            result = "red"
            reason = f"r > max(g,b) + 30 ({r} > {max(g,b)} + 30 = {max(g,b)+30})"
        elif b > max(r, g) + 30:
            result = "blue"
            reason = f"b > max(r,g) + 30 ({b} > {max(r,g)} + 30 = {max(r,g)+30})"
        elif g > max(r, b) + 30:
            result = "green"
            reason = f"g > max(r,b) + 30 ({g} > {max(r,b)} + 30 = {max(r,b)+30})"
        elif r > 180 and g > 180 and b < 100:
            result = "yellow"
            reason = f"r>180 ({r}>180), g>180 ({g}>180), b<100 ({b}<100)"
        elif r > 200 and 100 < g < 200 and b < 100:
            result = "orange"
            reason = f"r>200 ({r}>200), 100<g<200 (100<{g}<200), b<100 ({b}<100)"
        elif r > 100 and b > 100 and g < min(r, b):
            result = "purple"
            reason = f"r>100 ({r}>100), b>100 ({b}>100), g<min(r,b) ({g}<{min(r,b)})"
        elif r > 100 and g > 80 and b > 60 and (max(r,g,b) - min(r,g,b)) < 80:
            result = "beige"
            reason = f"r>100 ({r}>100), g>80 ({g}>80), b>60 ({b}>60), max-min<80 ({max(r,g,b)-min(r,g,b)}<80)"
        elif r > 150 and r > g + 20 and r > b + 20:
            result = "pink"
            reason = f"r>150 ({r}>150), r>g+20 ({r}>{g+20}), r>b+20 ({r}>{b+20})"
        else:
            result = "gray"  # Default fallback
            reason = "No specific rules matched, defaulting to gray"
        
        print(f"   ✅ CLASSIFIED AS: {result}")
        print(f"   📝 REASON: {reason}")
        
        return result
    
    # ============================================================================
    # 4️⃣ STYLE KEYWORD GENERATION
    # ============================================================================
    
    def generate_style_keywords(self, garment_type, dominant_colors):
        """
        🏷️ STEP 5: Generate style keywords based on garment type and colors
        
        This combines the garment type with color information to create
        searchable style keywords for matching similar items.
        """
        print(f"\n🏷️ STYLE KEYWORD GENERATION:")
        
        # Base keywords from garment type
        base_keywords = [garment_type.lower()]
        
        # Add color keywords (top 3 colors)
        color_keywords = []
        for color_info in dominant_colors[:3]:
            if color_info['percentage'] > 5:  # Only include colors > 5%
                color_keywords.append(color_info['color_name'])
        
        # Combine keywords
        all_keywords = base_keywords + color_keywords
        
        print(f"   📋 Generated keywords: {', '.join(all_keywords)}")
        
        return all_keywords
    
    # ============================================================================
    # 5️⃣ COMPLETE ANALYSIS PIPELINE
    # ============================================================================
    
    def analyze_image_complete(self, image_path):
        """
        🔄 COMPLETE AI ANALYSIS PIPELINE
        
        This runs the full analysis that the AI performs on every uploaded image:
        1. Extract features (aspect ratio, edges, colors)
        2. Classify garment type using rules
        3. Extract and classify dominant colors
        4. Generate style keywords
        """
        print("\n" + "="*80)
        print("🤖 COMPLETE AI IMAGE ANALYSIS PIPELINE")
        print("="*80)
        
        # Step 1: Extract features
        features = self.extract_image_features(image_path)
        if not features:
            print("❌ Could not load image")
            return None
        
        # Step 2: Classify garment type
        garment_type = self.classify_garment_type(features)
        
        # Step 3: Extract colors
        dominant_colors = self.extract_dominant_colors(image_path)
        
        # Step 4: Generate keywords
        keywords = self.generate_style_keywords(garment_type, dominant_colors)
        
        # Summary
        print("\n" + "="*80)
        print("📊 FINAL AI ANALYSIS RESULTS")
        print("="*80)
        print(f"👔 Garment Type: {garment_type}")
        print(f"🎨 Primary Color: {dominant_colors[0]['color_name']} ({dominant_colors[0]['percentage']:.1f}%)")
        print(f"🏷️ Style Keywords: {', '.join(keywords)}")
        print("\n💡 This is exactly how the AI 'sees' and classifies your image!")
        
        return {
            'garment_type': garment_type,
            'dominant_colors': dominant_colors,
            'keywords': keywords,
            'features': features
        }

# ============================================================================
# 🚀 HOW TO IMPROVE THE AI
# ============================================================================

def how_to_improve_ai():
    """
    💡 GUIDE: How to improve the AI's accuracy
    
    PROBLEM: Trench coat classified as white instead of beige/tan
    
    SOLUTIONS:
    
    1️⃣ ADJUST COLOR THRESHOLDS:
       - Current beige rule: r>100, g>80, b>60, max-min<80
       - For tan/khaki: Add specific RGB ranges
       - Example: if 150<r<220 and 120<g<180 and 80<b<140: return "tan"
    
    2️⃣ ADD MORE COLOR CATEGORIES:
       - tan, khaki, cream, ivory, sand, camel
       - Each with specific RGB ranges
    
    3️⃣ USE COLOR SPACE CONVERSION:
       - Convert RGB to HSV (Hue, Saturation, Value)
       - HSV is better for distinguishing similar colors
    
    4️⃣ MACHINE LEARNING APPROACH:
       - Train a color classifier on labeled color data
       - Use deep learning for more accurate color recognition
    
    5️⃣ CONTEXT-AWARE CLASSIFICATION:
       - Consider garment type when classifying colors
       - Coats are more likely to be tan/beige than pure white
    """
    print("\n" + "="*80)
    print("💡 HOW TO IMPROVE THE AI")
    print("="*80)
    print("\n🎯 CURRENT ISSUE: Trench coat shows as 'white' instead of 'beige/tan'")
    print("\n🔧 SOLUTIONS:")
    print("\n1️⃣ ADJUST COLOR THRESHOLDS in get_color_name():")
    print("   • Current beige rule is too restrictive")
    print("   • Add specific tan/khaki RGB ranges")
    print("   • Example: if 150<r<220 and 120<g<180 and 80<b<140: return 'tan'")
    print("\n2️⃣ ADD MORE COLOR CATEGORIES:")
    print("   • tan, khaki, cream, ivory, sand, camel")
    print("   • Each with specific RGB ranges")
    print("\n3️⃣ USE HSV COLOR SPACE:")
    print("   • Convert RGB to HSV for better color distinction")
    print("   • HSV separates hue from brightness")
    print("\n4️⃣ MACHINE LEARNING:")
    print("   • Train a color classifier on labeled data")
    print("   • Use deep learning for more accuracy")
    print("\n5️⃣ CONTEXT-AWARE:")
    print("   • Consider garment type when classifying colors")
    print("   • Coats are more likely tan/beige than pure white")

# ============================================================================
# 🧪 DEMO USAGE
# ============================================================================

if __name__ == "__main__":
    print("🤖 AI Image Analysis Guide - Complete System Overview")
    print("="*80)
    
    # Initialize the analysis system
    analyzer = AIImageAnalysisGuide()
    
    # Show improvement guide
    how_to_improve_ai()
    
    # Analyze the trench coat image
    test_image = "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/uploads/user_3_1756552528/user_3_1756552528_78f28430528b4dbfbee77eff8952d8e5.webp"
    
    if os.path.exists(test_image):
        print(f"\n🔍 ANALYZING YOUR TRENCH COAT IMAGE...")
        results = analyzer.analyze_image_complete(test_image)
    else:
        print(f"\n❌ Image not found: {test_image}")
        print("💡 Update the path to your image and run again")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("\n📚 You now have the complete code and logic the AI uses.")
    print("🔧 Use the improvement suggestions to enhance accuracy.")
    print("="*80)