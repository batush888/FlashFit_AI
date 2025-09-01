#!/usr/bin/env python3
"""
调试特征提取 - 查看实际特征值
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.classifier import get_classifier
from PIL import Image

def debug_feature_extraction():
    """
    调试特征提取过程，显示实际特征值
    """
    print("🔧 特征提取调试")
    print("=" * 50)
    
    # 获取分类器实例
    classifier = get_classifier()
    
    # 测试图像
    test_images = [
        ("/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/static/jackets/black_blazer.png", "jacket"),
        ("/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/static/pants/blue_jeans.png", "pants"),
        ("/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/static/dresses/red_cocktail.png", "dress"),
        ("/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/static/skirts/blue_a.png", "skirt"),
        ("/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/static/shorts/khaki_casual.png", "shorts")
    ]
    
    for image_path, expected_type in test_images:
        if not os.path.exists(image_path):
            print(f"❌ 图像不存在: {image_path}")
            continue
            
        print(f"\n📸 分析图像: {os.path.basename(image_path)} (预期: {expected_type})")
        print("-" * 40)
        
        try:
            # 加载图像
            image = Image.open(image_path)
            print(f"图像尺寸: {image.size}")
            
            # 提取特征
            features = classifier._extract_features(image)
            
            print(f"特征值:")
            print(f"  长宽比 (aspect_ratio): {features['aspect_ratio']:.4f}")
            print(f"  边缘密度 (edge_density): {features['edge_density']:.4f}")
            print(f"  颜色方差 (color_variance): {features['color_variance']:.1f}")
            
            # 显示分类规则检查
            aspect_ratio = features["aspect_ratio"]
            edge_density = features["edge_density"]
            color_variance = features["color_variance"]
            
            print(f"\n分类规则检查:")
            print(f"  鞋子 (aspect_ratio < 0.7 and edge_density > 0.08): {aspect_ratio < 0.7 and edge_density > 0.08}")
            print(f"  裤子 (aspect_ratio > 1.6): {aspect_ratio > 1.6}")
            print(f"  连衣裙 (aspect_ratio > 1.4 and color_variance > 500): {aspect_ratio > 1.4 and (color_variance > 500 or edge_density > 0.08)}")
            print(f"  短裤 (aspect_ratio <= 1.0 and edge_density > 0.05 and color_variance < 19): {aspect_ratio <= 1.0 and edge_density > 0.05 and color_variance < 19}")
            print(f"  裙子1 (aspect_ratio == 1.0 and edge_density > 0.08 and color_variance >= 19): {aspect_ratio == 1.0 and edge_density > 0.08 and color_variance >= 19}")
            print(f"  配饰 (edge_density > 0.2): {edge_density > 0.2}")
            print(f"  裙子2 (1.0 < aspect_ratio <= 1.4 and edge_density > 0.08 and color_variance < 100): {1.0 < aspect_ratio <= 1.4 and edge_density > 0.08 and edge_density < 0.18 and color_variance < 100}")
            print(f"  外套 (jacket conditions): {(0.8 <= aspect_ratio <= 1.3 and edge_density > 0.08 and color_variance > 15) or (aspect_ratio > 1.0 and color_variance > 800)}")
            
            # 执行分类
            result = classifier.classify_garment(image_path)
            print(f"\n实际分类结果: {result['category']} (置信度: {result['confidence']:.1%})")
            
        except Exception as e:
            print(f"❌ 处理图像时出错: {str(e)}")

if __name__ == "__main__":
    debug_feature_extraction()