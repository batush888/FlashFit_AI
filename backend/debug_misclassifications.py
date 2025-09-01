#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.classifier import GarmentClassifier
import cv2
import numpy as np

def debug_specific_image(image_path, expected_category):
    """调试特定图像的分类问题"""
    print(f"\n🔍 调试图像: {os.path.basename(image_path)} (预期: {expected_category})")
    print("=" * 60)
    
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法加载图像: {image_path}")
        return
    
    print(f"图像尺寸: {image.shape}")
    
    # 初始化分类器
    classifier = GarmentClassifier()
    
    # 提取特征
    features = classifier._extract_features(image)
    aspect_ratio = features['aspect_ratio']
    edge_density = features['edge_density']
    color_variance = features['color_variance']
    
    print(f"\n📊 特征值:")
    print(f"  宽高比 (aspect_ratio): {aspect_ratio:.4f}")
    print(f"  边缘密度 (edge_density): {edge_density:.4f}")
    print(f"  颜色方差 (color_variance): {color_variance:.4f}")
    
    # 进行分类
    result = classifier.classify_garment(image_path)
    predicted_category = result['category']
    confidence = result['confidence']
    
    print(f"\n🎯 分类结果:")
    print(f"  预测类别: {predicted_category}")
    print(f"  置信度: {confidence:.1f}%")
    print(f"  是否正确: {'✅ 是' if predicted_category == expected_category else '❌ 否'}")
    
    # 详细检查每个规则
    print(f"\n📋 详细规则匹配分析:")
    
    # 规则1: dress
    dress_match = aspect_ratio > 1.2 and edge_density < 0.15
    print(f"  规则1 (dress): AR > 1.2 AND ED < 0.15")
    print(f"    {aspect_ratio:.4f} > 1.2 = {aspect_ratio > 1.2}, {edge_density:.4f} < 0.15 = {edge_density < 0.15}")
    print(f"    匹配: {'✅' if dress_match else '❌'}")
    
    # 规则2: pants
    pants_match = aspect_ratio > 1.1 and edge_density >= 0.15
    print(f"  规则2 (pants): AR > 1.1 AND ED >= 0.15")
    print(f"    {aspect_ratio:.4f} > 1.1 = {aspect_ratio > 1.1}, {edge_density:.4f} >= 0.15 = {edge_density >= 0.15}")
    print(f"    匹配: {'✅' if pants_match else '❌'}")
    
    # 规则3: sweater
    sweater_match = aspect_ratio == 1.0 and edge_density < 0.105 and color_variance < 15
    print(f"  规则3 (sweater): AR == 1.0 AND ED < 0.105 AND CV < 15")
    print(f"    {aspect_ratio:.4f} == 1.0 = {aspect_ratio == 1.0}, {edge_density:.4f} < 0.105 = {edge_density < 0.105}, {color_variance:.4f} < 15 = {color_variance < 15}")
    print(f"    匹配: {'✅' if sweater_match else '❌'}")
    
    # 规则4: shorts
    shorts_match = aspect_ratio == 1.0 and color_variance < 19
    print(f"  规则4 (shorts): AR == 1.0 AND CV < 19")
    print(f"    {aspect_ratio:.4f} == 1.0 = {aspect_ratio == 1.0}, {color_variance:.4f} < 19 = {color_variance < 19}")
    print(f"    匹配: {'✅' if shorts_match else '❌'}")
    
    # 规则5: skirt
    skirt_match = aspect_ratio == 1.0 and color_variance >= 19
    print(f"  规则5 (skirt): AR == 1.0 AND CV >= 19")
    print(f"    {aspect_ratio:.4f} == 1.0 = {aspect_ratio == 1.0}, {color_variance:.4f} >= 19 = {color_variance >= 19}")
    print(f"    匹配: {'✅' if skirt_match else '❌'}")
    
    # 规则6: skirt2
    skirt2_match = 0.8 <= aspect_ratio < 1.0 and edge_density >= 0.1
    print(f"  规则6 (skirt2): 0.8 <= AR < 1.0 AND ED >= 0.1")
    print(f"    0.8 <= {aspect_ratio:.4f} < 1.0 = {0.8 <= aspect_ratio < 1.0}, {edge_density:.4f} >= 0.1 = {edge_density >= 0.1}")
    print(f"    匹配: {'✅' if skirt2_match else '❌'}")
    
    # 规则7: jacket
    jacket_match = aspect_ratio <= 1.1 and edge_density >= 0.12
    print(f"  规则7 (jacket): AR <= 1.1 AND ED >= 0.12")
    print(f"    {aspect_ratio:.4f} <= 1.1 = {aspect_ratio <= 1.1}, {edge_density:.4f} >= 0.12 = {edge_density >= 0.12}")
    print(f"    匹配: {'✅' if jacket_match else '❌'}")
    
    # 规则8: blouse
    blouse_match = 0.9 <= aspect_ratio <= 1.1 and edge_density < 0.12
    print(f"  规则8 (blouse): 0.9 <= AR <= 1.1 AND ED < 0.12")
    print(f"    0.9 <= {aspect_ratio:.4f} <= 1.1 = {0.9 <= aspect_ratio <= 1.1}, {edge_density:.4f} < 0.12 = {edge_density < 0.12}")
    print(f"    匹配: {'✅' if blouse_match else '❌'}")
    
    # 默认: shirt
    print(f"  默认规则 (shirt): 如果没有其他规则匹配")
    
    # 统计匹配的规则数量
    matching_rules = sum([dress_match, pants_match, sweater_match, shorts_match, 
                        skirt_match, skirt2_match, jacket_match, blouse_match])
    print(f"\n📊 总计匹配规则数: {matching_rules}")
    
    if matching_rules > 1:
        print(f"⚠️  警告: 匹配了多个规则，可能存在规则重叠问题")
    elif matching_rules == 0:
        print(f"ℹ️  信息: 没有匹配任何规则，将使用默认分类 (shirt)")

def main():
    # 测试图像路径
    test_images = [
        ("data/static/sweaters/pink_knit.png", "shirt"),  # 被错误分类为sweater的图像
    ]
    
    for image_path, expected_category in test_images:
        debug_specific_image(image_path, expected_category)

if __name__ == "__main__":
    main()