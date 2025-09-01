#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析分类错误的详细脚本
检查特征提取和分类规则的问题
"""

import os
import sys
from PIL import Image
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.classifier import get_classifier

def analyze_image_features(image_path, expected_category):
    """分析单个图像的详细特征"""
    classifier = get_classifier()
    
    print(f"\n{'='*60}")
    print(f"📸 分析图像: {os.path.basename(image_path)}")
    print(f"预期类别: {expected_category}")
    print(f"{'='*60}")
    
    try:
        # 加载图像
        image = Image.open(image_path)
        print(f"图像尺寸: {image.size}")
        
        # 提取特征
        features = classifier._extract_features(image)
        aspect_ratio = features['aspect_ratio']
        edge_density = features['edge_density']
        color_variance = features['color_variance']
        
        print(f"\n🔍 提取的特征:")
        print(f"  长宽比 (aspect_ratio): {aspect_ratio:.4f}")
        print(f"  边缘密度 (edge_density): {edge_density:.4f}")
        print(f"  颜色方差 (color_variance): {color_variance:.4f}")
        
        # 执行分类
        result = classifier.classify_garment(image_path)
        predicted_category = result['category']
        confidence = result['confidence']
        
        print(f"\n🎯 分类结果:")
        print(f"  预测类别: {predicted_category}")
        print(f"  置信度: {confidence:.1%}")
        print(f"  是否正确: {'✅ 是' if predicted_category == expected_category else '❌ 否'}")
        
        # 检查每个规则的匹配情况（按当前分类器的顺序）
        print(f"\n📋 规则匹配分析:")
        
        # 规则1: dress
        dress_match = aspect_ratio > 1.2 and edge_density < 0.15
        print(f"  规则1 (dress): aspect_ratio > 1.2 AND edge_density < 0.15")
        print(f"    条件: {aspect_ratio:.4f} > 1.2 = {aspect_ratio > 1.2}, {edge_density:.4f} < 0.15 = {edge_density < 0.15}")
        print(f"    匹配: {'✅' if dress_match else '❌'}")
        
        # 规则2: pants
        pants_match = aspect_ratio > 1.1 and edge_density >= 0.15
        print(f"  规则2 (pants): aspect_ratio > 1.1 AND edge_density >= 0.15")
        print(f"    条件: {aspect_ratio:.4f} > 1.1 = {aspect_ratio > 1.1}, {edge_density:.4f} >= 0.15 = {edge_density >= 0.15}")
        print(f"    匹配: {'✅' if pants_match else '❌'}")
        
        # 规则3: sweater (更新的条件)
        sweater_match = aspect_ratio == 1.0 and edge_density < 0.105 and color_variance < 15
        print(f"  规则3 (sweater): aspect_ratio == 1.0 AND edge_density < 0.105 AND color_variance < 15")
        print(f"    条件: {aspect_ratio:.4f} == 1.0 = {aspect_ratio == 1.0}, {edge_density:.4f} < 0.105 = {edge_density < 0.105}, {color_variance:.4f} < 15 = {color_variance < 15}")
        print(f"    匹配: {'✅' if sweater_match else '❌'}")
        
        # 规则4: shorts
        shorts_match = aspect_ratio == 1.0 and color_variance < 19
        print(f"  规则4 (shorts): aspect_ratio == 1.0 AND color_variance < 19")
        print(f"    条件: {aspect_ratio:.4f} == 1.0 = {aspect_ratio == 1.0}, {color_variance:.4f} < 19 = {color_variance < 19}")
        print(f"    匹配: {'✅' if shorts_match else '❌'}")
        
        # 规则5: skirt
        skirt_match = aspect_ratio == 1.0 and color_variance >= 19
        print(f"  规则5 (skirt): aspect_ratio == 1.0 AND color_variance >= 19")
        print(f"    条件: {aspect_ratio:.4f} == 1.0 = {aspect_ratio == 1.0}, {color_variance:.4f} >= 19 = {color_variance >= 19}")
        print(f"    匹配: {'✅' if skirt_match else '❌'}")
        
        # 规则6: skirt2
        skirt2_match = 0.8 <= aspect_ratio < 1.0 and edge_density >= 0.1
        print(f"  规则6 (skirt2): 0.8 <= aspect_ratio < 1.0 AND edge_density >= 0.1")
        print(f"    条件: 0.8 <= {aspect_ratio:.4f} < 1.0 = {0.8 <= aspect_ratio < 1.0}, {edge_density:.4f} >= 0.1 = {edge_density >= 0.1}")
        print(f"    匹配: {'✅' if skirt2_match else '❌'}")
        
        # 规则7: jacket
        jacket_match = aspect_ratio <= 1.1 and edge_density >= 0.12
        print(f"  规则7 (jacket): aspect_ratio <= 1.1 AND edge_density >= 0.12")
        print(f"    条件: {aspect_ratio:.4f} <= 1.1 = {aspect_ratio <= 1.1}, {edge_density:.4f} >= 0.12 = {edge_density >= 0.12}")
        print(f"    匹配: {'✅' if jacket_match else '❌'}")
        
        # 规则8: blouse
        blouse_match = 0.9 <= aspect_ratio <= 1.1 and edge_density < 0.12
        print(f"  规则8 (blouse): 0.9 <= aspect_ratio <= 1.1 AND edge_density < 0.12")
        print(f"    条件: 0.9 <= {aspect_ratio:.4f} <= 1.1 = {0.9 <= aspect_ratio <= 1.1}, {edge_density:.4f} < 0.12 = {edge_density < 0.12}")
        print(f"    匹配: {'✅' if blouse_match else '❌'}")
        
        # 统计匹配的规则数量
        matching_rules = sum([dress_match, pants_match, sweater_match, shorts_match, 
                            skirt_match, skirt2_match, jacket_match, blouse_match])
        print(f"\n📊 总计匹配规则数: {matching_rules}")
        
        if matching_rules == 0:
            print("⚠️  警告: 没有匹配任何规则，将默认分类为 'shirt'")
        elif matching_rules > 1:
            print(f"⚠️  警告: 匹配了多个规则，可能存在规则重叠问题")
        
        return {
            'image_path': image_path,
            'expected': expected_category,
            'predicted': predicted_category,
            'correct': predicted_category == expected_category,
            'features': features,
            'matching_rules': matching_rules
        }
        
    except Exception as e:
        print(f"❌ 分析图像时出错: {str(e)}")
        return None

def main():
    """主函数"""
    print("🔍 分类错误详细分析")
    print("=" * 60)
    
    # 测试用例 - 重点关注错误分类的图像
    test_cases = [
        ("sweaters/pink_knit.png", "sweater"),  # 被错误分类为 shorts
        ("sweaters/gray_wool.png", "sweater"),  # 被错误分类为 skirt
        ("shirts/white_cotton.png", "shirt"),   # 被错误分类为 skirt
        ("shorts/khaki_casual.png", "shorts"),  # 正确分类
        ("skirts/blue_a.png", "skirt"),         # 正确分类
    ]
    
    test_images_dir = "data/static"
    results = []
    
    for relative_path, expected_category in test_cases:
        image_path = os.path.join(test_images_dir, relative_path)
        if os.path.exists(image_path):
            result = analyze_image_features(image_path, expected_category)
            if result:
                results.append(result)
        else:
            print(f"❌ 文件不存在: {image_path}")
    
    # 总结分析
    print(f"\n{'='*60}")
    print("📊 分析总结")
    print(f"{'='*60}")
    
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print(f"总体准确率: {correct_count}/{total_count} ({accuracy:.1f}%)")
    
    print("\n❌ 错误分类详情:")
    for result in results:
        if not result['correct']:
            features = result['features']
            print(f"  {os.path.basename(result['image_path'])}: {result['expected']} → {result['predicted']}")
            print(f"    特征: AR={features['aspect_ratio']:.3f}, ED={features['edge_density']:.3f}, CV={features['color_variance']:.1f}")
    
    print("\n💡 改进建议:")
    print("  1. 检查规则重叠问题 - 多个规则匹配同一图像")
    print("  2. 调整特征阈值以更好地区分相似类别")
    print("  3. 考虑添加更多特征来提高区分度")
    print("  4. 优化规则优先级和匹配逻辑")

if __name__ == "__main__":
    main()