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

from models.advanced_classifier import get_advanced_classifier

def analyze_image_features(image_path, expected_category):
    """分析单个图像的详细特征"""
    classifier = get_advanced_classifier()
    
    print(f"\n{'='*60}")
    print(f"📸 分析图像: {os.path.basename(image_path)}")
    print(f"预期类别: {expected_category}")
    print(f"{'='*60}")
    
    try:
        # 加载图像
        image = Image.open(image_path)
        print(f"图像尺寸: {image.size}")
        
        # 使用高级分类器进行分析
        print(f"\n🔍 使用高级分类器进行特征分析")
        
        # 执行分类
        result = classifier.classify_garment(image_path)
        predicted_category = result['category']
        confidence = result['confidence']
        
        print(f"\n🎯 分类结果:")
        print(f"  预测类别: {predicted_category}")
        print(f"  置信度: {confidence:.1%}")
        print(f"  是否正确: {'✅ 是' if predicted_category == expected_category else '❌ 否'}")
        
        # 高级分类器分析
        print(f"\n🧠 高级分类器使用机器学习进行智能分类")
        print(f"   ✓ 使用CLIP嵌入和神经网络特征进行分类")
        
        # 显示分类结果分析
        print(f"\n💡 分类结果分析:")
        if predicted_category != expected_category:
            print(f"   - 分类差异: 预期 {expected_category}，实际 {predicted_category}")
            print(f"   - 置信度: {result['confidence']:.1%}")
            print(f"   - 建议: 高级分类器使用深度学习，可能识别出更细致的特征")
        else:
            print(f"   ✅ 分类正确！置信度: {result['confidence']:.1%}")
        
        return {
            'image_path': image_path,
            'expected': expected_category,
            'predicted': predicted_category,
            'correct': predicted_category == expected_category,
            'confidence': result['confidence']
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