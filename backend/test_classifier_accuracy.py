#!/usr/bin/env python3
"""
测试分类器准确性 - 使用多个测试图像
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.classifier import get_classifier
from PIL import Image
import glob

def test_classifier_accuracy():
    """
    测试分类器在多个图像上的准确性
    """
    classifier = get_classifier()
    
    # 测试图像路径 - 使用有标签的静态图像
    test_images_dir = "data/static"
    
    # 定义测试图像及其预期分类
    test_cases = [
        ("blouses/floral_spring.png", "shirt"),  # blouse通常归类为shirt
        ("dresses/red_cocktail.png", "dress"),
        ("jackets/black_blazer.png", "jacket"),
        ("pants/blue_jeans.png", "pants"),
        ("shirts/white_formal.png", "shirt"),
        ("shorts/khaki_casual.png", "shorts"),
        ("skirts/blue_a.png", "skirt"),
        ("sweaters/pink_knit.png", "shirt"),  # sweater通常归类为shirt
    ]
    
    print(f"🔍 测试 {len(test_cases)} 个标记图像")
    print("=" * 60)
    
    results = {}
    correct_predictions = 0
    total_processed = 0
    
    for relative_path, expected_category in test_cases:
        image_path = os.path.join(test_images_dir, relative_path)
        filename = os.path.basename(relative_path)
        try:
            print(f"\n📸 分析图像: {filename} (预期: {expected_category})")
            print("-" * 50)
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"❌ 文件不存在: {image_path}")
                continue
            
            # 加载图像
            image = Image.open(image_path)
            print(f"图像尺寸: {image.size}")
            
            # 执行分类
            result = classifier.classify_garment(image_path)
            predicted_category = result['category']
            confidence = result['confidence']
            
            # 检查预测是否正确
            is_correct = predicted_category == expected_category
            status = "✅ 正确" if is_correct else "❌ 错误"
            
            print(f"预期分类: {expected_category}")
            print(f"实际分类: {predicted_category} (置信度: {confidence:.1%}) {status}")
            
            # 统计结果
            if predicted_category not in results:
                results[predicted_category] = 0
            results[predicted_category] += 1
            
            if is_correct:
                correct_predictions += 1
            total_processed += 1
            
        except Exception as e:
            print(f"❌ 处理图像 {filename} 时出错: {str(e)}")
    
    # 显示统计结果
    print("\n" + "=" * 60)
    print("📊 分类准确性报告:")
    print("-" * 30)
    
    if total_processed > 0:
        accuracy = (correct_predictions / total_processed) * 100
        print(f"总体准确率: {correct_predictions}/{total_processed} ({accuracy:.1f}%)")
        
        print("\n预测分布:")
        for category, count in sorted(results.items()):
            percentage = (count / total_processed) * 100
            print(f"  {category}: {count} 个 ({percentage:.1f}%)")
        
        # 检查分类多样性
        unique_categories = len(results)
        print(f"\n识别出的不同类别: {unique_categories}")
        
        if accuracy >= 80:
            print("✅ 优秀: 分类器准确率很高")
        elif accuracy >= 60:
            print("⚠️  一般: 分类器准确率中等，需要改进")
        else:
            print("❌ 较差: 分类器准确率较低，需要重大改进")
            
        if unique_categories == 1:
            print("⚠️  警告: 所有图像都被分类为同一类别，可能存在分类器问题")
        elif unique_categories >= 5:
            print("✅ 良好: 分类器能够识别多种不同的服装类别")
    else:
        print("❌ 没有成功处理任何图像")

if __name__ == "__main__":
    test_classifier_accuracy()