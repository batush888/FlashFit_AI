#!/usr/bin/env python3
"""
测试分类器问题的调试脚本
"""

import sys
import os
sys.path.append('backend')

from PIL import Image
import numpy as np
from models.classifier import get_classifier

def test_classification_issues():
    """
    测试分类器的问题
    """
    print("🔍 测试分类器问题...")
    
    classifier = get_classifier()
    
    # 测试用例：创建模拟图像来测试分类逻辑
    test_cases = [
        {
            "name": "短裤 (正方形, 中等颜色方差)",
            "aspect_ratio": 1.0,
            "edge_density": 0.10,
            "color_variance": 16.0,
            "expected": "shorts"
        },
        {
            "name": "衬衫 (正方形, 低边缘密度)", 
            "aspect_ratio": 1.0,
            "edge_density": 0.08,
            "color_variance": 12.0,
            "expected": "shirt"
        },
        {
            "name": "风衣 (长方形, 高边缘密度)",
            "aspect_ratio": 1.3,
            "edge_density": 0.13,
            "color_variance": 20.0,
            "expected": "jacket"
        }
    ]
    
    print("\n📊 测试分类规则:")
    for case in test_cases:
        features = {
            "aspect_ratio": case["aspect_ratio"],
            "edge_density": case["edge_density"],
            "color_variance": case["color_variance"],
            "width": 224,
            "height": int(224 * case["aspect_ratio"])
        }
        
        result = classifier._rule_based_classify(features)
        status = "✅" if result == case["expected"] else "❌"
        
        print(f"  {status} {case['name']}")
        print(f"     特征: 长宽比={features['aspect_ratio']:.1f}, 边缘密度={features['edge_density']:.3f}, 颜色方差={features['color_variance']:.1f}")
        print(f"     预期: {case['expected']}, 实际: {result}")
        print()

def test_color_recognition():
    """
    测试颜色识别问题
    """
    print("🎨 测试颜色识别...")
    
    classifier = get_classifier()
    
    # 测试颜色用例
    color_test_cases = [
        {
            "name": "米色/驼色",
            "rgb": (194, 154, 108),  # 典型的米色
            "expected": "beige"
        },
        {
            "name": "卡其色", 
            "rgb": (195, 176, 145),  # 卡其色
            "expected": "khaki"
        },
        {
            "name": "白色",
            "rgb": (240, 240, 240),  # 白色
            "expected": "white"
        },
        {
            "name": "奶油色",
            "rgb": (255, 253, 208),  # 奶油色
            "expected": "cream"
        }
    ]
    
    print("\n🎯 测试颜色识别:")
    for case in color_test_cases:
        result = classifier._get_color_name(case["rgb"], "jacket")
        status = "✅" if result == case["expected"] else "❌"
        
        print(f"  {status} {case['name']}")
        print(f"     RGB: {case['rgb']}, 预期: {case['expected']}, 实际: {result}")
        print()

if __name__ == "__main__":
    test_classification_issues()
    test_color_recognition()