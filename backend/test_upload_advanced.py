#!/usr/bin/env python3
"""
测试上传API使用高级分类器的效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from api.upload import UploadHandler
from models.advanced_classifier import get_advanced_classifier
import asyncio
from unittest.mock import Mock

def create_mock_upload_file(image_path: str):
    """创建模拟的UploadFile对象"""
    mock_file = Mock()
    mock_file.filename = os.path.basename(image_path)
    mock_file.size = os.path.getsize(image_path)
    
    # 创建一个可以读取的文件对象
    with open(image_path, 'rb') as f:
        mock_file.file = Mock()
        mock_file.file.read = lambda: f.read()
        # 重置文件指针
        f.seek(0)
        mock_file.file.seek = lambda pos: f.seek(pos)
    
    return mock_file

async def test_upload_with_advanced_classifier():
    """测试上传API使用高级分类器"""
    print("=== 测试上传API使用高级分类器 ===")
    
    # 初始化上传处理器
    upload_handler = UploadHandler()
    
    # 使用实际上传的图像进行测试
    test_images = [
        "data/uploads/user_5_1756715906/user_5_1756715906_2a6cd59f40194899913f407c52efb5bc.webp",
        "data/uploads/user_3_1756552528/user_3_1756552528_400586b132f34e6dbcc8241bf5a3ba31.jpg", 
        "data/uploads/user_2_1756506140/user_2_1756506140_e896348dfccd472f8cabd4a95e661bd3.jpg"
    ]
    
    # 不设置期望结果，只观察分类效果
    expected_results = {}
    
    results = []
    
    for image_path in test_images:
        filename = os.path.basename(image_path)
        
        if not os.path.exists(image_path):
            print(f"⚠️  图像文件不存在: {image_path}")
            continue
            
        print(f"\n📸 测试图像: {image_path}")
        
        try:
            # 直接测试图像处理功能
            processing_result = upload_handler._process_image(image_path)
            classification = processing_result["classification"]
            
            print(f"分类结果:")
            print(f"  类别: {classification.get('category', 'unknown')} ({classification.get('category_cn', 'unknown')})")
            print(f"  置信度: {classification.get('confidence', 0):.3f}")
            
            if "dominant_colors" in classification:
                print(f"  主要颜色:")
                for i, color in enumerate(classification["dominant_colors"][:3]):
                    print(f"    {i+1}. {color.get('name', 'unknown')} ({color.get('percentage', 0):.1f}%)")
            
            # 检查结果
            expected = expected_results.get(filename, {})
            actual_type = classification.get('category', 'unknown')
            actual_colors = [c.get('name', '') for c in classification.get('dominant_colors', [])]
            
            type_correct = actual_type == expected.get('type', '')
            color_correct = expected.get('color', '') in actual_colors
            
            print(f"\n✅ 结果评估:")
            print(f"  类型正确: {'✓' if type_correct else '✗'} (期望: {expected.get('type', 'N/A')}, 实际: {actual_type})")
            print(f"  颜色正确: {'✓' if color_correct else '✗'} (期望: {expected.get('color', 'N/A')}, 实际: {actual_colors})")
            
            results.append({
                'image': filename,
                'type_correct': type_correct,
                'color_correct': color_correct,
                'classification': classification
            })
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            results.append({
                'image': filename,
                'type_correct': False,
                'color_correct': False,
                'error': str(e)
            })
    
    # 总结结果
    print(f"\n=== 测试总结 ===")
    total_tests = len(results)
    type_correct_count = sum(1 for r in results if r.get('type_correct', False))
    color_correct_count = sum(1 for r in results if r.get('color_correct', False))
    
    print(f"总测试数: {total_tests}")
    if total_tests > 0:
        print(f"类型识别准确率: {type_correct_count}/{total_tests} ({type_correct_count/total_tests*100:.1f}%)")
        print(f"颜色识别准确率: {color_correct_count}/{total_tests} ({color_correct_count/total_tests*100:.1f}%)")
    else:
        print("没有成功处理的测试图像")
    
    if type_correct_count == total_tests and color_correct_count == total_tests:
        print("🎉 所有测试通过！高级分类器工作正常。")
    else:
        print("⚠️  仍有分类错误，需要进一步优化。")
    
    return results

def test_direct_classifier():
    """直接测试高级分类器"""
    print("\n=== 直接测试高级分类器 ===")
    
    classifier = get_advanced_classifier()
    
    test_images = [
        "data/uploads/user_5_1756715906/user_5_1756715906_2a6cd59f40194899913f407c52efb5bc.webp",
        "data/uploads/user_3_1756552528/user_3_1756552528_400586b132f34e6dbcc8241bf5a3ba31.jpg", 
        "data/uploads/user_2_1756506140/user_2_1756506140_e896348dfccd472f8cabd4a95e661bd3.jpg"
    ]
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"⚠️  图像文件不存在: {image_path}")
            continue
            
        print(f"\n📸 测试图像: {image_path}")
        
        try:
            result = classifier.classify_garment(image_path, debug=True)
            print(f"分类结果: {result}")
        except Exception as e:
            print(f"❌ 分类失败: {e}")

if __name__ == "__main__":
    print("开始测试上传API使用高级分类器...")
    
    # 测试直接分类器
    test_direct_classifier()
    
    # 测试上传处理器
    asyncio.run(test_upload_with_advanced_classifier())