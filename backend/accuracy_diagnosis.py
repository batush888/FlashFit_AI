#!/usr/bin/env python3
"""
图像分析准确性诊断工具
测试多个图像以识别分类准确性问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.classifier import get_classifier
from PIL import Image
import glob
import json

def diagnose_accuracy_issues():
    """
    诊断图像分析准确性问题
    """
    print("🔍 FlashFit AI 图像分析准确性诊断")
    print("=" * 60)
    
    # 获取分类器实例
    classifier = get_classifier()
    
    # 测试图像路径
    test_images = [
        # 用户上传的图像
        "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/uploads/user_3_1756552528/user_3_1756552528_cb9d721ca28a412eb9991a8ccc7b5f13.webp",  # 风衣
        "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/uploads/user_3_1756552528/user_3_1756552528_2d2027d690ec48ff81a6f0f0b0971840.webp",  # 未知
        "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/uploads/user_3_1756552528/user_3_1756552528_3cc376e1d1a8411685981291ec5caa9a.jpg",   # 未知
        
        # 静态测试图像
        "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/static/jackets/black_blazer.png",
        "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/static/pants/blue_jeans.png",
        "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/static/dresses/red_cocktail.png",
        "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/static/shirts/white_formal.png",
        "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/static/skirts/blue_a.png",
        "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/static/shorts/khaki_casual.png"
    ]
    
    # 预期结果（基于文件名推断）
    expected_results = {
        "black_blazer.png": {"category": "jacket", "colors": ["black"]},
        "blue_jeans.png": {"category": "pants", "colors": ["blue"]},
        "red_cocktail.png": {"category": "dress", "colors": ["red"]},
        "white_formal.png": {"category": "shirt", "colors": ["white"]},
        "blue_a.png": {"category": "skirt", "colors": ["blue"]},
        "khaki_casual.png": {"category": "shorts", "colors": ["khaki", "tan", "beige"]}
    }
    
    results = []
    accuracy_issues = []
    
    for i, image_path in enumerate(test_images, 1):
        if not os.path.exists(image_path):
            print(f"⚠️  图像 {i}: {os.path.basename(image_path)} - 文件不存在")
            continue
            
        print(f"\n📸 测试图像 {i}: {os.path.basename(image_path)}")
        print("-" * 50)
        
        try:
            # 分析图像
            result = classifier.classify_garment(image_path, debug=True)
            
            # 显示结果
            print(f"分类: {result['category_cn']} ({result['category']})")
            print(f"置信度: {result['confidence']:.1%}")
            print(f"主要颜色: {', '.join([c['name'] for c in result['colors'][:3]])}")
            
            # 检查准确性
            filename = os.path.basename(image_path)
            if filename in expected_results:
                expected = expected_results[filename]
                
                # 检查分类准确性
                category_correct = result['category'].lower() == expected['category'].lower()
                if not category_correct:
                    accuracy_issues.append({
                        "image": filename,
                        "issue": "分类错误",
                        "expected": expected['category'],
                        "actual": result['category'],
                        "confidence": result['confidence']
                    })
                    print(f"❌ 分类错误: 预期 {expected['category']}, 实际 {result['category']}")
                else:
                    print(f"✅ 分类正确: {result['category']}")
                
                # 检查颜色准确性
                detected_colors = [c['name'].lower() for c in result['colors']]
                expected_colors = [c.lower() for c in expected['colors']]
                color_match = any(ec in detected_colors for ec in expected_colors)
                
                if not color_match:
                    accuracy_issues.append({
                        "image": filename,
                        "issue": "颜色检测错误",
                        "expected_colors": expected['colors'],
                        "detected_colors": [c['name'] for c in result['colors'][:3]]
                    })
                    print(f"❌ 颜色错误: 预期 {expected['colors']}, 检测到 {detected_colors[:3]}")
                else:
                    print(f"✅ 颜色正确: 检测到预期颜色")
            
            results.append({
                "image": filename,
                "result": result,
                "features": result['features']
            })
            
        except Exception as e:
            print(f"❌ 分析失败: {str(e)}")
            accuracy_issues.append({
                "image": os.path.basename(image_path),
                "issue": "分析异常",
                "error": str(e)
            })
    
    # 总结准确性问题
    print(f"\n\n📊 准确性诊断总结")
    print("=" * 60)
    
    if accuracy_issues:
        print(f"🚨 发现 {len(accuracy_issues)} 个准确性问题:")
        for issue in accuracy_issues:
            print(f"\n• 图像: {issue['image']}")
            print(f"  问题: {issue['issue']}")
            if 'expected' in issue:
                print(f"  预期: {issue['expected']}")
                print(f"  实际: {issue['actual']}")
                print(f"  置信度: {issue.get('confidence', 'N/A')}")
            elif 'expected_colors' in issue:
                print(f"  预期颜色: {issue['expected_colors']}")
                print(f"  检测颜色: {issue['detected_colors']}")
            elif 'error' in issue:
                print(f"  错误: {issue['error']}")
    else:
        print("✅ 未发现明显的准确性问题")
    
    # 分析常见问题模式
    print(f"\n🔍 问题模式分析:")
    
    category_errors = [i for i in accuracy_issues if i['issue'] == '分类错误']
    color_errors = [i for i in accuracy_issues if i['issue'] == '颜色检测错误']
    
    if category_errors:
        print(f"• 分类错误: {len(category_errors)} 个")
        for error in category_errors:
            print(f"  - {error['expected']} → {error['actual']} (置信度: {error.get('confidence', 'N/A')})")
    
    if color_errors:
        print(f"• 颜色检测错误: {len(color_errors)} 个")
        for error in color_errors:
            print(f"  - 预期: {error['expected_colors']} vs 检测: {error['detected_colors']}")
    
    # 建议改进措施
    print(f"\n💡 建议改进措施:")
    
    if category_errors:
        print("• 分类改进:")
        print("  - 调整特征提取参数 (长宽比、边缘密度阈值)")
        print("  - 增加更多训练样本")
        print("  - 优化分类规则")
    
    if color_errors:
        print("• 颜色检测改进:")
        print("  - 扩展颜色RGB范围")
        print("  - 改进HSV颜色空间转换")
        print("  - 增加颜色聚类数量")
    
    if not accuracy_issues:
        print("• 系统运行良好，继续监控性能")
    
    return results, accuracy_issues

if __name__ == "__main__":
    diagnose_accuracy_issues()