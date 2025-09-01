#!/usr/bin/env python3
"""
测试增强版分类器 - 展示从standalone代码集成的改进
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.classifier import get_classifier
from PIL import Image

def test_enhanced_classifier():
    """
    测试增强版分类器的新功能
    """
    print("🤖 测试增强版FlashFit AI分类器")
    print("=" * 50)
    
    # 测试图片路径
    test_image = "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/uploads/user_3_1756552528/user_3_1756552528_cb9d721ca28a412eb9991a8ccc7b5f13.webp"
    
    if not os.path.exists(test_image):
        print(f"❌ 图片未找到: {test_image}")
        return
    
    print(f"📸 分析图片: {os.path.basename(test_image)}")
    
    # 获取分类器实例
    classifier = get_classifier()
    
    # 使用调试模式进行分类
    print("\n🔍 启用调试模式分析:")
    result = classifier.classify_garment(test_image, debug=True)
    
    # 显示详细结果
    print(f"\n🏷️  分类结果:")
    print(f"   服装类型: {result['category_cn']} ({result['category']})")
    print(f"   置信度: {result['confidence']:.1%}")
    
    print(f"\n🎨 主要颜色:")
    for i, color in enumerate(result['colors']):
        print(f"   {i+1}. {color.get('name_display', color['name'])}: {color['percentage']}%")
        print(f"      RGB{color['rgb']} | {color['hex']}")
    
    print(f"\n🏷️  风格关键词: {', '.join(result['keywords'])}")
    
    print(f"\n📊 技术分析:")
    print(result['explanation'])
    
    print(f"\n📈 特征详情:")
    features = result['features']
    print(f"   • 图片尺寸: {features['width']} x {features['height']}")
    print(f"   • 长宽比: {features['aspect_ratio']:.3f}")
    print(f"   • 边缘密度: {features['edge_density']:.4f}")
    print(f"   • 颜色方差: {features['color_variance']:.1f}")
    print(f"   • 颜色复杂度: {features['color_complexity']:.6f}")
    print(f"   • 亮度: {features['brightness']:.1f}")
    print(f"   • 亮度方差: {features['brightness_variance']:.1f}")
    
    return result

if __name__ == "__main__":
    print("\n" + "="*80)
    print("增强版FlashFit AI分类器 - 集成Standalone代码改进")
    print("="*80)
    print("新功能:")
    print("• 改进的特征提取 (更精确的颜色方差计算)")
    print("• 增强的风衣检测 (基于长宽比和颜色复杂度)")
    print("• 详细的调试输出")
    print("• 分类决策解释")
    print("• 扩展的颜色关键词 (tan, khaki, beige, cream)")
    print("="*80)
    
    test_enhanced_classifier()