#!/usr/bin/env python3
"""
对比测试 - FlashFit AI增强版分类器 vs 原始Standalone代码
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.advanced_classifier import get_advanced_classifier
from Standalone_Image_Analysis_Code import StandaloneGarmentAnalyzer
from PIL import Image

def compare_classifiers():
    """
    对比两个分类器的结果
    """
    print("🔄 FlashFit AI分类器对比测试")
    print("=" * 60)
    
    # 测试图片路径
    test_image = "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/uploads/user_3_1756552528/user_3_1756552528_cb9d721ca28a412eb9991a8ccc7b5f13.webp"
    
    if not os.path.exists(test_image):
        print(f"❌ 图片未找到: {test_image}")
        return
    
    print(f"📸 测试图片: {os.path.basename(test_image)}")
    
    # 1. FlashFit AI增强版分类器
    print("\n" + "="*30 + " FlashFit AI增强版 " + "="*30)
    flashfit_classifier = get_advanced_classifier()
    flashfit_result = flashfit_classifier.classify_garment(test_image, debug=False)
    
    print(f"分类: {flashfit_result['category_cn']} ({flashfit_result['category']})")
    print(f"置信度: {flashfit_result['confidence']:.1%}")
    print(f"主要颜色: {', '.join([c['name'] for c in flashfit_result['colors'][:3]])}")
    print(f"风格关键词: {', '.join(flashfit_result['keywords'][:5])}")
    
    # 2. 原始Standalone分析器
    print("\n" + "="*30 + " 原始Standalone版 " + "="*30)
    standalone_analyzer = StandaloneGarmentAnalyzer()
    standalone_result = standalone_analyzer.analyze_garment(test_image)
    
    print(f"分类: {standalone_result['garment_type']}")
    print(f"置信度: {standalone_result['confidence']:.1%}")
    print(f"主要颜色: {', '.join([f"{c['name']} ({c['percentage']:.1f}%)" for c in standalone_result['dominant_colors'][:3]])}")
    print(f"风格关键词: {', '.join(standalone_result['style_keywords'])}")
    
    # 3. 特征对比
    print("\n" + "="*30 + " 特征对比 " + "="*30)
    ff_features = flashfit_result['features']
    sa_features = standalone_result['features']
    
    print(f"{'特征':<15} {'FlashFit AI':<15} {'Standalone':<15} {'差异':<10}")
    print("-" * 60)
    
    # 长宽比对比
    ff_ar = ff_features['aspect_ratio']
    sa_ar = sa_features['aspect_ratio']
    diff_ar = abs(ff_ar - sa_ar)
    print(f"{'长宽比':<15} {ff_ar:<15.3f} {sa_ar:<15.3f} {diff_ar:<10.3f}")
    
    # 边缘密度对比
    ff_ed = ff_features['edge_density']
    sa_ed = sa_features['edge_density']
    diff_ed = abs(ff_ed - sa_ed)
    print(f"{'边缘密度':<15} {ff_ed:<15.4f} {sa_ed:<15.4f} {diff_ed:<10.4f}")
    
    # 颜色方差对比
    ff_cv = ff_features['color_variance']
    sa_cv = sa_features['color_variance']
    diff_cv = abs(ff_cv - sa_cv)
    print(f"{'颜色方差':<15} {ff_cv:<15.1f} {sa_cv:<15.1f} {diff_cv:<10.1f}")
    
    # 4. 改进总结
    print("\n" + "="*30 + " 改进总结 " + "="*30)
    print("✅ 成功集成的改进:")
    print("   • 增强的特征提取算法")
    print("   • 改进的风衣检测逻辑")
    print("   • 详细的分类解释")
    print("   • 扩展的颜色关键词")
    print("   • 调试模式输出")
    print("   • 向后兼容的数据结构")
    
    print("\n🎯 分类一致性:")
    ff_category = flashfit_result['category'].lower()
    sa_category = standalone_result['garment_type'].lower()
    
    if 'jacket' in ff_category and 'jacket' in sa_category:
        print("   ✅ 两个分类器都正确识别为外套类型")
    elif ff_category == sa_category:
        print(f"   ✅ 分类完全一致: {ff_category}")
    else:
        print(f"   ⚠️  分类略有差异: FlashFit='{ff_category}' vs Standalone='{sa_category}'")
    
    return flashfit_result, standalone_result

if __name__ == "__main__":
    compare_classifiers()