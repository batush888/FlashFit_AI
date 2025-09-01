#!/usr/bin/env python3
"""
测试边缘检测功能
"""

import cv2
import numpy as np
from PIL import Image
import os

def test_edge_detection():
    """
    测试边缘检测是否正常工作
    """
    print("🔧 测试边缘检测功能")
    print("=" * 40)
    
    # 测试图像路径
    test_image = "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/backend/data/static/jackets/black_blazer.png"
    
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        return
    
    # 加载图像
    image = Image.open(test_image).convert('RGB')
    print(f"图像尺寸: {image.size}")
    
    # 转换为numpy数组
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    print(f"数组形状: {img_array.shape}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    print(f"灰度图形状: {gray.shape}")
    print(f"灰度图数据类型: {gray.dtype}")
    print(f"灰度图值范围: {gray.min()} - {gray.max()}")
    
    # 边缘检测
    edges = cv2.Canny(gray, 30, 100)
    print(f"边缘图形状: {edges.shape}")
    print(f"边缘图数据类型: {edges.dtype}")
    print(f"边缘图值范围: {edges.min()} - {edges.max()}")
    
    # 计算边缘密度
    edge_pixels = np.sum(edges > 0)
    total_pixels = height * width
    edge_density = edge_pixels / total_pixels
    
    print(f"\n边缘统计:")
    print(f"  边缘像素数: {edge_pixels}")
    print(f"  总像素数: {total_pixels}")
    print(f"  边缘密度: {edge_density:.6f}")
    
    # 测试不同的Canny参数
    print(f"\n测试不同Canny参数:")
    for low, high in [(50, 150), (100, 200), (10, 50)]:
        edges_test = cv2.Canny(gray, low, high)
        edge_density_test = np.sum(edges_test > 0) / total_pixels
        print(f"  参数({low}, {high}): 边缘密度 = {edge_density_test:.6f}")

if __name__ == "__main__":
    test_edge_detection()