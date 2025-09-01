#!/usr/bin/env python3
"""
Improved Fashion Classifier with better color detection and garment classification
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import numpy as np
from typing import List, Tuple, Union, Dict, Any
import cv2
from sklearn.cluster import KMeans
import os
from numpy.typing import NDArray

class ImprovedGarmentClassifier:
    """
    改进的服装分类器 - 更准确的颜色识别和服装类型分类
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 服装类别
        self.garment_categories = [
            "shirt",      # 衬衫
            "pants",      # 裤子
            "jacket",     # 外套
            "dress",      # 连衣裙
            "skirt",      # 裙子
            "shoes",      # 鞋子
            "accessory"   # 配饰
        ]
        
        # 中文类别映射
        self.category_cn = {
            "shirt": "衬衫",
            "pants": "裤子", 
            "jacket": "外套",
            "dress": "连衣裙",
            "skirt": "裙子",
            "shoes": "鞋子",
            "accessory": "配饰"
        }
        
        # 改进的颜色名称映射
        self.color_names = {
            "red": "红色",
            "blue": "蓝色",
            "green": "绿色",
            "yellow": "黄色",
            "orange": "橙色",
            "purple": "紫色",
            "pink": "粉色",
            "brown": "棕色",
            "black": "黑色",
            "white": "白色",
            "gray": "灰色",
            "beige": "米色",
            "navy": "深蓝色",
            "tan": "棕褐色",
            "cream": "奶油色"
        }
        
        print("改进的服装分类器初始化完成")
    
    def remove_background(self, image: Image.Image, threshold: int = 240) -> Image.Image:
        """
        移除白色背景，专注于服装区域
        
        Args:
            image: 输入图像
            threshold: 白色背景阈值
            
        Returns:
            移除背景后的图像
        """
        img_array = np.array(image)
        
        # 创建掩码：识别白色背景
        # 白色背景通常是RGB值都很高且相近的区域
        white_mask = (
            (img_array[:, :, 0] > threshold) & 
            (img_array[:, :, 1] > threshold) & 
            (img_array[:, :, 2] > threshold) &
            (np.abs(img_array[:, :, 0].astype(int) - img_array[:, :, 1].astype(int)) < 15) &
            (np.abs(img_array[:, :, 1].astype(int) - img_array[:, :, 2].astype(int)) < 15)
        )
        
        # 创建非背景掩码
        garment_mask = ~white_mask
        
        # 如果几乎整个图像都是背景，降低阈值
        if np.sum(garment_mask) < img_array.size * 0.1:  # 如果非背景区域小于10%
            threshold = 220
            white_mask = (
                (img_array[:, :, 0] > threshold) & 
                (img_array[:, :, 1] > threshold) & 
                (img_array[:, :, 2] > threshold) &
                (np.abs(img_array[:, :, 0].astype(int) - img_array[:, :, 1].astype(int)) < 20) &
                (np.abs(img_array[:, :, 1].astype(int) - img_array[:, :, 2].astype(int)) < 20)
            )
            garment_mask = ~white_mask
        
        return garment_mask
    
    def extract_dominant_colors_improved(self, image: Image.Image, n_colors: int = 3) -> List[dict]:
        """
        改进的主要颜色提取 - 忽略背景
        
        Args:
            image: PIL图像
            n_colors: 提取的颜色数量
            
        Returns:
            颜色信息列表
        """
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 移除白色背景
        garment_mask = self.remove_background(image)
        
        # 只使用非背景像素
        garment_pixels = img_array[garment_mask]
        
        if len(garment_pixels) < 100:  # 如果服装像素太少，使用全部像素
            pixels = img_array.reshape(-1, 3)
            print("警告：检测到的服装区域太小，使用全部像素")
        else:
            pixels = garment_pixels
            print(f"使用 {len(pixels)} 个服装像素进行颜色分析（忽略背景）")
        
        # 使用K-means聚类找到主要颜色
        n_clusters = min(n_colors, len(pixels) // 50)  # 确保有足够的像素
        if n_clusters < 1:
            n_clusters = 1
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(pixels)
        
        colors = []
        labels = kmeans.labels_
        if labels is not None:
            for i, color in enumerate(kmeans.cluster_centers_):
                # 计算该颜色的占比
                percentage = np.sum(labels == i) / len(labels)
                
                # 转换为整数RGB值
                rgb = (int(color[0]), int(color[1]), int(color[2]))
                
                # 获取颜色名称
                color_name = self._get_color_name_improved(rgb)
                
                colors.append({
                    "rgb": rgb,
                    "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                    "name": color_name,
                    "name_cn": self.color_names.get(color_name, color_name),
                    "percentage": round(percentage * 100, 1)
                })
        
        # 按占比排序
        colors.sort(key=lambda x: x["percentage"], reverse=True)
        
        return colors
    
    def _get_color_name_improved(self, rgb: Tuple[int, int, int]) -> str:
        """
        改进的颜色名称识别
        
        Args:
            rgb: RGB颜色值
            
        Returns:
            颜色名称
        """
        r, g, b = rgb
        
        # 改进的颜色分类规则
        # 1. 黑白灰色系
        if r > 240 and g > 240 and b > 240:
            return "white"
        elif r < 30 and g < 30 and b < 30:
            return "black"
        elif abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:
            if r > 200:
                return "white"
            elif r > 140:
                return "gray"
            elif r > 80:
                return "gray"
            else:
                return "black"
        
        # 2. 蓝色系 - 改进检测
        elif b > r + 20 and b > g + 10:
            if b > 100 and r < 80 and g < 80:
                if b > 150:
                    return "blue"
                else:
                    return "navy"  # 深蓝色
            elif b > 120 and r > 80:
                return "purple"
            else:
                return "blue"
        
        # 3. 红色系
        elif r > g + 30 and r > b + 30:
            if r > 180 and g < 80 and b < 80:
                return "red"
            elif r > 150 and g > 80 and b < 80:
                return "orange"
            elif r > 120 and g > 60 and b > 60:
                return "pink"
            else:
                return "brown"
        
        # 4. 绿色系
        elif g > r + 30 and g > b + 30:
            return "green"
        
        # 5. 黄色系
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        
        # 6. 紫色系
        elif r > 120 and b > 120 and g < 100:
            return "purple"
        
        # 7. 橙色系
        elif r > 180 and g > 100 and g < 150 and b < 100:
            return "orange"
        
        # 8. 棕色/米色系 - 改进检测
        elif r > 80 and g > 60 and b > 40:
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            if max_val - min_val < 60:  # 颜色相近
                if r > 160 and g > 140 and b > 100:
                    return "beige"  # 米色
                elif r > 120 and g > 100 and b > 70:
                    return "tan"    # 棕褐色
                else:
                    return "brown"  # 棕色
        
        # 9. 深色系
        elif max(r, g, b) < 100:
            if b > r and b > g:
                return "navy"
            else:
                return "black"
        
        # 默认为灰色
        else:
            return "gray"
    
    def _extract_features_improved(self, image: Image.Image) -> dict:
        """
        改进的特征提取
        
        Args:
            image: PIL图像
            
        Returns:
            特征字典
        """
        # 基本尺寸信息
        width, height = image.size
        aspect_ratio = height / width if width > 0 else 1.0
        
        # 移除背景后计算特征
        garment_mask = self.remove_background(image)
        img_array = np.array(image)
        
        # 计算服装区域的边界框
        if np.sum(garment_mask) > 0:
            rows, cols = np.where(garment_mask)
            if len(rows) > 0 and len(cols) > 0:
                top, bottom = rows.min(), rows.max()
                left, right = cols.min(), cols.max()
                
                # 服装区域的实际宽高比
                garment_width = right - left
                garment_height = bottom - top
                garment_aspect_ratio = garment_height / garment_width if garment_width > 0 else aspect_ratio
            else:
                garment_aspect_ratio = aspect_ratio
        else:
            garment_aspect_ratio = aspect_ratio
        
        # 边缘检测（只在服装区域）
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 只计算服装区域的边缘密度
        if np.sum(garment_mask) > 0:
            garment_edges = edges[garment_mask]
            edge_density = np.sum(garment_edges > 0) / len(garment_edges) if len(garment_edges) > 0 else 0
        else:
            edge_density = np.sum(edges > 0) / edges.size
        
        # 颜色方差（只在服装区域）
        if np.sum(garment_mask) > 0:
            garment_pixels = img_array[garment_mask]
            color_variance = np.var(garment_pixels) if len(garment_pixels) > 0 else 0
        else:
            pixels = img_array.reshape(-1, 3)
            color_variance = np.var(pixels)
        
        return {
            "width": width,
            "height": height,
            "aspect_ratio": garment_aspect_ratio,  # 使用服装区域的宽高比
            "edge_density": edge_density,
            "color_variance": color_variance,
            "garment_area_ratio": np.sum(garment_mask) / garment_mask.size if garment_mask.size > 0 else 1.0
        }
    
    def _rule_based_classify_improved(self, features: dict) -> str:
        """
        改进的基于规则的分类方法
        
        Args:
            features: 图像特征
            
        Returns:
            分类结果
        """
        aspect_ratio = features["aspect_ratio"]
        edge_density = features["edge_density"]
        color_variance = features["color_variance"]
        garment_area_ratio = features.get("garment_area_ratio", 1.0)
        
        print(f"分类特征: aspect_ratio={aspect_ratio:.2f}, edge_density={edge_density:.3f}, color_variance={color_variance:.1f}")
        
        # 改进的分类规则
        # 1. 鞋子识别 - 通常宽扁，边缘复杂
        if aspect_ratio < 0.9 and edge_density > 0.15:
            return "shoes"
        
        # 2. 裤子识别 - 长条形，改进检测
        elif aspect_ratio > 1.8 or (aspect_ratio > 1.5 and edge_density < 0.12):
            return "pants"
        
        # 3. 连衣裙识别 - 较长，通常有更多细节
        elif aspect_ratio > 1.6 and (color_variance > 1000 or edge_density > 0.20):
            return "dress"
        
        # 4. 外套识别 - 方形到略长，边缘复杂（口袋、拉链等）
        elif aspect_ratio >= 0.8 and aspect_ratio <= 1.3 and edge_density > 0.18:
            return "jacket"
        
        # 5. 裙子识别 - 中等长度，但比裤子短
        elif aspect_ratio > 1.2 and aspect_ratio <= 1.7 and edge_density < 0.18 and color_variance < 800:
            return "skirt"
        
        # 6. 配饰识别 - 复杂形状或非常丰富的颜色
        elif edge_density > 0.30 or color_variance > 2000 or garment_area_ratio < 0.3:
            return "accessory"
        
        # 7. 衬衫识别 - 默认类别，通常是方形到略长
        else:
            return "shirt"
    
    def classify_garment(self, image: Union[Image.Image, str, np.ndarray]) -> dict:
        """
        改进的服装分类
        
        Args:
            image: 输入图像
            
        Returns:
            分类结果字典
        """
        # 处理输入图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # 获取改进的图像特征
        features = self._extract_features_improved(image)
        
        # 改进的分类
        category = self._rule_based_classify_improved(features)
        
        # 提取改进的主要颜色
        dominant_colors = self.extract_dominant_colors_improved(image)
        
        return {
            "category": category,
            "category_cn": self.category_cn.get(category, category),
            "confidence": 0.85,  # 改进后的置信度
            "dominant_colors": dominant_colors,
            "features": features
        }
    
    def get_style_keywords(self, classification_result: dict) -> list:
        """
        生成风格关键词
        
        Args:
            classification_result: 分类结果字典
            
        Returns:
            风格关键词列表
        """
        keywords = []
        
        # 添加服装类型关键词
        category = classification_result.get('category', '')
        if category:
            keywords.append(category)
        
        # 添加颜色关键词
        colors = classification_result.get('dominant_colors', [])
        for color in colors[:2]:  # 只取前两个主要颜色
            if color.get('percentage', 0) > 10:  # 只有占比超过10%的颜色才加入关键词
                keywords.append(color['name'])
        
        # 根据服装类型添加特定风格关键词
        style_mapping = {
            'dress': ['elegant', 'feminine', 'formal'],
            'shirt': ['casual', 'professional', 'versatile'],
            'pants': ['comfortable', 'practical', 'everyday'],
            'skirt': ['feminine', 'stylish', 'versatile'],
            'jacket': ['layering', 'outerwear', 'structured'],
            'shoes': ['footwear', 'accessory', 'style'],
            'accessory': ['decorative', 'accent', 'detail']
        }
        
        if category in style_mapping:
            keywords.extend(style_mapping[category])
        
        return list(set(keywords))  # 去重

# 全局改进分类器实例
_improved_classifier = None

def get_improved_classifier() -> ImprovedGarmentClassifier:
    """获取改进的服装分类器实例（单例模式）"""
    global _improved_classifier
    if _improved_classifier is None:
        _improved_classifier = ImprovedGarmentClassifier()
    return _improved_classifier