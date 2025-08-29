import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Union
import cv2
from sklearn.cluster import KMeans
import os

class GarmentClassifier:
    """
    服装分类器 - 识别服装类型和主要颜色
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
        
        # 颜色名称映射
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
            "beige": "米色"
        }
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 初始化简单分类器（基于规则的方法作为MVP）
        self._init_rule_based_classifier()
        
        print("服装分类器初始化完成")
    
    def _init_rule_based_classifier(self):
        """
        初始化基于规则的分类器（MVP版本）
        """
        # 基于颜色和形状特征的简单规则
        self.classification_rules = {
            "aspect_ratio_thresholds": {
                "tall": 1.5,  # 高度/宽度 > 1.5 (裤子、连衣裙)
                "wide": 0.7   # 高度/宽度 < 0.7 (鞋子)
            },
            "color_based_hints": {
                "black_shoes_threshold": 0.6,  # 黑色占比超过60%可能是鞋子
                "colorful_accessory_threshold": 0.8  # 颜色丰富可能是配饰
            }
        }
    
    def classify_garment(self, image: Union[Image.Image, str, np.ndarray]) -> dict:
        """
        分类服装类型
        
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
        
        # 获取图像特征
        features = self._extract_features(image)
        
        # 基于规则的分类（MVP版本）
        category = self._rule_based_classify(features)
        
        # 提取主要颜色
        dominant_colors = self.extract_dominant_colors(image)
        
        return {
            "category": category,
            "category_cn": self.category_cn.get(category, category),
            "confidence": features.get("confidence", 0.7),  # MVP默认置信度
            "dominant_colors": dominant_colors,
            "features": features
        }
    
    def _extract_features(self, image: Image.Image) -> dict:
        """
        提取图像特征
        
        Args:
            image: PIL图像
            
        Returns:
            特征字典
        """
        # 转换为numpy数组进行分析
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # 计算长宽比
        aspect_ratio = height / width
        
        # 计算颜色分布
        colors = img_array.reshape(-1, 3)
        color_variance = np.var(colors, axis=0).mean()
        
        # 计算边缘密度（简单的形状复杂度指标）
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        return {
            "aspect_ratio": aspect_ratio,
            "color_variance": color_variance,
            "edge_density": edge_density,
            "width": width,
            "height": height,
            "confidence": 0.7  # MVP默认置信度
        }
    
    def _rule_based_classify(self, features: dict) -> str:
        """
        基于规则的分类方法（MVP版本）
        
        Args:
            features: 图像特征
            
        Returns:
            分类结果
        """
        aspect_ratio = features["aspect_ratio"]
        edge_density = features["edge_density"]
        color_variance = features["color_variance"]
        
        # 简单的规则分类
        if aspect_ratio > 1.8 and edge_density < 0.1:
            return "pants"  # 长条形，边缘简单 -> 裤子
        elif aspect_ratio > 1.5 and color_variance > 1000:
            return "dress"  # 较长，颜色丰富 -> 连衣裙
        elif aspect_ratio < 0.6:
            return "shoes"  # 宽扁形 -> 鞋子
        elif aspect_ratio < 1.2 and edge_density > 0.15:
            return "jacket"  # 方形，边缘复杂 -> 外套
        elif aspect_ratio > 1.3 and aspect_ratio < 1.8:
            return "skirt"  # 中等长度 -> 裙子
        elif edge_density > 0.2 or color_variance > 2000:
            return "accessory"  # 复杂形状或颜色丰富 -> 配饰
        else:
            return "shirt"  # 默认分类
    
    def extract_dominant_colors(self, image: Image.Image, n_colors: int = 3) -> List[dict]:
        """
        提取图像的主要颜色
        
        Args:
            image: PIL图像
            n_colors: 提取的颜色数量
            
        Returns:
            颜色信息列表
        """
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 重塑为像素列表
        pixels = img_array.reshape(-1, 3)
        
        # 使用K-means聚类找到主要颜色
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = []
        for i, color in enumerate(kmeans.cluster_centers_):
            # 计算该颜色的占比
            labels = kmeans.labels_
            percentage = np.sum(labels == i) / len(labels)
            
            # 转换为整数RGB值
            rgb = (int(color[0]), int(color[1]), int(color[2]))
            
            # 获取颜色名称
            color_name = self._get_color_name(rgb)
            
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
    
    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """
        根据RGB值获取颜色名称
        
        Args:
            rgb: RGB颜色值
            
        Returns:
            颜色名称
        """
        r, g, b = rgb
        
        # 简单的颜色分类规则
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
            if r > 150:
                return "white"
            elif r > 100:
                return "gray"
            else:
                return "black"
        elif r > g + 50 and r > b + 50:
            if r > 200 and g < 100:
                return "red"
            elif r > 150 and g > 100:
                return "orange"
            else:
                return "brown"
        elif g > r + 50 and g > b + 50:
            return "green"
        elif b > r + 50 and b > g + 50:
            return "blue"
        elif r > 150 and g > 100 and b < 100:
            return "yellow"
        elif r > 150 and b > 100 and g < 150:
            return "purple"
        elif r > 150 and g > 100 and b > 100:
            return "pink"
        elif r > 100 and g > 80 and b > 60:
            return "beige"
        else:
            return "gray"
    
    def get_style_keywords(self, classification_result: dict) -> List[str]:
        """
        根据分类结果生成风格关键词
        
        Args:
            classification_result: 分类结果
            
        Returns:
            风格关键词列表
        """
        category = classification_result["category"]
        colors = classification_result["dominant_colors"]
        
        keywords = []
        
        # 基于类别的关键词
        category_keywords = {
            "shirt": ["休闲", "商务", "日常"],
            "pants": ["修身", "舒适", "百搭"],
            "jacket": ["时尚", "保暖", "外搭"],
            "dress": ["优雅", "女性", "正式"],
            "skirt": ["甜美", "淑女", "轻盈"],
            "shoes": ["舒适", "搭配", "实用"],
            "accessory": ["点缀", "个性", "精致"]
        }
        
        keywords.extend(category_keywords.get(category, []))
        
        # 基于颜色的关键词
        if colors:
            main_color = colors[0]["name"]
            color_keywords = {
                "black": ["经典", "百搭", "正式"],
                "white": ["清新", "简约", "纯净"],
                "red": ["热情", "醒目", "活力"],
                "blue": ["稳重", "清爽", "专业"],
                "green": ["自然", "清新", "活泼"],
                "yellow": ["明亮", "活泼", "温暖"],
                "pink": ["甜美", "温柔", "浪漫"],
                "gray": ["中性", "低调", "百搭"]
            }
            keywords.extend(color_keywords.get(main_color, []))
        
        return list(set(keywords))  # 去重

# 全局分类器实例
_classifier = None

def get_classifier() -> GarmentClassifier:
    """
    获取全局分类器实例（单例模式）
    """
    global _classifier
    if _classifier is None:
        _classifier = GarmentClassifier()
    return _classifier