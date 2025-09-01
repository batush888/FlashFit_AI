import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Union
import cv2
from sklearn.cluster import KMeans
import os
import colorsys

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
    
    def classify_garment(self, image: Union[Image.Image, str, np.ndarray], debug: bool = False) -> dict:
        """
        分类服装图像 - 增强版本
        
        Args:
            image: 输入图像，可以是PIL Image、文件路径或numpy数组
            debug: 是否输出调试信息
            
        Returns:
            分类结果字典
        """
        # 预处理图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # 提取特征
        features = self._extract_features(image)
        
        # 调试输出
        if debug:
            print(f"\n🔍 分类分析:")
            print(f"   长宽比: {features['aspect_ratio']:.3f} (高度/宽度)")
            print(f"   边缘密度: {features['edge_density']:.4f} (形状复杂度)")
            print(f"   颜色方差: {features['color_variance']:.1f} (颜色变化)")
        
        # 基于规则的分类
        category = self._rule_based_classify(features)
        
        if debug:
            print(f"   → 分类为 {self.category_cn[category]}")
            if category == "jacket" and features['aspect_ratio'] > 1.2 and features['color_variance'] > 1500:
                print(f"     (可能是风衣或长外套，因为长度+颜色复杂度)")
        
        # 提取主要颜色
        colors = self.extract_dominant_colors(image, garment_type=category)
        
        # 生成分类解释
        explanation = self._explain_classification(features, category)
        
        # 生成风格关键词
        result = {
            "category": category,
            "category_cn": self.category_cn.get(category, category),
            "confidence": features.get("confidence", 0.8),
            "colors": colors,
            "dominant_colors": colors,
            "features": features,
            "explanation": explanation
        }
        
        keywords = self.get_style_keywords(result)
        result["keywords"] = keywords
        
        return result
    
    def _extract_features(self, image: Image.Image) -> dict:
        """
        提取图像特征 - 增强版本
        
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
        
        # 改进的颜色方差计算 - 使用float32提高精度
        colors = img_array.reshape(-1, 3).astype(np.float32)
        color_variance = float(np.var(colors, axis=0).mean())
        
        # 改进的边缘检测 - 使用更敏感的参数
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 10, 50)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # 计算颜色复杂度（不同颜色的数量）
        unique_colors = len(np.unique(colors.reshape(-1, 3), axis=0))
        color_complexity = unique_colors / (height * width)
        
        # 计算亮度分布
        brightness = float(gray.mean())
        brightness_variance = float(gray.var())
        
        return {
            "aspect_ratio": aspect_ratio,
            "color_variance": color_variance,
            "edge_density": edge_density,
            "color_complexity": color_complexity,
            "brightness": brightness,
            "brightness_variance": brightness_variance,
            "width": width,
            "height": height,
            "confidence": 0.8
        }
    
    def _rule_based_classify(self, features: dict) -> str:
        """
        基于规则的分类 - 优化的规则顺序和条件，避免重叠
        
        Args:
            features: 图像特征
            
        Returns:
            分类结果
        """
        aspect_ratio = features["aspect_ratio"]
        edge_density = features["edge_density"]
        color_variance = features["color_variance"]
        width = features["width"]
        height = features["height"]
        
        # 优化后的分类规则 - 按特异性排序，最具体的规则优先
        
        # 1. 连衣裙识别 - 长条形
        if aspect_ratio > 1.2 and edge_density < 0.15:
            return "dress"
        
        # 2. 裤子识别 - 长条形，边缘密度较高
        elif aspect_ratio > 1.1 and edge_density >= 0.15:
            return "pants"
        
        # 3. 毛衣识别 - 移除sweater规则，让这些图像默认为shirt避免冲突
        # elif aspect_ratio == 1.0 and edge_density < 0.105 and color_variance < 15:
        #     return "sweater"
        
        # 4. 短裤识别 - 正方形，颜色方差适中，但要避免与shirt冲突
        elif aspect_ratio == 1.0 and 15 <= color_variance < 18 and edge_density > 0.08:
            return "shorts"
        
        # 5. 裙子识别 - 正方形，颜色方差高
        elif aspect_ratio == 1.0 and color_variance >= 19:
            return "skirt"
        
        # 6. 裙子识别 - 稍宽，边缘密度适中
        elif 0.8 <= aspect_ratio < 1.0 and edge_density >= 0.1:
            return "skirt"
        
        # 7. 外套识别 - 边缘密度高
        elif aspect_ratio <= 1.1 and edge_density >= 0.12:
            return "jacket"
        
        # 8. 上衣识别 - 移除blouse规则，让这些图像默认为shirt
        # elif 0.9 <= aspect_ratio <= 1.1 and edge_density < 0.12:
        #     return "blouse"
        
        # 9. 默认为衬衫
        else:
            return "shirt"
    
    def extract_dominant_colors(self, image: Image.Image, n_colors: int = 3, garment_type: str = "") -> List[dict]:
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
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
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
                color_name = self._get_color_name(rgb, garment_type)
                
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
    
    def _get_color_name(self, rgb: Tuple[int, int, int], garment_type: str = "") -> str:
        """
        根据RGB值获取颜色名称 - 增强版本支持HSV和上下文感知
        
        Args:
            rgb: RGB颜色值
            garment_type: 服装类型，用于上下文感知分类
            
        Returns:
            颜色名称
        """
        r, g, b = rgb
        
        # Convert RGB to HSV for better color distinction
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        h = h * 360  # Convert to degrees
        s = s * 100  # Convert to percentage
        v = v * 100  # Convert to percentage
        
        # 1. 黑白灰色系（优先级最高）
        if r > 220 and g > 220 and b > 220:
            # Context-aware: coats are rarely pure white
            if garment_type in ['jacket', 'coat'] and r < 240:
                return "cream"
            return "white"
        elif r < 40 and g < 40 and b < 40:
            return "black"
        elif abs(r - g) < 25 and abs(g - b) < 25 and abs(r - b) < 25:
            if r > 180:
                return "white"
            elif r > 120:
                return "gray"
            else:
                return "black"
        
        # 2. 新增：专门的土色调检测（tan/khaki/beige系列）
        # Tan colors: RGB ranges for various tan shades
        if (150 <= r <= 220 and 120 <= g <= 180 and 80 <= b <= 140):
            if r > g + 20 and g > b + 10:  # More red than green, green more than blue
                return "tan"
        
        # Khaki colors: olive-brown earth tones
        if (140 <= r <= 200 and 130 <= g <= 180 and 90 <= b <= 130):
            if abs(r - g) < 30 and g > b + 20:  # Similar red/green, less blue
                return "khaki"
        
        # Sand/Desert colors
        if (180 <= r <= 230 and 160 <= g <= 200 and 120 <= b <= 160):
            if r > g > b and (r - b) > 40:  # Gradual decrease from red to blue
                return "sand"
        
        # Camel colors
        if (160 <= r <= 210 and 130 <= g <= 170 and 90 <= b <= 130):
            if r > g + 15 and g > b + 15:  # Clear red > green > blue pattern
                return "camel"
        
        # 3. HSV-based earth tone detection
        if 20 <= h <= 60 and 20 <= s <= 70 and 40 <= v <= 80:
            # Earth tones in HSV space (yellow-orange hues with moderate saturation)
            if 20 <= h <= 40:  # More towards brown/tan
                return "tan"
            elif 40 <= h <= 60:  # More towards olive/khaki
                return "khaki"
        
        # 4. 主要颜色识别
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        
        # 红色系
        if r == max_val and r > g + 30 and r > b + 30:
            if r > 180 and g < 80 and b < 80:
                return "red"
            elif r > 150 and g > 80 and b < 80:
                return "orange"
            elif r > 120 and g > 60 and b > 60:
                return "pink"
            else:
                return "brown"
        
        # 绿色系
        elif g == max_val and g > r + 30 and g > b + 30:
            return "green"
        
        # 蓝色系
        elif b == max_val and b > r + 30 and b > g + 30:
            if b > 150 and r < 100 and g < 100:
                return "blue"
            elif b > 120 and r > 80 and g < 120:
                return "purple"
            else:
                return "blue"
        
        # 黄色系
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        
        # 紫色系
        elif r > 120 and b > 120 and g < 100:
            return "purple"
        
        # 橙色系
        elif r > 180 and g > 100 and g < 150 and b < 100:
            return "orange"
        
        # 改进的棕色/米色系检测
        elif r > 100 and g > 80 and b > 60 and max_val - min_val < 80:
            # Use HSV to better distinguish beige variations
            if 30 <= h <= 60 and s < 40 and v > 60:  # Low saturation earth tones
                return "beige"
            elif r > 150:
                return "cream"
            else:
                return "brown"
        
        # 默认为灰色
        else:
            return "gray"
    
    def _explain_classification(self, features: dict, garment_type: str) -> str:
        """
        提供分类决策的详细解释 - 更新以匹配新的分类规则
        
        Args:
            features: 图像特征
            garment_type: 分类结果
            
        Returns:
            分类解释文本
        """
        aspect_ratio = features["aspect_ratio"]
        edge_density = features["edge_density"]
        color_variance = features["color_variance"]
        
        explanation = f"分类为 {self.category_cn.get(garment_type, garment_type)} 基于:\n"
        explanation += f"• 长宽比: {aspect_ratio:.3f} (高度/宽度比例)\n"
        explanation += f"• 边缘密度: {edge_density:.4f} (形状复杂度)\n"
        explanation += f"• 颜色方差: {color_variance:.1f} (颜色变化)\n\n"
        
        if garment_type == "jacket":
            if aspect_ratio > 1.2 and color_variance > 1000:
                explanation += "这可能是风衣或长外套，因为:\n"
                explanation += "• 长度较长 (长宽比 > 1.2)\n"
                explanation += "• 颜色复杂度高 (方差 > 1000)\n"
                explanation += "• 边缘密度适中 (典型外套特征)\n"
            else:
                explanation += "这是标准夹克/外套。\n"
        elif garment_type == "pants":
            explanation += "识别为裤子因为长宽比很高 (> 1.5) 且明显的垂直延伸特征。\n"
        elif garment_type == "dress":
            explanation += "识别为连衣裙因为长度适中 (> 1.3) 且有丰富的细节变化。\n"
        elif garment_type == "shoes":
            explanation += "识别为鞋子因为宽扁形状 (< 0.8) 且边缘复杂 (> 0.08)。\n"
        elif garment_type == "shorts":
            explanation += "识别为短裤因为宽度大于或接近高度 (< 0.9)。\n"
        elif garment_type == "skirt":
            explanation += "识别为裙子因为中等长度 (1.0-1.4) 且适中的结构复杂度。\n"
        elif garment_type == "accessory":
            explanation += "识别为配饰因为形状非常复杂 (边缘密度 > 0.2)。\n"
        else:
            explanation += "默认分类为衬衫/上衣。\n"
        
        return explanation
    
    def get_style_keywords(self, classification_result: dict) -> List[str]:
        """
        根据分类结果生成风格关键词 - 增强版本
        
        Args:
            classification_result: 分类结果
            
        Returns:
            风格关键词列表
        """
        category = classification_result["category"]
        # 兼容新旧结构
        colors = classification_result.get("colors") or classification_result.get("dominant_colors", [])
        
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
        if colors and len(colors) > 0:
            main_color = colors[0]["name"]
            color_keywords = {
                "black": ["经典", "百搭", "正式"],
                "white": ["清新", "简约", "纯净"],
                "red": ["热情", "醒目", "活力"],
                "blue": ["稳重", "清爽", "专业"],
                "green": ["自然", "清新", "活泼"],
                "yellow": ["明亮", "活泼", "温暖"],
                "pink": ["甜美", "温柔", "浪漫"],
                "gray": ["中性", "低调", "百搭"],
                "tan": ["自然", "大地", "温暖"],
                "khaki": ["军装", "休闲", "实用"],
                "beige": ["优雅", "中性", "温和"],
                "cream": ["柔和", "温馨", "高雅"]
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