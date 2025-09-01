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
        
        # 获取主要颜色
        primary_color = colors[0]['name'] if colors else 'unknown'
        
        # 生成风格关键词
        result = {
            "category": category,
            "garment_type": category,  # 添加 garment_type 字段
            "category_cn": self.category_cn.get(category, category),
            "primary_color": primary_color,  # 添加 primary_color 字段
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
        
        # 1. 外套识别 - 优先处理，避免与连衣裙混淆
        # 长外套/风衣：长条形 + 高边缘密度
        if aspect_ratio > 1.2 and edge_density >= 0.12:
            return "jacket"
        
        # 短外套：正方形或稍长 + 高边缘密度
        elif aspect_ratio <= 1.2 and edge_density >= 0.13:
            return "jacket"
        
        # 2. 连衣裙识别 - 长条形但边缘密度较低（布料较柔软）
        elif aspect_ratio > 1.3 and edge_density < 0.12:
            return "dress"
        
        # 3. 裤子识别 - 长条形，边缘密度适中到高
        elif aspect_ratio > 1.1 and edge_density >= 0.15:
            return "pants"
        
        # 4. 短裤识别 - 正方形，颜色方差适中，但要避免与shirt冲突
        elif aspect_ratio == 1.0 and 15 <= color_variance < 18 and edge_density > 0.08:
            return "shorts"
        
        # 5. 裙子识别 - 正方形，颜色方差高
        elif aspect_ratio == 1.0 and color_variance >= 19:
            return "skirt"
        
        # 6. 裙子识别 - 稍宽，边缘密度适中
        elif 0.8 <= aspect_ratio < 1.0 and edge_density >= 0.1:
            return "skirt"
        
        # 7. 默认为衬衫
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
    
    def _rgb_to_lab(self, rgb):
        """Convert RGB to Lab color space for better perceptual accuracy"""
        import cv2
        import numpy as np
        r, g, b = rgb
        rgb_array = np.array([[[r, g, b]]], dtype=np.uint8)
        lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)[0][0]
        return lab.astype(float)
    
    def _rgb_to_hsv_precise(self, rgb):
        """Convert RGB to HSV with precise calculation"""
        import cv2
        import numpy as np
        r, g, b = rgb
        rgb_array = np.array([[[r, g, b]]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)[0][0]
        return hsv.astype(float)
    
    def _calculate_delta_e(self, lab1, lab2):
        """Calculate Delta E (CIE76) color difference"""
        import numpy as np
        return np.sqrt(np.sum((lab1 - lab2) ** 2))
    
    def _get_color_features(self, rgb):
        """Extract comprehensive color features from RGB"""
        import numpy as np
        r, g, b = rgb
        
        # Lab features
        lab = self._rgb_to_lab(rgb)
        L, a, b_lab = lab
        
        # HSV features
        hsv = self._rgb_to_hsv_precise(rgb)
        h, s, v = hsv
        
        # Combined feature vector
        return np.array([L, a, b_lab, h, s, v, r, g, b])
    
    def _get_color_name(self, rgb: Tuple[int, int, int], garment_type: str = "") -> str:
        """Advanced color recognition using Lab+HSV color spaces and Delta E"""
        import numpy as np
        
        r, g, b = rgb
        
        # Get comprehensive color features
        features = self._get_color_features(rgb)
        L, a, b_lab, h, s, v = features[:6]
        
        # Define reference colors using actual OpenCV Lab values
        # OpenCV Lab: L(0-255), a(0-255), b(0-255) where 128 is neutral
        reference_colors = {
             'white': {'lab': [255, 128, 128], 'rgb': [255, 255, 255]},
             'black': {'lab': [0, 128, 128], 'rgb': [0, 0, 0]},
             'gray': {'lab': [128, 128, 128], 'rgb': [128, 128, 128]},
             'beige': {'lab': [169, 137, 158], 'rgb': [194, 154, 108]},
             'khaki': {'lab': [186, 130, 146], 'rgb': [195, 176, 145]},
             'cream': {'lab': [251, 121, 150], 'rgb': [255, 253, 208]},
             'brown': {'lab': [97, 142, 155], 'rgb': [101, 67, 33]},
             'red': {'lab': [136, 208, 172], 'rgb': [255, 0, 0]},
             'blue': {'lab': [82, 169, 42], 'rgb': [0, 0, 255]},
             'green': {'lab': [222, 42, 214], 'rgb': [0, 255, 0]},
             'yellow': {'lab': [247, 107, 222], 'rgb': [255, 255, 0]},
             'orange': {'lab': [191, 152, 206], 'rgb': [255, 165, 0]},
             'purple': {'lab': [76, 187, 92], 'rgb': [128, 0, 128]},
             'pink': {'lab': [213, 152, 133], 'rgb': [255, 192, 203]}
         }
        
        current_lab = np.array([L, a, b_lab])
        
        # 1. 优先处理黑白灰（基于亮度和色度）
        if L > 90 and abs(a) < 5 and abs(b_lab) < 10:
            return "white"
        elif L < 20:
            return "black"
        elif L > 30 and L < 70 and abs(a) < 10 and abs(b_lab) < 10:
            return "gray"
        
        # 2. 使用Delta E进行精确颜色匹配
        min_delta_e = float('inf')
        best_color = 'gray'
        
        # 特别处理pink检测，防止与beige混淆
        pink_lab = np.array(reference_colors['pink']['lab'])
        pink_delta_e = self._calculate_delta_e(current_lab, pink_lab)
        
        # Pink detection: 粉色通常有较高的红色分量和特定的色调
        if (h <= 15 or h >= 330) and s >= 20 and v >= 150:  # 红色调范围
            if pink_delta_e < 25:  # 接近粉色的Lab值
                return "pink"
        
        # 特别处理beige和khaki的区分
        beige_lab = np.array(reference_colors['beige']['lab'])
        khaki_lab = np.array(reference_colors['khaki']['lab'])
        
        beige_delta_e = self._calculate_delta_e(current_lab, beige_lab)
        khaki_delta_e = self._calculate_delta_e(current_lab, khaki_lab)
        
        # 如果在beige/khaki的范围内，使用更精确的判断
        if beige_delta_e < 30 or khaki_delta_e < 30:
            # 更精确的beige/khaki区分逻辑
            
            # 1. 首先检查是否为beige系列（包括浅beige）
            if h <= 40 and s >= 10:  # 橙黄色调范围，有一定饱和度
                if v >= 150:  # 较亮的颜色
                    # 浅beige: 高亮度 + 低饱和度 + 暖色调
                    if s <= 90 and a >= -5 and b_lab >= 10:
                        return "beige"
                elif v >= 100:  # 中等亮度
                    # 标准beige: 中等亮度 + 暖色调
                    if a >= -2 and b_lab >= 12:
                        return "beige"
            
            # 2. 检查是否为khaki（偏绿的土色）
            elif 40 <= h <= 60 and s >= 8:  # 黄绿色调范围
                if a <= 5 and 8 <= b_lab <= 20:  # Lab空间中偏绿偏黄
                    return "khaki"
            
            # 3. 边界情况：使用Delta E决定
            if beige_delta_e < khaki_delta_e:
                return "beige"
            else:
                return "khaki"
        
        # 3. 对其他颜色使用Delta E匹配
        for color_name, color_data in reference_colors.items():
            if color_name in ['beige', 'khaki']:  # 已经处理过
                continue
                
            ref_lab = np.array(color_data['lab'])
            delta_e = self._calculate_delta_e(current_lab, ref_lab)
            
            if delta_e < min_delta_e:
                min_delta_e = delta_e
                best_color = color_name
        
        # 4. 基于HSV的补充判断（用于Delta E不够准确的情况）
        if min_delta_e > 30:  # Delta E差异较大时，使用HSV规则
            if s < 15:  # 低饱和度
                if v > 80:
                    return "white"
                elif v < 30:
                    return "black"
                else:
                    return "gray"
            
            # 高饱和度颜色的HSV判断
            if s > 30:
                if h < 15 or h > 345:
                    return "red"
                elif 15 <= h <= 45:
                    return "orange"
                elif 45 <= h <= 75:
                    return "yellow"
                elif 75 <= h <= 165:
                    return "green"
                elif 165 <= h <= 270:
                    return "blue"
                elif 270 <= h <= 345:
                    return "purple"
        
        # 5. 置信度检查
        if min_delta_e < 15:  # 高置信度
            return best_color
        elif min_delta_e < 30:  # 中等置信度
            return best_color
        else:  # 低置信度，返回最接近的基础颜色
            if L > 70:
                return "white" if s < 20 else best_color
            elif L < 30:
                return "black"
            else:
                return "gray" if s < 20 else best_color
    
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