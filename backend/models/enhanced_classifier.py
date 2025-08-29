import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Union, Optional
import cv2
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from .clip_model import get_clip_model
from .classifier import GarmentClassifier

class EnhancedGarmentClassifier:
    """
    增强版服装分类器 - 结合CLIP特征和传统CV特征的混合模型
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化基础分类器和CLIP模型
        self.base_classifier = GarmentClassifier()
        self.clip_model = get_clip_model()
        
        # 服装类别和置信度阈值
        self.categories = [
            "shirt", "pants", "jacket", "dress", 
            "skirt", "shoes", "accessory", "top", "bottom"
        ]
        
        self.category_cn = {
            "shirt": "衬衫",
            "pants": "裤子", 
            "jacket": "外套",
            "dress": "连衣裙",
            "skirt": "裙子",
            "shoes": "鞋子",
            "accessory": "配饰",
            "top": "上装",
            "bottom": "下装"
        }
        
        # 初始化机器学习模型
        self.ml_classifier = None
        self.feature_scaler = StandardScaler()
        self.confidence_threshold = 0.6
        
        # 预定义的文本描述用于CLIP分类
        self.text_descriptions = {
            "shirt": ["a shirt", "a t-shirt", "a blouse", "a top"],
            "pants": ["pants", "trousers", "jeans", "bottoms"],
            "jacket": ["a jacket", "a coat", "outerwear", "a blazer"],
            "dress": ["a dress", "a gown", "a frock"],
            "skirt": ["a skirt", "a mini skirt", "a long skirt"],
            "shoes": ["shoes", "sneakers", "boots", "sandals"],
            "accessory": ["an accessory", "jewelry", "a bag", "a hat"]
        }
        
        print("增强版服装分类器初始化完成")
    
    def extract_enhanced_features(self, image: Union[Image.Image, str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        提取增强特征，包括CLIP特征和传统CV特征
        
        Args:
            image: 输入图像
            
        Returns:
            特征字典
        """
        # 处理输入图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        features = {}
        
        # 1. CLIP视觉特征
        clip_features = self.clip_model.encode_image(image)
        features['clip'] = clip_features
        
        # 2. 传统CV特征
        cv_features = self._extract_cv_features(image)
        features['cv'] = cv_features
        
        # 3. 颜色特征
        color_features = self._extract_color_features(image)
        features['color'] = color_features
        
        # 4. 形状特征
        shape_features = self._extract_shape_features(image)
        features['shape'] = shape_features
        
        return features
    
    def _extract_cv_features(self, image: Image.Image) -> np.ndarray:
        """
        提取传统计算机视觉特征
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # 转换为灰度图
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # 1. 基本几何特征
        aspect_ratio = height / width
        area = height * width
        features.extend([aspect_ratio, area / 10000])  # 归一化面积
        
        # 2. 边缘特征
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        edge_variance = np.var(edges)
        features.extend([edge_density, edge_variance / 1000])  # 归一化方差
        
        # 3. 纹理特征 (LBP-like)
        texture_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(texture_variance / 1000)
        
        # 4. 轮廓特征
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            contour_perimeter = cv2.arcLength(largest_contour, True)
            if contour_perimeter > 0:
                compactness = 4 * np.pi * contour_area / (contour_perimeter ** 2)
            else:
                compactness = 0
            features.append(compactness)
        else:
            features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_color_features(self, image: Image.Image) -> np.ndarray:
        """
        提取颜色特征
        """
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3)
        
        features = []
        
        # 1. 颜色统计特征
        color_mean = np.mean(pixels, axis=0)
        color_std = np.std(pixels, axis=0)
        features.extend(color_mean / 255.0)  # 归一化到[0,1]
        features.extend(color_std / 255.0)
        
        # 2. 颜色分布特征
        hist_r = np.histogram(pixels[:, 0], bins=8, range=(0, 256))[0]
        hist_g = np.histogram(pixels[:, 1], bins=8, range=(0, 256))[0]
        hist_b = np.histogram(pixels[:, 2], bins=8, range=(0, 256))[0]
        
        # 归一化直方图
        total_pixels = len(pixels)
        hist_features = np.concatenate([hist_r, hist_g, hist_b]) / total_pixels
        features.extend(hist_features)
        
        # 3. 主要颜色特征
        try:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_ / 255.0  # 归一化
            features.extend(dominant_colors.flatten())
        except:
            features.extend([0] * 9)  # 3个颜色 × 3个通道
        
        return np.array(features, dtype=np.float32)
    
    def _extract_shape_features(self, image: Image.Image) -> np.ndarray:
        """
        提取形状特征
        """
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # 1. Hu矩特征
        moments = cv2.moments(gray)
        if moments['m00'] != 0:
            hu_moments = cv2.HuMoments(moments).flatten()
            # 对数变换以减小数值范围
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            features.extend(hu_moments)
        else:
            features.extend([0] * 7)
        
        # 2. 方向梯度直方图 (HOG-like简化版)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        orientation = np.arctan2(sobely, sobelx)
        
        # 计算方向直方图
        hist, _ = np.histogram(orientation, bins=8, range=(-np.pi, np.pi), weights=magnitude)
        hist = hist / (np.sum(hist) + 1e-10)  # 归一化
        features.extend(hist)
        
        return np.array(features, dtype=np.float32)
    
    def classify_with_clip(self, image: Union[Image.Image, str, np.ndarray]) -> Dict[str, float]:
        """
        使用CLIP进行零样本分类
        
        Args:
            image: 输入图像
            
        Returns:
            各类别的置信度分数
        """
        # 处理输入图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # 编码图像
        image_features = self.clip_model.encode_image(image)
        
        category_scores = {}
        
        for category, descriptions in self.text_descriptions.items():
            scores = []
            for desc in descriptions:
                text_features = self.clip_model.encode_text(desc)
                similarity = self.clip_model.compute_similarity(image_features, text_features)
                scores.append(similarity)
            
            # 取最高相似度作为该类别的分数
            category_scores[category] = max(scores)
        
        return category_scores
    
    def classify_garment_enhanced(self, image: Union[Image.Image, str, np.ndarray]) -> Dict[str, any]:
        """
        增强版服装分类
        
        Args:
            image: 输入图像
            
        Returns:
            分类结果字典
        """
        # 1. 基础分类器结果
        base_result = self.base_classifier.classify_garment(image)
        
        # 2. CLIP分类结果
        clip_scores = self.classify_with_clip(image)
        
        # 3. 提取增强特征
        enhanced_features = self.extract_enhanced_features(image)
        
        # 4. 融合多个分类结果
        final_category, final_confidence = self._fuse_classification_results(
            base_result, clip_scores, enhanced_features
        )
        
        # 5. 生成最终结果
        result = {
            "category": final_category,
            "category_cn": self.category_cn.get(final_category, final_category),
            "confidence": final_confidence,
            "dominant_colors": base_result["dominant_colors"],
            "clip_scores": clip_scores,
            "base_prediction": base_result["category"],
            "features": {
                "clip_features": enhanced_features['clip'].tolist(),
                "cv_features": enhanced_features['cv'].tolist(),
                "color_features": enhanced_features['color'].tolist(),
                "shape_features": enhanced_features['shape'].tolist()
            }
        }
        
        return result
    
    def _fuse_classification_results(self, base_result: Dict, clip_scores: Dict, 
                                   enhanced_features: Dict) -> Tuple[str, float]:
        """
        融合多个分类结果
        
        Args:
            base_result: 基础分类器结果
            clip_scores: CLIP分类分数
            enhanced_features: 增强特征
            
        Returns:
            (最终类别, 置信度)
        """
        # 权重设置
        base_weight = 0.3
        clip_weight = 0.7
        
        # 获取CLIP最高分类别
        clip_category = max(clip_scores.items(), key=lambda x: x[1])[0]
        clip_confidence = clip_scores[clip_category]
        
        base_category = base_result["category"]
        base_confidence = base_result["confidence"]
        
        # 如果两个分类器结果一致，提高置信度
        if base_category == clip_category:
            final_category = base_category
            final_confidence = min(0.95, base_confidence * base_weight + clip_confidence * clip_weight + 0.1)
        else:
            # 如果不一致，选择置信度更高的
            if clip_confidence > base_confidence:
                final_category = clip_category
                final_confidence = clip_confidence * clip_weight
            else:
                final_category = base_category
                final_confidence = base_confidence * base_weight
        
        # 确保置信度在合理范围内
        final_confidence = max(0.1, min(0.95, final_confidence))
        
        return final_category, final_confidence
    
    def train_ml_classifier(self, training_data: List[Tuple[Dict, str]]):
        """
        训练机器学习分类器
        
        Args:
            training_data: 训练数据列表，每个元素为(特征字典, 标签)
        """
        if not training_data:
            print("警告: 没有训练数据，跳过ML分类器训练")
            return
        
        # 准备训练数据
        X = []
        y = []
        
        for features_dict, label in training_data:
            # 合并所有特征
            combined_features = np.concatenate([
                features_dict['clip'],
                features_dict['cv'],
                features_dict['color'],
                features_dict['shape']
            ])
            X.append(combined_features)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # 特征标准化
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # 训练随机森林分类器
        self.ml_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.ml_classifier.fit(X_scaled, y)
        
        print(f"ML分类器训练完成，训练样本数: {len(X)}")
    
    def save_model(self, model_path: str):
        """
        保存训练好的模型
        
        Args:
            model_path: 模型保存路径
        """
        if self.ml_classifier is not None:
            model_data = {
                'classifier': self.ml_classifier,
                'scaler': self.feature_scaler,
                'categories': self.categories
            }
            joblib.dump(model_data, model_path)
            print(f"模型已保存到: {model_path}")
        else:
            print("警告: 没有训练好的模型可保存")
    
    def load_model(self, model_path: str):
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
        """
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.ml_classifier = model_data['classifier']
            self.feature_scaler = model_data['scaler']
            self.categories = model_data['categories']
            print(f"模型已从 {model_path} 加载")
        else:
            print(f"警告: 模型文件 {model_path} 不存在")

# 全局增强分类器实例
_enhanced_classifier = None

def get_enhanced_classifier() -> EnhancedGarmentClassifier:
    """
    获取全局增强分类器实例（单例模式）
    """
    global _enhanced_classifier
    if _enhanced_classifier is None:
        _enhanced_classifier = EnhancedGarmentClassifier()
    return _enhanced_classifier