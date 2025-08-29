import torch
import open_clip
from PIL import Image
import numpy as np
from typing import List, Union
import os
from sklearn.metrics.pairwise import cosine_similarity

class CLIPModel:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        """
        初始化CLIP模型
        
        Args:
            model_name: CLIP模型名称
            pretrained: 预训练权重来源
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载CLIP模型
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # 设置为评估模式
        self.model.eval()
        
        print(f"CLIP模型 {model_name} 加载完成")
    
    def encode_image(self, image: Union[Image.Image, str, np.ndarray]) -> np.ndarray:
        """
        将图像编码为特征向量
        
        Args:
            image: PIL图像、图像路径或numpy数组
            
        Returns:
            归一化的特征向量
        """
        # 处理不同类型的输入
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("不支持的图像格式")
        
        # 预处理图像
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 编码
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()
    
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        将文本编码为特征向量
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            归一化的特征向量
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # 分词
        text_tokens = self.tokenizer(texts).to(self.device)
        
        # 编码
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        计算两个特征向量的余弦相似度
        
        Args:
            features1: 第一个特征向量
            features2: 第二个特征向量
            
        Returns:
            余弦相似度分数
        """
        # 确保是2D数组
        if features1.ndim == 1:
            features1 = features1.reshape(1, -1)
        if features2.ndim == 1:
            features2 = features2.reshape(1, -1)
        
        similarity = cosine_similarity(features1, features2)[0, 0]
        return float(similarity)
    
    def find_most_similar(self, query_features: np.ndarray, 
                         candidate_features: List[np.ndarray], 
                         top_k: int = 5) -> List[tuple]:
        """
        找到最相似的候选项
        
        Args:
            query_features: 查询特征向量
            candidate_features: 候选特征向量列表
            top_k: 返回前k个最相似的结果
            
        Returns:
            (索引, 相似度分数)的列表，按相似度降序排列
        """
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        
        similarities = []
        for i, candidate in enumerate(candidate_features):
            if candidate.ndim == 1:
                candidate = candidate.reshape(1, -1)
            sim = cosine_similarity(query_features, candidate)[0, 0]
            similarities.append((i, float(sim)))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def batch_encode_images(self, images: List[Union[Image.Image, str]]) -> np.ndarray:
        """
        批量编码图像
        
        Args:
            images: 图像列表
            
        Returns:
            特征矩阵 (n_images, feature_dim)
        """
        features_list = []
        
        for image in images:
            try:
                features = self.encode_image(image)
                features_list.append(features)
            except Exception as e:
                print(f"编码图像失败: {e}")
                continue
        
        if not features_list:
            raise ValueError("没有成功编码的图像")
        
        return np.vstack(features_list)
    
    def save_features(self, features: np.ndarray, filepath: str):
        """
        保存特征向量到文件
        
        Args:
            features: 特征向量
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, features)
        print(f"特征向量已保存到: {filepath}")
    
    def load_features(self, filepath: str) -> np.ndarray:
        """
        从文件加载特征向量
        
        Args:
            filepath: 文件路径
            
        Returns:
            特征向量
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"特征文件不存在: {filepath}")
        
        features = np.load(filepath)
        print(f"特征向量已从 {filepath} 加载")
        return features

# 全局CLIP模型实例
_clip_model = None

def get_clip_model() -> CLIPModel:
    """
    获取全局CLIP模型实例（单例模式）
    """
    global _clip_model
    if _clip_model is None:
        _clip_model = CLIPModel()
    return _clip_model