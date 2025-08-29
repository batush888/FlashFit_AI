import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os
from datetime import datetime
from .clip_model import get_clip_model
from .enhanced_classifier import get_enhanced_classifier

class StyleCompatibilityNetwork(nn.Module):
    """
    风格兼容性神经网络 - 学习服装搭配的兼容性
    """
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.compatibility_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, item1_features: torch.Tensor, item2_features: torch.Tensor) -> torch.Tensor:
        """
        计算两个服装项目的兼容性分数
        
        Args:
            item1_features: 第一个服装项目的特征
            item2_features: 第二个服装项目的特征
            
        Returns:
            兼容性分数 (0-1)
        """
        # 编码特征
        encoded1 = self.feature_encoder(item1_features)
        encoded2 = self.feature_encoder(item2_features)
        
        # 计算特征交互
        interaction = encoded1 * encoded2  # 元素级乘法
        difference = torch.abs(encoded1 - encoded2)  # 差异特征
        
        # 合并特征
        combined = torch.cat([interaction, difference], dim=-1)
        
        # 计算兼容性分数
        compatibility = self.compatibility_head(combined)
        
        return compatibility

class AdvancedStyleMatcher:
    """
    高级风格匹配器 - 使用深度学习和规则结合的方法
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化组件
        self.clip_model = get_clip_model()
        self.enhanced_classifier = get_enhanced_classifier()
        
        # 初始化兼容性网络
        self.compatibility_net = StyleCompatibilityNetwork()
        self.compatibility_net.to(self.device)
        
        # 颜色兼容性规则
        self.color_harmony_rules = self._init_color_harmony_rules()
        
        # 风格规则
        self.style_rules = self._init_style_rules()
        
        # 场合规则
        self.occasion_rules = self._init_occasion_rules()
        
        # 季节规则
        self.season_rules = self._init_season_rules()
        
        print("高级风格匹配器初始化完成")
    
    def _init_color_harmony_rules(self) -> Dict[str, Any]:
        """
        初始化颜色和谐规则
        """
        return {
            # 互补色搭配
            "complementary": {
                "red": ["green", "blue"],
                "blue": ["orange", "yellow"],
                "green": ["red", "pink"],
                "yellow": ["purple", "blue"],
                "orange": ["blue", "navy"],
                "purple": ["yellow", "green"]
            },
            
            # 类似色搭配
            "analogous": {
                "red": ["orange", "pink", "burgundy"],
                "blue": ["navy", "teal", "purple"],
                "green": ["teal", "lime", "olive"],
                "yellow": ["orange", "gold", "cream"],
                "purple": ["pink", "lavender", "magenta"]
            },
            
            # 中性色（万能搭配）
            "neutral": ["black", "white", "gray", "beige", "brown", "navy"],
            
            # 冲突色（避免搭配）
            "conflicting": {
                "red": ["pink", "orange"],
                "blue": ["green"],
                "yellow": ["orange"],
                "purple": ["red"]
            },
            
            # 季节性颜色偏好
            "seasonal": {
                "spring": ["pink", "light_blue", "yellow", "green"],
                "summer": ["white", "light_blue", "coral", "mint"],
                "autumn": ["brown", "orange", "burgundy", "olive"],
                "winter": ["black", "navy", "gray", "red"]
            }
        }
    
    def _init_style_rules(self) -> Dict[str, Any]:
        """
        初始化风格规则
        """
        return {
            "casual": {
                "preferred_items": ["shirt", "pants", "shoes", "jacket"],
                "avoid_items": ["dress"],
                "color_preference": ["blue", "white", "gray", "black"],
                "formality_level": 0.3
            },
            "formal": {
                "preferred_items": ["shirt", "jacket", "pants", "dress", "shoes"],
                "avoid_items": ["accessory"],
                "color_preference": ["black", "navy", "white", "gray"],
                "formality_level": 0.9
            },
            "business": {
                "preferred_items": ["shirt", "jacket", "pants", "shoes"],
                "avoid_items": ["dress", "skirt"],
                "color_preference": ["navy", "black", "white", "gray"],
                "formality_level": 0.8
            },
            "party": {
                "preferred_items": ["dress", "skirt", "jacket", "accessory"],
                "avoid_items": [],
                "color_preference": ["red", "black", "gold", "silver"],
                "formality_level": 0.7
            },
            "sport": {
                "preferred_items": ["shirt", "pants", "shoes"],
                "avoid_items": ["dress", "skirt", "jacket"],
                "color_preference": ["blue", "black", "white", "red"],
                "formality_level": 0.2
            }
        }
    
    def _init_occasion_rules(self) -> Dict[str, Any]:
        """
        初始化场合规则
        """
        return {
            "work": {
                "required_formality": 0.7,
                "preferred_colors": ["navy", "black", "white", "gray"],
                "avoid_colors": ["bright_pink", "neon"],
                "style_keywords": ["professional", "conservative", "polished"]
            },
            "date": {
                "required_formality": 0.6,
                "preferred_colors": ["red", "black", "navy", "pink"],
                "avoid_colors": ["brown", "olive"],
                "style_keywords": ["attractive", "stylish", "confident"]
            },
            "casual": {
                "required_formality": 0.3,
                "preferred_colors": ["blue", "white", "gray"],
                "avoid_colors": [],
                "style_keywords": ["comfortable", "relaxed", "easy"]
            },
            "party": {
                "required_formality": 0.8,
                "preferred_colors": ["black", "red", "gold", "silver"],
                "avoid_colors": ["brown", "beige"],
                "style_keywords": ["glamorous", "eye-catching", "festive"]
            }
        }
    
    def _init_season_rules(self) -> Dict[str, Any]:
        """
        初始化季节规则
        """
        return {
            "spring": {
                "preferred_colors": ["pink", "light_blue", "yellow", "green"],
                "preferred_materials": ["cotton", "linen"],
                "layering_preference": "light",
                "temperature_range": (15, 25)
            },
            "summer": {
                "preferred_colors": ["white", "light_blue", "coral", "mint"],
                "preferred_materials": ["cotton", "linen", "silk"],
                "layering_preference": "minimal",
                "temperature_range": (25, 35)
            },
            "autumn": {
                "preferred_colors": ["brown", "orange", "burgundy", "olive"],
                "preferred_materials": ["wool", "cotton", "denim"],
                "layering_preference": "medium",
                "temperature_range": (10, 20)
            },
            "winter": {
                "preferred_colors": ["black", "navy", "gray", "red"],
                "preferred_materials": ["wool", "cashmere", "leather"],
                "layering_preference": "heavy",
                "temperature_range": (-5, 15)
            }
        }
    
    def calculate_color_compatibility(self, color1: str, color2: str, 
                                   season: Optional[str] = None) -> float:
        """
        计算两个颜色的兼容性分数
        
        Args:
            color1: 第一个颜色
            color2: 第二个颜色
            season: 季节（可选）
            
        Returns:
            兼容性分数 (0-1)
        """
        # 基础兼容性分数
        base_score = 0.5
        
        # 中性色加分
        if color1 in self.color_harmony_rules["neutral"] or color2 in self.color_harmony_rules["neutral"]:
            base_score += 0.3
        
        # 互补色加分
        if (color1 in self.color_harmony_rules["complementary"] and 
            color2 in self.color_harmony_rules["complementary"][color1]):
            base_score += 0.2
        
        # 类似色加分
        if (color1 in self.color_harmony_rules["analogous"] and 
            color2 in self.color_harmony_rules["analogous"][color1]):
            base_score += 0.15
        
        # 冲突色扣分
        if (color1 in self.color_harmony_rules["conflicting"] and 
            color2 in self.color_harmony_rules["conflicting"][color1]):
            base_score -= 0.3
        
        # 季节性加分
        if season and season in self.color_harmony_rules["seasonal"]:
            seasonal_colors = self.color_harmony_rules["seasonal"][season]
            if color1 in seasonal_colors and color2 in seasonal_colors:
                base_score += 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def calculate_style_compatibility(self, item1: Dict, item2: Dict, 
                                    style: Optional[str] = None,
                                    occasion: Optional[str] = None) -> float:
        """
        计算两个服装项目的风格兼容性
        
        Args:
            item1: 第一个服装项目
            item2: 第二个服装项目
            style: 目标风格
            occasion: 场合
            
        Returns:
            风格兼容性分数 (0-1)
        """
        compatibility_score = 0.5
        
        # 1. 类别兼容性
        category1 = item1.get("category", "")
        category2 = item2.get("category", "")
        
        # 避免同类别重复（除非是配饰）
        if category1 == category2 and category1 != "accessory":
            compatibility_score -= 0.4
        
        # 2. 颜色兼容性
        colors1 = [c["name"] for c in item1.get("dominant_colors", [])]
        colors2 = [c["name"] for c in item2.get("dominant_colors", [])]
        
        if colors1 and colors2:
            color_scores = []
            for c1 in colors1[:2]:  # 只考虑前两个主要颜色
                for c2 in colors2[:2]:
                    color_scores.append(self.calculate_color_compatibility(c1, c2))
            
            if color_scores:
                compatibility_score += (max(color_scores) - 0.5) * 0.6
        
        # 3. 风格规则兼容性
        if style and style in self.style_rules:
            style_rule = self.style_rules[style]
            
            # 检查偏好项目
            if category1 in style_rule["preferred_items"]:
                compatibility_score += 0.1
            if category2 in style_rule["preferred_items"]:
                compatibility_score += 0.1
            
            # 检查避免项目
            if category1 in style_rule["avoid_items"]:
                compatibility_score -= 0.2
            if category2 in style_rule["avoid_items"]:
                compatibility_score -= 0.2
        
        # 4. 场合兼容性
        if occasion and occasion in self.occasion_rules:
            occasion_rule = self.occasion_rules[occasion]
            
            # 检查颜色偏好
            preferred_colors = occasion_rule["preferred_colors"]
            avoid_colors = occasion_rule["avoid_colors"]
            
            for color in colors1 + colors2:
                if color in preferred_colors:
                    compatibility_score += 0.05
                if color in avoid_colors:
                    compatibility_score -= 0.1
        
        return max(0.0, min(1.0, compatibility_score))
    
    def calculate_clip_similarity(self, item1: Dict, item2: Dict) -> float:
        """
        使用CLIP特征计算相似度
        
        Args:
            item1: 第一个服装项目
            item2: 第二个服装项目
            
        Returns:
            CLIP相似度分数 (0-1)
        """
        try:
            # 获取CLIP特征
            features1 = np.array(item1.get("features", {}).get("clip_features", []))
            features2 = np.array(item2.get("features", {}).get("clip_features", []))
            
            if len(features1) == 0 or len(features2) == 0:
                return 0.5  # 默认中等相似度
            
            # 计算余弦相似度
            similarity = cosine_similarity([features1], [features2])[0, 0]
            
            # 转换到0-1范围
            return (similarity + 1) / 2
            
        except Exception as e:
            print(f"CLIP相似度计算错误: {e}")
            return 0.5
    
    def generate_outfit_recommendations(self, target_item: Dict, 
                                      wardrobe: List[Dict],
                                      style: Optional[str] = None,
                                      occasion: Optional[str] = None,
                                      season: Optional[str] = None,
                                      max_items: int = 5,
                                      top_k: int = 3) -> List[Dict]:
        """
        生成服装搭配推荐
        
        Args:
            target_item: 目标服装项目
            wardrobe: 用户衣橱
            style: 目标风格
            occasion: 场合
            season: 季节
            max_items: 每个推荐的最大项目数
            top_k: 返回推荐数量
            
        Returns:
            推荐列表
        """
        recommendations = []
        
        # 1. 计算所有衣橱项目与目标项目的兼容性
        compatibility_scores = []
        
        for item in wardrobe:
            if item.get("id") == target_item.get("id"):
                continue  # 跳过目标项目本身
            
            # 计算综合兼容性分数
            style_score = self.calculate_style_compatibility(
                target_item, item, style, occasion
            )
            
            clip_score = self.calculate_clip_similarity(target_item, item)
            
            # 加权综合分数
            total_score = style_score * 0.7 + clip_score * 0.3
            
            compatibility_scores.append({
                "item": item,
                "score": total_score,
                "style_score": style_score,
                "clip_score": clip_score
            })
        
        # 2. 按分数排序
        compatibility_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # 3. 生成多个推荐组合
        for i in range(min(top_k, len(compatibility_scores))):
            recommendation = {
                "id": f"outfit_{i+1}",
                "title": f"搭配建议 {i+1}",
                "items": [target_item],
                "total_score": 0.0,
                "style_tags": [],
                "occasion_fit": 0.0
            }
            
            # 选择兼容的项目
            selected_items = [target_item]
            current_score = compatibility_scores[i]["score"]
            
            # 添加最兼容的项目
            if i < len(compatibility_scores):
                selected_items.append(compatibility_scores[i]["item"])
            
            # 尝试添加更多兼容项目
            for j, comp_item in enumerate(compatibility_scores[i+1:], i+1):
                if len(selected_items) >= max_items:
                    break
                
                # 检查与已选项目的兼容性
                is_compatible = True
                for selected in selected_items:
                    if self.calculate_style_compatibility(
                        selected, comp_item["item"], style, occasion
                    ) < 0.4:
                        is_compatible = False
                        break
                
                if is_compatible:
                    selected_items.append(comp_item["item"])
                    current_score += comp_item["score"] * 0.5  # 递减权重
            
            recommendation["items"] = selected_items
            recommendation["total_score"] = current_score / len(selected_items)
            
            # 生成风格标签
            recommendation["style_tags"] = self._generate_style_tags(
                selected_items, style, occasion
            )
            
            # 计算场合适合度
            recommendation["occasion_fit"] = self._calculate_occasion_fit(
                selected_items, occasion
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_style_tags(self, items: List[Dict], 
                           style: Optional[str] = None,
                           occasion: Optional[str] = None) -> List[str]:
        """
        生成风格标签
        
        Args:
            items: 服装项目列表
            style: 风格
            occasion: 场合
            
        Returns:
            风格标签列表
        """
        tags = set()
        
        # 基于风格的标签
        if style and style in self.style_rules:
            style_rule = self.style_rules[style]
            if style_rule["formality_level"] > 0.7:
                tags.add("正式")
            elif style_rule["formality_level"] < 0.4:
                tags.add("休闲")
            else:
                tags.add("半正式")
        
        # 基于场合的标签
        if occasion and occasion in self.occasion_rules:
            occasion_rule = self.occasion_rules[occasion]
            tags.update(occasion_rule["style_keywords"])
        
        # 基于颜色的标签
        all_colors = []
        for item in items:
            colors = [c["name"] for c in item.get("dominant_colors", [])]
            all_colors.extend(colors)
        
        if "black" in all_colors:
            tags.add("经典")
        if "white" in all_colors:
            tags.add("简约")
        if "red" in all_colors:
            tags.add("醒目")
        
        return list(tags)[:5]  # 限制标签数量
    
    def _calculate_occasion_fit(self, items: List[Dict], 
                              occasion: Optional[str] = None) -> float:
        """
        计算场合适合度
        
        Args:
            items: 服装项目列表
            occasion: 场合
            
        Returns:
            适合度分数 (0-1)
        """
        if not occasion or occasion not in self.occasion_rules:
            return 0.7  # 默认适合度
        
        occasion_rule = self.occasion_rules[occasion]
        fit_score = 0.5
        
        # 检查正式度要求
        required_formality = occasion_rule["required_formality"]
        
        # 简单的正式度评估（基于类别）
        formality_scores = {
            "dress": 0.8,
            "jacket": 0.7,
            "shirt": 0.6,
            "pants": 0.5,
            "skirt": 0.6,
            "shoes": 0.5,
            "accessory": 0.3
        }
        
        avg_formality = np.mean([
            formality_scores.get(item.get("category", ""), 0.5) 
            for item in items
        ])
        
        # 计算正式度匹配分数
        formality_diff = abs(avg_formality - required_formality)
        fit_score += (1 - formality_diff) * 0.3
        
        # 检查颜色偏好
        all_colors = []
        for item in items:
            colors = [c["name"] for c in item.get("dominant_colors", [])]
            all_colors.extend(colors)
        
        preferred_colors = occasion_rule["preferred_colors"]
        avoid_colors = occasion_rule["avoid_colors"]
        
        color_bonus = 0
        for color in all_colors:
            if color in preferred_colors:
                color_bonus += 0.1
            if color in avoid_colors:
                color_bonus -= 0.2
        
        fit_score += min(0.2, max(-0.2, color_bonus))
        
        return max(0.0, min(1.0, fit_score))

# 全局风格匹配器实例
_style_matcher = None

def get_style_matcher() -> AdvancedStyleMatcher:
    """
    获取全局风格匹配器实例（单例模式）
    """
    global _style_matcher
    if _style_matcher is None:
        _style_matcher = AdvancedStyleMatcher()
    return _style_matcher