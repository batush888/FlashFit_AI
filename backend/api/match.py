import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
import random
import numpy as np
import logging

# 导入模型
from models.clip_model import get_clip_model
from models.advanced_classifier import get_advanced_classifier
from models.fashion_encoder import FashionEncoder
from services.recommend_service import get_recommendation_service

# 导入生成式匹配处理器
try:
    from api.generative_match import GenerativeMatchHandler
except ImportError:
    GenerativeMatchHandler = None

class MatchHandler:
    """
    服装搭配匹配处理器
    """
    
    def __init__(self):
        self.users_file = Path("data/users.json")
        self.templates_file = Path("data/style_templates.json")
        self.logger = logging.getLogger(__name__)
        
        # 初始化风格模板
        self._init_style_templates()
        
        # 初始化AI推荐服务
        try:
            self.recommendation_service = get_recommendation_service()
            print("AI推荐服务初始化成功")
        except Exception as e:
            print(f"AI推荐服务初始化失败，将使用模板匹配: {e}")
            self.recommendation_service = None
        
        # 初始化生成式匹配处理器（可选）
        self.generative_handler = None
        self.fashion_encoder = None
        try:
            if GenerativeMatchHandler:
                self.generative_handler = GenerativeMatchHandler()
                self.fashion_encoder = FashionEncoder()
                print("生成式匹配处理器初始化成功")
        except Exception as e:
            print(f"生成式匹配处理器初始化失败，将使用模板匹配: {e}")
        
        # LLM提示模板
        self.llm_prompt_template = """
你是一个中文时尚助理。用户上传了一件衣服：
- 类别: {category}
- 主色: {color}
- 风格关键词: {style_keywords}
- 场合: {occasion}

生成3个搭配建议，每个包含：
1) 吸睛标题（≤10字）
2) 2-3条提示（≤18字）
3) 推荐鞋/外套/配饰

输出JSON数组格式。
"""
        
        print("匹配处理器初始化完成")
    
    def _load_users(self) -> dict:
        """
        加载用户数据
        """
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_users(self, users: dict):
        """
        保存用户数据
        """
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    
    def _init_style_templates(self):
        """
        初始化风格模板数据
        """
        if not self.templates_file.exists():
            # 创建基础风格模板
            templates = {
                "casual": {
                    "name": "休闲风",
                    "combinations": [
                        {
                            "title": "舒适日常",
                            "items": ["shirt", "pants", "shoes"],
                            "colors": ["blue", "white", "black"],
                            "tips": ["选择舒适面料", "颜色搭配简洁", "适合日常出行"]
                        },
                        {
                            "title": "轻松周末",
                            "items": ["shirt", "jacket", "pants"],
                            "colors": ["gray", "white", "blue"],
                            "tips": ["层次搭配有趣", "适合休闲场合", "舒适度优先"]
                        }
                    ]
                },
                "formal": {
                    "name": "正式风",
                    "combinations": [
                        {
                            "title": "商务精英",
                            "items": ["shirt", "jacket", "pants", "shoes"],
                            "colors": ["black", "white", "gray"],
                            "tips": ["经典商务搭配", "颜色沉稳大气", "适合工作场合"]
                        },
                        {
                            "title": "优雅正装",
                            "items": ["dress", "jacket", "shoes"],
                            "colors": ["navy", "black", "white"],
                            "tips": ["简约而不简单", "展现专业形象", "细节决定成败"]
                        }
                    ]
                },
                "trendy": {
                    "name": "时尚风",
                    "combinations": [
                        {
                            "title": "街头潮流",
                            "items": ["shirt", "jacket", "pants", "accessory"],
                            "colors": ["red", "black", "white"],
                            "tips": ["大胆色彩搭配", "个性配饰点缀", "展现时尚态度"]
                        },
                        {
                            "title": "都市摩登",
                            "items": ["dress", "jacket", "shoes", "accessory"],
                            "colors": ["pink", "gray", "black"],
                            "tips": ["现代感十足", "配饰提升层次", "适合都市生活"]
                        }
                    ]
                }
            }
            
            self.templates_file.parent.mkdir(exist_ok=True)
            with open(self.templates_file, 'w', encoding='utf-8') as f:
                json.dump(templates, f, ensure_ascii=False, indent=2)
    
    def _load_style_templates(self) -> dict:
        """
        加载风格模板
        """
        try:
            with open(self.templates_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._init_style_templates()
            return self._load_style_templates()
    
    def _get_user_item(self, item_id: str, user_id: str) -> Optional[dict]:
        """
        获取用户的特定物品
        """
        users = self._load_users()
        
        for user in users.values():
            if user.get("user_id") == user_id:
                for item in user.get("wardrobe_items", []):
                    if item.get("item_id") == item_id:
                        return item
        return None
    
    def _get_user_wardrobe(self, user_id: str) -> List[dict]:
        """
        获取用户衣橱
        """
        users = self._load_users()
        
        for user in users.values():
            if user.get("user_id") == user_id:
                return user.get("wardrobe_items", [])
        return []
    
    def _find_matching_items(self, target_item: dict, wardrobe: List[dict], 
                           target_categories: List[str]) -> List[dict]:
        """
        在用户衣橱中找到匹配的物品
        """
        matching_items = []
        
        for item in wardrobe:
            if item["garment_type"] in target_categories and item["item_id"] != target_item["item_id"]:
                # 计算颜色匹配度（简单实现）
                color_match_score = self._calculate_color_compatibility(
                    target_item["colors"], item["colors"]
                )
                
                item_with_score = item.copy()
                item_with_score["match_score"] = color_match_score
                matching_items.append(item_with_score)
        
        # 按匹配分数排序
        matching_items.sort(key=lambda x: x["match_score"], reverse=True)
        
        return matching_items
    
    def _calculate_color_compatibility(self, colors1: List[dict], colors2: List[dict]) -> float:
        """
        计算颜色兼容性分数
        """
        if not colors1 or not colors2:
            return 0.5
        
        # 简单的颜色匹配规则
        color_compatibility = {
            "black": ["white", "gray", "red", "blue", "yellow"],
            "white": ["black", "blue", "red", "green", "pink"],
            "gray": ["black", "white", "blue", "pink", "yellow"],
            "blue": ["white", "black", "gray", "yellow", "brown"],
            "red": ["black", "white", "gray", "blue"],
            "green": ["white", "brown", "beige", "black"],
            "yellow": ["black", "blue", "gray", "white"],
            "pink": ["white", "gray", "black", "blue"],
            "brown": ["beige", "white", "green", "blue"],
            "beige": ["brown", "white", "black", "blue"]
        }
        
        main_color1 = colors1[0]["name"] if colors1 else "gray"
        main_color2 = colors2[0]["name"] if colors2 else "gray"
        
        if main_color2 in color_compatibility.get(main_color1, []):
            return 0.8
        elif main_color1 == main_color2:
            return 0.6  # 同色系，中等匹配
        else:
            return 0.3  # 低匹配度
    
    def _generate_outfit_suggestions(self, target_item: dict, wardrobe: List[dict], 
                                   occasion: Optional[str], target_count: int) -> List[dict]:
        """
        生成搭配建议
        """
        suggestions = []
        templates = self._load_style_templates()
        
        # 根据目标物品类型确定需要的其他类别
        category_combinations = {
            "shirt": ["pants", "jacket", "shoes"],
            "pants": ["shirt", "jacket", "shoes"],
            "jacket": ["shirt", "pants", "shoes"],
            "dress": ["jacket", "shoes", "accessory"],
            "skirt": ["shirt", "jacket", "shoes"],
            "shoes": ["shirt", "pants", "jacket"],
            "accessory": ["shirt", "pants", "shoes"]
        }
        
        target_category = target_item["garment_type"]
        needed_categories = category_combinations.get(target_category, ["shirt", "pants", "shoes"])
        
        # 在用户衣橱中找匹配物品
        matching_items = self._find_matching_items(target_item, wardrobe, needed_categories)
        
        # 生成搭配建议
        for i in range(min(target_count, 3)):
            suggestion_id = f"suggestion_{uuid.uuid4().hex}"
            
            # 选择风格模板
            style_keys = list(templates.keys())
            style_key = style_keys[i % len(style_keys)]
            style = templates[style_key]
            
            # 构建搭配
            outfit_items = [target_item]
            
            # 从匹配物品中选择
            used_categories = {target_category}
            for item in matching_items:
                if len(outfit_items) >= 4:  # 最多4件物品
                    break
                if item["garment_type"] not in used_categories:
                    outfit_items.append(item)
                    used_categories.add(item["garment_type"])
            
            # 生成中文建议
            main_color = target_item["colors"][0]["name_cn"] if target_item["colors"] else "经典"
            
            suggestion = {
                "suggestion_id": suggestion_id,
                "title_cn": f"{main_color}{style['name']}",
                "style_name": style["name"],
                "occasion": occasion or "日常",
                "items": outfit_items,
                "tips_cn": [
                    f"以{main_color}为主色调",
                    f"适合{occasion or '日常'}场合",
                    "注意整体协调性"
                ],
                "similarity_score": round(random.uniform(0.7, 0.9), 2),  # MVP模拟分数
                "created_at": datetime.now().isoformat(),
                "collage_url": f"/api/collages/{suggestion_id}.jpg"  # 模拟拼图URL
            }
            
            suggestions.append(suggestion)
        
        return suggestions
    
    async def _try_generative_matching(self, target_item: dict, user_id: str, 
                                     occasion: Optional[str] = None, 
                                     target_count: int = 3) -> Optional[List[dict]]:
        """
        尝试使用生成式匹配获取建议
        """
        if not self.generative_handler or not self.fashion_encoder:
            return None
            
        try:
            # 获取目标物品的图片路径
            image_path = target_item.get('image_path')
            if not image_path or not os.path.exists(image_path):
                return None
            
            # 使用时尚编码器生成嵌入
            query_embedding = self.fashion_encoder.embed_image(image_path)
            if query_embedding is None:
                return None
            
            # 转换为列表格式
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # 调用生成式匹配
            result = await self.generative_handler.generate_compatible_embeddings(
                query_embedding=query_embedding,
                occasion=occasion,
                top_k=target_count * 2  # 获取更多候选以提高质量
            )
            
            if result.get('success') and result.get('suggestions'):
                # 转换生成式匹配结果为标准格式
                suggestions = []
                for item in result['suggestions'][:target_count]:
                    suggestion = {
                        'suggestion_id': str(uuid.uuid4()),
                        'title': f"AI推荐搭配 {len(suggestions) + 1}",
                        'items': [target_item, item],
                        'tips': [
                            "AI生成的智能搭配",
                            f"相似度: {item.get('similarity', 0.8):.1%}",
                            "基于深度学习推荐"
                        ],
                        'occasion': occasion or '日常',
                        'style_score': item.get('similarity', 0.8),
                        'created_at': datetime.now().isoformat(),
                        'source': 'generative_ai'
                    }
                    suggestions.append(suggestion)
                
                return suggestions
                
        except Exception as e:
            print(f"生成式匹配失败: {e}")
            return None
        
        return None
    
    async def generate_suggestions(self, item_id: str, user_id: str, 
                                 occasion: Optional[str] = None, 
                                 target_count: int = 3) -> Dict[str, Any]:
        """
        生成搭配建议
        
        Args:
            item_id: 目标物品ID
            user_id: 用户ID
            occasion: 场合（可选）
            target_count: 目标建议数量
            
        Returns:
            搭配建议结果
        """
        # 获取目标物品
        target_item = self._get_user_item(item_id, user_id)
        if not target_item:
            raise ValueError(f"未找到物品 {item_id}")
        
        # 获取用户衣橱
        user_wardrobe = self._get_user_wardrobe(user_id)
        
        suggestions = []
        
        # 尝试使用AI推荐服务
        if self.recommendation_service and target_item.get('file_path'):
            try:
                print(f"使用AI推荐服务为物品 {item_id} 生成建议")
                ai_result = await self.recommendation_service.generate_recommendations(
                    query_image_path=target_item['file_path'],
                    user_wardrobe=user_wardrobe,
                    occasion=occasion,
                    top_k=target_count
                )
                
                # 转换AI推荐结果为标准格式
                if ai_result and 'recommendations' in ai_result:
                    for i, rec in enumerate(ai_result['recommendations'][:target_count]):
                        suggestion = {
                            "suggestion_id": f"ai_suggestion_{uuid.uuid4().hex}",
                            "title_cn": rec.get('title_cn', f"AI推荐搭配 {i+1}"),
                            "style_name": rec.get('style_name', 'AI智能搭配'),
                            "occasion": occasion or "日常",
                            "items": [target_item] + rec.get('matching_items', []),
                            "tips_cn": rec.get('tips_cn', ["AI智能推荐", "基于深度学习", "个性化匹配"]),
                            "similarity_score": rec.get('fusion_score', {}).get('final', 0.8),
                            "created_at": datetime.now().isoformat(),
                            "collage_url": f"/api/collages/ai_{uuid.uuid4().hex}.jpg",
                            "source": "ai_recommendation"
                        }
                        suggestions.append(suggestion)
                
                print(f"AI推荐服务生成了 {len(suggestions)} 个建议")
                
            except Exception as e:
                print(f"AI推荐服务失败: {e}")
        
        # 如果AI推荐失败或建议不足，尝试生成式匹配
        if len(suggestions) < target_count:
            generative_suggestions = await self._try_generative_matching(
                target_item, user_id, occasion, target_count - len(suggestions)
            )
            if generative_suggestions:
                suggestions.extend(generative_suggestions)
        
        # 如果仍然建议不足，使用模板匹配
        if len(suggestions) < target_count:
            template_suggestions = self._generate_outfit_suggestions(
                target_item, user_wardrobe, occasion, target_count - len(suggestions)
            )
            suggestions.extend(template_suggestions)
        
        # 保存建议历史
        users = self._load_users()
        for user in users.values():
            if user.get("user_id") == user_id:
                if "suggestions_history" not in user:
                    user["suggestions_history"] = []
                
                for suggestion in suggestions:
                    user["suggestions_history"].append({
                        "suggestion_id": suggestion["suggestion_id"],
                        "target_item_id": item_id,
                        "created_at": suggestion["created_at"],
                        "occasion": occasion
                    })
                break
        
        self._save_users(users)
        
        return {
            "message": "搭配建议生成成功",
            "target_item": target_item,
            "suggestions": suggestions[:target_count],
            "total_count": len(suggestions[:target_count]),
            "ai_enabled": self.recommendation_service is not None
        }
    
    def get_suggestion_by_id(self, suggestion_id: str, user_id: str) -> Optional[dict]:
        """
        根据ID获取建议详情
        """
        users = self._load_users()
        
        for user in users.values():
            if user.get("user_id") == user_id:
                for suggestion in user.get("suggestions_history", []):
                    if suggestion.get("suggestion_id") == suggestion_id:
                        return suggestion
        return None