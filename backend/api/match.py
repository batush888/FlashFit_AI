import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
import random

# 导入模型
from models.clip_model import get_clip_model
from models.classifier import get_classifier

class MatchHandler:
    """
    服装搭配匹配处理器
    """
    
    def __init__(self):
        self.users_file = Path("data/users.json")
        self.templates_file = Path("data/style_templates.json")
        
        # 初始化风格模板
        self._init_style_templates()
        
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
    
    async def generate_suggestions(self, item_id: str, user_id: str, 
                                 occasion: Optional[str] = None, 
                                 target_count: int = 3) -> Dict[str, Any]:
        """
        生成搭配建议的主要方法
        """
        # 获取目标物品
        target_item = self._get_user_item(item_id, user_id)
        if not target_item:
            raise ValueError("物品不存在")
        
        # 获取用户衣橱
        wardrobe = self._get_user_wardrobe(user_id)
        if len(wardrobe) < 2:
            raise ValueError("衣橱物品太少，无法生成搭配建议")
        
        # 生成搭配建议
        suggestions = self._generate_outfit_suggestions(
            target_item, wardrobe, occasion, target_count
        )
        
        if not suggestions:
            raise ValueError("无法生成搭配建议")
        
        # 保存建议到用户数据（用于反馈）
        users = self._load_users()
        for email, user in users.items():
            if user["user_id"] == user_id:
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
            "suggestions": suggestions,
            "total_count": len(suggestions)
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