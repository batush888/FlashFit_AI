import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
import random
import asyncio

# 导入增强模型
try:
    from models.enhanced_classifier import get_enhanced_classifier
    from models.style_matcher import get_style_matcher
    ENHANCED_MODELS_AVAILABLE = True
except ImportError:
    # 如果增强模型不可用，回退到高级分类器
    from models.clip_model import get_clip_model
    from models.advanced_classifier import get_advanced_classifier
    ENHANCED_MODELS_AVAILABLE = False
    
    def get_enhanced_classifier():
        return get_advanced_classifier()
    
    def get_style_matcher():
        return None

class EnhancedMatchHandler:
    """
    增强版服装搭配匹配处理器
    """
    
    def __init__(self):
        self.users_file = Path("data/users.json")
        self.templates_file = Path("data/style_templates.json")
        self.feedback_file = Path("data/feedback.json")
        
        # 初始化增强模型
        self.enhanced_classifier = get_enhanced_classifier()
        self.style_matcher = get_style_matcher()
        self.enhanced_models_available = ENHANCED_MODELS_AVAILABLE
        
        # 初始化风格模板
        self._init_enhanced_templates()
        
        # 增强的LLM提示模板
        self.enhanced_llm_prompt = """
你是一个专业的中文时尚顾问。用户上传了一件衣服，请基于以下信息生成个性化搭配建议：

服装信息：
- 类别: {category} ({category_cn})
- 主要颜色: {dominant_colors}
- 置信度: {confidence:.2f}
- 风格关键词: {style_keywords}
- CLIP分析: {clip_analysis}

用户偏好：
- 场合: {occasion}
- 风格偏好: {style_preference}
- 季节: {season}
- 个人风格: {personal_style}

请生成3个高质量的搭配建议，每个包含：
1. 吸引人的标题（8-12字）
2. 3-4条具体的搭配提示（每条15-20字）
3. 推荐的具体单品（鞋子、外套、配饰等）
4. 适合的场合和时间
5. 整体风格描述

输出格式为JSON数组，确保建议实用且时尚。
"""
        
        print("增强版匹配处理器初始化完成")
    
    def _load_users(self) -> dict:
        """加载用户数据"""
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_users(self, users: dict):
        """保存用户数据"""
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    
    def _load_feedback(self) -> dict:
        """加载用户反馈数据"""
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"feedback": [], "ratings": {}}
    
    def _save_feedback(self, feedback_data: dict):
        """保存用户反馈数据"""
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
    
    def _init_enhanced_templates(self):
        """初始化增强风格模板"""
        if not self.templates_file.exists():
            enhanced_templates = {
                "casual_chic": {
                    "name": "休闲时尚",
                    "description": "轻松舒适又不失时尚感",
                    "formality_level": 0.4,
                    "combinations": [
                        {
                            "title": "都市休闲",
                            "items": ["shirt", "pants", "shoes", "accessory"],
                            "colors": ["blue", "white", "gray"],
                            "tips": ["选择修身剪裁", "配色简洁大方", "细节提升质感", "舒适度优先"]
                        }
                    ]
                },
                "business_professional": {
                    "name": "商务专业",
                    "description": "专业形象与个人魅力并重",
                    "formality_level": 0.9,
                    "combinations": [
                        {
                            "title": "精英范儿",
                            "items": ["shirt", "jacket", "pants", "shoes"],
                            "colors": ["navy", "white", "black"],
                            "tips": ["剪裁合身重要", "颜色搭配经典", "细节决定品质", "配饰画龙点睛"]
                        }
                    ]
                },
                "romantic_feminine": {
                    "name": "浪漫女性",
                    "description": "优雅柔美的女性魅力",
                    "formality_level": 0.6,
                    "combinations": [
                        {
                            "title": "温柔淑女",
                            "items": ["dress", "jacket", "shoes", "accessory"],
                            "colors": ["pink", "white", "beige"],
                            "tips": ["面料质感重要", "色彩温柔和谐", "剪裁突出身材", "配饰精致小巧"]
                        }
                    ]
                },
                "street_fashion": {
                    "name": "街头时尚",
                    "description": "个性张扬的潮流风格",
                    "formality_level": 0.3,
                    "combinations": [
                        {
                            "title": "潮流达人",
                            "items": ["shirt", "jacket", "pants", "shoes", "accessory"],
                            "colors": ["black", "white", "red"],
                            "tips": ["层次搭配有趣", "色彩对比鲜明", "单品个性突出", "整体协调统一"]
                        }
                    ]
                }
            }
            
            with open(self.templates_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_templates, f, ensure_ascii=False, indent=2)
    
    def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """获取用户偏好设置"""
        users = self._load_users()
        user_data = users.get(user_id, {})
        
        return {
            "style_preference": user_data.get("style_preference", "casual"),
            "color_preferences": user_data.get("color_preferences", []),
            "body_type": user_data.get("body_type", "average"),
            "lifestyle": user_data.get("lifestyle", "casual"),
            "budget_range": user_data.get("budget_range", "medium"),
            "favorite_brands": user_data.get("favorite_brands", []),
            "avoid_categories": user_data.get("avoid_categories", [])
        }
    
    def _analyze_item_with_enhanced_classifier(self, item: Dict) -> Dict[str, Any]:
        """使用增强分类器分析服装项目"""
        try:
            # 如果有图片路径，使用分类器
            if "image_path" in item and os.path.exists(item["image_path"]):
                if self.enhanced_models_available:
                    # 尝试使用增强方法
                    try:
                        enhanced_result = getattr(self.enhanced_classifier, 'classify_garment_enhanced', None)
                        if enhanced_result:
                            return enhanced_result(item["image_path"])
                    except Exception:
                        pass
                
                # 使用基础分类器
                basic_result = self.enhanced_classifier.classify_garment(item["image_path"])
                return {
                    "category": basic_result.get("category", "unknown"),
                    "category_cn": basic_result.get("category_cn", "未知"),
                    "confidence": basic_result.get("confidence", 0.7),
                    "dominant_colors": basic_result.get("dominant_colors", []),
                    "clip_scores": {},
                    "features": {}
                }
            else:
                # 回退到基础信息
                return {
                    "category": item.get("category", "unknown"),
                    "category_cn": item.get("category_cn", "未知"),
                    "confidence": 0.7,
                    "dominant_colors": item.get("dominant_colors", []),
                    "clip_scores": {},
                    "features": {}
                }
        except Exception as e:
            print(f"增强分类器分析失败: {e}")
            return {
                "category": item.get("category", "unknown"),
                "category_cn": item.get("category_cn", "未知"),
                "confidence": 0.5,
                "dominant_colors": item.get("dominant_colors", []),
                "clip_scores": {},
                "features": {}
            }
    
    def _generate_enhanced_suggestions(self, target_item: Dict, 
                                     user_wardrobe: List[Dict],
                                     user_preferences: Dict,
                                     occasion: Optional[str] = None,
                                     season: Optional[str] = None) -> List[Dict]:
        """生成增强版搭配建议"""
        suggestions = []
        
        try:
            # 使用增强分类器分析目标项目
            enhanced_analysis = self._analyze_item_with_enhanced_classifier(target_item)
            
            # 如果有风格匹配器，使用它生成建议
            if self.style_matcher:
                style_recommendations = self.style_matcher.generate_outfit_recommendations(
                    target_item=enhanced_analysis,
                    wardrobe=user_wardrobe,
                    style=user_preferences.get("style_preference"),
                    occasion=occasion,
                    season=season,
                    top_k=3
                )
                
                for i, rec in enumerate(style_recommendations):
                    suggestion = {
                        "id": str(uuid.uuid4()),
                        "title": rec.get("title", f"搭配建议 {i+1}"),
                        "confidence": rec.get("total_score", 0.7),
                        "items": rec.get("items", [target_item]),
                        "style_tags": rec.get("style_tags", []),
                        "occasion_fit": rec.get("occasion_fit", 0.7),
                        "tips": self._generate_enhanced_tips(rec, enhanced_analysis, user_preferences),
                        "reasoning": self._generate_reasoning(rec, enhanced_analysis),
                        "created_at": datetime.now().isoformat()
                    }
                    suggestions.append(suggestion)
            
            # 如果没有风格匹配器或生成的建议不足，使用传统方法补充
            if len(suggestions) < 3:
                traditional_suggestions = self._generate_traditional_suggestions(
                    target_item, user_wardrobe, user_preferences, occasion
                )
                suggestions.extend(traditional_suggestions[:3-len(suggestions)])
        
        except Exception as e:
            print(f"增强建议生成失败，使用传统方法: {e}")
            suggestions = self._generate_traditional_suggestions(
                target_item, user_wardrobe, user_preferences, occasion
            )
        
        return suggestions[:3]  # 确保只返回3个建议
    
    def _generate_enhanced_tips(self, recommendation: Dict, 
                              analysis: Dict, 
                              user_preferences: Dict) -> List[str]:
        """生成增强版搭配提示"""
        tips = []
        
        # 基于分析结果的提示
        category = analysis.get("category", "")
        confidence = analysis.get("confidence", 0.7)
        colors = analysis.get("dominant_colors", [])
        
        # 类别特定提示
        category_tips = {
            "shirt": ["选择合适的领型", "注意袖长搭配", "考虑面料质感"],
            "pants": ["确保腰线合适", "选择合身剪裁", "注意长度比例"],
            "jacket": ["外套是整体亮点", "注意肩线位置", "考虑层次搭配"],
            "dress": ["突出身材优势", "选择合适长度", "配饰画龙点睛"],
            "shoes": ["鞋子决定风格", "考虑舒适度", "与整体协调"]
        }
        
        if category in category_tips:
            tips.extend(category_tips[category][:2])
        
        # 颜色搭配提示
        if colors:
            main_color = colors[0].get("name", "")
            color_tips = {
                "black": "黑色百搭经典",
                "white": "白色清新简约",
                "red": "红色醒目热情",
                "blue": "蓝色稳重大方",
                "gray": "灰色低调优雅"
            }
            if main_color in color_tips:
                tips.append(color_tips[main_color])
        
        # 用户偏好提示
        style_pref = user_preferences.get("style_preference", "")
        if style_pref == "casual":
            tips.append("保持轻松舒适感")
        elif style_pref == "formal":
            tips.append("注重细节和质感")
        elif style_pref == "trendy":
            tips.append("加入时尚元素")
        
        return tips[:4]  # 限制提示数量
    
    def _generate_reasoning(self, recommendation: Dict, analysis: Dict) -> str:
        """生成搭配推理说明"""
        category = analysis.get("category_cn", "服装")
        confidence = analysis.get("confidence", 0.7)
        style_tags = recommendation.get("style_tags", [])
        
        reasoning_parts = []
        
        # 基础分析
        reasoning_parts.append(f"这件{category}")
        
        # 置信度说明
        if confidence > 0.8:
            reasoning_parts.append("识别度很高")
        elif confidence > 0.6:
            reasoning_parts.append("特征明显")
        else:
            reasoning_parts.append("风格独特")
        
        # 风格标签
        if style_tags:
            tags_str = "、".join(style_tags[:3])
            reasoning_parts.append(f"适合{tags_str}风格")
        
        # 搭配逻辑
        reasoning_parts.append("通过色彩和谐、风格统一的原则进行搭配")
        
        return "，".join(reasoning_parts) + "。"
    
    def _generate_traditional_suggestions(self, target_item: Dict,
                                        user_wardrobe: List[Dict],
                                        user_preferences: Dict,
                                        occasion: Optional[str] = None) -> List[Dict]:
        """生成传统搭配建议（回退方案）"""
        suggestions = []
        
        # 基础搭配模板
        templates = [
            {
                "title": "经典搭配",
                "tips": ["选择经典色彩", "注重剪裁合身", "配饰简约精致", "整体协调统一"],
                "style_tags": ["经典", "百搭", "实用"]
            },
            {
                "title": "时尚搭配",
                "tips": ["加入流行元素", "色彩搭配大胆", "层次丰富有趣", "突出个人特色"],
                "style_tags": ["时尚", "个性", "潮流"]
            },
            {
                "title": "优雅搭配",
                "tips": ["选择优质面料", "色调温和协调", "剪裁精致合身", "细节精心处理"],
                "style_tags": ["优雅", "精致", "女性"]
            }
        ]
        
        for i, template in enumerate(templates):
            suggestion = {
                "id": str(uuid.uuid4()),
                "title": template["title"],
                "confidence": 0.7 - i * 0.05,  # 递减置信度
                "items": [target_item],
                "style_tags": template["style_tags"],
                "occasion_fit": 0.7,
                "tips": template["tips"],
                "reasoning": f"基于{template['title']}原则，注重整体协调性和实用性。",
                "created_at": datetime.now().isoformat()
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    async def generate_enhanced_suggestions(self, item_id: str, user_id: str,
                                          occasion: Optional[str] = None,
                                          season: Optional[str] = None,
                                          style_preference: Optional[str] = None) -> Dict[str, Any]:
        """生成增强版搭配建议"""
        try:
            # 获取用户数据
            users = self._load_users()
            user_data = users.get(user_id, {})
            user_wardrobe = user_data.get("wardrobe_items", [])
            
            # 找到目标服装项目
            target_item = None
            for item in user_wardrobe:
                if item.get("id") == item_id:
                    target_item = item
                    break
            
            if not target_item:
                return {
                    "success": False,
                    "error": "未找到指定的服装项目",
                    "suggestions": []
                }
            
            # 获取用户偏好
            user_preferences = self._get_user_preferences(user_id)
            if style_preference:
                user_preferences["style_preference"] = style_preference
            
            # 生成增强建议
            suggestions = self._generate_enhanced_suggestions(
                target_item=target_item,
                user_wardrobe=user_wardrobe,
                user_preferences=user_preferences,
                occasion=occasion,
                season=season
            )
            
            # 保存建议历史
            suggestion_history = {
                "user_id": user_id,
                "item_id": item_id,
                "occasion": occasion,
                "season": season,
                "suggestions": suggestions,
                "created_at": datetime.now().isoformat()
            }
            
            # 更新用户数据
            if "suggestion_history" not in user_data:
                user_data["suggestion_history"] = []
            user_data["suggestion_history"].append(suggestion_history)
            
            # 保持历史记录在合理范围内
            if len(user_data["suggestion_history"]) > 50:
                user_data["suggestion_history"] = user_data["suggestion_history"][-50:]
            
            users[user_id] = user_data
            self._save_users(users)
            
            return {
                "success": True,
                "suggestions": suggestions,
                "target_item": target_item,
                "user_preferences": user_preferences,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"生成增强建议时出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "suggestions": []
            }
    
    def save_user_feedback(self, user_id: str, suggestion_id: str, 
                          rating: int, feedback_text: Optional[str] = None) -> bool:
        """保存用户反馈"""
        try:
            feedback_data = self._load_feedback()
            
            new_feedback = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "suggestion_id": suggestion_id,
                "rating": rating,
                "feedback_text": feedback_text,
                "created_at": datetime.now().isoformat()
            }
            
            feedback_data["feedback"].append(new_feedback)
            
            # 更新评分统计
            if suggestion_id not in feedback_data["ratings"]:
                feedback_data["ratings"][suggestion_id] = []
            feedback_data["ratings"][suggestion_id].append(rating)
            
            self._save_feedback(feedback_data)
            return True
            
        except Exception as e:
            print(f"保存反馈时出错: {e}")
            return False
    
    def get_suggestion_analytics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取建议分析数据"""
        try:
            feedback_data = self._load_feedback()
            
            analytics = {
                "total_suggestions": 0,
                "total_feedback": len(feedback_data["feedback"]),
                "average_rating": 0.0,
                "rating_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                "popular_styles": {},
                "user_satisfaction": 0.0
            }
            
            # 过滤用户特定数据
            feedback_list = feedback_data["feedback"]
            if user_id:
                feedback_list = [f for f in feedback_list if f["user_id"] == user_id]
            
            if feedback_list:
                # 计算平均评分
                ratings = [f["rating"] for f in feedback_list]
                analytics["average_rating"] = sum(ratings) / len(ratings)
                
                # 评分分布
                for rating in ratings:
                    analytics["rating_distribution"][rating] += 1
                
                # 用户满意度（4-5分为满意）
                satisfied_count = sum(1 for r in ratings if r >= 4)
                analytics["user_satisfaction"] = satisfied_count / len(ratings)
            
            return analytics
            
        except Exception as e:
            print(f"获取分析数据时出错: {e}")
            return {}

# 全局增强匹配处理器实例
_enhanced_match_handler = None

def get_enhanced_match_handler() -> EnhancedMatchHandler:
    """获取全局增强匹配处理器实例（单例模式）"""
    global _enhanced_match_handler
    if _enhanced_match_handler is None:
        _enhanced_match_handler = EnhancedMatchHandler()
    return _enhanced_match_handler