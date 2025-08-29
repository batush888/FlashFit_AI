import os
import json
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutfitMatchingPipeline:
    """
    服装搭配推荐管道
    """
    
    def __init__(self, 
                 clip_model_path: Optional[str] = None,
                 classifier_model_path: Optional[str] = None,
                 templates_path: str = "data/templates",
                 device: str = "auto"):
        """
        初始化管道
        
        Args:
            clip_model_path: CLIP模型路径
            classifier_model_path: 分类器模型路径
            templates_path: 模板数据路径
            device: 设备类型
        """
        self.device = self._setup_device(device)
        self.templates_path = templates_path
        
        # 初始化组件
        self.clip_model = None
        self.classifier = None
        self.preprocessor = None
        
        # 加载数据
        self.templates = self._load_templates()
        self.color_rules = self._load_color_rules()
        self.outfit_rules = self._load_outfit_rules()
        
        logger.info(f"管道初始化完成，设备: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """
        设置计算设备
        
        Args:
            device: 设备类型
            
        Returns:
            PyTorch设备对象
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
            else:
                device = "cpu"
        
        return torch.device(device)
    
    def _load_templates(self) -> List[Dict[str, Any]]:
        """
        加载风格模板
        
        Returns:
            模板列表
        """
        templates = []
        
        # 加载基础模板
        base_templates_path = os.path.join(self.templates_path, "style_templates.json")
        if os.path.exists(base_templates_path):
            with open(base_templates_path, "r", encoding="utf-8") as f:
                templates.extend(json.load(f))
        
        # 加载生成的模板
        generated_templates_path = os.path.join(self.templates_path, "generated_templates.json")
        if os.path.exists(generated_templates_path):
            with open(generated_templates_path, "r", encoding="utf-8") as f:
                templates.extend(json.load(f))
        
        logger.info(f"加载了 {len(templates)} 个风格模板")
        return templates
    
    def _load_color_rules(self) -> Dict[str, Any]:
        """
        加载颜色规则
        
        Returns:
            颜色规则字典
        """
        color_rules_path = os.path.join(self.templates_path, "color_rules.json")
        if os.path.exists(color_rules_path):
            with open(color_rules_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _load_outfit_rules(self) -> Dict[str, Any]:
        """
        加载搭配规则
        
        Returns:
            搭配规则字典
        """
        outfit_rules_path = os.path.join(self.templates_path, "outfit_rules.json")
        if os.path.exists(outfit_rules_path):
            with open(outfit_rules_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def initialize_models(self):
        """
        初始化所有模型
        """
        try:
            # 延迟导入以避免循环导入
            from .clip_model import get_clip_model
            from .classifier import get_classifier
            from .preprocessing import get_preprocessor
            
            self.clip_model = get_clip_model()
            self.classifier = get_classifier()
            self.preprocessor = get_preprocessor()
            
            logger.info("所有模型初始化完成")
            
        except ImportError as e:
            logger.error(f"模型导入失败: {e}")
            raise
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            处理结果
        """
        if not self.preprocessor:
            self.initialize_models()
        
        try:
            # 加载和预处理图像
            image = self.preprocessor.load_and_validate_image(image_path)
            
            # 增强图像质量
            enhanced_image = self.preprocessor.enhance_image_quality(image)
            
            # 提取特征
            clip_features = self._extract_clip_features(enhanced_image)
            classification_result = self._classify_garment(enhanced_image)
            color_analysis = self.preprocessor.extract_dominant_colors(enhanced_image)
            
            result = {
                "image_path": image_path,
                "clip_features": clip_features.tolist() if isinstance(clip_features, torch.Tensor) else clip_features,
                "classification": classification_result,
                "colors": color_analysis,
                "processed_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"图像处理失败 {image_path}: {e}")
            raise
    
    def _extract_clip_features(self, image) -> torch.Tensor:
        """
        提取CLIP特征
        
        Args:
            image: PIL图像对象
            
        Returns:
            CLIP特征向量
        """
        if not self.clip_model:
            raise RuntimeError("CLIP模型未初始化")
        
        # 预处理图像
        image_tensor = self.preprocessor.preprocess_for_clip(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.clip_model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)  # 归一化
        
        return features.squeeze(0).cpu()
    
    def _classify_garment(self, image) -> Dict[str, Any]:
        """
        分类服装
        
        Args:
            image: PIL图像对象
            
        Returns:
            分类结果
        """
        if not self.classifier:
            raise RuntimeError("分类器未初始化")
        
        return self.classifier.classify_image(image)
    
    def find_matching_templates(self, 
                              item_features: Dict[str, Any],
                              occasion: Optional[str] = None,
                              season: Optional[str] = None,
                              style_preference: Optional[str] = None,
                              top_k: int = 10) -> List[Dict[str, Any]]:
        """
        查找匹配的模板
        
        Args:
            item_features: 物品特征
            occasion: 场合
            season: 季节
            style_preference: 风格偏好
            top_k: 返回数量
            
        Returns:
            匹配的模板列表
        """
        matching_templates = []
        
        for template in self.templates:
            score = self._calculate_template_similarity(item_features, template, 
                                                      occasion, season, style_preference)
            if score > 0.3:  # 最低相似度阈值
                template_copy = template.copy()
                template_copy["similarity_score"] = score
                matching_templates.append(template_copy)
        
        # 按相似度排序
        matching_templates.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return matching_templates[:top_k]
    
    def _calculate_template_similarity(self, 
                                     item_features: Dict[str, Any],
                                     template: Dict[str, Any],
                                     occasion: Optional[str] = None,
                                     season: Optional[str] = None,
                                     style_preference: Optional[str] = None) -> float:
        """
        计算模板相似度
        
        Args:
            item_features: 物品特征
            template: 模板
            occasion: 场合
            season: 季节
            style_preference: 风格偏好
            
        Returns:
            相似度分数
        """
        score = 0.0
        
        # 场合匹配
        if occasion and template.get("occasion") == occasion:
            score += 0.3
        
        # 季节匹配
        if season and season in template.get("season", []):
            score += 0.2
        
        # 风格匹配
        if style_preference and template.get("style") == style_preference:
            score += 0.2
        
        # 颜色兼容性
        item_color = item_features.get("classification", {}).get("primary_color")
        if item_color:
            template_colors = [item.get("color") for item in template.get("items", [])]
            if self._check_color_compatibility(item_color, template_colors):
                score += 0.2
        
        # 类别匹配（检查是否有相同类别的物品）
        item_category = item_features.get("classification", {}).get("garment_type")
        if item_category:
            template_categories = [item.get("category") for item in template.get("items", [])]
            if item_category in template_categories:
                score += 0.1
        
        return min(score, 1.0)  # 确保分数不超过1.0
    
    def _check_color_compatibility(self, base_color: str, template_colors: List[str]) -> bool:
        """
        检查颜色兼容性
        
        Args:
            base_color: 基础颜色
            template_colors: 模板颜色列表
            
        Returns:
            是否兼容
        """
        if not base_color or not template_colors:
            return False
        
        # 检查颜色规则
        for rule_type, rules in self.color_rules.items():
            if base_color in rules:
                compatible_colors = rules[base_color]
                for template_color in template_colors:
                    if template_color in compatible_colors:
                        return True
        
        return False
    
    def generate_outfit_suggestions(self, 
                                  item_features: Dict[str, Any],
                                  user_wardrobe: List[Dict[str, Any]] = None,
                                  occasion: Optional[str] = None,
                                  season: Optional[str] = None,
                                  count: int = 3) -> List[Dict[str, Any]]:
        """
        生成搭配建议
        
        Args:
            item_features: 目标物品特征
            user_wardrobe: 用户衣橱
            occasion: 场合
            season: 季节
            count: 建议数量
            
        Returns:
            搭配建议列表
        """
        suggestions = []
        
        # 查找匹配的模板
        matching_templates = self.find_matching_templates(
            item_features, occasion, season, top_k=count * 2
        )
        
        for i, template in enumerate(matching_templates[:count]):
            suggestion = self._create_outfit_suggestion(
                item_features, template, user_wardrobe, i + 1
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _create_outfit_suggestion(self, 
                                item_features: Dict[str, Any],
                                template: Dict[str, Any],
                                user_wardrobe: List[Dict[str, Any]] = None,
                                suggestion_id: int = 1) -> Dict[str, Any]:
        """
        创建单个搭配建议
        
        Args:
            item_features: 物品特征
            template: 模板
            user_wardrobe: 用户衣橱
            suggestion_id: 建议ID
            
        Returns:
            搭配建议
        """
        # 基础信息
        suggestion = {
            "id": f"suggestion_{suggestion_id:03d}",
            "template_id": template.get("id"),
            "template_name": template.get("name"),
            "similarity_score": template.get("similarity_score", 0.0),
            "style": template.get("style"),
            "occasion": template.get("occasion"),
            "season": template.get("season"),
            "items": [],
            "missing_items": [],
            "chinese_advice": self._generate_chinese_advice(item_features, template)
        }
        
        # 匹配用户衣橱中的物品
        if user_wardrobe:
            for template_item in template.get("items", []):
                matched_item = self._find_matching_wardrobe_item(
                    template_item, user_wardrobe
                )
                
                if matched_item:
                    suggestion["items"].append(matched_item)
                else:
                    suggestion["missing_items"].append(template_item)
        else:
            # 如果没有用户衣橱，使用模板物品
            suggestion["items"] = template.get("items", [])
        
        return suggestion
    
    def _find_matching_wardrobe_item(self, 
                                   template_item: Dict[str, Any],
                                   wardrobe: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        在用户衣橱中查找匹配的物品
        
        Args:
            template_item: 模板物品
            wardrobe: 用户衣橱
            
        Returns:
            匹配的物品或None
        """
        best_match = None
        best_score = 0.0
        
        for item in wardrobe:
            score = 0.0
            
            # 类别匹配
            if item.get("garment_type") == template_item.get("category"):
                score += 0.5
            
            # 颜色匹配
            if item.get("primary_color") == template_item.get("color"):
                score += 0.3
            elif self._check_color_compatibility(
                item.get("primary_color", ""), [template_item.get("color", "")]
            ):
                score += 0.2
            
            # 风格匹配
            if item.get("style") == template_item.get("style"):
                score += 0.2
            
            if score > best_score and score > 0.5:  # 最低匹配阈值
                best_score = score
                best_match = item.copy()
                best_match["match_score"] = score
        
        return best_match
    
    def _generate_chinese_advice(self, 
                               item_features: Dict[str, Any],
                               template: Dict[str, Any]) -> Dict[str, str]:
        """
        生成中文搭配建议
        
        Args:
            item_features: 物品特征
            template: 模板
            
        Returns:
            中文建议字典
        """
        # 这里是简化版本，实际应该调用LLM API
        style_advice = {
            "casual": "舒适自然，适合日常穿搭",
            "formal": "优雅正式，展现专业形象",
            "business": "干练大方，彰显职场魅力",
            "sporty": "活力十足，运动休闲风",
            "elegant": "高贵典雅，气质出众",
            "vintage": "复古经典，别具韵味",
            "modern": "时尚前卫，潮流感十足",
            "bohemian": "自由奔放，艺术气息浓厚",
            "minimalist": "简约大方，低调有品味",
            "trendy": "紧跟潮流，时尚感强烈"
        }
        
        style = template.get("style", "casual")
        occasion = template.get("occasion", "casual")
        
        title = f"{template.get('name', '时尚搭配')}"
        tips = [
            style_advice.get(style, "时尚搭配建议"),
            f"适合{occasion}场合",
            "颜色搭配和谐统一"
        ]
        
        return {
            "title_cn": title,
            "tips_cn": tips
        }
    
    def batch_process_images(self, 
                           image_paths: List[str],
                           output_dir: str = "data/processed") -> List[Dict[str, Any]]:
        """
        批量处理图像
        
        Args:
            image_paths: 图像路径列表
            output_dir: 输出目录
            
        Returns:
            处理结果列表
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"处理图像 {i+1}/{len(image_paths)}: {image_path}")
                result = self.process_image(image_path)
                results.append(result)
                
                # 保存结果
                result_path = os.path.join(output_dir, f"result_{i:04d}.json")
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                logger.error(f"处理失败 {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "status": "failed"
                })
        
        return results
    
    def save_pipeline_state(self, filepath: str):
        """
        保存管道状态
        
        Args:
            filepath: 保存路径
        """
        state = {
            "device": str(self.device),
            "templates_count": len(self.templates),
            "color_rules_count": len(self.color_rules),
            "outfit_rules_count": len(self.outfit_rules),
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        logger.info(f"管道状态已保存到 {filepath}")

# 全局管道实例
_pipeline = None

def get_pipeline() -> OutfitMatchingPipeline:
    """
    获取全局管道实例（单例模式）
    
    Returns:
        管道实例
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = OutfitMatchingPipeline()
    return _pipeline

def initialize_pipeline():
    """
    初始化管道
    """
    pipeline = get_pipeline()
    pipeline.initialize_models()
    logger.info("管道初始化完成")

if __name__ == "__main__":
    # 测试管道
    pipeline = OutfitMatchingPipeline()
    pipeline.initialize_models()
    
    # 保存管道状态
    pipeline.save_pipeline_state("models/pipeline_state.json")
    
    print("管道测试完成")