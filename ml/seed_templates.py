import json
import os
import random
from typing import List, Dict, Any, Tuple
import numpy as np
from itertools import combinations

class TemplateGenerator:
    """
    风格模板生成器
    """
    
    def __init__(self):
        self.categories = {
            "tops": ["shirt", "sweater", "jacket", "coat"],
            "bottoms": ["pants", "skirt", "shorts"],
            "dresses": ["dress"],
            "shoes": ["shoes"],
            "accessories": ["accessory"]
        }
        
        self.colors = {
            "neutral": ["black", "white", "gray", "beige", "brown"],
            "warm": ["red", "orange", "yellow", "pink"],
            "cool": ["blue", "green", "purple"],
            "earth": ["brown", "beige", "olive", "tan"]
        }
        
        self.styles = [
            "casual", "formal", "business", "sporty", "elegant",
            "vintage", "modern", "bohemian", "minimalist", "trendy"
        ]
        
        self.occasions = [
            "work", "party", "casual", "formal", "sport",
            "date", "travel", "home", "shopping", "meeting"
        ]
        
        self.seasons = ["spring", "summer", "autumn", "winter"]
        
        # 加载现有数据
        self.load_existing_data()
    
    def load_existing_data(self):
        """
        加载现有的模板和规则数据
        """
        try:
            with open("data/templates/style_templates.json", "r", encoding="utf-8") as f:
                self.existing_templates = json.load(f)
        except FileNotFoundError:
            self.existing_templates = []
        
        try:
            with open("data/templates/color_rules.json", "r", encoding="utf-8") as f:
                self.color_rules = json.load(f)
        except FileNotFoundError:
            self.color_rules = {}
        
        try:
            with open("data/mock/fashion_items.json", "r", encoding="utf-8") as f:
                self.fashion_items = json.load(f)
        except FileNotFoundError:
            self.fashion_items = []
    
    def get_compatible_colors(self, base_color: str) -> List[str]:
        """
        获取与基础颜色兼容的颜色
        
        Args:
            base_color: 基础颜色
            
        Returns:
            兼容颜色列表
        """
        compatible = set()
        
        # 检查所有颜色规则
        for rule_type, rules in self.color_rules.items():
            if base_color in rules:
                compatible.update(rules[base_color])
        
        # 如果没有找到规则，使用中性色
        if not compatible:
            compatible = set(self.colors["neutral"])
        
        return list(compatible)
    
    def generate_outfit_combination(self, style: str, occasion: str, season: str) -> Dict[str, Any]:
        """
        生成单个搭配组合
        
        Args:
            style: 风格
            occasion: 场合
            season: 季节
            
        Returns:
            搭配组合
        """
        outfit_items = []
        
        # 选择基础颜色
        if season == "winter":
            base_colors = self.colors["neutral"] + self.colors["earth"]
        elif season == "summer":
            base_colors = self.colors["cool"] + ["white", "beige"]
        elif season in ["spring", "autumn"]:
            base_colors = self.colors["warm"] + self.colors["neutral"]
        else:
            base_colors = sum(self.colors.values(), [])
        
        base_color = random.choice(base_colors)
        compatible_colors = self.get_compatible_colors(base_color)
        
        # 决定是否包含连衣裙
        include_dress = random.choice([True, False]) and occasion in ["party", "date", "formal"]
        
        if include_dress:
            # 连衣裙搭配
            outfit_items.append({
                "category": "dress",
                "color": base_color,
                "style": style
            })
            
            # 添加鞋子
            shoe_color = random.choice(compatible_colors)
            outfit_items.append({
                "category": "shoes",
                "color": shoe_color,
                "style": style
            })
            
            # 可能添加外套
            if season in ["autumn", "winter"] or random.choice([True, False]):
                jacket_color = random.choice(compatible_colors)
                jacket_type = "coat" if season == "winter" else "jacket"
                outfit_items.append({
                    "category": jacket_type,
                    "color": jacket_color,
                    "style": style
                })
        else:
            # 分体搭配
            # 上装
            top_category = random.choice(self.categories["tops"])
            if season == "winter" and top_category in ["shirt"]:
                top_category = "sweater"  # 冬天更可能选择毛衣
            
            outfit_items.append({
                "category": top_category,
                "color": base_color,
                "style": style
            })
            
            # 下装
            bottom_category = random.choice(self.categories["bottoms"])
            if season == "winter" and bottom_category == "shorts":
                bottom_category = "pants"  # 冬天不穿短裤
            
            bottom_color = random.choice(compatible_colors)
            outfit_items.append({
                "category": bottom_category,
                "color": bottom_color,
                "style": style
            })
            
            # 鞋子
            shoe_color = random.choice(compatible_colors)
            outfit_items.append({
                "category": "shoes",
                "color": shoe_color,
                "style": style
            })
        
        # 可能添加配饰
        if random.choice([True, False]):
            accessory_color = random.choice(compatible_colors)
            outfit_items.append({
                "category": "accessory",
                "color": accessory_color,
                "style": style
            })
        
        return {
            "items": outfit_items,
            "base_color": base_color,
            "compatible_colors": compatible_colors
        }
    
    def generate_style_keywords(self, style: str, occasion: str, season: str) -> List[str]:
        """
        生成风格关键词
        
        Args:
            style: 风格
            occasion: 场合
            season: 季节
            
        Returns:
            关键词列表
        """
        keyword_map = {
            "casual": ["comfortable", "relaxed", "everyday", "easy-going"],
            "formal": ["elegant", "sophisticated", "polished", "refined"],
            "business": ["professional", "sharp", "confident", "authoritative"],
            "sporty": ["athletic", "dynamic", "energetic", "functional"],
            "elegant": ["graceful", "classy", "timeless", "sophisticated"],
            "vintage": ["retro", "classic", "nostalgic", "timeless"],
            "modern": ["contemporary", "sleek", "cutting-edge", "fresh"],
            "bohemian": ["free-spirited", "artistic", "eclectic", "flowing"],
            "minimalist": ["clean", "simple", "understated", "refined"],
            "trendy": ["fashionable", "current", "stylish", "on-trend"]
        }
        
        occasion_keywords = {
            "work": ["professional", "appropriate", "polished"],
            "party": ["festive", "eye-catching", "glamorous"],
            "casual": ["comfortable", "relaxed", "effortless"],
            "formal": ["elegant", "sophisticated", "refined"],
            "sport": ["functional", "comfortable", "practical"],
            "date": ["romantic", "charming", "attractive"],
            "travel": ["comfortable", "versatile", "practical"]
        }
        
        season_keywords = {
            "spring": ["fresh", "light", "renewed"],
            "summer": ["breezy", "cool", "vibrant"],
            "autumn": ["cozy", "warm", "rich"],
            "winter": ["warm", "layered", "cozy"]
        }
        
        keywords = []
        keywords.extend(random.sample(keyword_map.get(style, []), min(2, len(keyword_map.get(style, [])))))
        keywords.extend(random.sample(occasion_keywords.get(occasion, []), min(1, len(occasion_keywords.get(occasion, [])))))
        keywords.extend(random.sample(season_keywords.get(season, []), min(1, len(season_keywords.get(season, [])))))
        
        return list(set(keywords))[:4]  # 最多4个关键词
    
    def generate_chinese_names(self, style: str, occasion: str, season: str) -> Tuple[str, str]:
        """
        生成中文名称和描述
        
        Args:
            style: 风格
            occasion: 场合
            season: 季节
            
        Returns:
            (名称, 描述)
        """
        style_names = {
            "casual": "休闲",
            "formal": "正装",
            "business": "商务",
            "sporty": "运动",
            "elegant": "优雅",
            "vintage": "复古",
            "modern": "现代",
            "bohemian": "波西米亚",
            "minimalist": "极简",
            "trendy": "时尚"
        }
        
        occasion_names = {
            "work": "工作",
            "party": "派对",
            "casual": "日常",
            "formal": "正式",
            "sport": "运动",
            "date": "约会",
            "travel": "旅行",
            "home": "居家",
            "shopping": "购物",
            "meeting": "会议"
        }
        
        season_names = {
            "spring": "春季",
            "summer": "夏季",
            "autumn": "秋季",
            "winter": "冬季"
        }
        
        style_cn = style_names.get(style, style)
        occasion_cn = occasion_names.get(occasion, occasion)
        season_cn = season_names.get(season, season)
        
        name = f"{season_cn}{style_cn}"
        description = f"适合{occasion_cn}场合的{season_cn}{style_cn}搭配"
        
        return name, description
    
    def generate_templates(self, count: int = 50) -> List[Dict[str, Any]]:
        """
        生成多个风格模板
        
        Args:
            count: 生成数量
            
        Returns:
            模板列表
        """
        templates = []
        
        for i in range(count):
            # 随机选择风格、场合和季节
            style = random.choice(self.styles)
            occasion = random.choice(self.occasions)
            season = random.choice(self.seasons)
            
            # 生成搭配组合
            outfit = self.generate_outfit_combination(style, occasion, season)
            
            # 生成关键词
            keywords = self.generate_style_keywords(style, occasion, season)
            
            # 生成中文名称
            name_cn, description_cn = self.generate_chinese_names(style, occasion, season)
            
            # 创建模板
            template = {
                "id": f"generated_{i:04d}",
                "name": name_cn,
                "name_en": f"{season}_{style}_{occasion}",
                "description": description_cn,
                "items": outfit["items"],
                "occasion": occasion,
                "season": [season] if season else self.seasons,
                "style": style,
                "style_keywords": keywords,
                "base_color": outfit["base_color"],
                "compatible_colors": outfit["compatible_colors"],
                "popularity_score": random.uniform(0.6, 1.0),  # 模拟流行度
                "difficulty_level": random.choice(["easy", "medium", "advanced"]),
                "tags": [style, occasion, season] + keywords[:2]
            }
            
            templates.append(template)
        
        return templates
    
    def generate_outfit_rules(self) -> Dict[str, Any]:
        """
        生成搭配规则
        
        Returns:
            搭配规则字典
        """
        rules = {
            "category_combinations": {
                "must_have": {
                    "casual": ["tops", "bottoms", "shoes"],
                    "formal": ["tops", "bottoms", "shoes"],
                    "party": ["dresses", "shoes"],
                    "sport": ["tops", "bottoms", "shoes"]
                },
                "optional": {
                    "casual": ["accessories"],
                    "formal": ["accessories", "jacket"],
                    "party": ["accessories", "jacket"],
                    "sport": ["accessories"]
                }
            },
            "color_harmony": {
                "max_colors": 4,
                "neutral_ratio": 0.6,  # 中性色占比
                "accent_ratio": 0.4    # 强调色占比
            },
            "seasonal_preferences": {
                "spring": {
                    "preferred_colors": ["green", "pink", "yellow", "white"],
                    "avoid_colors": ["black", "brown"],
                    "preferred_materials": ["cotton", "linen"]
                },
                "summer": {
                    "preferred_colors": ["blue", "white", "yellow", "orange"],
                    "avoid_colors": ["black", "brown"],
                    "preferred_materials": ["cotton", "linen", "silk"]
                },
                "autumn": {
                    "preferred_colors": ["brown", "orange", "red", "yellow"],
                    "avoid_colors": ["bright_pink", "neon"],
                    "preferred_materials": ["wool", "cotton", "denim"]
                },
                "winter": {
                    "preferred_colors": ["black", "white", "gray", "red"],
                    "avoid_colors": ["bright_yellow", "neon"],
                    "preferred_materials": ["wool", "cashmere", "leather"]
                }
            },
            "occasion_guidelines": {
                "work": {
                    "formality_level": "high",
                    "preferred_styles": ["business", "formal", "minimalist"],
                    "avoid_styles": ["bohemian", "sporty"]
                },
                "party": {
                    "formality_level": "medium-high",
                    "preferred_styles": ["elegant", "trendy", "vintage"],
                    "avoid_styles": ["sporty", "casual"]
                },
                "casual": {
                    "formality_level": "low",
                    "preferred_styles": ["casual", "modern", "minimalist"],
                    "avoid_styles": ["formal"]
                }
            }
        }
        
        return rules
    
    def save_templates(self, templates: List[Dict[str, Any]], filename: str = "generated_templates.json"):
        """
        保存生成的模板
        
        Args:
            templates: 模板列表
            filename: 文件名
        """
        filepath = f"data/templates/{filename}"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(templates, f, ensure_ascii=False, indent=2)
        
        print(f"保存了 {len(templates)} 个模板到 {filepath}")
    
    def save_rules(self, rules: Dict[str, Any], filename: str = "outfit_rules.json"):
        """
        保存搭配规则
        
        Args:
            rules: 规则字典
            filename: 文件名
        """
        filepath = f"data/templates/{filename}"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
        
        print(f"保存了搭配规则到 {filepath}")

def create_embedding_templates():
    """
    创建用于嵌入的模板数据
    """
    # 这些是用于CLIP嵌入的文本模板
    text_templates = [
        # 基础描述模板
        "a photo of a {category} in {color} color",
        "a {style} {category} that is {color}",
        "a {color} {category} for {occasion} occasions",
        "a {season} {category} in {color}",
        
        # 风格描述模板
        "a {style} and {adjective} {category}",
        "a {category} with {style} design in {color}",
        "a {color} {category} perfect for {occasion}",
        "a {season} {style} {category}",
        
        # 详细描述模板
        "a {adjective} {color} {category} suitable for {occasion}",
        "a {style} {category} in {color} color for {season} season",
        "a {color} {category} with {style} aesthetic",
        "a {season} {category} in {color} that looks {adjective}"
    ]
    
    # 形容词列表
    adjectives = [
        "elegant", "casual", "comfortable", "stylish", "modern",
        "classic", "trendy", "sophisticated", "simple", "chic",
        "versatile", "practical", "fashionable", "timeless", "contemporary"
    ]
    
    embedding_data = {
        "text_templates": text_templates,
        "adjectives": adjectives,
        "usage": "These templates are used to generate text descriptions for CLIP embeddings"
    }
    
    with open("data/templates/embedding_templates.json", "w", encoding="utf-8") as f:
        json.dump(embedding_data, f, ensure_ascii=False, indent=2)
    
    print("创建了嵌入模板数据")

def main():
    """
    主函数
    """
    print("开始生成风格模板...")
    
    # 创建模板生成器
    generator = TemplateGenerator()
    
    # 生成模板
    templates = generator.generate_templates(count=100)  # 生成100个模板
    
    # 生成规则
    rules = generator.generate_outfit_rules()
    
    # 创建嵌入模板
    create_embedding_templates()
    
    # 保存数据
    generator.save_templates(templates)
    generator.save_rules(rules)
    
    # 统计信息
    print("\n生成统计:")
    print(f"- 总模板数: {len(templates)}")
    
    # 按风格统计
    style_count = {}
    for template in templates:
        style = template["style"]
        style_count[style] = style_count.get(style, 0) + 1
    
    print("- 按风格分布:")
    for style, count in sorted(style_count.items()):
        print(f"  {style}: {count}")
    
    # 按场合统计
    occasion_count = {}
    for template in templates:
        occasion = template["occasion"]
        occasion_count[occasion] = occasion_count.get(occasion, 0) + 1
    
    print("- 按场合分布:")
    for occasion, count in sorted(occasion_count.items()):
        print(f"  {occasion}: {count}")
    
    print("\n模板生成完成！")
    print("\n生成的文件:")
    print("- data/templates/generated_templates.json")
    print("- data/templates/outfit_rules.json")
    print("- data/templates/embedding_templates.json")

if __name__ == "__main__":
    main()