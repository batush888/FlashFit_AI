import os
import requests
import json
from typing import List, Dict, Any
import zipfile
from pathlib import Path
import shutil

def create_directories():
    """
    创建必要的目录结构
    """
    directories = [
        "data/raw",
        "data/processed",
        "data/templates",
        "data/mock",
        "models/checkpoints",
        "models/embeddings"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

def download_file(url: str, filepath: str, chunk_size: int = 8192) -> bool:
    """
    下载文件
    
    Args:
        url: 下载链接
        filepath: 保存路径
        chunk_size: 块大小
        
    Returns:
        是否下载成功
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        
        print(f"下载完成: {filepath}")
        return True
        
    except Exception as e:
        print(f"下载失败 {url}: {str(e)}")
        return False

def create_mock_fashion_data():
    """
    创建模拟时尚数据集
    """
    # 服装类别和颜色
    categories = [
        "shirt", "pants", "dress", "skirt", "jacket", 
        "shoes", "accessory", "sweater", "shorts", "coat"
    ]
    
    colors = [
        "black", "white", "gray", "red", "blue", "green", 
        "yellow", "orange", "purple", "pink", "brown", "beige"
    ]
    
    styles = [
        "casual", "formal", "business", "sporty", "elegant", 
        "vintage", "modern", "bohemian", "minimalist", "trendy"
    ]
    
    occasions = [
        "work", "party", "casual", "formal", "sport", 
        "date", "travel", "home", "shopping", "meeting"
    ]
    
    # 生成模拟数据
    mock_items = []
    
    for i in range(200):  # 生成200个模拟物品
        item = {
            "id": f"mock_{i:04d}",
            "category": categories[i % len(categories)],
            "primary_color": colors[i % len(colors)],
            "secondary_color": colors[(i + 3) % len(colors)] if i % 3 == 0 else None,
            "style": styles[i % len(styles)],
            "occasion": occasions[i % len(occasions)],
            "brand": f"Brand_{(i % 20) + 1}",
            "price_range": ["low", "medium", "high"][i % 3],
            "season": ["spring", "summer", "autumn", "winter"][i % 4],
            "material": ["cotton", "polyester", "wool", "silk", "denim", "leather"][i % 6],
            "size": ["XS", "S", "M", "L", "XL"][i % 5],
            "tags": [styles[i % len(styles)], occasions[i % len(occasions)]],
            "description": f"A {styles[i % len(styles)]} {colors[i % len(colors)]} {categories[i % len(categories)]} perfect for {occasions[i % len(occasions)]} occasions.",
            "image_url": f"https://example.com/images/mock_{i:04d}.jpg",
            "embedding": [0.1 * (i % 100) for _ in range(512)]  # 模拟CLIP嵌入
        }
        mock_items.append(item)
    
    # 保存模拟数据
    with open("data/mock/fashion_items.json", "w", encoding="utf-8") as f:
        json.dump(mock_items, f, ensure_ascii=False, indent=2)
    
    print(f"生成了 {len(mock_items)} 个模拟时尚物品")
    
    return mock_items

def create_style_templates():
    """
    创建风格模板数据
    """
    templates = [
        {
            "id": "casual_everyday",
            "name": "休闲日常",
            "description": "舒适的日常穿搭",
            "items": [
                {"category": "shirt", "color": "white", "style": "casual"},
                {"category": "pants", "color": "blue", "style": "casual"},
                {"category": "shoes", "color": "white", "style": "casual"}
            ],
            "occasion": "casual",
            "season": ["spring", "summer", "autumn"],
            "style_keywords": ["comfortable", "relaxed", "everyday"]
        },
        {
            "id": "business_formal",
            "name": "商务正装",
            "description": "专业的商务装扮",
            "items": [
                {"category": "shirt", "color": "white", "style": "formal"},
                {"category": "pants", "color": "black", "style": "formal"},
                {"category": "jacket", "color": "black", "style": "formal"},
                {"category": "shoes", "color": "black", "style": "formal"}
            ],
            "occasion": "work",
            "season": ["spring", "summer", "autumn", "winter"],
            "style_keywords": ["professional", "elegant", "sophisticated"]
        },
        {
            "id": "party_chic",
            "name": "派对时尚",
            "description": "适合聚会的时尚搭配",
            "items": [
                {"category": "dress", "color": "black", "style": "elegant"},
                {"category": "shoes", "color": "black", "style": "elegant"},
                {"category": "accessory", "color": "gold", "style": "elegant"}
            ],
            "occasion": "party",
            "season": ["spring", "summer", "autumn", "winter"],
            "style_keywords": ["glamorous", "stylish", "eye-catching"]
        },
        {
            "id": "sporty_active",
            "name": "运动活力",
            "description": "运动和健身的搭配",
            "items": [
                {"category": "shirt", "color": "gray", "style": "sporty"},
                {"category": "shorts", "color": "black", "style": "sporty"},
                {"category": "shoes", "color": "white", "style": "sporty"}
            ],
            "occasion": "sport",
            "season": ["spring", "summer", "autumn"],
            "style_keywords": ["athletic", "comfortable", "functional"]
        },
        {
            "id": "romantic_date",
            "name": "浪漫约会",
            "description": "适合约会的浪漫搭配",
            "items": [
                {"category": "dress", "color": "pink", "style": "elegant"},
                {"category": "shoes", "color": "beige", "style": "elegant"},
                {"category": "accessory", "color": "silver", "style": "elegant"}
            ],
            "occasion": "date",
            "season": ["spring", "summer"],
            "style_keywords": ["romantic", "feminine", "charming"]
        },
        {
            "id": "winter_cozy",
            "name": "冬日温暖",
            "description": "温暖舒适的冬季搭配",
            "items": [
                {"category": "sweater", "color": "beige", "style": "casual"},
                {"category": "pants", "color": "brown", "style": "casual"},
                {"category": "coat", "color": "brown", "style": "casual"},
                {"category": "shoes", "color": "brown", "style": "casual"}
            ],
            "occasion": "casual",
            "season": ["winter"],
            "style_keywords": ["cozy", "warm", "comfortable"]
        },
        {
            "id": "summer_fresh",
            "name": "夏日清新",
            "description": "清爽的夏季搭配",
            "items": [
                {"category": "shirt", "color": "blue", "style": "casual"},
                {"category": "shorts", "color": "white", "style": "casual"},
                {"category": "shoes", "color": "white", "style": "casual"}
            ],
            "occasion": "casual",
            "season": ["summer"],
            "style_keywords": ["fresh", "light", "breezy"]
        },
        {
            "id": "vintage_retro",
            "name": "复古怀旧",
            "description": "经典的复古风格",
            "items": [
                {"category": "shirt", "color": "white", "style": "vintage"},
                {"category": "skirt", "color": "red", "style": "vintage"},
                {"category": "shoes", "color": "black", "style": "vintage"},
                {"category": "accessory", "color": "black", "style": "vintage"}
            ],
            "occasion": "casual",
            "season": ["spring", "summer", "autumn"],
            "style_keywords": ["retro", "classic", "timeless"]
        },
        {
            "id": "minimalist_modern",
            "name": "极简现代",
            "description": "简约现代的搭配",
            "items": [
                {"category": "shirt", "color": "white", "style": "minimalist"},
                {"category": "pants", "color": "black", "style": "minimalist"},
                {"category": "shoes", "color": "white", "style": "minimalist"}
            ],
            "occasion": "casual",
            "season": ["spring", "summer", "autumn", "winter"],
            "style_keywords": ["clean", "simple", "modern"]
        },
        {
            "id": "bohemian_free",
            "name": "波西米亚",
            "description": "自由奔放的波西米亚风格",
            "items": [
                {"category": "dress", "color": "orange", "style": "bohemian"},
                {"category": "shoes", "color": "brown", "style": "bohemian"},
                {"category": "accessory", "color": "brown", "style": "bohemian"}
            ],
            "occasion": "casual",
            "season": ["spring", "summer", "autumn"],
            "style_keywords": ["free-spirited", "artistic", "flowing"]
        }
    ]
    
    # 保存模板数据
    with open("data/templates/style_templates.json", "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)
    
    print(f"创建了 {len(templates)} 个风格模板")
    
    return templates

def create_color_compatibility_rules():
    """
    创建颜色搭配规则
    """
    color_rules = {
        "complementary": {
            "red": ["green", "white", "black", "beige"],
            "blue": ["orange", "white", "black", "beige"],
            "green": ["red", "white", "black", "beige"],
            "yellow": ["purple", "white", "black", "gray"],
            "orange": ["blue", "white", "black", "brown"],
            "purple": ["yellow", "white", "black", "gray"]
        },
        "neutral_base": {
            "white": ["black", "gray", "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown"],
            "black": ["white", "gray", "red", "blue", "green", "yellow", "orange", "purple", "pink"],
            "gray": ["white", "black", "red", "blue", "green", "yellow", "orange", "purple", "pink"],
            "beige": ["white", "black", "brown", "green", "blue", "red"]
        },
        "monochromatic": {
            "blue": ["blue", "white", "black"],
            "red": ["red", "white", "black"],
            "green": ["green", "white", "black"],
            "brown": ["brown", "beige", "white", "black"]
        },
        "seasonal": {
            "spring": ["green", "pink", "yellow", "white", "beige"],
            "summer": ["blue", "white", "yellow", "orange", "pink"],
            "autumn": ["brown", "orange", "red", "yellow", "beige"],
            "winter": ["black", "white", "gray", "red", "blue"]
        }
    }
    
    # 保存颜色规则
    with open("data/templates/color_rules.json", "w", encoding="utf-8") as f:
        json.dump(color_rules, f, ensure_ascii=False, indent=2)
    
    print("创建了颜色搭配规则")
    
    return color_rules

def create_sample_images_metadata():
    """
    创建示例图片的元数据
    """
    # 这里创建一些示例图片的元数据，实际图片可以后续添加
    sample_metadata = [
        {
            "filename": "sample_shirt_001.jpg",
            "category": "shirt",
            "color": "white",
            "style": "casual",
            "description": "白色休闲衬衫",
            "tags": ["basic", "versatile", "cotton"]
        },
        {
            "filename": "sample_pants_001.jpg",
            "category": "pants",
            "color": "blue",
            "style": "casual",
            "description": "蓝色牛仔裤",
            "tags": ["denim", "classic", "comfortable"]
        },
        {
            "filename": "sample_dress_001.jpg",
            "category": "dress",
            "color": "black",
            "style": "elegant",
            "description": "黑色优雅连衣裙",
            "tags": ["formal", "elegant", "versatile"]
        },
        {
            "filename": "sample_shoes_001.jpg",
            "category": "shoes",
            "color": "black",
            "style": "formal",
            "description": "黑色正装皮鞋",
            "tags": ["leather", "formal", "business"]
        },
        {
            "filename": "sample_jacket_001.jpg",
            "category": "jacket",
            "color": "gray",
            "style": "business",
            "description": "灰色西装外套",
            "tags": ["wool", "business", "professional"]
        }
    ]
    
    # 保存示例元数据
    with open("data/mock/sample_images.json", "w", encoding="utf-8") as f:
        json.dump(sample_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"创建了 {len(sample_metadata)} 个示例图片元数据")
    
    return sample_metadata

def download_open_clip_model():
    """
    下载OpenCLIP模型（如果需要）
    """
    print("OpenCLIP模型将在首次使用时自动下载")
    print("如需手动下载，请运行: pip install open_clip_torch")
    
    # 创建模型配置文件
    model_config = {
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "cache_dir": "models/checkpoints",
        "device": "auto"  # 自动检测GPU/CPU
    }
    
    with open("models/clip_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)
    
    print("创建了CLIP模型配置文件")

def main():
    """
    主函数 - 执行所有数据准备步骤
    """
    print("开始准备AI Outfit Matcher数据...")
    
    # 创建目录结构
    create_directories()
    
    # 创建模拟数据
    create_mock_fashion_data()
    
    # 创建风格模板
    create_style_templates()
    
    # 创建颜色规则
    create_color_compatibility_rules()
    
    # 创建示例图片元数据
    create_sample_images_metadata()
    
    # 配置CLIP模型
    download_open_clip_model()
    
    print("\n数据准备完成！")
    print("\n创建的文件:")
    print("- data/mock/fashion_items.json (200个模拟时尚物品)")
    print("- data/templates/style_templates.json (10个风格模板)")
    print("- data/templates/color_rules.json (颜色搭配规则)")
    print("- data/mock/sample_images.json (示例图片元数据)")
    print("- models/clip_config.json (CLIP模型配置)")
    print("\n下一步: 运行 python ml/seed_templates.py 来生成更多模板数据")

if __name__ == "__main__":
    main()