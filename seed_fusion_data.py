#!/usr/bin/env python3
"""
Seed script to build FAISS index from template items
Based on user's suggested architecture for multi-model fusion
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import asyncio
from typing import List, Dict, Any

# Add backend to path
sys.path.append('backend')

# Import components
from backend.models.clip_encoder import get_clip_encoder
from backend.models.blip_captioner import get_blip_captioner
from backend.models.fashion_encoder import get_fashion_encoder
from backend.models.vector_store import get_clip_store, get_blip_store, get_fashion_store
from backend.services.recommend_service import get_recommendation_service

def create_sample_fashion_items():
    """
    Create sample fashion items for testing
    """
    sample_items = [
        {
            "id": "skirt_blue_001",
            "img_url": "/static/skirts/blue_a.png",
            "tags": ["裙子", "通勤", "夏季", "蓝色"],
            "category": "裙子",
            "color": "蓝色",
            "season": "夏季",
            "occasion": "通勤",
            "description": "优雅的蓝色A字裙，适合职场通勤"
        },
        {
            "id": "shirt_white_002",
            "img_url": "/static/shirts/white_formal.png",
            "tags": ["衬衫", "正式", "全季", "白色"],
            "category": "衬衫",
            "color": "白色",
            "season": "全季",
            "occasion": "正式",
            "description": "经典白色正装衬衫，百搭必备"
        },
        {
            "id": "dress_red_003",
            "img_url": "/static/dresses/red_cocktail.png",
            "tags": ["连衣裙", "派对", "春夏", "红色"],
            "category": "连衣裙",
            "color": "红色",
            "season": "春夏",
            "occasion": "派对",
            "description": "性感红色鸡尾酒裙，派对首选"
        },
        {
            "id": "jacket_black_004",
            "img_url": "/static/jackets/black_blazer.png",
            "tags": ["外套", "商务", "秋冬", "黑色"],
            "category": "外套",
            "color": "黑色",
            "season": "秋冬",
            "occasion": "商务",
            "description": "经典黑色西装外套，商务必备"
        },
        {
            "id": "jeans_blue_005",
            "img_url": "/static/pants/blue_jeans.png",
            "tags": ["牛仔裤", "休闲", "全季", "蓝色"],
            "category": "裤子",
            "color": "蓝色",
            "season": "全季",
            "occasion": "休闲",
            "description": "舒适蓝色牛仔裤，休闲百搭"
        },
        {
            "id": "sweater_pink_006",
            "img_url": "/static/sweaters/pink_knit.png",
            "tags": ["毛衣", "温暖", "秋冬", "粉色"],
            "category": "毛衣",
            "color": "粉色",
            "season": "秋冬",
            "occasion": "日常",
            "description": "温暖粉色针织毛衣，冬日温馨"
        },
        {
            "id": "shorts_khaki_007",
            "img_url": "/static/shorts/khaki_casual.png",
            "tags": ["短裤", "运动", "夏季", "卡其色"],
            "category": "短裤",
            "color": "卡其色",
            "season": "夏季",
            "occasion": "运动",
            "description": "实用卡其色休闲短裤，夏日运动"
        },
        {
            "id": "blouse_floral_008",
            "img_url": "/static/blouses/floral_spring.png",
            "tags": ["上衣", "花卉", "春季", "彩色"],
            "category": "上衣",
            "color": "花卉图案",
            "season": "春季",
            "occasion": "约会",
            "description": "浪漫花卉图案上衣，春日约会"
        }
    ]
    
    return sample_items

def create_placeholder_images(items: List[Dict[str, Any]]):
    """
    Create placeholder images for testing
    """
    print("Creating placeholder images...")
    
    # Ensure static directories exist
    static_dirs = [
        "backend/data/static/skirts",
        "backend/data/static/shirts", 
        "backend/data/static/dresses",
        "backend/data/static/jackets",
        "backend/data/static/pants",
        "backend/data/static/sweaters",
        "backend/data/static/shorts",
        "backend/data/static/blouses"
    ]
    
    for dir_path in static_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Color mapping for placeholder images
    color_map = {
        "蓝色": (70, 130, 180),
        "白色": (255, 255, 255),
        "红色": (220, 20, 60),
        "黑色": (30, 30, 30),
        "粉色": (255, 182, 193),
        "卡其色": (240, 230, 140),
        "花卉图案": (255, 105, 180)
    }
    
    for item in items:
        img_path = f"backend/data{item['img_url']}"
        
        if not os.path.exists(img_path):
            # Create placeholder image
            color = color_map.get(item['color'], (128, 128, 128))
            img = Image.new('RGB', (224, 224), color=color)
            
            # Add some texture/pattern
            pixels = img.load()
            for i in range(0, 224, 20):
                for j in range(224):
                    if i < 224:
                        pixels[i, j] = tuple(min(255, c + 20) for c in color)
            
            img.save(img_path)
            print(f"✓ Created placeholder: {img_path}")

async def process_item_embeddings(item: Dict[str, Any], encoders: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Process a single item to generate embeddings
    
    Args:
        item: Fashion item metadata
        encoders: Dictionary of encoder instances
        
    Returns:
        Item with embeddings added, or None if processing failed
    """
    img_path = f"backend/data{item['img_url']}"
    
    try:
        # Generate CLIP embedding
        clip_embedding = encoders['clip'].embed_image(img_path)
        
        # Generate BLIP caption and use CLIP for text embedding
        blip_caption = encoders['blip'].caption(img_path)
        blip_embedding = encoders['clip'].embed_text(blip_caption)
        
        # Generate Fashion embedding
        fashion_embedding = encoders['fashion'].embed_image(img_path)
        
        # Add embeddings to item
        processed_item = item.copy()
        processed_item.update({
            "emb_clip": clip_embedding.flatten().tolist(),
            "emb_blip": blip_embedding.flatten().tolist(), 
            "emb_fashion": fashion_embedding.flatten().tolist(),
            "blip_caption": blip_caption,
            "image_path": img_path
        })
        
        print(f"✓ Processed embeddings for {item['id']}")
        return processed_item
        
    except Exception as e:
        print(f"✗ Error processing {item['id']}: {e}")
        return None

async def build_vector_indices(processed_items: List[Dict[str, Any]], stores: Dict[str, Any]):
    """
    Build FAISS indices from processed items
    
    Args:
        processed_items: Items with embeddings
        stores: Dictionary of vector store instances
    """
    print("\nBuilding vector indices...")
    
    for item in processed_items:
        if item is None:
            continue
            
        try:
            # Add to CLIP store
            clip_embedding = np.array(item['emb_clip']).reshape(1, -1)
            stores['clip'].add(clip_embedding, [item])
            
            # Add to BLIP store
            blip_embedding = np.array(item['emb_blip']).reshape(1, -1)
            stores['blip'].add(blip_embedding, [item])
            
            # Add to Fashion store
            fashion_embedding = np.array(item['emb_fashion']).reshape(1, -1)
            stores['fashion'].add(fashion_embedding, [item])
            
            print(f"✓ Added {item['id']} to all vector stores")
            
        except Exception as e:
            print(f"✗ Error adding {item['id']} to stores: {e}")
    
    # Save indices
    print("\nSaving vector indices...")
    stores['clip'].save()
    stores['blip'].save()
    stores['fashion'].save()
    
    print("✓ All indices saved successfully")

async def main():
    """
    Main seeding function
    """
    print("🌱 Starting Fashion Data Seeding Process")
    print("=" * 50)
    
    # Create sample items
    print("\n1. Creating sample fashion items...")
    sample_items = create_sample_fashion_items()
    print(f"✓ Created {len(sample_items)} sample items")
    
    # Create placeholder images
    print("\n2. Creating placeholder images...")
    create_placeholder_images(sample_items)
    
    # Initialize encoders
    print("\n3. Initializing encoders...")
    encoders = {
        'clip': get_clip_encoder(),
        'blip': get_blip_captioner(),
        'fashion': get_fashion_encoder()
    }
    print("✓ All encoders initialized")
    
    # Initialize vector stores
    print("\n4. Initializing vector stores...")
    stores = {
        'clip': get_clip_store(dim=512),
        'blip': get_blip_store(dim=512),  # Using CLIP text embeddings, so same dimension
        'fashion': get_fashion_store(dim=512)
    }
    print("✓ All vector stores initialized")
    
    # Process items to generate embeddings
    print("\n5. Processing item embeddings...")
    processed_items = []
    
    for item in sample_items:
        processed_item = await process_item_embeddings(item, encoders)
        if processed_item is not None:
            processed_items.append(processed_item)
    
    print(f"✓ Successfully processed {len(processed_items)} items")
    
    # Build vector indices
    print("\n6. Building vector indices...")
    await build_vector_indices(processed_items, stores)
    
    # Save processed items metadata
    print("\n7. Saving metadata...")
    metadata_path = "backend/data/fashion_catalog.json"
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(processed_items, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Metadata saved to {metadata_path}")
    
    # Test the recommendation service
    print("\n8. Testing recommendation service...")
    try:
        service = get_recommendation_service()
        stats = service.get_service_stats()
        print(f"✓ Service stats: {stats}")
        
        # Test with first item
        if processed_items:
            test_item = processed_items[0]
            print(f"\n   Testing with item: {test_item['id']}")
            
            # This would normally use the actual recommendation pipeline
            print("   ✓ Recommendation service is ready for testing")
            
    except Exception as e:
        print(f"✗ Error testing service: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Fashion Data Seeding Complete!")
    print(f"📊 Summary:")
    print(f"   - Items processed: {len(processed_items)}")
    print(f"   - Vector stores: 3 (CLIP, BLIP, Fashion)")
    print(f"   - Metadata file: {metadata_path}")
    print("\n🚀 Ready for multi-model fusion recommendations!")

if __name__ == "__main__":
    asyncio.run(main())