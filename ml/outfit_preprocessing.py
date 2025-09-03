import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import json
from typing import List, Tuple, Dict, Optional
import random
from pathlib import Path

class OutfitDataPreprocessor:
    """Preprocessor for outfit compatibility dataset"""
    
    def __init__(self, 
                 dataset_root: str,
                 train_images_path: str,
                 test_images_path: str,
                 image_size: Tuple[int, int] = (224, 224),
                 normalize_stats: Optional[Dict] = None):
        """
        Initialize the preprocessor
        
        Args:
            dataset_root: Path to outfit_items_dataset
            train_images_path: Path to train_images directory
            test_images_path: Path to test_images directory
            image_size: Target image size (height, width)
            normalize_stats: Custom normalization statistics
        """
        self.dataset_root = Path(dataset_root)
        self.train_images_path = Path(train_images_path)
        self.test_images_path = Path(test_images_path)
        self.image_size = image_size
        
        # Default ImageNet normalization stats
        self.normalize_stats = normalize_stats or {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
        
        # Category mappings
        self.categories = {
            'accessories': ['bag', 'hat'],
            'bottomwear': ['pants', 'shorts', 'skirt'],
            'footwear': ['flats', 'heels', 'shoes', 'sneakers'],
            'one-piece': ['dress'],
            'upperwear': ['jacket', 'shirt', 'tshirt']
        }
        
        self.label_encoder = LabelEncoder()
        self.item_metadata = []
        
    def get_transform(self, is_training: bool = True) -> transforms.Compose:
        """Get image transformation pipeline"""
        if is_training:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.normalize_stats['mean'],
                    std=self.normalize_stats['std']
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.normalize_stats['mean'],
                    std=self.normalize_stats['std']
                )
            ])
    
    def load_and_preprocess_image(self, image_path: str, transform: transforms.Compose) -> torch.Tensor:
        """Load and preprocess a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            return transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image tensor if loading fails
            return torch.zeros(3, *self.image_size)
    
    def scan_outfit_items_dataset(self) -> List[Dict]:
        """Scan the outfit_items_dataset and create metadata"""
        items = []
        
        for main_category, subcategories in self.categories.items():
            main_cat_path = self.dataset_root / main_category
            if not main_cat_path.exists():
                continue
                
            for subcategory in subcategories:
                subcat_path = main_cat_path / subcategory
                if not subcat_path.exists():
                    continue
                    
                # Scan for images in subcategory
                for image_file in subcat_path.glob('*.png'):
                    items.append({
                        'image_path': str(image_file),
                        'main_category': main_category,
                        'subcategory': subcategory,
                        'filename': image_file.name,
                        'item_id': image_file.stem
                    })
        
        return items
    
    def scan_train_images(self) -> List[Dict]:
        """Scan train_images directory and extract metadata from filenames"""
        items = []
        
        for image_file in self.train_images_path.glob('*.png'):
            filename = image_file.name
            # Parse filename: MEN-Category-id_XXXXX-XX_X_view.png
            parts = filename.replace('.png', '').split('-')
            
            if len(parts) >= 3:
                gender = parts[0]  # MEN/WOMEN
                category = parts[1]  # Denim, Pants, Shorts, etc.
                
                # Map category to our standard categories
                main_category = self._map_category(category)
                
                items.append({
                    'image_path': str(image_file),
                    'main_category': main_category,
                    'subcategory': category.lower(),
                    'gender': gender.lower(),
                    'filename': filename,
                    'item_id': filename.replace('.png', '')
                })
        
        return items
    
    def _map_category(self, category: str) -> str:
        """Map category names to main categories"""
        category_lower = category.lower()
        
        if category_lower in ['denim', 'pants']:
            return 'bottomwear'
        elif category_lower in ['shorts']:
            return 'bottomwear'
        elif category_lower in ['shirt', 'tshirt', 'jacket']:
            return 'upperwear'
        elif category_lower in ['dress']:
            return 'one-piece'
        elif category_lower in ['shoes', 'sneakers', 'heels', 'flats']:
            return 'footwear'
        elif category_lower in ['bag', 'hat']:
            return 'accessories'
        else:
            return 'unknown'
    
    def create_outfit_combinations(self, items: List[Dict], num_positive: int = 1000, num_negative: int = 1000) -> List[Dict]:
        """Create positive and negative outfit combinations"""
        combinations = []
        
        # Group items by category
        items_by_category = {}
        for item in items:
            category = item['main_category']
            if category not in items_by_category:
                items_by_category[category] = []
            items_by_category[category].append(item)
        
        # Create positive combinations (compatible outfits)
        for _ in range(num_positive):
            outfit = self._create_compatible_outfit(items_by_category)
            if outfit:
                combinations.append({
                    'items': outfit,
                    'label': 1,  # Compatible
                    'type': 'positive'
                })
        
        # Create negative combinations (incompatible outfits)
        for _ in range(num_negative):
            outfit = self._create_incompatible_outfit(items_by_category)
            if outfit:
                combinations.append({
                    'items': outfit,
                    'label': 0,  # Incompatible
                    'type': 'negative'
                })
        
        return combinations
    
    def _create_compatible_outfit(self, items_by_category: Dict) -> Optional[List[Dict]]:
        """Create a compatible outfit combination"""
        outfit = []
        
        # Basic outfit: upperwear + bottomwear + footwear
        required_categories = ['upperwear', 'bottomwear', 'footwear']
        
        for category in required_categories:
            if category in items_by_category and items_by_category[category]:
                item = random.choice(items_by_category[category])
                outfit.append(item)
        
        # Optionally add accessories
        if 'accessories' in items_by_category and random.random() < 0.3:
            accessory = random.choice(items_by_category['accessories'])
            outfit.append(accessory)
        
        return outfit if len(outfit) >= 3 else None
    
    def _create_incompatible_outfit(self, items_by_category: Dict) -> Optional[List[Dict]]:
        """Create an incompatible outfit combination"""
        outfit = []
        
        # Create incompatible combinations by mixing formal/casual or conflicting styles
        categories = list(items_by_category.keys())
        
        # Randomly select 2-4 items from different categories
        num_items = random.randint(2, 4)
        selected_categories = random.sample(categories, min(num_items, len(categories)))
        
        for category in selected_categories:
            if items_by_category[category]:
                item = random.choice(items_by_category[category])
                outfit.append(item)
        
        return outfit if len(outfit) >= 2 else None
    
    def save_metadata(self, items: List[Dict], combinations: List[Dict], output_path: str):
        """Save processed metadata to JSON file"""
        metadata = {
            'items': items,
            'combinations': combinations,
            'categories': self.categories,
            'image_size': self.image_size,
            'normalize_stats': self.normalize_stats,
            'total_items': len(items),
            'total_combinations': len(combinations)
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {output_path}")
        print(f"Total items: {len(items)}")
        print(f"Total combinations: {len(combinations)}")
    
    def process_dataset(self, output_dir: str = 'processed_data'):
        """Main processing pipeline"""
        print("Starting dataset preprocessing...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Scan datasets
        print("Scanning outfit items dataset...")
        outfit_items = self.scan_outfit_items_dataset()
        
        print("Scanning train images...")
        train_items = self.scan_train_images()
        
        # Combine all items
        all_items = outfit_items + train_items
        print(f"Found {len(all_items)} total items")
        
        # Create outfit combinations
        print("Creating outfit combinations...")
        combinations = self.create_outfit_combinations(all_items)
        
        # Save metadata
        metadata_path = output_path / 'outfit_metadata.json'
        self.save_metadata(all_items, combinations, str(metadata_path))
        
        print("Dataset preprocessing completed!")
        return all_items, combinations

class OutfitCompatibilityDataset(Dataset):
    """PyTorch Dataset for outfit compatibility"""
    
    def __init__(self, combinations: List[Dict], transform: transforms.Compose):
        self.combinations = combinations
        self.transform = transform
    
    def __len__(self):
        return len(self.combinations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        combination = self.combinations[idx]
        items = combination['items']
        label = combination['label']
        
        # Load and transform images
        images = []
        for item in items:
            try:
                image = Image.open(item['image_path']).convert('RGB')
                image_tensor = self.transform(image)
                images.append(image_tensor)
            except Exception as e:
                print(f"Error loading image {item['image_path']}: {e}")
                # Add a blank tensor if image fails to load
                images.append(torch.zeros(3, 224, 224))
        
        # Pad or truncate to fixed number of items (e.g., 4)
        max_items = 4
        while len(images) < max_items:
            images.append(torch.zeros(3, 224, 224))
        images = images[:max_items]
        
        # Stack images
        images_tensor = torch.stack(images)
        
        return images_tensor, torch.tensor(label, dtype=torch.float32)

if __name__ == "__main__":
    # Example usage
    dataset_root = "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/data/datasets/outfit_items_dataset"
    train_images_path = "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/data/datasets/train_images"
    test_images_path = "/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/data/test_images"
    
    preprocessor = OutfitDataPreprocessor(
        dataset_root=dataset_root,
        train_images_path=train_images_path,
        test_images_path=test_images_path
    )
    
    # Process the dataset
    items, combinations = preprocessor.process_dataset()