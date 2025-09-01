import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import json
import random
from pathlib import Path

class FashionDataset(Dataset):
    """Fashion dataset for loading and preprocessing fashion images"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 target_size: Tuple[int, int] = (256, 256),
                 load_annotations: bool = True):
        """
        Args:
            data_dir: Path to the dataset directory
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
            target_size: Target image size (height, width)
            load_annotations: Whether to load shape and texture annotations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_size = target_size
        self.load_annotations = load_annotations
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transforms(split)
        else:
            self.transform = transform
            
        # Load dataset
        self.samples = self._load_samples()
        self.category_to_idx = self._build_category_mapping()
        
        # Load annotations if requested
        if load_annotations:
            self.shape_annotations = self._load_shape_annotations()
            self.texture_annotations = self._load_texture_annotations()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _get_default_transforms(self, split: str) -> transforms.Compose:
        """Get default transforms based on split"""
        if split == 'train':
            return transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _load_samples(self) -> List[Dict[str, str]]:
        """Load image samples from the dataset directory"""
        samples = []
        
        # Determine image directory based on split
        if self.split == 'train':
            img_dir = self.data_dir / 'train_images'
        elif self.split == 'test':
            img_dir = self.data_dir / 'test_images'
        else:  # validation - we'll use a subset of train images
            img_dir = self.data_dir / 'train_images'
        
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        
        # Get all image files
        image_files = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
        
        # For validation split, use a subset of training images
        if self.split == 'val':
            random.seed(42)  # For reproducibility
            random.shuffle(image_files)
            # Use 20% of training images for validation
            val_size = len(image_files) // 5
            image_files = image_files[:val_size]
        elif self.split == 'train':
            # Use remaining 80% for training
            random.seed(42)
            random.shuffle(image_files)
            train_size = (len(image_files) * 4) // 5
            image_files = image_files[len(image_files) - train_size:]
        
        for img_path in image_files:
            # Parse filename to extract metadata
            filename = img_path.name
            parts = filename.replace('.png', '').replace('.jpg', '').split('-')
            
            if len(parts) >= 3:
                gender = parts[0]  # MEN or WOMEN
                category = parts[1]  # e.g., Denim, Shirts_Polos
                
                samples.append({
                    'image_path': str(img_path),
                    'filename': filename,
                    'gender': gender,
                    'category': category,
                    'full_category': f"{gender}-{category}"
                })
        
        return samples
    
    def _build_category_mapping(self) -> Dict[str, int]:
        """Build category to index mapping"""
        categories = set()
        for sample in self.samples:
            categories.add(sample['full_category'])
        
        categories = sorted(list(categories))
        return {cat: idx for idx, cat in enumerate(categories)}
    
    def _load_shape_annotations(self) -> Dict[str, List[float]]:
        """Load shape annotations"""
        annotations = {}
        
        # Load shape annotations
        shape_dir = self.data_dir / 'shape_ann'
        ann_file = shape_dir / f'{self.split}_ann_file.txt'
        
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        filename = parts[0]
                        # Convert string annotations to float values
                        shape_values = [float(x) for x in parts[1:] if x.replace('.', '').replace('-', '').isdigit()]
                        annotations[filename] = shape_values
        
        return annotations
    
    def _load_texture_annotations(self) -> Dict[str, int]:
        """Load texture annotations"""
        annotations = {}
        
        # Load texture annotations
        texture_dir = self.data_dir / 'texture_ann' / self.split
        
        if texture_dir.exists():
            for ann_file in texture_dir.glob('*.txt'):
                with open(ann_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            filename = parts[0]
                            texture_label = int(parts[1])
                            annotations[filename] = texture_label
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset"""
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        category_idx = self.category_to_idx[sample['full_category']]
        gender_idx = 0 if sample['gender'] == 'MEN' else 1
        
        result = {
            'image': image,
            'category_label': torch.tensor(category_idx, dtype=torch.long),
            'gender_label': torch.tensor(gender_idx, dtype=torch.long),
            'filename': sample['filename']
        }
        
        # Add annotations if available
        if self.load_annotations:
            filename = sample['filename']
            
            # Shape annotations
            if filename in self.shape_annotations:
                shape_features = torch.tensor(self.shape_annotations[filename], dtype=torch.float32)
                result['shape_features'] = shape_features
            else:
                result['shape_features'] = torch.zeros(10, dtype=torch.float32)  # Default size
            
            # Texture annotations
            if filename in self.texture_annotations:
                texture_label = torch.tensor(self.texture_annotations[filename], dtype=torch.long)
                result['texture_label'] = texture_label
            else:
                result['texture_label'] = torch.tensor(0, dtype=torch.long)  # Default
        
        return result
    
    def get_category_names(self) -> List[str]:
        """Get list of category names"""
        return list(self.category_to_idx.keys())
    
    def get_sample_weights(self) -> torch.Tensor:
        """Calculate sample weights for balanced training"""
        category_counts = {}
        for sample in self.samples:
            cat = sample['full_category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Calculate weights (inverse frequency)
        total_samples = len(self.samples)
        weights = []
        
        for sample in self.samples:
            cat = sample['full_category']
            weight = total_samples / (len(category_counts) * category_counts[cat])
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)

class FashionDataModule:
    """Data module for managing fashion dataset loading and preprocessing"""
    
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 target_size: Tuple[int, int] = (256, 256),
                 pin_memory: bool = True):
        """
        Args:
            data_dir: Path to dataset directory
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            target_size: Target image size
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.pin_memory = pin_memory
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def setup(self):
        """Setup datasets and data loaders"""
        # Create datasets
        self.train_dataset = FashionDataset(
            data_dir=self.data_dir,
            split='train',
            target_size=self.target_size
        )
        
        self.val_dataset = FashionDataset(
            data_dir=self.data_dir,
            split='val',
            target_size=self.target_size
        )
        
        self.test_dataset = FashionDataset(
            data_dir=self.data_dir,
            split='test',
            target_size=self.target_size
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        print(f"Data module setup complete:")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
        print(f"  Categories: {len(self.train_dataset.category_to_idx)}")
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for balanced training"""
        if self.train_dataset is None:
            raise RuntimeError("Must call setup() first")
        
        return self.train_dataset.get_sample_weights()
    
    def get_category_mapping(self) -> Dict[str, int]:
        """Get category to index mapping"""
        if self.train_dataset is None:
            raise RuntimeError("Must call setup() first")
        
        return self.train_dataset.category_to_idx

class FashionAugmentation:
    """Advanced augmentation techniques for fashion images"""
    
    @staticmethod
    def get_training_transforms(target_size: Tuple[int, int] = (256, 256)) -> transforms.Compose:
        """Get training transforms with advanced augmentation"""
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1), 
                scale=(0.8, 1.2),
                shear=10
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_validation_transforms(target_size: Tuple[int, int] = (256, 256)) -> transforms.Compose:
        """Get validation transforms (no augmentation)"""
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_test_time_augmentation() -> List[transforms.Compose]:
        """Get multiple transforms for test-time augmentation"""
        base_transforms = [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        # Original
        tta_transforms = [transforms.Compose(base_transforms)]
        
        # Horizontal flip
        tta_transforms.append(transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
        # Slight rotations
        for angle in [-5, 5]:
            tta_transforms.append(transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomRotation(degrees=(angle, angle)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
        
        return tta_transforms

# Utility functions
def create_fashion_dataloaders(data_dir: str, 
                              batch_size: int = 32,
                              num_workers: int = 4,
                              target_size: Tuple[int, int] = (256, 256)) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create fashion data loaders"""
    data_module = FashionDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        target_size=target_size
    )
    
    data_module.setup()
    
    return data_module.train_loader, data_module.val_loader, data_module.test_loader

def analyze_dataset_statistics(data_dir: str) -> Dict[str, any]:
    """Analyze dataset statistics"""
    dataset = FashionDataset(data_dir, split='train')
    
    # Category distribution
    category_counts = {}
    gender_counts = {'MEN': 0, 'WOMEN': 0}
    
    for sample in dataset.samples:
        cat = sample['full_category']
        gender = sample['gender']
        
        category_counts[cat] = category_counts.get(cat, 0) + 1
        gender_counts[gender] += 1
    
    return {
        'total_samples': len(dataset),
        'num_categories': len(category_counts),
        'category_distribution': category_counts,
        'gender_distribution': gender_counts,
        'category_mapping': dataset.category_to_idx
    }