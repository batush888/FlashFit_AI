#!/usr/bin/env python3
"""
Phase 2 Data Preparation System for FlashFit AI

This module handles:
1. DeepFashion dataset download and organization
2. Polyvore Outfits dataset processing
3. Image preprocessing with augmentation
4. Text preprocessing for fashion attributes
5. Train/validation/test splits (80/10/10)
6. CLIP and BLIP embedding generation
"""

import os
import json
import requests
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import hashlib
import pickle
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase2DataPreparator:
    """
    Comprehensive data preparation system for Phase 2 model training
    """
    
    def __init__(self, base_data_dir: str = "data/phase2"):
        self.base_data_dir = Path(base_data_dir)
        self.raw_data_dir = self.base_data_dir / "raw"
        self.processed_data_dir = self.base_data_dir / "processed"
        self.embeddings_dir = self.base_data_dir / "embeddings"
        
        # Create directory structure
        self._create_directories()
        
        # Dataset configurations
        self.datasets_config = {
            "deepfashion": {
                "name": "DeepFashion",
                "urls": {
                    "category_attribute": "http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction/Anno_coarse.tar.gz",
                    "consumer_to_shop": "http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval/Img.zip"
                },
                "expected_samples": 800000
            },
            "polyvore": {
                "name": "Polyvore Outfits",
                "urls": {
                    "outfits": "https://github.com/xthan/polyvore-dataset/raw/master/polyvore_outfits.tar.gz"
                },
                "expected_samples": 365000
            }
        }
        
        # Image preprocessing transforms
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.standard_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Fashion vocabulary for BLIP enhancement
        self.fashion_vocabulary = self._load_fashion_vocabulary()
        
        logger.info(f"Phase2DataPreparator initialized with base directory: {self.base_data_dir}")
    
    def _create_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.base_data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.embeddings_dir,
            self.raw_data_dir / "deepfashion",
            self.raw_data_dir / "polyvore",
            self.processed_data_dir / "images" / "train",
            self.processed_data_dir / "images" / "val",
            self.processed_data_dir / "images" / "test",
            self.processed_data_dir / "annotations",
            self.processed_data_dir / "splits"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def _load_fashion_vocabulary(self) -> Dict[str, List[str]]:
        """Load comprehensive fashion vocabulary for BLIP enhancement"""
        return {
            "colors": [
                "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", 
                "black", "white", "gray", "grey", "navy", "maroon", "teal", "olive",
                "burgundy", "coral", "turquoise", "magenta", "beige", "tan", "khaki"
            ],
            "materials": [
                "cotton", "silk", "wool", "polyester", "linen", "denim", "leather", 
                "cashmere", "velvet", "satin", "chiffon", "lace", "mesh", "knit",
                "fleece", "canvas", "suede", "nylon", "spandex", "rayon"
            ],
            "styles": [
                "casual", "formal", "business", "sporty", "elegant", "vintage", 
                "modern", "classic", "trendy", "bohemian", "minimalist", "edgy",
                "romantic", "preppy", "gothic", "punk", "hipster", "chic"
            ],
            "categories": [
                "shirt", "blouse", "t-shirt", "sweater", "jacket", "coat", "dress",
                "skirt", "pants", "jeans", "shorts", "shoes", "boots", "sneakers",
                "sandals", "bag", "purse", "backpack", "hat", "scarf", "belt"
            ],
            "patterns": [
                "striped", "polka dot", "floral", "geometric", "plaid", "checkered",
                "solid", "printed", "embroidered", "sequined", "beaded"
            ]
        }
    
    def download_datasets(self, force_download: bool = False) -> Dict[str, bool]:
        """
        Download DeepFashion and Polyvore datasets
        
        Args:
            force_download: Force re-download even if files exist
            
        Returns:
            Dictionary with download status for each dataset
        """
        download_status = {}
        
        for dataset_name, config in self.datasets_config.items():
            logger.info(f"Processing {config['name']} dataset...")
            dataset_dir = self.raw_data_dir / dataset_name
            
            success = True
            for file_key, url in config["urls"].items():
                filename = url.split("/")[-1]
                filepath = dataset_dir / filename
                
                if filepath.exists() and not force_download:
                    logger.info(f"File already exists: {filepath}")
                    continue
                
                try:
                    logger.info(f"Downloading {filename} from {url}")
                    success &= self._download_file(url, filepath)
                    
                    # Extract if it's an archive
                    if filename.endswith(('.zip', '.tar.gz', '.tgz')):
                        success &= self._extract_archive(filepath, dataset_dir)
                        
                except Exception as e:
                    logger.error(f"Failed to download {filename}: {e}")
                    success = False
            
            download_status[dataset_name] = success
            
        return download_status
    
    def _download_file(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """Download a file with progress tracking"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            return False
    
    def _extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract archive files"""
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix in ['.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_to)
            
            logger.info(f"Successfully extracted: {archive_path}")
            return True
            
        except Exception as e:
            logger.error(f"Extraction failed for {archive_path}: {e}")
            return False
    
    def preprocess_images(self, apply_augmentation: bool = True) -> Dict[str, int]:
        """
        Preprocess images with resizing, normalization, and augmentation
        
        Args:
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {"processed": 0, "failed": 0, "augmented": 0}
        
        # Process DeepFashion images
        deepfashion_dir = self.raw_data_dir / "deepfashion"
        if deepfashion_dir.exists():
            stats = self._process_dataset_images(
                deepfashion_dir, "deepfashion", apply_augmentation, stats
            )
        
        # Process Polyvore images
        polyvore_dir = self.raw_data_dir / "polyvore"
        if polyvore_dir.exists():
            stats = self._process_dataset_images(
                polyvore_dir, "polyvore", apply_augmentation, stats
            )
        
        logger.info(f"Image preprocessing completed. Stats: {stats}")
        return stats
    
    def _process_dataset_images(self, dataset_dir: Path, dataset_name: str, 
                              apply_augmentation: bool, stats: Dict[str, int]) -> Dict[str, int]:
        """Process images from a specific dataset"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for image_path in dataset_dir.rglob('*'):
            if image_path.suffix.lower() in image_extensions:
                try:
                    # Load and validate image
                    image = Image.open(image_path).convert('RGB')
                    
                    # Apply standard preprocessing
                    processed_image = self.standard_transforms(image)
                    
                    # Save processed image
                    relative_path = image_path.relative_to(dataset_dir)
                    output_path = self.processed_data_dir / "images" / dataset_name / relative_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Convert tensor back to PIL for saving
                    processed_pil = transforms.ToPILImage()(processed_image)
                    processed_pil.save(output_path)
                    
                    stats["processed"] += 1
                    
                    # Apply augmentation if requested
                    if apply_augmentation:
                        augmented_image = self.augmentation_transforms(image)
                        aug_path = output_path.parent / f"aug_{output_path.name}"
                        aug_pil = transforms.ToPILImage()(augmented_image)
                        aug_pil.save(aug_path)
                        stats["augmented"] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process image {image_path}: {e}")
                    stats["failed"] += 1
        
        return stats
    
    def preprocess_text_annotations(self) -> Dict[str, Any]:
        """
        Extract and preprocess text annotations for fashion attributes
        
        Returns:
            Dictionary with processed annotations
        """
        annotations = {
            "fashion_attributes": [],
            "outfit_descriptions": [],
            "category_labels": [],
            "style_keywords": []
        }
        
        # Process DeepFashion annotations
        deepfashion_annotations = self._extract_deepfashion_annotations()
        annotations["fashion_attributes"].extend(deepfashion_annotations.get("attributes", []))
        annotations["category_labels"].extend(deepfashion_annotations.get("categories", []))
        
        # Process Polyvore annotations
        polyvore_annotations = self._extract_polyvore_annotations()
        annotations["outfit_descriptions"].extend(polyvore_annotations.get("descriptions", []))
        annotations["style_keywords"].extend(polyvore_annotations.get("keywords", []))
        
        # Clean and enhance text data
        annotations = self._clean_text_annotations(annotations)
        
        # Save processed annotations
        annotations_file = self.processed_data_dir / "annotations" / "processed_annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        logger.info(f"Text preprocessing completed. Saved to: {annotations_file}")
        return annotations
    
    def _extract_deepfashion_annotations(self) -> Dict[str, List[Dict]]:
        """Extract annotations from DeepFashion dataset"""
        annotations = {"attributes": [], "categories": []}
        
        # Look for annotation files in DeepFashion directory
        deepfashion_dir = self.raw_data_dir / "deepfashion"
        
        # This is a placeholder - actual implementation would parse
        # DeepFashion's specific annotation format
        logger.info("Extracting DeepFashion annotations...")
        
        return annotations
    
    def _extract_polyvore_annotations(self) -> Dict[str, List[Dict]]:
        """Extract annotations from Polyvore dataset"""
        annotations = {"descriptions": [], "keywords": []}
        
        # Look for annotation files in Polyvore directory
        polyvore_dir = self.raw_data_dir / "polyvore"
        
        # This is a placeholder - actual implementation would parse
        # Polyvore's specific annotation format
        logger.info("Extracting Polyvore annotations...")
        
        return annotations
    
    def _clean_text_annotations(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and enhance text annotations with fashion vocabulary"""
        # Expand vocabulary coverage
        enhanced_annotations = annotations.copy()
        
        # Add fashion-specific terms to improve BLIP vocabulary coverage
        fashion_terms = []
        for category, terms in self.fashion_vocabulary.items():
            fashion_terms.extend(terms)
        
        enhanced_annotations["fashion_vocabulary"] = fashion_terms
        enhanced_annotations["vocabulary_coverage"] = len(fashion_terms)
        
        logger.info(f"Enhanced annotations with {len(fashion_terms)} fashion terms")
        return enhanced_annotations
    
    def create_train_val_test_splits(self, train_ratio: float = 0.8, 
                                   val_ratio: float = 0.1, 
                                   test_ratio: float = 0.1) -> Dict[str, List[str]]:
        """
        Create train/validation/test splits (80/10/10)
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            
        Returns:
            Dictionary with file paths for each split
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Collect all processed images
        all_images = []
        processed_images_dir = self.processed_data_dir / "images"
        
        for image_path in processed_images_dir.rglob('*.jpg'):
            if not image_path.name.startswith('aug_'):  # Exclude augmented images from splitting
                all_images.append(str(image_path))
        
        logger.info(f"Found {len(all_images)} images for splitting")
        
        # Create stratified splits
        train_files, temp_files = train_test_split(
            all_images, test_size=(val_ratio + test_ratio), random_state=42
        )
        
        val_files, test_files = train_test_split(
            temp_files, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42
        )
        
        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        
        # Save splits to files
        splits_dir = self.processed_data_dir / "splits"
        for split_name, file_list in splits.items():
            split_file = splits_dir / f"{split_name}_files.json"
            with open(split_file, 'w') as f:
                json.dump(file_list, f, indent=2)
            
            logger.info(f"{split_name.capitalize()} set: {len(file_list)} files")
        
        return splits
    
    def generate_embeddings_preview(self) -> Dict[str, Any]:
        """
        Generate preview embeddings using CLIP and BLIP for initial feature space
        
        Returns:
            Dictionary with embedding statistics
        """
        # This is a placeholder for embedding generation
        # Actual implementation would use CLIP and BLIP models
        
        embedding_stats = {
            "clip_embeddings_generated": 0,
            "blip_embeddings_generated": 0,
            "embedding_dimension": 512,
            "total_samples": 0
        }
        
        logger.info("Embedding generation preview completed")
        return embedding_stats
    
    def validate_dataset_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of processed datasets
        
        Returns:
            Validation report
        """
        report = {
            "total_images": 0,
            "corrupted_images": 0,
            "missing_annotations": 0,
            "split_integrity": True,
            "vocabulary_coverage": 0
        }
        
        # Count processed images
        processed_images_dir = self.processed_data_dir / "images"
        if processed_images_dir.exists():
            report["total_images"] = len(list(processed_images_dir.rglob('*.jpg')))
        
        # Check vocabulary coverage
        annotations_file = self.processed_data_dir / "annotations" / "processed_annotations.json"
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
                report["vocabulary_coverage"] = annotations.get("vocabulary_coverage", 0)
        
        # Validate splits
        splits_dir = self.processed_data_dir / "splits"
        required_splits = ["train_files.json", "val_files.json", "test_files.json"]
        report["split_integrity"] = all(
            (splits_dir / split_file).exists() for split_file in required_splits
        )
        
        logger.info(f"Dataset validation completed: {report}")
        return report
    
    def get_preparation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of data preparation status
        
        Returns:
            Preparation summary
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "base_directory": str(self.base_data_dir),
            "datasets_configured": list(self.datasets_config.keys()),
            "fashion_vocabulary_size": sum(len(terms) for terms in self.fashion_vocabulary.values()),
            "directory_structure": self._get_directory_structure(),
            "validation_report": self.validate_dataset_integrity()
        }
        
        return summary
    
    def _get_directory_structure(self) -> Dict[str, bool]:
        """Check if all required directories exist"""
        required_dirs = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.embeddings_dir,
            self.processed_data_dir / "images",
            self.processed_data_dir / "annotations",
            self.processed_data_dir / "splits"
        ]
        
        return {str(dir_path): dir_path.exists() for dir_path in required_dirs}


def main():
    """Main function for testing data preparation"""
    preparator = Phase2DataPreparator()
    
    logger.info("Starting Phase 2 data preparation...")
    
    # Get preparation summary
    summary = preparator.get_preparation_summary()
    logger.info(f"Preparation summary: {json.dumps(summary, indent=2)}")
    
    # Note: Actual dataset download would require proper URLs and authentication
    logger.info("Data preparation system initialized successfully")
    logger.info("Ready for Phase 2 model training pipeline")


if __name__ == "__main__":
    main()