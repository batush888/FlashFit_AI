#!/usr/bin/env python3
"""
Build FAISS indices from wardrobe items using multi-model encoders.
This script processes clothing images and builds searchable vector indices.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import logging
from datetime import datetime
from tqdm import tqdm

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

from models.clip_encoder import get_clip_encoder
from models.blip_captioner import get_blip_captioner
from models.fashion_encoder import get_fashion_encoder
from models.vector_store import get_clip_store, get_blip_store, get_fashion_store

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndexBuilder:
    """
    Build FAISS indices from clothing images using multi-model encoders
    """
    
    def __init__(self, data_dir: str = "data", force_rebuild: bool = False):
        """
        Initialize the index builder
        
        Args:
            data_dir: Directory containing clothing images and metadata
            force_rebuild: Whether to rebuild existing indices
        """
        self.data_dir = Path(data_dir)
        self.force_rebuild = force_rebuild
        
        # Initialize encoders
        logger.info("Initializing encoders...")
        self.clip_encoder = get_clip_encoder()
        self.blip_captioner = get_blip_captioner()
        self.fashion_encoder = get_fashion_encoder()
        
        # Initialize vector stores
        logger.info("Initializing vector stores...")
        self.clip_store = get_clip_store(dim=512)
        self.blip_store = get_blip_store(dim=768)
        self.fashion_store = get_fashion_store(dim=512)
        
        logger.info("IndexBuilder initialized successfully")
    
    def find_image_files(self, directory: Path) -> List[Path]:
        """
        Find all image files in directory and subdirectories
        
        Args:
            directory: Directory to search
            
        Returns:
            List of image file paths
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory.rglob(f'*{ext}'))
            image_files.extend(directory.rglob(f'*{ext.upper()}'))
        
        return sorted(image_files)
    
    def load_metadata(self, metadata_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
        """
        Load metadata for clothing items
        
        Args:
            metadata_path: Path to metadata JSON file
            
        Returns:
            Dictionary mapping item IDs to metadata
        """
        if metadata_path and metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        
        return {}
    
    def generate_item_metadata(self, image_path: Path, existing_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate metadata for a clothing item
        
        Args:
            image_path: Path to the image file
            existing_metadata: Existing metadata if available
            
        Returns:
            Item metadata dictionary
        """
        item_id = image_path.stem
        
        # Start with existing metadata or create new
        metadata = existing_metadata or {}
        
        # Add basic information
        metadata.update({
            'item_id': item_id,
            'image_path': str(image_path.absolute()),
            'filename': image_path.name,
            'file_size': image_path.stat().st_size,
            'processed_at': datetime.now().isoformat()
        })
        
        # Add image dimensions
        try:
            with Image.open(image_path) as img:
                metadata['image_width'] = img.width
                metadata['image_height'] = img.height
                metadata['image_mode'] = img.mode
        except Exception as e:
            logger.warning(f"Failed to get image info for {image_path}: {e}")
        
        return metadata
    
    def process_single_item(self, image_path: Path, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single clothing item and generate all embeddings
        
        Args:
            image_path: Path to the image file
            metadata: Item metadata
            
        Returns:
            Dictionary with embeddings and enhanced metadata, or None if failed
        """
        try:
            item_id = metadata['item_id']
            
            # Generate CLIP embedding
            clip_embedding = self.clip_encoder.embed_image(str(image_path))
            
            # Generate BLIP caption and embedding
            blip_caption = self.blip_captioner.caption(str(image_path))
            blip_description = self.blip_captioner.generate_fashion_description(str(image_path))
            blip_embedding = self.blip_captioner.get_text_embedding(blip_caption)
            
            # Generate fashion embedding and attributes
            fashion_embedding = self.fashion_encoder.embed_fashion_image(str(image_path))
            fashion_attributes = self.fashion_encoder.analyze_fashion_attributes(str(image_path))
            garment_classification = self.fashion_encoder.classify_garment_type(str(image_path))
            
            # Enhanced metadata with AI analysis
            enhanced_metadata = {
                **metadata,
                'blip_caption': blip_caption,
                'blip_description': blip_description,
                'fashion_attributes': fashion_attributes,
                'garment_type': garment_classification.get('top_category', 'unknown'),
                'garment_confidence': garment_classification.get('top_score', 0.0),
                'garment_categories': garment_classification.get('categories', {})
            }
            
            return {
                'item_id': item_id,
                'clip_embedding': clip_embedding,
                'blip_embedding': blip_embedding,
                'fashion_embedding': fashion_embedding,
                'metadata': enhanced_metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            return None
    
    def build_indices(self, 
                     image_directory: str,
                     metadata_file: Optional[str] = None,
                     batch_size: int = 32,
                     max_items: Optional[int] = None) -> Dict[str, Any]:
        """
        Build FAISS indices from clothing images
        
        Args:
            image_directory: Directory containing clothing images
            metadata_file: Optional metadata JSON file
            batch_size: Batch size for processing
            max_items: Maximum number of items to process (for testing)
            
        Returns:
            Build statistics
        """
        image_dir = Path(image_directory)
        if not image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_directory}")
        
        # Load existing metadata
        metadata_path = Path(metadata_file) if metadata_file else None
        existing_metadata = self.load_metadata(metadata_path)
        
        # Find all image files
        logger.info(f"Scanning for images in {image_dir}...")
        image_files = self.find_image_files(image_dir)
        
        if max_items:
            image_files = image_files[:max_items]
        
        logger.info(f"Found {len(image_files)} image files to process")
        
        if not image_files:
            logger.warning("No image files found")
            return {'processed': 0, 'failed': 0, 'skipped': 0}
        
        # Check if indices already exist and not forcing rebuild
        if not self.force_rebuild:
            clip_stats = self.clip_store.get_stats()
            if clip_stats['total_vectors'] > 0:
                logger.info(f"Indices already exist with {clip_stats['total_vectors']} items. Use --force-rebuild to rebuild.")
                return {'processed': 0, 'failed': 0, 'skipped': clip_stats['total_vectors']}
        
        # Process items in batches
        processed_count = 0
        failed_count = 0
        
        # Prepare batch containers
        clip_embeddings = []
        blip_embeddings = []
        fashion_embeddings = []
        batch_metadata = []
        
        logger.info("Starting image processing...")
        
        for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            try:
                # Generate metadata for this item
                item_metadata = self.generate_item_metadata(
                    image_path, 
                    existing_metadata.get(image_path.stem, {})
                )
                
                # Process the item
                result = self.process_single_item(image_path, item_metadata)
                
                if result is None:
                    failed_count += 1
                    continue
                
                # Add to batch
                clip_embeddings.append(result['clip_embedding'])
                blip_embeddings.append(result['blip_embedding'])
                fashion_embeddings.append(result['fashion_embedding'])
                batch_metadata.append(result['metadata'])
                
                processed_count += 1
                
                # Process batch when full or at end
                if len(clip_embeddings) >= batch_size or i == len(image_files) - 1:
                    if clip_embeddings:  # Only process if we have items
                        self._add_batch_to_stores(
                            clip_embeddings,
                            blip_embeddings,
                            fashion_embeddings,
                            batch_metadata
                        )
                        
                        # Clear batch containers
                        clip_embeddings = []
                        blip_embeddings = []
                        fashion_embeddings = []
                        batch_metadata = []
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                failed_count += 1
                continue
        
        # Save all indices
        logger.info("Saving indices...")
        self.clip_store.save()
        self.blip_store.save()
        self.fashion_store.save()
        
        # Generate build report
        stats = {
            'processed': processed_count,
            'failed': failed_count,
            'skipped': 0,
            'total_files': len(image_files),
            'build_time': datetime.now().isoformat(),
            'vector_store_stats': {
                'clip': self.clip_store.get_stats(),
                'blip': self.blip_store.get_stats(),
                'fashion': self.fashion_store.get_stats()
            }
        }
        
        logger.info(f"Index building complete: {processed_count} processed, {failed_count} failed")
        return stats
    
    def _add_batch_to_stores(self, 
                           clip_embeddings: List[np.ndarray],
                           blip_embeddings: List[np.ndarray],
                           fashion_embeddings: List[np.ndarray],
                           metadata_list: List[Dict[str, Any]]):
        """
        Add a batch of embeddings to all vector stores
        
        Args:
            clip_embeddings: List of CLIP embeddings
            blip_embeddings: List of BLIP embeddings
            fashion_embeddings: List of fashion embeddings
            metadata_list: List of metadata dictionaries
        """
        try:
            # Convert to numpy arrays
            clip_batch = np.vstack(clip_embeddings)
            blip_batch = np.vstack(blip_embeddings)
            fashion_batch = np.vstack(fashion_embeddings)
            
            # Add to stores
            self.clip_store.add(clip_batch, metadata_list)
            self.blip_store.add(blip_batch, metadata_list)
            self.fashion_store.add(fashion_batch, metadata_list)
            
        except Exception as e:
            logger.error(f"Failed to add batch to stores: {e}")
            raise
    
    def export_build_report(self, stats: Dict[str, Any], output_path: str = "build_report.json"):
        """
        Export build statistics to JSON file
        
        Args:
            stats: Build statistics
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Build report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save build report: {e}")

def main():
    """
    Main function for command-line usage
    """
    parser = argparse.ArgumentParser(description="Build FAISS indices from clothing images")
    parser.add_argument("image_directory", help="Directory containing clothing images")
    parser.add_argument("--metadata", help="Optional metadata JSON file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--max-items", type=int, help="Maximum number of items to process (for testing)")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild existing indices")
    parser.add_argument("--output-report", default="build_report.json", help="Output file for build report")
    parser.add_argument("--data-dir", default="data", help="Data directory for indices")
    
    args = parser.parse_args()
    
    try:
        # Initialize builder
        builder = IndexBuilder(data_dir=args.data_dir, force_rebuild=args.force_rebuild)
        
        # Build indices
        stats = builder.build_indices(
            image_directory=args.image_directory,
            metadata_file=args.metadata,
            batch_size=args.batch_size,
            max_items=args.max_items
        )
        
        # Export report
        builder.export_build_report(stats, args.output_report)
        
        # Print summary
        print(f"\nBuild Summary:")
        print(f"  Processed: {stats['processed']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Report saved to: {args.output_report}")
        
        if stats['failed'] > 0:
            print(f"\nWarning: {stats['failed']} items failed to process. Check logs for details.")
        
    except Exception as e:
        logger.error(f"Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()