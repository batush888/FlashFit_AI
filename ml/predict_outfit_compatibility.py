import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Optional
import random
from collections import defaultdict

from outfit_preprocessing import OutfitDataPreprocessor
from outfit_compatibility_model import OutfitCompatibilityModel, create_model
from evaluate_model import load_model

class OutfitCompatibilityPredictor:
    """Outfit compatibility prediction and visualization class"""
    
    def __init__(self, model: OutfitCompatibilityModel, device: torch.device, 
                 image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize predictor
        
        Args:
            model: Trained outfit compatibility model
            device: Device to run predictions on
            image_size: Size to resize images to
        """
        self.model = model.to(device)
        self.device = device
        self.image_size = image_size
        self.model.eval()
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def predict_compatibility(self, image_paths: List[str]) -> Dict:
        """Predict compatibility for a set of outfit items"""
        # Load and preprocess images
        images = []
        valid_paths = []
        
        for path in image_paths:
            img_tensor = self.load_and_preprocess_image(path)
            if img_tensor is not None:
                images.append(img_tensor)
                valid_paths.append(path)
        
        if len(images) == 0:
            return {'error': 'No valid images found'}
        
        # Pad or truncate to model's expected number of items
        num_items = self.model.num_items
        
        if len(images) < num_items:
            # Pad with zeros (black images)
            while len(images) < num_items:
                images.append(torch.zeros_like(images[0]))
        elif len(images) > num_items:
            # Truncate
            images = images[:num_items]
            valid_paths = valid_paths[:num_items]
        
        # Stack images and add batch dimension
        outfit_tensor = torch.stack(images).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            predictions, item_features = self.model(outfit_tensor)
            probability = torch.sigmoid(predictions).item()
            is_compatible = probability > 0.5
        
        return {
            'compatibility_score': probability,
            'is_compatible': is_compatible,
            'confidence': abs(probability - 0.5) * 2,  # Distance from decision boundary
            'valid_image_paths': valid_paths,
            'item_features': item_features.cpu().numpy()
        }
    
    def predict_single_item_compatibility(self, base_outfit: List[str], candidate_item: str) -> Dict:
        """Predict how well a candidate item fits with a base outfit"""
        base_result = self.predict_compatibility(base_outfit)
        
        if 'error' in base_result:
            return base_result
        
        # Try adding the candidate item
        extended_outfit = base_outfit + [candidate_item]
        extended_result = self.predict_compatibility(extended_outfit)
        
        if 'error' in extended_result:
            return extended_result
        
        # Calculate improvement
        improvement = extended_result['compatibility_score'] - base_result['compatibility_score']
        
        return {
            'base_compatibility': base_result['compatibility_score'],
            'extended_compatibility': extended_result['compatibility_score'],
            'improvement': improvement,
            'recommendation': 'Add item' if improvement > 0 else 'Skip item',
            'candidate_item': candidate_item
        }
    
    def find_best_combinations(self, item_categories: Dict[str, List[str]], 
                             max_combinations: int = 10) -> List[Dict]:
        """Find best outfit combinations from available items"""
        results = []
        
        # Generate random combinations
        for _ in range(max_combinations * 3):  # Generate more to filter best ones
            combination = []
            
            # Sample one item from each category
            for category, items in item_categories.items():
                if items:
                    combination.append(random.choice(items))
            
            if len(combination) >= 2:  # Need at least 2 items
                result = self.predict_compatibility(combination)
                if 'error' not in result:
                    result['combination'] = combination
                    results.append(result)
        
        # Sort by compatibility score and return top results
        results.sort(key=lambda x: x['compatibility_score'], reverse=True)
        return results[:max_combinations]
    
    def visualize_outfit_prediction(self, image_paths: List[str], prediction_result: Dict, 
                                  save_path: Optional[str] = None, title: Optional[str] = None):
        """Visualize outfit with prediction results"""
        valid_paths = prediction_result.get('valid_image_paths', image_paths)
        
        if not valid_paths:
            print("No valid images to visualize")
            return
        
        # Calculate grid size
        n_items = len(valid_paths)
        cols = min(4, n_items)
        rows = (n_items + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows + 2))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Load and display images
        for i, img_path in enumerate(valid_paths):
            try:
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"Item {i+1}\n{Path(img_path).stem}", fontsize=10)
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading\n{Path(img_path).name}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(valid_paths), len(axes)):
            axes[i].axis('off')
        
        # Add prediction information
        compatibility_score = prediction_result['compatibility_score']
        is_compatible = prediction_result['is_compatible']
        confidence = prediction_result['confidence']
        
        # Color based on compatibility
        color = 'green' if is_compatible else 'red'
        status = 'COMPATIBLE' if is_compatible else 'INCOMPATIBLE'
        
        # Main title
        if title is None:
            title = f"Outfit Compatibility Prediction"
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Add prediction text
        prediction_text = (
            f"Status: {status}\n"
            f"Compatibility Score: {compatibility_score:.3f}\n"
            f"Confidence: {confidence:.3f}"
        )
        
        fig.text(0.5, 0.02, prediction_text, ha='center', va='bottom', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def create_outfit_grid(self, combinations: List[Dict], save_path: Optional[str] = None):
        """Create a grid showing multiple outfit combinations"""
        n_outfits = len(combinations)
        if n_outfits == 0:
            print("No combinations to visualize")
            return
        
        # Calculate grid size
        cols = min(3, n_outfits)
        rows = (n_outfits + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 8*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, combo in enumerate(combinations):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Create subplot for this outfit
            valid_paths = combo.get('valid_image_paths', combo.get('combination', []))
            
            if valid_paths:
                # Create mini grid for outfit items
                n_items = len(valid_paths)
                item_cols = min(2, n_items)
                item_rows = (n_items + item_cols - 1) // item_cols
                
                # Load and arrange images
                outfit_images = []
                for img_path in valid_paths:
                    try:
                        img = Image.open(img_path)
                        img = img.resize((100, 100))
                        outfit_images.append(np.array(img))
                    except:
                        # Create placeholder
                        outfit_images.append(np.zeros((100, 100, 3), dtype=np.uint8))
                
                # Arrange images in grid
                if len(outfit_images) == 1:
                    combined_img = outfit_images[0]
                elif len(outfit_images) == 2:
                    combined_img = np.hstack(outfit_images)
                elif len(outfit_images) == 3:
                    top_row = np.hstack([outfit_images[0], outfit_images[1]])
                    bottom_row = np.hstack([outfit_images[2], np.zeros_like(outfit_images[2])])
                    combined_img = np.vstack([top_row, bottom_row])
                else:  # 4 or more items
                    top_row = np.hstack(outfit_images[:2])
                    bottom_row = np.hstack(outfit_images[2:4] if len(outfit_images) >= 4 
                                         else [outfit_images[2], np.zeros_like(outfit_images[2])])
                    combined_img = np.vstack([top_row, bottom_row])
                
                ax.imshow(combined_img)
            
            # Add compatibility information
            score = combo['compatibility_score']
            is_compatible = combo['is_compatible']
            
            color = 'green' if is_compatible else 'red'
            status = 'COMPATIBLE' if is_compatible else 'INCOMPATIBLE'
            
            ax.set_title(f"Outfit {i+1}\n{status}\nScore: {score:.3f}", 
                        fontsize=10, color=color, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(combinations), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Outfit Compatibility Predictions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Outfit grid saved to {save_path}")
        
        plt.show()
    
    def analyze_item_features(self, image_paths: List[str], save_path: Optional[str] = None):
        """Analyze and visualize item features"""
        result = self.predict_compatibility(image_paths)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        features = result['item_features'].squeeze()
        valid_paths = result['valid_image_paths']
        
        # Create feature visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature heatmap
        im = ax1.imshow(features, cmap='viridis', aspect='auto')
        ax1.set_title('Item Features Heatmap')
        ax1.set_xlabel('Feature Dimension')
        ax1.set_ylabel('Item Index')
        
        # Add item labels
        item_labels = [f"Item {i+1}\n{Path(path).stem}" for i, path in enumerate(valid_paths)]
        ax1.set_yticks(range(len(item_labels)))
        ax1.set_yticklabels(item_labels)
        
        plt.colorbar(im, ax=ax1)
        
        # Feature statistics
        feature_stats = {
            'mean': np.mean(features, axis=1),
            'std': np.std(features, axis=1),
            'max': np.max(features, axis=1),
            'min': np.min(features, axis=1)
        }
        
        x = np.arange(len(valid_paths))
        width = 0.2
        
        for i, (stat_name, values) in enumerate(feature_stats.items()):
            ax2.bar(x + i*width, values, width, label=stat_name)
        
        ax2.set_title('Feature Statistics by Item')
        ax2.set_xlabel('Item Index')
        ax2.set_ylabel('Feature Value')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels([f"Item {i+1}" for i in range(len(valid_paths))])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature analysis saved to {save_path}")
        
        plt.show()
        
        return features

def collect_images_by_category(dataset_root: str) -> Dict[str, List[str]]:
    """Collect images organized by category"""
    categories = defaultdict(list)
    dataset_path = Path(dataset_root)
    
    if dataset_path.exists():
        for category_dir in dataset_path.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                for img_file in category_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        categories[category_name].append(str(img_file))
    
    return dict(categories)

def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Predict Outfit Compatibility')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--images', type=str, nargs='+', help='Paths to outfit images')
    parser.add_argument('--dataset_root', type=str, help='Root directory of dataset for random combinations')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory for results')
    parser.add_argument('--mode', type=str, choices=['single', 'combinations', 'analysis'], 
                       default='single', help='Prediction mode')
    parser.add_argument('--num_combinations', type=int, default=10, help='Number of combinations to generate')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Create predictor
    predictor = OutfitCompatibilityPredictor(model, device)
    
    if args.mode == 'single':
        # Single outfit prediction
        if not args.images:
            print("Error: --images required for single mode")
            return
        
        print(f"Predicting compatibility for {len(args.images)} items...")
        result = predictor.predict_compatibility(args.images)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        # Print results
        print("\nPrediction Results:")
        print("=" * 50)
        print(f"Compatibility Score: {result['compatibility_score']:.3f}")
        print(f"Status: {'COMPATIBLE' if result['is_compatible'] else 'INCOMPATIBLE'}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        # Visualize
        predictor.visualize_outfit_prediction(
            args.images, result,
            save_path=str(output_dir / 'single_prediction.png')
        )
        
        # Analyze features
        predictor.analyze_item_features(
            args.images,
            save_path=str(output_dir / 'feature_analysis.png')
        )
    
    elif args.mode == 'combinations':
        # Generate and evaluate combinations
        if not args.dataset_root:
            print("Error: --dataset_root required for combinations mode")
            return
        
        print(f"Collecting images from {args.dataset_root}...")
        categories = collect_images_by_category(args.dataset_root)
        
        if not categories:
            print("No images found in dataset")
            return
        
        print(f"Found categories: {list(categories.keys())}")
        for cat, items in categories.items():
            print(f"  {cat}: {len(items)} items")
        
        print(f"\nGenerating {args.num_combinations} outfit combinations...")
        combinations = predictor.find_best_combinations(categories, args.num_combinations)
        
        if not combinations:
            print("No valid combinations found")
            return
        
        # Print top combinations
        print("\nTop Outfit Combinations:")
        print("=" * 60)
        for i, combo in enumerate(combinations[:5]):
            print(f"\nRank {i+1}:")
            print(f"  Score: {combo['compatibility_score']:.3f}")
            print(f"  Status: {'COMPATIBLE' if combo['is_compatible'] else 'INCOMPATIBLE'}")
            print(f"  Items: {[Path(p).name for p in combo['combination']]}")
        
        # Visualize combinations
        predictor.create_outfit_grid(
            combinations,
            save_path=str(output_dir / 'outfit_combinations.png')
        )
        
        # Save results
        results_data = []
        for combo in combinations:
            results_data.append({
                'items': [str(Path(p).name) for p in combo['combination']],
                'compatibility_score': combo['compatibility_score'],
                'is_compatible': combo['is_compatible'],
                'confidence': combo['confidence']
            })
        
        with open(output_dir / 'combinations_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
    
    elif args.mode == 'analysis':
        # Detailed analysis mode
        if not args.images:
            print("Error: --images required for analysis mode")
            return
        
        print(f"Performing detailed analysis for {len(args.images)} items...")
        
        # Basic prediction
        result = predictor.predict_compatibility(args.images)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        # Print detailed results
        print("\nDetailed Analysis Results:")
        print("=" * 60)
        print(f"Compatibility Score: {result['compatibility_score']:.3f}")
        print(f"Status: {'COMPATIBLE' if result['is_compatible'] else 'INCOMPATIBLE'}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        # Individual item analysis
        if len(args.images) > 1:
            print("\nIndividual Item Contributions:")
            base_outfit = args.images[:-1]
            for i, item in enumerate(args.images):
                if i == 0:
                    continue
                
                test_outfit = args.images[:i] + args.images[i+1:]
                if len(test_outfit) > 0:
                    without_item = predictor.predict_compatibility(test_outfit)
                    if 'error' not in without_item:
                        contribution = result['compatibility_score'] - without_item['compatibility_score']
                        print(f"  {Path(item).name}: {contribution:+.3f}")
        
        # Visualizations
        predictor.visualize_outfit_prediction(
            args.images, result,
            save_path=str(output_dir / 'detailed_prediction.png'),
            title="Detailed Outfit Analysis"
        )
        
        predictor.analyze_item_features(
            args.images,
            save_path=str(output_dir / 'detailed_features.png')
        )
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()