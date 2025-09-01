import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import cv2
from datetime import datetime

# Import our models and data
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.fashion_ai_model import FashionAIModel, FashionGAN, FashionAISystem
from data.fashion_dataset import FashionAugmentation

class FashionPredictor:
    """Fashion AI predictor for classifying and analyzing fashion images"""
    
    def __init__(self,
                 model_path: str,
                 category_mapping: Dict[str, int] = None,
                 device: str = 'auto',
                 confidence_threshold: float = 0.5):
        """
        Args:
            model_path: Path to the trained model checkpoint
            category_mapping: Mapping from category names to indices
            device: Device to use for inference
            confidence_threshold: Minimum confidence for predictions
        """
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.confidence_threshold = confidence_threshold
        
        # Load model and mappings
        self.model, self.category_mapping = self._load_model(model_path, category_mapping)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create reverse mapping
        self.idx_to_category = {v: k for k, v in self.category_mapping.items()}
        
        # Setup transforms
        self.transform = FashionAugmentation.get_validation_transforms()
        self.tta_transforms = FashionAugmentation.get_test_time_augmentation()
        
        print(f"Fashion predictor loaded on {self.device}")
        print(f"Categories: {len(self.category_mapping)}")
    
    def _load_model(self, model_path: str, category_mapping: Dict[str, int] = None) -> Tuple[nn.Module, Dict[str, int]]:
        """Load trained model and category mapping"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load category mapping from checkpoint or use provided
        if category_mapping is None:
            # Try to load from checkpoint config
            if 'category_mapping' in checkpoint:
                category_mapping = checkpoint['category_mapping']
            else:
                # Default mapping - this should be updated based on your dataset
                category_mapping = {
                    'MEN-Denim': 0, 'MEN-Pants': 1, 'MEN-Shirts_Polos': 2, 'MEN-Sweaters': 3,
                    'WOMEN-Dresses': 4, 'WOMEN-Pants': 5, 'WOMEN-Shirts_Blouses': 6, 'WOMEN-Shorts': 7
                }
        
        # Create model with correct number of classes
        num_classes = len(category_mapping)
        model = FashionAIModel(num_classes=num_classes)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, category_mapping
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for inference"""
        # Convert to PIL Image if needed
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def predict_single(self, image: Union[str, Image.Image, np.ndarray], 
                      use_tta: bool = False) -> Dict[str, Any]:
        """Predict on a single image"""
        with torch.no_grad():
            if use_tta:
                return self._predict_with_tta(image)
            else:
                return self._predict_simple(image)
    
    def _predict_simple(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Simple prediction without test-time augmentation"""
        # Preprocess image
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Forward pass
        outputs = self.model(input_tensor)
        
        # Get predictions
        category_probs = F.softmax(outputs['category_logits'], dim=1)
        gender_probs = F.softmax(outputs['gender_logits'], dim=1)
        
        # Get top predictions
        category_conf, category_idx = torch.max(category_probs, 1)
        gender_conf, gender_idx = torch.max(gender_probs, 1)
        
        # Convert to numpy
        category_probs_np = category_probs.cpu().numpy()[0]
        gender_probs_np = gender_probs.cpu().numpy()[0]
        
        # Get category name
        category_name = self.idx_to_category.get(category_idx.item(), 'Unknown')
        gender_name = 'MEN' if gender_idx.item() == 0 else 'WOMEN'
        
        # Get top-k predictions
        top_k = min(5, len(self.category_mapping))
        top_k_indices = np.argsort(category_probs_np)[-top_k:][::-1]
        top_k_predictions = [
            {
                'category': self.idx_to_category.get(idx, 'Unknown'),
                'confidence': float(category_probs_np[idx])
            }
            for idx in top_k_indices
        ]
        
        return {
            'category': {
                'predicted': category_name,
                'confidence': float(category_conf.item()),
                'top_k': top_k_predictions
            },
            'gender': {
                'predicted': gender_name,
                'confidence': float(gender_conf.item())
            },
            'is_confident': category_conf.item() > self.confidence_threshold,
            'raw_outputs': {
                'category_logits': outputs['category_logits'].cpu().numpy(),
                'gender_logits': outputs['gender_logits'].cpu().numpy()
            }
        }
    
    def _predict_with_tta(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Prediction with test-time augmentation"""
        # Convert to PIL Image if needed
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        all_category_probs = []
        all_gender_probs = []
        
        # Apply multiple augmentations
        for transform in self.tta_transforms:
            input_tensor = transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                
                category_probs = F.softmax(outputs['category_logits'], dim=1)
                gender_probs = F.softmax(outputs['gender_logits'], dim=1)
                
                all_category_probs.append(category_probs.cpu().numpy())
                all_gender_probs.append(gender_probs.cpu().numpy())
        
        # Average predictions
        avg_category_probs = np.mean(all_category_probs, axis=0)[0]
        avg_gender_probs = np.mean(all_gender_probs, axis=0)[0]
        
        # Get predictions
        category_idx = np.argmax(avg_category_probs)
        gender_idx = np.argmax(avg_gender_probs)
        
        category_conf = avg_category_probs[category_idx]
        gender_conf = avg_gender_probs[gender_idx]
        
        # Get category and gender names
        category_name = self.idx_to_category.get(category_idx, 'Unknown')
        gender_name = 'MEN' if gender_idx == 0 else 'WOMEN'
        
        # Get top-k predictions
        top_k = min(5, len(self.category_mapping))
        top_k_indices = np.argsort(avg_category_probs)[-top_k:][::-1]
        top_k_predictions = [
            {
                'category': self.idx_to_category.get(idx, 'Unknown'),
                'confidence': float(avg_category_probs[idx])
            }
            for idx in top_k_indices
        ]
        
        return {
            'category': {
                'predicted': category_name,
                'confidence': float(category_conf),
                'top_k': top_k_predictions
            },
            'gender': {
                'predicted': gender_name,
                'confidence': float(gender_conf)
            },
            'is_confident': category_conf > self.confidence_threshold,
            'tta_used': True,
            'num_augmentations': len(self.tta_transforms)
        }
    
    def predict_batch(self, images: List[Union[str, Image.Image, np.ndarray]], 
                     batch_size: int = 32) -> List[Dict[str, Any]]:
        """Predict on a batch of images"""
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_tensors = []
            
            # Preprocess batch
            for image in batch_images:
                tensor = self.preprocess_image(image)
                batch_tensors.append(tensor)
            
            # Stack tensors
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            with torch.no_grad():
                # Forward pass
                outputs = self.model(batch_tensor)
                
                # Get predictions
                category_probs = F.softmax(outputs['category_logits'], dim=1)
                gender_probs = F.softmax(outputs['gender_logits'], dim=1)
                
                # Process each sample in batch
                for j in range(len(batch_images)):
                    category_conf, category_idx = torch.max(category_probs[j], 0)
                    gender_conf, gender_idx = torch.max(gender_probs[j], 0)
                    
                    category_name = self.idx_to_category.get(category_idx.item(), 'Unknown')
                    gender_name = 'MEN' if gender_idx.item() == 0 else 'WOMEN'
                    
                    results.append({
                        'category': {
                            'predicted': category_name,
                            'confidence': float(category_conf.item())
                        },
                        'gender': {
                            'predicted': gender_name,
                            'confidence': float(gender_conf.item())
                        },
                        'is_confident': category_conf.item() > self.confidence_threshold
                    })
        
        return results
    
    def analyze_image_features(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Extract and analyze image features"""
        # Preprocess image
        input_tensor = self.preprocess_image(image).to(self.device)
        
        with torch.no_grad():
            # Get intermediate features
            features = self.model.extract_features(input_tensor)
            
            # Get predictions
            outputs = self.model(input_tensor)
            prediction = self._predict_simple(image)
            
            return {
                'prediction': prediction,
                'features': {
                    'backbone_features': features['backbone_features'].cpu().numpy(),
                    'attention_weights': features.get('attention_weights', {}) if isinstance(features.get('attention_weights'), dict) else {},
                    'feature_dimensions': features['backbone_features'].shape
                },
                'analysis': {
                    'dominant_colors': self._extract_dominant_colors(image),
                    'texture_analysis': self._analyze_texture(image),
                    'shape_analysis': self._analyze_shape(image)
                }
            }
    
    def _extract_dominant_colors(self, image: Union[str, Image.Image, np.ndarray], k: int = 5) -> List[Dict[str, Any]]:
        """Extract dominant colors from image"""
        # Convert to numpy array
        if isinstance(image, str):
            img_array = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            img_array = np.array(image.convert('RGB'))
        else:
            img_array = image
        
        # Reshape for k-means
        pixels = img_array.reshape(-1, 3)
        
        # Use k-means to find dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = []
        for i, color in enumerate(kmeans.cluster_centers_):
            colors.append({
                'rgb': [int(c) for c in color],
                'hex': '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2])),
                'percentage': float(np.sum(kmeans.labels_ == i) / len(kmeans.labels_))
            })
        
        # Sort by percentage
        colors.sort(key=lambda x: x['percentage'], reverse=True)
        return colors
    
    def _analyze_texture(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, float]:
        """Analyze texture properties of the image"""
        # Convert to grayscale numpy array
        if isinstance(image, str):
            gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, Image.Image):
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Calculate texture features
        # Variance (texture roughness)
        variance = float(np.var(gray))
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)
        
        # Local binary pattern (simplified)
        lbp_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
        
        return {
            'variance': variance,
            'edge_density': edge_density,
            'lbp_variance': lbp_var,
            'smoothness': 1.0 / (1.0 + variance) if variance > 0 else 1.0
        }
    
    def _analyze_shape(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, float]:
        """Analyze shape properties of the image"""
        # Convert to numpy array
        if isinstance(image, str):
            img_array = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            img_array = np.array(image.convert('RGB'))
        else:
            img_array = image
        
        height, width = img_array.shape[:2]
        
        return {
            'aspect_ratio': float(width / height),
            'width': width,
            'height': height,
            'area': width * height
        }
    
    def compare_images(self, image1: Union[str, Image.Image, np.ndarray], 
                      image2: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Compare two images and find similarities"""
        # Get features for both images
        features1 = self.analyze_image_features(image1)
        features2 = self.analyze_image_features(image2)
        
        # Calculate feature similarity
        feat1 = features1['features']['backbone_features'].flatten()
        feat2 = features2['features']['backbone_features'].flatten()
        
        # Cosine similarity
        cosine_sim = float(np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2)))
        
        # Euclidean distance (normalized)
        euclidean_dist = float(np.linalg.norm(feat1 - feat2))
        
        return {
            'similarity_score': cosine_sim,
            'distance': euclidean_dist,
            'predictions': {
                'image1': features1['prediction'],
                'image2': features2['prediction']
            },
            'same_category': features1['prediction']['category']['predicted'] == features2['prediction']['category']['predicted'],
            'same_gender': features1['prediction']['gender']['predicted'] == features2['prediction']['gender']['predicted']
        }
    
    def find_similar_patterns(self, query_image: Union[str, Image.Image, np.ndarray],
                             image_database: List[Union[str, Image.Image, np.ndarray]],
                             top_k: int = 5) -> List[Dict[str, Any]]:
        """Find images with similar patterns to the query image"""
        # Get query features
        query_features = self.analyze_image_features(query_image)
        query_feat = query_features['features']['backbone_features'].flatten()
        
        similarities = []
        
        # Compare with each image in database
        for i, db_image in enumerate(image_database):
            try:
                db_features = self.analyze_image_features(db_image)
                db_feat = db_features['features']['backbone_features'].flatten()
                
                # Calculate similarity
                cosine_sim = float(np.dot(query_feat, db_feat) / (np.linalg.norm(query_feat) * np.linalg.norm(db_feat)))
                
                similarities.append({
                    'index': i,
                    'image': db_image,
                    'similarity': cosine_sim,
                    'prediction': db_features['prediction']
                })
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

class FashionPatternLearner:
    """Learn and analyze patterns from fashion images"""
    
    def __init__(self, predictor: FashionPredictor):
        self.predictor = predictor
        self.learned_patterns = {}
        self.category_features = {}
    
    def learn_from_images(self, images: List[Union[str, Image.Image, np.ndarray]], 
                         labels: List[str] = None) -> Dict[str, Any]:
        """Learn patterns from a collection of images"""
        print(f"Learning patterns from {len(images)} images...")
        
        category_features = {}
        all_features = []
        all_predictions = []
        
        for i, image in enumerate(images):
            try:
                # Analyze image
                analysis = self.predictor.analyze_image_features(image)
                prediction = analysis['prediction']
                features = analysis['features']['backbone_features'].flatten()
                
                # Store features by category
                category = prediction['category']['predicted']
                if category not in category_features:
                    category_features[category] = []
                category_features[category].append(features)
                
                all_features.append(features)
                all_predictions.append(prediction)
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue
        
        # Calculate category centroids
        category_centroids = {}
        for category, features_list in category_features.items():
            if features_list:
                category_centroids[category] = np.mean(features_list, axis=0)
        
        # Store learned patterns
        self.category_features = category_features
        self.learned_patterns = {
            'category_centroids': category_centroids,
            'num_samples_per_category': {cat: len(feats) for cat, feats in category_features.items()},
            'learning_timestamp': datetime.now().isoformat()
        }
        
        print(f"Learned patterns for {len(category_centroids)} categories")
        return self.learned_patterns
    
    def predict_next_trend(self, recent_images: List[Union[str, Image.Image, np.ndarray]]) -> Dict[str, Any]:
        """Predict next fashion trend based on recent images"""
        if not self.learned_patterns:
            raise ValueError("No patterns learned yet. Call learn_from_images first.")
        
        # Analyze recent images
        recent_features = []
        recent_predictions = []
        
        for image in recent_images:
            analysis = self.predictor.analyze_image_features(image)
            recent_features.append(analysis['features']['backbone_features'].flatten())
            recent_predictions.append(analysis['prediction'])
        
        # Calculate trend direction
        if len(recent_features) > 1:
            # Calculate average change in features
            feature_changes = []
            for i in range(1, len(recent_features)):
                change = recent_features[i] - recent_features[i-1]
                feature_changes.append(change)
            
            avg_change = np.mean(feature_changes, axis=0)
            
            # Project future trend
            last_features = recent_features[-1]
            predicted_features = last_features + avg_change
            
            # Find closest category to predicted features
            best_category = None
            best_similarity = -1
            
            for category, centroid in self.learned_patterns['category_centroids'].items():
                similarity = np.dot(predicted_features, centroid) / (np.linalg.norm(predicted_features) * np.linalg.norm(centroid))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_category = category
            
            return {
                'predicted_trend_category': best_category,
                'confidence': float(best_similarity),
                'trend_direction': 'evolving' if np.linalg.norm(avg_change) > 0.1 else 'stable',
                'recent_categories': [pred['category']['predicted'] for pred in recent_predictions]
            }
        
        return {
            'predicted_trend_category': recent_predictions[0]['category']['predicted'] if recent_predictions else None,
            'confidence': 0.5,
            'trend_direction': 'insufficient_data'
        }

# Utility functions
def create_predictor(model_path: str, 
                   category_mapping: Dict[str, int] = None,
                   device: str = 'auto') -> FashionPredictor:
    """Create a fashion predictor"""
    return FashionPredictor(
        model_path=model_path,
        category_mapping=category_mapping,
        device=device
    )

def batch_predict_directory(predictor: FashionPredictor, 
                           directory: str,
                           output_file: str = None) -> List[Dict[str, Any]]:
    """Predict on all images in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(directory).glob(f'*{ext}'))
        image_paths.extend(Path(directory).glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_paths)} images in {directory}")
    
    results = []
    for img_path in image_paths:
        try:
            prediction = predictor.predict_single(str(img_path))
            results.append({
                'image_path': str(img_path),
                'filename': img_path.name,
                'prediction': prediction
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return results