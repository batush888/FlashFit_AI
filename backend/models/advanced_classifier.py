import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Union, Dict
import cv2
from sklearn.cluster import KMeans
import os
try:
    from transformers import CLIPModel, CLIPProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
import colorsys

class AdvancedGarmentClassifier:
    """
    Advanced Garment Classifier that addresses the issues with:
    1. Misidentifying garment types (shorts/shirts as jackets)
    2. Color misidentification (black as white, beige as pink)
    3. Better texture and shape analysis
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize CLIP for semantic understanding
        self.clip_model = None
        self.clip_processor = None
        self.clip_available = False
        
        if CLIP_AVAILABLE:
            try:
                from transformers import CLIPModel, CLIPProcessor
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                # Move model to appropriate device
                self.clip_model = self.clip_model.to(self.device)
                self.clip_available = True
            except Exception as e:
                print(f"CLIP model not available, using traditional CV methods only: {e}")
                self.clip_available = False
                self.clip_model = None
                self.clip_processor = None
        else:
            print("CLIP library not installed, using traditional CV methods only")
        
        # Garment categories with better definitions
        self.categories = {
            "shirt": "衬衫",
            "t-shirt": "T恤", 
            "blouse": "女式衬衫",
            "pants": "长裤",
            "jeans": "牛仔裤",
            "shorts": "短裤",
            "jacket": "夹克",
            "coat": "外套",
            "blazer": "西装外套",
            "dress": "连衣裙",
            "skirt": "裙子",
            "shoes": "鞋子",
            "accessory": "配饰"
        }
        
        # Improved color definitions with better thresholds
        self.color_definitions = {
            'black': {'rgb_range': [(0, 0, 0), (50, 50, 50)], 'hsv_range': [(0, 0, 0), (360, 100, 20)]},
            'white': {'rgb_range': [(200, 200, 200), (255, 255, 255)], 'hsv_range': [(0, 0, 80), (360, 20, 100)]},
            'gray': {'rgb_range': [(50, 50, 50), (200, 200, 200)], 'hsv_range': [(0, 0, 20), (360, 20, 80)]},
            'beige': {'rgb_range': [(180, 150, 120), (220, 190, 160)], 'hsv_range': [(20, 10, 70), (40, 40, 90)]},
            'brown': {'rgb_range': [(80, 50, 30), (150, 100, 70)], 'hsv_range': [(10, 30, 30), (30, 70, 60)]},
            'red': {'rgb_range': [(150, 0, 0), (255, 100, 100)], 'hsv_range': [(350, 50, 50), (10, 100, 100)]},
            'blue': {'rgb_range': [(0, 50, 150), (100, 150, 255)], 'hsv_range': [(200, 50, 50), (250, 100, 100)]},
            'navy': {'rgb_range': [(0, 20, 80), (50, 70, 120)], 'hsv_range': [(210, 60, 30), (240, 100, 50)]},
            'light_blue': {'rgb_range': [(100, 150, 200), (180, 220, 255)], 'hsv_range': [(190, 20, 70), (220, 50, 100)]},
            'green': {'rgb_range': [(0, 100, 0), (100, 200, 100)], 'hsv_range': [(90, 30, 40), (150, 100, 80)]},
            'yellow': {'rgb_range': [(200, 200, 0), (255, 255, 150)], 'hsv_range': [(50, 50, 70), (70, 100, 100)]},
            'orange': {'rgb_range': [(200, 100, 0), (255, 180, 100)], 'hsv_range': [(10, 60, 70), (35, 100, 100)]},
            'pink': {'rgb_range': [(200, 150, 170), (255, 200, 220)], 'hsv_range': [(320, 15, 75), (350, 40, 100)]},
            'purple': {'rgb_range': [(100, 0, 150), (180, 100, 220)], 'hsv_range': [(270, 50, 40), (320, 100, 90)]}
        }
        
        # CLIP text prompts for better semantic understanding
        self.clip_prompts = {
            "shirt": ["a shirt", "a button-up shirt", "a dress shirt", "a casual shirt"],
            "t-shirt": ["a t-shirt", "a tee shirt", "a casual t-shirt", "a cotton t-shirt"],
            "blouse": ["a blouse", "a women's blouse", "a feminine top"],
            "pants": ["pants", "trousers", "long pants", "dress pants"],
            "jeans": ["jeans", "denim pants", "blue jeans", "denim trousers"],
            "shorts": ["shorts", "short pants", "summer shorts", "casual shorts"],
            "jacket": ["a jacket", "a casual jacket", "a light jacket"],
            "coat": ["a coat", "a long coat", "an overcoat", "outerwear"],
            "blazer": ["a blazer", "a suit jacket", "a formal jacket"],
            "dress": ["a dress", "a women's dress", "a long dress", "a casual dress"],
            "skirt": ["a skirt", "a women's skirt", "a short skirt", "a long skirt"],
            "shoes": ["shoes", "footwear", "sneakers", "dress shoes"],
            "accessory": ["an accessory", "jewelry", "a bag", "a hat"]
        }
    
    def classify_garment(self, image: Union[Image.Image, str, np.ndarray], debug: bool = False) -> Dict:
        """
        Advanced garment classification with improved accuracy
        """
        # Preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Extract comprehensive features
        features = self._extract_advanced_features(image)
        
        if debug:
            print(f"\n🔍 Advanced Classification Analysis:")
            print(f"   Aspect Ratio: {features['aspect_ratio']:.3f}")
            print(f"   Edge Density: {features['edge_density']:.4f}")
            print(f"   Texture Complexity: {features['texture_complexity']:.4f}")
            print(f"   Color Uniformity: {features['color_uniformity']:.3f}")
            print(f"   Shape Regularity: {features['shape_regularity']:.3f}")
        
        # Multi-stage classification
        category_scores = {}
        
        # 1. Traditional CV-based classification
        cv_category = self._cv_based_classify(features)
        category_scores['cv'] = cv_category
        
        # 2. CLIP-based semantic classification (if available)
        if self.clip_available:
            clip_scores = self._clip_classify(image)
            category_scores['clip'] = clip_scores
        
        # 3. Fusion of results
        final_category = self._fuse_classifications(category_scores, features)
        
        # Extract colors with improved accuracy
        colors = self._extract_colors_advanced(image, final_category)
        
        if debug:
            print(f"   → Final Classification: {self.categories.get(final_category, final_category)}")
            print(f"   → Primary Color: {colors[0]['name'] if colors else 'unknown'}")
        
        return {
            "category": final_category,
            "category_cn": self.categories.get(final_category, final_category),
            "confidence": features.get('confidence', 0.85),
            "colors": colors,
            "dominant_colors": colors,
            "primary_color": colors[0]['name'] if colors else 'unknown',
            "features": features,
            "classification_scores": category_scores
        }
    
    def _extract_advanced_features(self, image: Image.Image) -> Dict:
        """
        Extract comprehensive features for better classification
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Basic geometric features
        aspect_ratio = height / width
        
        # Advanced edge analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Texture analysis using gradient-based method (alternative to LBP)
        # Calculate texture using Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        texture_complexity = np.var(gradient_mag)
        
        # Color analysis
        colors = img_array.reshape(-1, 3).astype(np.float32)
        color_variance = float(np.var(colors))
        color_uniformity = 1.0 / (1.0 + color_variance / 1000.0)  # Normalized
        
        # Shape regularity using contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                shape_regularity = min(circularity, 1.0)
            else:
                shape_regularity = 0.0
        else:
            shape_regularity = 0.0
        
        # Brightness and contrast
        brightness = float(gray.mean())
        contrast = float(gray.std())
        
        # Gradient magnitude for structure analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2).mean()
        
        return {
            'aspect_ratio': aspect_ratio,
            'edge_density': edge_density,
            'texture_complexity': texture_complexity / 1000.0,  # Normalized
            'color_variance': color_variance,
            'color_uniformity': color_uniformity,
            'shape_regularity': shape_regularity,
            'brightness': brightness,
            'contrast': contrast,
            'gradient_magnitude': gradient_magnitude / 100.0,  # Normalized
            'width': width,
            'height': height
        }
    
    def _cv_based_classify(self, features: Dict) -> str:
        """
        Improved rule-based classification using comprehensive features
        """
        aspect_ratio = features['aspect_ratio']
        edge_density = features['edge_density']
        texture_complexity = features['texture_complexity']
        color_uniformity = features['color_uniformity']
        shape_regularity = features['shape_regularity']
        brightness = features.get('brightness', 128)
        width = features.get('width', 200)
        height = features.get('height', 200)
        
        # More sophisticated rules with priority order
        
        # Shoes: typically wide and have complex texture/edges
        if (aspect_ratio < 0.8 and edge_density > 0.15 and 
            texture_complexity > 0.1):
            return "shoes"
        
        # Coats/Trench coats: long, structured, high edge density, moderate texture
        if (aspect_ratio >= 1.4 and aspect_ratio <= 2.2 and 
            edge_density > 0.06 and texture_complexity > 0.02 and 
            shape_regularity < 0.8):
            return "coat"
        
        # Hoodies: moderate aspect ratio, high texture complexity, lower edge density
        if (1.1 <= aspect_ratio <= 1.5 and 
            texture_complexity > 0.05 and edge_density < 0.08 and 
            color_uniformity < 0.7):
            return "jacket"  # Classify hoodies as jackets for simplicity
        
        # Jackets: moderate aspect ratio with structured edges
        if ((aspect_ratio >= 1.0 and aspect_ratio <= 1.4 and edge_density > 0.08) or
            (aspect_ratio >= 1.25 and aspect_ratio <= 1.4 and texture_complexity > 0.015)):
            return "jacket"
        
        # Shorts: square-ish, low aspect ratio, usually uniform color
        if (0.5 <= aspect_ratio < 0.95 and 
            color_uniformity > 0.6 and texture_complexity < 0.06):
            return "shorts"
        
        # T-shirts: moderate aspect ratio, high uniformity, low texture
        if (0.9 <= aspect_ratio <= 1.2 and 
            color_uniformity > 0.75 and texture_complexity < 0.04):
            return "t-shirt"
        
        # Shirts: moderate aspect ratio, less uniform than t-shirts
        if (0.95 <= aspect_ratio <= 1.3 and 
            color_uniformity > 0.4 and texture_complexity < 0.08):
            return "shirt"
        
        # Pants: tall and narrow
        if (aspect_ratio > 1.5 and aspect_ratio < 3.0 and 
            color_uniformity > 0.5):
            return "pants"
        
        # Dresses: very tall, various textures
        if (aspect_ratio > 1.6 and aspect_ratio < 2.5):
            return "dress"
        
        # Skirts: moderate height with specific characteristics
        if (1.2 <= aspect_ratio <= 1.8 and 
            texture_complexity > 0.15 and 
            color_uniformity < 0.5):
            return "skirt"
        
        # Default fallback based on aspect ratio with better logic
        if aspect_ratio < 0.9:
            return "shorts"
        elif aspect_ratio < 1.3:
            return "shirt"
        elif aspect_ratio < 1.8:
            return "jacket"
        else:
            return "pants"
    
    def _clip_classify(self, image: Image.Image) -> Dict[str, float]:
        """
        Use CLIP for semantic classification
        """
        if not self.clip_available or self.clip_model is None or self.clip_processor is None:
            return {}
        
        try:
            # Prepare image and text inputs
            inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get image features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            category_scores = {}
            
            for category, prompts in self.clip_prompts.items():
                scores = []
                for prompt in prompts:
                    text_inputs = self.clip_processor(text=prompt, return_tensors="pt", padding=True)
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                    
                    with torch.no_grad():
                        text_features = self.clip_model.get_text_features(**text_inputs)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        
                        similarity = torch.cosine_similarity(image_features, text_features)
                        scores.append(float(similarity))
                
                category_scores[category] = max(scores)
            
            return category_scores
        
        except Exception as e:
            print(f"CLIP classification error: {e}")
            return {}
    
    def _fuse_classifications(self, category_scores: Dict, features: Dict) -> str:
        """
        Intelligently fuse different classification results
        """
        cv_result = category_scores.get('cv', 'shirt')
        clip_scores = category_scores.get('clip', {})
        
        if not clip_scores:
            return cv_result
        
        # Get top CLIP prediction
        clip_result = max(clip_scores.items(), key=lambda x: x[1])
        clip_category, clip_confidence = clip_result
        
        # Fusion logic
        if clip_confidence > 0.3:  # High CLIP confidence
            # Check if CV and CLIP agree
            if cv_result == clip_category:
                return cv_result  # Both agree
            else:
                # Resolve conflicts based on features
                aspect_ratio = features['aspect_ratio']
                edge_density = features['edge_density']
                
                # Use domain knowledge to resolve conflicts
                if clip_category in ['shorts', 't-shirt'] and cv_result == 'jacket':
                    # Likely misclassified by CV due to edges
                    if aspect_ratio < 1.3 and edge_density < 0.2:
                        return clip_category
                
                # Default to CLIP if confidence is high
                if clip_confidence > 0.4:
                    return clip_category
        
        return cv_result
    
    def _extract_colors_advanced(self, image: Image.Image, garment_type: str = "") -> List[Dict]:
        """
        Advanced color extraction with better accuracy
        """
        img_array = np.array(image)
        
        # Remove background using simple thresholding and morphology
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Create mask to focus on the garment
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to focus on garment pixels
        masked_img = img_array[mask > 0]
        
        if len(masked_img) == 0:
            masked_img = img_array.reshape(-1, 3)
        
        # Use K-means clustering for color extraction
        unique_colors = np.unique(masked_img.reshape(-1, 3), axis=0)
        n_colors = min(5, len(unique_colors))
        if n_colors < 2:
            n_colors = 2
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
        kmeans.fit(masked_img)
        
        colors = []
        labels = kmeans.labels_
        
        for i, color in enumerate(kmeans.cluster_centers_):
            if labels is None or len(labels) == 0:
                continue
            percentage = np.sum(labels == i) / len(labels)
            
            if percentage < 0.05:  # Skip very small color regions
                continue
            
            rgb = (int(color[0]), int(color[1]), int(color[2]))
            color_name = self._get_color_name_advanced(rgb)
            
            colors.append({
                "rgb": rgb,
                "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                "name": color_name,
                "percentage": round(percentage * 100, 1)
            })
        
        # Sort by percentage
        colors.sort(key=lambda x: x["percentage"], reverse=True)
        
        return colors[:3]  # Return top 3 colors
    
    def _get_color_name_advanced(self, rgb: Tuple[int, int, int]) -> str:
        """
        Advanced color name detection with better accuracy
        """
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        
        # Convert to HSV for better color analysis
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        h = h * 360  # Convert to degrees
        s = s * 100  # Convert to percentage
        v = v * 100  # Convert to percentage
        
        # Improved color detection with more precise thresholds
        
        # Black detection (very strict)
        if v < 20 and s < 30:
            return "black"
        
        # White detection (very bright with low saturation)
        elif v > 80 and s < 15:
            return "white"
        
        # Gray detection
        elif s < 20 and 20 <= v <= 80:
            if v < 40:
                return "dark gray"
            elif v > 60:
                return "light gray"
            else:
                return "gray"
        
        # Beige/cream detection (more specific)
        elif ((20 <= h <= 60 and 10 <= s <= 50 and v > 65) or 
              (r > 200 and g > 180 and b > 150 and abs(r-g) < 40 and r > b and g > b)):
            return "beige"
        
        # Blue detection (more specific for steel blue/light blue)
        elif 180 <= h <= 240 and s > 25:
            if v > 70:
                return "blue"  # Light blue
            elif v > 40:
                return "blue"  # Medium blue
            else:
                return "navy"  # Dark blue
        
        # Cyan detection (separate from blue)
        elif 150 <= h < 180 and s > 30:
            return "cyan"
        
        # Color detection based on hue for saturated colors
        elif s > 25:  # Only for reasonably saturated colors
            if h < 15 or h > 345:
                return "red"
            elif 15 <= h < 45:
                return "orange"
            elif 45 <= h < 75:
                return "yellow"
            elif 75 <= h < 150:
                return "green"
            elif 270 <= h < 330:
                return "purple"
            elif 330 <= h <= 345:
                return "pink"
        
        # Fallback: check if it's a desaturated version of a color
        if s > 10:  # Some saturation
            if 330 <= h <= 360 or 0 <= h < 30:
                return "pink" if s < 40 else "red"
            elif 180 <= h < 240:
                return "blue"
            elif 75 <= h < 150:
                return "green"
        
        # Final fallback
        return "gray"

# Global instance
_advanced_classifier = None

def get_advanced_classifier() -> AdvancedGarmentClassifier:
    """Get the global advanced classifier instance"""
    global _advanced_classifier
    if _advanced_classifier is None:
        _advanced_classifier = AdvancedGarmentClassifier()
    return _advanced_classifier