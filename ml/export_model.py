import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import argparse
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    onnx = None
    ort = None
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available. Install with: pip install onnx onnxruntime")
from datetime import datetime

from outfit_compatibility_model import OutfitCompatibilityModel, create_model
from evaluate_model import load_model

class ModelExporter:
    """Export trained model for production use"""
    
    def __init__(self, model: OutfitCompatibilityModel, config: dict, device: torch.device):
        """
        Initialize model exporter
        
        Args:
            model: Trained outfit compatibility model
            config: Model configuration
            device: Device model is on
        """
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
    
    def export_pytorch_model(self, output_path: str) -> Dict:
        """Export model as PyTorch state dict with metadata"""
        export_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.config,
            'model_architecture': {
                'backbone': self.config.get('backbone', 'resnet50'),
                'num_items': self.config.get('num_items', 4),
                'feature_dim': self.config.get('feature_dim', 512),
                'hidden_dim': self.config.get('hidden_dim', 256),
                'dropout_rate': self.config.get('dropout_rate', 0.3)
            },
            'preprocessing': {
                'image_size': self.config.get('image_size', [224, 224]),
                'normalize_mean': [0.485, 0.456, 0.406],
                'normalize_std': [0.229, 0.224, 0.225]
            },
            'export_info': {
                'export_date': datetime.now().isoformat(),
                'pytorch_version': torch.__version__,
                'model_parameters': sum(p.numel() for p in self.model.parameters()),
                'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            }
        }
        
        torch.save(export_data, output_path)
        print(f"PyTorch model exported to: {output_path}")
        print(f"Model size: {export_data['export_info']['model_size_mb']:.2f} MB")
        
        return export_data['export_info']
    
    def export_onnx_model(self, output_path: str, input_shape: Optional[Tuple[int, ...]] = None) -> Dict:
        """Export model to ONNX format"""
        if input_shape is None:
            num_items = self.config.get('num_items', 4)
            image_size = self.config.get('image_size', [224, 224])
            input_shape = (1, num_items, 3, image_size[0], image_size[1])
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install with: pip install onnx onnxruntime")
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            (dummy_input,),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['outfit_images'],
            output_names=['compatibility_score', 'item_features'],
            dynamic_axes={
                'outfit_images': {0: 'batch_size'},
                'compatibility_score': {0: 'batch_size'},
                'item_features': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        if onnx is not None:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
        
        # Test ONNX runtime
        if ort is not None:
            ort_session = ort.InferenceSession(output_path)
        
        # Get model info
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        
        export_info = {
            'onnx_file_size_mb': file_size,
            'input_shape': input_shape,
            'opset_version': 11
        }
        
        print(f"ONNX model exported to: {output_path}")
        print(f"ONNX model size: {file_size:.2f} MB")
        
        return export_info
    
    def export_torchscript_model(self, output_path: str, input_shape: Optional[Tuple[int, ...]] = None) -> Dict:
        """Export model to TorchScript format"""
        if input_shape is None:
            num_items = self.config.get('num_items', 4)
            image_size = self.config.get('image_size', [224, 224])
            input_shape = (1, num_items, 3, image_size[0], image_size[1])
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Trace the model
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        # Save traced model
        torch.jit.save(traced_model, output_path)
        
        # Get model info
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        
        export_info = {
            'torchscript_file_size_mb': file_size,
            'input_shape': input_shape,
            'traced': True
        }
        
        print(f"TorchScript model exported to: {output_path}")
        print(f"TorchScript model size: {file_size:.2f} MB")
        
        return export_info
    
    def create_inference_wrapper(self, output_path: str) -> str:
        """Create a standalone inference wrapper class"""
        wrapper_code = f'''
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Union
from pathlib import Path

class OutfitCompatibilityInference:
    """Standalone inference wrapper for outfit compatibility model"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize inference wrapper
        
        Args:
            model_path: Path to exported PyTorch model
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model data
        model_data = torch.load(model_path, map_location=self.device)
        
        # Extract configuration
        self.config = model_data['model_config']
        self.model_architecture = model_data['model_architecture']
        self.preprocessing_config = model_data['preprocessing']
        
        # Recreate model architecture
        self.model = self._create_model()
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessing
        self.transform = self._create_transform()
        
        print(f"Model loaded on {{self.device}}")
        print(f"Model parameters: {{sum(p.numel() for p in self.model.parameters()):,}}")
    
    def _create_model(self):
        """Recreate model architecture"""
        # Import model architecture (you'll need to include the model definition)
        from outfit_compatibility_model import create_model
        return create_model(self.config)
    
    def _create_transform(self):
        """Create image preprocessing transform"""
        image_size = self.preprocessing_config['image_size']
        mean = self.preprocessing_config['normalize_mean']
        std = self.preprocessing_config['normalize_std']
        
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            raise ValueError(f"Error processing image {{image_path}}: {{e}}")
    
    def predict_compatibility(self, image_paths: List[str]) -> Dict:
        """Predict outfit compatibility
        
        Args:
            image_paths: List of paths to outfit item images
            
        Returns:
            Dictionary with compatibility score and metadata
        """
        # Preprocess images
        images = []
        valid_paths = []
        
        for path in image_paths:
            try:
                img_tensor = self.preprocess_image(path)
                images.append(img_tensor)
                valid_paths.append(path)
            except Exception as e:
                print(f"Warning: Skipping {{path}} - {{e}}")
        
        if len(images) == 0:
            raise ValueError("No valid images provided")
        
        # Pad or truncate to expected number of items
        num_items = self.model_architecture['num_items']
        
        if len(images) < num_items:
            # Pad with zeros
            while len(images) < num_items:
                images.append(torch.zeros_like(images[0]))
        elif len(images) > num_items:
            # Truncate
            images = images[:num_items]
            valid_paths = valid_paths[:num_items]
        
        # Create batch
        outfit_tensor = torch.stack(images).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            predictions, item_features = self.model(outfit_tensor)
            probability = torch.sigmoid(predictions).item()
        
        return {{
            'compatibility_score': probability,
            'is_compatible': probability > 0.5,
            'confidence': abs(probability - 0.5) * 2,
            'valid_image_paths': valid_paths,
            'raw_prediction': predictions.item(),
            'item_features_shape': item_features.shape
        }}
    
    def batch_predict(self, outfit_lists: List[List[str]]) -> List[Dict]:
        """Predict compatibility for multiple outfits
        
        Args:
            outfit_lists: List of outfit image path lists
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for outfit in outfit_lists:
            try:
                result = self.predict_compatibility(outfit)
                results.append(result)
            except Exception as e:
                results.append({{'error': str(e)}})
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {{
            'architecture': self.model_architecture,
            'preprocessing': self.preprocessing_config,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        }}

# Example usage:
if __name__ == "__main__":
    # Initialize inference
    inference = OutfitCompatibilityInference('outfit_compatibility_model.pth')
    
    # Predict single outfit
    outfit_images = ['item1.jpg', 'item2.jpg', 'item3.jpg']
    result = inference.predict_compatibility(outfit_images)
    
    print(f"Compatibility Score: {{result['compatibility_score']:.3f}}")
    print(f"Compatible: {{result['is_compatible']}}")
    print(f"Confidence: {{result['confidence']:.3f}}")
'''
        
        with open(output_path, 'w') as f:
            f.write(wrapper_code)
        
        print(f"Inference wrapper created: {output_path}")
        return output_path
    
    def create_integration_guide(self, output_path: str) -> str:
        """Create integration guide for FlashFit AI system"""
        guide_content = f'''
# Outfit Compatibility Model Integration Guide

## Overview
This guide explains how to integrate the trained outfit compatibility model into the existing FlashFit AI system.

## Model Information
- **Architecture**: {self.config.get('backbone', 'resnet50')}-based CNN
- **Input**: {self.config.get('num_items', 4)} fashion item images
- **Output**: Compatibility score (0-1) and item features
- **Image Size**: {self.config.get('image_size', [224, 224])}
- **Parameters**: {sum(p.numel() for p in self.model.parameters()):,}

## Files Exported
1. `outfit_compatibility_model.pth` - PyTorch model with full metadata
2. `outfit_compatibility_model.onnx` - ONNX format for cross-platform deployment
3. `outfit_compatibility_model.pt` - TorchScript format for production
4. `outfit_inference.py` - Standalone inference wrapper
5. `integration_guide.md` - This guide

## Integration Steps

### 1. Backend Integration (Python/FastAPI)

```python
# Add to backend/models/
from outfit_inference import OutfitCompatibilityInference

# Initialize model (do this once at startup)
outfit_model = OutfitCompatibilityInference('models/outfit_compatibility_model.pth')

# Add new API endpoint in app.py
@app.post("/api/outfit/compatibility")
async def predict_outfit_compatibility(outfit_images: List[str]):
    try:
        result = outfit_model.predict_compatibility(outfit_images)
        return {{
            "success": True,
            "compatibility_score": result['compatibility_score'],
            "is_compatible": result['is_compatible'],
            "confidence": result['confidence']
        }}
    except Exception as e:
        return {{"success": False, "error": str(e)}}
```

### 2. Frontend Integration (React/TypeScript)

```typescript
// Add to frontend/src/services/
export interface OutfitCompatibilityResult {{
  success: boolean;
  compatibility_score?: number;
  is_compatible?: boolean;
  confidence?: number;
  error?: string;
}}

export const outfitService = {{
  async checkCompatibility(imagePaths: string[]): Promise<OutfitCompatibilityResult> {{
    const response = await fetch('/api/outfit/compatibility', {{
      method: 'POST',
      headers: {{
        'Content-Type': 'application/json',
      }},
      body: JSON.stringify({{ outfit_images: imagePaths }}),
    }});
    
    return await response.json();
  }}
}};
```

### 3. UI Components

```tsx
// Add compatibility indicator component
interface CompatibilityIndicatorProps {{
  score: number;
  isCompatible: boolean;
  confidence: number;
}}

const CompatibilityIndicator: React.FC<CompatibilityIndicatorProps> = ({{ 
  score, isCompatible, confidence 
}}) => {{
  const color = isCompatible ? 'green' : 'red';
  const percentage = Math.round(score * 100);
  
  return (
    <div className="compatibility-indicator">
      <div className="score" style={{{{ color }}}}>
        {{percentage}}% Compatible
      </div>
      <div className="confidence">
        Confidence: {{Math.round(confidence * 100)}}%
      </div>
    </div>
  );
}};
```

### 4. Database Schema Updates

```sql
-- Add compatibility tracking table
CREATE TABLE outfit_compatibility (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    item_ids INTEGER[],
    compatibility_score FLOAT,
    is_compatible BOOLEAN,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add index for performance
CREATE INDEX idx_outfit_compatibility_user_id ON outfit_compatibility(user_id);
CREATE INDEX idx_outfit_compatibility_score ON outfit_compatibility(compatibility_score);
```

### 5. Existing System Integration Points

#### A. Fashion Generator Integration
Update `backend/generation/fashion_generator.py`:

```python
class FashionGenerator:
    def __init__(self):
        # ... existing code ...
        self.outfit_model = OutfitCompatibilityInference('models/outfit_compatibility_model.pth')
    
    def generate_compatible_outfit(self, base_items: List[str], num_suggestions: int = 5):
        """Generate outfit suggestions that are compatible with base items"""
        # Use existing generation methods but filter by compatibility
        candidates = self.generate_similar_items(base_items)
        
        compatible_outfits = []
        for candidate_set in candidates:
            result = self.outfit_model.predict_compatibility(base_items + candidate_set)
            if result['is_compatible'] and result['confidence'] > 0.7:
                compatible_outfits.append({{
                    'items': candidate_set,
                    'compatibility_score': result['compatibility_score'],
                    'confidence': result['confidence']
                }})
        
        # Sort by compatibility score
        compatible_outfits.sort(key=lambda x: x['compatibility_score'], reverse=True)
        return compatible_outfits[:num_suggestions]
```

#### B. Recommendation System Integration
Update `backend/services/match_service.py`:

```python
class MatchService:
    def __init__(self):
        # ... existing code ...
        self.outfit_model = OutfitCompatibilityInference('models/outfit_compatibility_model.pth')
    
    def get_enhanced_recommendations(self, user_preferences: dict, items: List[dict]):
        """Enhanced recommendations with compatibility scoring"""
        # Get base recommendations using existing logic
        base_recommendations = self.get_recommendations(user_preferences, items)
        
        # Add compatibility scoring
        enhanced_recommendations = []
        for rec in base_recommendations:
            item_paths = [item['image_path'] for item in rec['items']]
            compatibility = self.outfit_model.predict_compatibility(item_paths)
            
            rec['compatibility'] = {{
                'score': compatibility['compatibility_score'],
                'is_compatible': compatibility['is_compatible'],
                'confidence': compatibility['confidence']
            }}
            
            enhanced_recommendations.append(rec)
        
        # Re-rank by compatibility
        enhanced_recommendations.sort(
            key=lambda x: (x['compatibility']['score'], x['match_score']), 
            reverse=True
        )
        
        return enhanced_recommendations
```

## Performance Considerations

1. **Model Loading**: Load the model once at application startup, not per request
2. **Batch Processing**: Use `batch_predict()` for multiple outfits
3. **Caching**: Cache compatibility scores for frequently requested combinations
4. **Async Processing**: Use async/await for non-blocking inference

## Monitoring and Logging

```python
# Add logging for model predictions
import logging

logger = logging.getLogger(__name__)

def log_prediction(outfit_images: List[str], result: dict):
    logger.info(f"Outfit compatibility prediction: {{
        'num_items': len(outfit_images),
        'compatibility_score': result['compatibility_score'],
        'is_compatible': result['is_compatible'],
        'confidence': result['confidence']
    }}")
```

## Testing

```python
# Unit tests for integration
import unittest
from outfit_inference import OutfitCompatibilityInference

class TestOutfitCompatibility(unittest.TestCase):
    def setUp(self):
        self.model = OutfitCompatibilityInference('models/outfit_compatibility_model.pth')
    
    def test_single_prediction(self):
        result = self.model.predict_compatibility(['test1.jpg', 'test2.jpg'])
        self.assertIn('compatibility_score', result)
        self.assertIsInstance(result['is_compatible'], bool)
    
    def test_batch_prediction(self):
        outfits = [['test1.jpg', 'test2.jpg'], ['test3.jpg', 'test4.jpg']]
        results = self.model.batch_predict(outfits)
        self.assertEqual(len(results), 2)
```

## Deployment Notes

1. **Model Files**: Ensure model files are included in deployment package
2. **Dependencies**: Add required packages to requirements.txt:
   ```
   torch>=1.9.0
   torchvision>=0.10.0
   Pillow>=8.0.0
   numpy>=1.21.0
   ```
3. **GPU Support**: For GPU deployment, ensure CUDA compatibility
4. **Memory**: Model requires ~{sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024):.0f}MB RAM

## API Documentation

Add to your API documentation:

```yaml
/api/outfit/compatibility:
  post:
    summary: Check outfit compatibility
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              outfit_images:
                type: array
                items:
                  type: string
                description: List of image file paths
    responses:
      200:
        description: Compatibility prediction
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                compatibility_score:
                  type: number
                  minimum: 0
                  maximum: 1
                is_compatible:
                  type: boolean
                confidence:
                  type: number
                  minimum: 0
                  maximum: 1
```

## Next Steps

1. Implement the backend API endpoint
2. Create frontend components for compatibility display
3. Update existing recommendation logic
4. Add compatibility tracking to user analytics
5. Consider A/B testing the enhanced recommendations
6. Monitor model performance and retrain as needed

For questions or issues, refer to the model training documentation or contact the ML team.
'''
        
        with open(output_path, 'w') as f:
            f.write(guide_content)
        
        print(f"Integration guide created: {output_path}")
        return output_path
    
    def export_all_formats(self, output_dir: str) -> Dict:
        """Export model in all supported formats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        export_info = {
            'export_directory': str(output_path),
            'export_date': datetime.now().isoformat(),
            'formats': {}
        }
        
        # PyTorch format
        pytorch_info = self.export_pytorch_model(str(output_path / 'outfit_compatibility_model.pth'))
        export_info['formats']['pytorch'] = pytorch_info
        
        # ONNX format
        try:
            onnx_info = self.export_onnx_model(str(output_path / 'outfit_compatibility_model.onnx'))
            export_info['formats']['onnx'] = onnx_info
        except Exception as e:
            print(f"Warning: ONNX export failed: {e}")
            export_info['formats']['onnx'] = {'error': str(e)}
        
        # TorchScript format
        try:
            torchscript_info = self.export_torchscript_model(str(output_path / 'outfit_compatibility_model.pt'))
            export_info['formats']['torchscript'] = torchscript_info
        except Exception as e:
            print(f"Warning: TorchScript export failed: {e}")
            export_info['formats']['torchscript'] = {'error': str(e)}
        
        # Create inference wrapper
        wrapper_path = self.create_inference_wrapper(str(output_path / 'outfit_inference.py'))
        export_info['inference_wrapper'] = wrapper_path
        
        # Create integration guide
        guide_path = self.create_integration_guide(str(output_path / 'integration_guide.md'))
        export_info['integration_guide'] = guide_path
        
        # Save export metadata
        with open(output_path / 'export_info.json', 'w') as f:
            json.dump(export_info, f, indent=2)
        
        return export_info

def main():
    """Main export function"""
    parser = argparse.ArgumentParser(description='Export Outfit Compatibility Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='exported_models', help='Output directory')
    parser.add_argument('--format', type=str, choices=['pytorch', 'onnx', 'torchscript', 'all'], 
                       default='all', help='Export format')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Create exporter
    exporter = ModelExporter(model, config, device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nExporting model to {output_dir}...")
    
    if args.format == 'all':
        # Export all formats
        export_info = exporter.export_all_formats(str(output_dir))
        
        print("\nExport Summary:")
        print("=" * 50)
        for format_name, info in export_info['formats'].items():
            if 'error' in info:
                print(f"{format_name.upper()}: Failed - {info['error']}")
            else:
                size_key = f"{format_name}_file_size_mb"
                if size_key in info:
                    print(f"{format_name.upper()}: {info[size_key]:.2f} MB")
                else:
                    print(f"{format_name.upper()}: Exported successfully")
        
        print(f"\nAdditional files:")
        print(f"- Inference wrapper: {export_info['inference_wrapper']}")
        print(f"- Integration guide: {export_info['integration_guide']}")
        print(f"- Export metadata: {output_dir / 'export_info.json'}")
        
    else:
        # Export single format
        if args.format == 'pytorch':
            exporter.export_pytorch_model(str(output_dir / 'outfit_compatibility_model.pth'))
        elif args.format == 'onnx':
            exporter.export_onnx_model(str(output_dir / 'outfit_compatibility_model.onnx'))
        elif args.format == 'torchscript':
            exporter.export_torchscript_model(str(output_dir / 'outfit_compatibility_model.pt'))
    
    print(f"\nModel export completed successfully!")
    print(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    main()