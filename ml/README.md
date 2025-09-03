# Outfit Compatibility ML System

A deep learning system for predicting fashion outfit compatibility using computer vision and neural networks.

## Overview

This system uses a CNN-based approach to analyze fashion items and predict whether they form a compatible outfit. The model is trained on fashion datasets and can be integrated into the FlashFit AI recommendation system.

## Features

- **Deep Learning Model**: CNN-based architecture with attention mechanisms
- **Multi-Item Analysis**: Supports 2-4 fashion items per outfit
- **Compatibility Scoring**: Outputs probability scores (0-1) for outfit compatibility
- **Multiple Export Formats**: PyTorch, ONNX, and TorchScript support
- **Production Ready**: Includes inference wrapper and integration guide
- **Comprehensive Evaluation**: Multiple metrics and visualization tools

## Project Structure

```
ml/
├── config.json                    # Training configuration
├── requirements.txt               # Python dependencies
├── README.md                     # This file
│
├── outfit_preprocessing.py        # Data preprocessing pipeline
├── outfit_compatibility_model.py  # Model architecture
├── train_outfit_compatibility.py  # Training pipeline
├── evaluate_model.py             # Model evaluation
├── predict_outfit_compatibility.py # Prediction examples
├── export_model.py               # Model export utilities
│
└── exported_models/              # Exported model files (created after export)
    ├── outfit_compatibility_model.pth
    ├── outfit_compatibility_model.onnx
    ├── outfit_compatibility_model.pt
    ├── outfit_inference.py
    ├── integration_guide.md
    └── export_info.json
```

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv outfit_ml_env
source outfit_ml_env/bin/activate  # On Windows: outfit_ml_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Ensure your data is organized as follows:
```
data/
├── datasets/
│   ├── train_images/          # Training images
│   └── outfit_items_dataset/  # Categorized items
│       ├── accessories/
│       ├── bottomwear/
│       ├── footwear/
│       ├── one-piece/
│       └── upperwear/
└── test_images/               # Test images
```

### 3. Training

```bash
# Train the model with default configuration
python train_outfit_compatibility.py

# Train with custom config
python train_outfit_compatibility.py --config custom_config.json

# Resume training from checkpoint
python train_outfit_compatibility.py --resume checkpoints/model_epoch_10.pth
```

### 4. Evaluation

```bash
# Evaluate trained model
python evaluate_model.py --checkpoint checkpoints/best_model.pth

# Generate detailed evaluation report
python evaluate_model.py --checkpoint checkpoints/best_model.pth --detailed
```

### 5. Prediction

```bash
# Single outfit prediction
python predict_outfit_compatibility.py --mode single --images item1.jpg item2.jpg item3.jpg

# Batch prediction
python predict_outfit_compatibility.py --mode batch --input_dir test_outfits/

# Find best combinations
python predict_outfit_compatibility.py --mode combinations --items_dir fashion_items/
```

### 6. Model Export

```bash
# Export all formats
python export_model.py --checkpoint checkpoints/best_model.pth --output_dir exported_models/

# Export specific format
python export_model.py --checkpoint checkpoints/best_model.pth --format pytorch
```

## Model Architecture

### Core Components

1. **Feature Extractor**: ResNet50/EfficientNet backbone for image feature extraction
2. **Attention Mechanism**: Focuses on important fashion elements
3. **Compatibility Network**: Analyzes relationships between items
4. **Classifier**: Outputs compatibility probability

### Model Specifications

- **Input**: 4 fashion item images (224x224 RGB)
- **Output**: Compatibility score (0-1) + item features
- **Parameters**: ~25M (ResNet50 backbone)
- **Memory**: ~100MB model size

### Loss Function

Combines Binary Cross-Entropy and Triplet Loss:
- **BCE Loss**: For compatibility classification
- **Triplet Loss**: For feature space organization
- **Total Loss**: `α * BCE + β * Triplet`

## Configuration

Edit `config.json` to customize training:

```json
{
  "model": {
    "backbone": "resnet50",
    "num_items": 4,
    "feature_dim": 512,
    "hidden_dim": 256,
    "dropout_rate": 0.3
  },
  "training": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam"
  },
  "data": {
    "image_size": [224, 224],
    "augmentation": true,
    "train_split": 0.8,
    "val_split": 0.1
  }
}
```

## Performance Metrics

The system tracks multiple metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **F1-Score**: Harmonic mean of precision/recall
- **ROC AUC**: Area under ROC curve
- **PR AUC**: Area under Precision-Recall curve

## Integration with FlashFit AI

### Backend Integration

```python
from exported_models.outfit_inference import OutfitCompatibilityInference

# Initialize model
outfit_model = OutfitCompatibilityInference('exported_models/outfit_compatibility_model.pth')

# Predict compatibility
result = outfit_model.predict_compatibility(['item1.jpg', 'item2.jpg', 'item3.jpg'])
print(f"Compatibility: {result['compatibility_score']:.3f}")
```

### API Endpoint

```python
@app.post("/api/outfit/compatibility")
async def check_compatibility(outfit_images: List[str]):
    result = outfit_model.predict_compatibility(outfit_images)
    return {
        "compatibility_score": result['compatibility_score'],
        "is_compatible": result['is_compatible'],
        "confidence": result['confidence']
    }
```

## Advanced Usage

### Custom Data Preprocessing

```python
from outfit_preprocessing import OutfitDataPreprocessor

preprocessor = OutfitDataPreprocessor(
    image_size=(224, 224),
    augmentation=True,
    normalize=True
)

# Process custom dataset
processed_data = preprocessor.process_directory('custom_fashion_data/')
```

### Model Fine-tuning

```python
from outfit_compatibility_model import create_model
from train_outfit_compatibility import OutfitCompatibilityTrainer

# Load pre-trained model
model = create_model(config)
model.load_state_dict(torch.load('pretrained_model.pth'))

# Fine-tune on new data
trainer = OutfitCompatibilityTrainer(model, config)
trainer.train(new_train_loader, new_val_loader)
```

### Batch Processing

```python
# Process multiple outfits
outfit_lists = [
    ['outfit1_item1.jpg', 'outfit1_item2.jpg'],
    ['outfit2_item1.jpg', 'outfit2_item2.jpg']
]

results = outfit_model.batch_predict(outfit_lists)
for i, result in enumerate(results):
    print(f"Outfit {i+1}: {result['compatibility_score']:.3f}")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config.json
   - Use gradient accumulation
   - Enable mixed precision training

2. **Low Accuracy**
   - Increase training epochs
   - Adjust learning rate
   - Add more training data
   - Try different backbone (EfficientNet)

3. **Slow Training**
   - Use GPU if available
   - Increase batch size
   - Use DataLoader with multiple workers

4. **Model Export Errors**
   - Ensure model is in eval mode
   - Check input tensor shapes
   - Verify ONNX/TorchScript compatibility

### Debug Mode

```bash
# Enable debug logging
export PYTHONPATH=$PYTHONPATH:.
python train_outfit_compatibility.py --debug
```

## Performance Optimization

### Training Optimization

- **Mixed Precision**: Use `--amp` flag for faster training
- **Gradient Accumulation**: For larger effective batch sizes
- **Learning Rate Scheduling**: Cosine annealing or step decay
- **Early Stopping**: Prevent overfitting

### Inference Optimization

- **Model Quantization**: Reduce model size
- **ONNX Runtime**: Faster inference
- **Batch Processing**: Process multiple outfits together
- **Caching**: Cache frequently used predictions

## Monitoring and Logging

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/
```

### Weights & Biases (Optional)

```bash
# Login to W&B
wandb login

# Train with W&B logging
python train_outfit_compatibility.py --wandb
```

## Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=. tests/

# Test specific component
pytest tests/test_model.py
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## Model Versioning

- **v1.0**: Initial CNN-based model
- **v1.1**: Added attention mechanism
- **v1.2**: Improved loss function with triplet loss
- **v2.0**: Multi-backbone support (ResNet, EfficientNet)

## License

This project is part of the FlashFit AI system. See main project license for details.

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the integration guide
3. Contact the ML team
4. Create an issue in the project repository

## Acknowledgments

- PyTorch team for the deep learning framework
- Fashion dataset contributors
- FlashFit AI development team

---

**Note**: This system is designed for fashion compatibility prediction and should be integrated with the existing FlashFit AI recommendation system for optimal performance.