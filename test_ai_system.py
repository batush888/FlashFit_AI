#!/usr/bin/env python3
"""
Simplified Fashion AI System Test
Tests basic functionality of the AI components
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image

# Add backend to path
sys.path.append('backend')

# Import our modules
from backend.data.fashion_dataset import FashionDataModule, create_fashion_dataloaders
from backend.models.fashion_ai_model import FashionAISystem
from backend.training.fashion_trainer import FashionTrainer
from backend.inference.fashion_predictor import FashionPredictor
from backend.generation.fashion_generator import FashionGenerator

class SimpleFashionAITester:
    def __init__(self):
        self.dataset_path = Path("/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/data/datasets")
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
    def test_data_loading(self):
        """Test basic data loading functionality"""
        print("\n=== Testing Data Loading ===")
        try:
            # Test data module creation
            data_module = FashionDataModule(
                data_dir=str(self.dataset_path),
                batch_size=4
            )
            data_module.setup()
            
            print(f"✓ Data module created successfully")
            if data_module.train_dataset is not None:
                print(f"  Train samples: {len(data_module.train_dataset)}")
            if data_module.val_dataset is not None:
                print(f"  Val samples: {len(data_module.val_dataset)}")
            if data_module.test_dataset is not None:
                print(f"  Test samples: {len(data_module.test_dataset)}")
            
            # Test getting a batch
            if data_module.train_loader is not None:
                train_batch = next(iter(data_module.train_loader))
                print(f"✓ Successfully loaded batch with shape: {train_batch['image'].shape}")
            
            return True
        except Exception as e:
            print(f"✗ Data loading test failed: {e}")
            return False
    
    def test_model_creation(self):
        """Test model creation and basic forward pass"""
        print("\n=== Testing Model Creation ===")
        try:
            # Create AI system with correct number of classes
            ai_system = FashionAISystem(num_classes=23, device=self.device)
            
            # Test with dummy input on the classifier model
            dummy_input = torch.randn(2, 3, 256, 256).to(self.device)
            
            with torch.no_grad():
                output = ai_system.classifier(dummy_input)
                print(f"✓ Model forward pass successful")
                print(f"  Input shape: {dummy_input.shape}")
                print(f"  Output keys: {list(output.keys())}")
            
            return True
        except Exception as e:
            print(f"✗ Model creation test failed: {e}")
            return False
    
    def test_training_setup(self):
        """Test training pipeline setup"""
        print("\n=== Testing Training Setup ===")
        try:
            # Create AI system and data module
            ai_system = FashionAISystem(num_classes=23, device=self.device)
            data_module = FashionDataModule(
                data_dir=str(self.dataset_path),
                batch_size=4
            )
            data_module.setup()
            
            if data_module.train_dataset is not None:
                print(f"Loaded {len(data_module.train_dataset)} samples for train split")
            if data_module.val_dataset is not None:
                print(f"Loaded {len(data_module.val_dataset)} samples for val split")
            if data_module.test_dataset is not None:
                print(f"Loaded {len(data_module.test_dataset)} samples for test split")
            print(f"Data module setup complete:")
            if data_module.train_dataset is not None:
                print(f"  Train samples: {len(data_module.train_dataset)}")
            if data_module.val_dataset is not None:
                print(f"  Val samples: {len(data_module.val_dataset)}")
            if data_module.test_dataset is not None:
                print(f"  Test samples: {len(data_module.test_dataset)}")
            print(f"  Categories: 23")  # Based on dataset analysis
            print(f"Using device: {self.device}")
            
            # Create config with proper structure
            config = {
                'learning_rate': 0.001,
                'batch_size': 4,
                'epochs': 1,
                'optimizer': {
                    'type': 'adam',
                    'lr': 0.001,
                    'weight_decay': 1e-4
                },
                'scheduler': {
                    'type': 'cosine',
                    'min_lr': 1e-6
                }
            }
            
            # Create trainer with the classifier model from AI system
            trainer = FashionTrainer(
                model=ai_system.classifier,
                data_module=data_module,
                config=config,
                device=self.device,
                experiment_name="test_training"
            )
            
            print("✓ Training pipeline setup successful")
            return True
        except Exception as e:
            print(f"✗ Training setup test failed: {e}")
            return False
    
    def test_prediction_setup(self):
        """Test prediction system setup"""
        print("\n=== Testing Prediction Setup ===")
        try:
            # Create a simple category mapping for testing
            category_mapping = {
                'MEN-Denim': 0, 'MEN-Jackets_Vests': 1, 'MEN-Pants': 2, 'MEN-Shirts_Polos': 3,
                'MEN-Shorts': 4, 'MEN-Sweaters': 5, 'MEN-Tees_Tanks': 6,
                'WOMEN-Blouses_Shirts': 7, 'WOMEN-Cardigans': 8, 'WOMEN-Denim': 9,
                'WOMEN-Dresses': 10, 'WOMEN-Graphic_Tees': 11, 'WOMEN-Jackets_Coats': 12,
                'WOMEN-Leggings': 13, 'WOMEN-Pants': 14, 'WOMEN-Shorts': 15,
                'WOMEN-Skirts': 16, 'WOMEN-Sweaters': 17, 'WOMEN-Tees_Tanks': 18
            }
            
            # Create AI system with 19 classes to match the default model config
            ai_system = FashionAISystem(num_classes=19, device=self.device)
            
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
                torch.save({
                    'model_state_dict': ai_system.classifier.state_dict(),
                    'num_classes': 19,
                    'model_config': {'num_classes': 19},
                    'category_mapping': category_mapping
                }, tmp_file.name)
                model_path = tmp_file.name
            
            # Create predictor
            predictor = FashionPredictor(
                model_path=model_path,
                category_mapping=category_mapping,  # Use the mapping we created
                device=self.device,
                confidence_threshold=0.5
            )
            
            print("✓ Prediction system setup successful")
            
            # Cleanup
            os.unlink(model_path)
            return True
        except Exception as e:
            print(f"✗ Prediction setup test failed: {e}")
            return False
    
    def test_generation_setup(self):
        """Test generation system setup"""
        print("\n=== Testing Generation Setup ===")
        try:
            # Create generator (no num_classes parameter needed)
            # Note: FashionGenerator can work without a pre-trained generator or predictor
            generator = FashionGenerator(
                latent_dim=128,
                device=self.device
            )
            
            print("✓ Generation system setup successful")
            return True
        except Exception as e:
            print(f"✗ Generation setup test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests and return summary"""
        print("\n" + "="*60)
        print("FASHION AI SYSTEM - SIMPLIFIED TESTS")
        print("="*60)
        
        tests = [
            ('Data Loading', self.test_data_loading),
            ('Model Creation', self.test_model_creation),
            ('Training Setup', self.test_training_setup),
            ('Prediction Setup', self.test_prediction_setup),
            ('Generation Setup', self.test_generation_setup)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
            except Exception as e:
                print(f"✗ {test_name} test crashed: {e}")
                results[test_name] = False
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\n🎉 All tests passed! The AI system is working correctly.")
            return True
        else:
            print(f"\n⚠️  {total - passed} tests failed. Please check the error messages above.")
            return False

def main():
    """Main test function"""
    tester = SimpleFashionAITester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ Fashion AI system is ready for use!")
        return 0
    else:
        print("\n❌ Fashion AI system needs fixes before use.")
        return 1

if __name__ == "__main__":
    exit(main())