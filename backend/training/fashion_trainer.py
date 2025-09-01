import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Import our models and data
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.fashion_ai_model import FashionAIModel, FashionGAN, FashionAISystem
from data.fashion_dataset import FashionDataModule, FashionAugmentation

class FashionTrainer:
    """Comprehensive trainer for fashion AI models"""
    
    def __init__(self,
                 model: nn.Module,
                 data_module: FashionDataModule,
                 config: Dict[str, Any],
                 device: str = 'auto',
                 experiment_name: str = None):
        """
        Args:
            model: The model to train
            data_module: Data module for loading data
            config: Training configuration
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            experiment_name: Name for the experiment
        """
        self.model = model
        self.data_module = data_module
        self.config = config
        
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
        
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup experiment tracking
        self.experiment_name = experiment_name or f"fashion_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints')) / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'logs'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Setup loss functions
        self._setup_loss_functions()
        
        print(f"Trainer initialized for experiment: {self.experiment_name}")
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        optimizer_config = self.config.get('optimizer', {'type': 'adam', 'lr': 0.001})
        
        if optimizer_config['type'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 0.001),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_config['type'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 0.01),
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_config['type'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 0.001),
                weight_decay=optimizer_config.get('weight_decay', 1e-2)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_config = self.config.get('scheduler', {'type': 'cosine'})
        
        if scheduler_config['type'].lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_config['type'].lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_config['type'].lower() == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        else:
            self.scheduler = None
    
    def _setup_loss_functions(self):
        """Setup loss functions"""
        # Get class weights for balanced training
        class_weights = self.data_module.get_class_weights()
        
        # Category classification loss
        self.category_criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None
        )
        
        # Gender classification loss
        self.gender_criterion = nn.CrossEntropyLoss()
        
        # Texture classification loss (if available)
        self.texture_criterion = nn.CrossEntropyLoss()
        
        # Shape regression loss (if available)
        self.shape_criterion = nn.MSELoss()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        category_correct = 0
        gender_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.data_module.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch['image'].to(self.device)
            category_labels = batch['category_label'].to(self.device)
            gender_labels = batch['gender_label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate losses
            category_loss = self.category_criterion(outputs['category_logits'], category_labels)
            gender_loss = self.gender_criterion(outputs['gender_logits'], gender_labels)
            
            # Total loss
            loss = category_loss + 0.5 * gender_loss
            
            # Add auxiliary losses if available
            if 'texture_label' in batch and 'texture_logits' in outputs:
                texture_labels = batch['texture_label'].to(self.device)
                texture_loss = self.texture_criterion(outputs['texture_logits'], texture_labels)
                loss += 0.3 * texture_loss
            
            if 'shape_features' in batch and 'shape_features' in outputs:
                shape_targets = batch['shape_features'].to(self.device)
                shape_loss = self.shape_criterion(outputs['shape_features'], shape_targets)
                loss += 0.2 * shape_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Calculate accuracy
            _, category_pred = torch.max(outputs['category_logits'], 1)
            _, gender_pred = torch.max(outputs['gender_logits'], 1)
            
            category_correct += (category_pred == category_labels).sum().item()
            gender_correct += (gender_pred == gender_labels).sum().item()
            total_samples += images.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Cat_Acc': f'{category_correct/total_samples:.3f}',
                'Gen_Acc': f'{gender_correct/total_samples:.3f}'
            })
            
            # Log to tensorboard
            global_step = self.current_epoch * len(self.data_module.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            self.writer.add_scalar('Train/CategoryLoss', category_loss.item(), global_step)
            self.writer.add_scalar('Train/GenderLoss', gender_loss.item(), global_step)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.data_module.train_loader)
        category_acc = category_correct / total_samples
        gender_acc = gender_correct / total_samples
        
        return {
            'loss': avg_loss,
            'category_accuracy': category_acc,
            'gender_accuracy': gender_acc
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        category_correct = 0
        gender_correct = 0
        total_samples = 0
        
        all_category_preds = []
        all_category_labels = []
        all_gender_preds = []
        all_gender_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_module.val_loader, desc="Validation"):
                # Move batch to device
                images = batch['image'].to(self.device)
                category_labels = batch['category_label'].to(self.device)
                gender_labels = batch['gender_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate losses
                category_loss = self.category_criterion(outputs['category_logits'], category_labels)
                gender_loss = self.gender_criterion(outputs['gender_logits'], gender_labels)
                
                loss = category_loss + 0.5 * gender_loss
                
                # Calculate accuracy
                _, category_pred = torch.max(outputs['category_logits'], 1)
                _, gender_pred = torch.max(outputs['gender_logits'], 1)
                
                category_correct += (category_pred == category_labels).sum().item()
                gender_correct += (gender_pred == gender_labels).sum().item()
                total_samples += images.size(0)
                total_loss += loss.item()
                
                # Store predictions for detailed analysis
                all_category_preds.extend(category_pred.cpu().numpy())
                all_category_labels.extend(category_labels.cpu().numpy())
                all_gender_preds.extend(gender_pred.cpu().numpy())
                all_gender_labels.extend(gender_labels.cpu().numpy())
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.data_module.val_loader)
        category_acc = category_correct / total_samples
        gender_acc = gender_correct / total_samples
        
        return {
            'loss': avg_loss,
            'category_accuracy': category_acc,
            'gender_accuracy': gender_acc,
            'category_predictions': all_category_preds,
            'category_labels': all_category_labels,
            'gender_predictions': all_gender_preds,
            'gender_labels': all_gender_labels
        }
    
    def train(self, epochs: int = None) -> Dict[str, List[float]]:
        """Main training loop"""
        if epochs is None:
            epochs = self.config.get('epochs', 100)
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['category_accuracy'])
                else:
                    self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_accuracies.append(train_metrics['category_accuracy'])
            self.val_accuracies.append(val_metrics['category_accuracy'])
            
            # Log to tensorboard
            self.writer.add_scalar('Epoch/TrainLoss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/TrainCategoryAcc', train_metrics['category_accuracy'], epoch)
            self.writer.add_scalar('Epoch/ValCategoryAcc', val_metrics['category_accuracy'], epoch)
            self.writer.add_scalar('Epoch/TrainGenderAcc', train_metrics['gender_accuracy'], epoch)
            self.writer.add_scalar('Epoch/ValGenderAcc', val_metrics['gender_accuracy'], epoch)
            self.writer.add_scalar('Epoch/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Category Acc: {train_metrics['category_accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Category Acc: {val_metrics['category_accuracy']:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_metrics['category_accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['category_accuracy']
            
            self.save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_acc': self.best_val_acc,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'config': self.config
            }, is_best)
            
            # Early stopping
            if self.config.get('early_stopping', False):
                patience = self.config.get('early_stopping_patience', 20)
                if epoch > patience:
                    recent_val_accs = self.val_accuracies[-patience:]
                    if all(acc <= self.best_val_acc for acc in recent_val_accs):
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Generate final report
        self.generate_training_report()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def save_checkpoint(self, state: Dict[str, Any], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{state['epoch']}.pth"
        torch.save(state, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(state, best_path)
            print(f"  New best model saved! Validation accuracy: {state['best_val_acc']:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
        return checkpoint
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.train_accuracies, label='Train Accuracy')
        axes[0, 1].plot(self.val_accuracies, label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate curve
        if hasattr(self, 'lr_history'):
            axes[1, 0].plot(self.lr_history)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)
        
        # Final validation metrics
        val_metrics = self.validate_epoch()
        
        # Confusion matrix for categories
        cm = confusion_matrix(val_metrics['category_labels'], val_metrics['category_predictions'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1], cmap='Blues')
        axes[1, 1].set_title('Category Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed classification report
        category_names = self.data_module.get_category_mapping()
        category_report = classification_report(
            val_metrics['category_labels'],
            val_metrics['category_predictions'],
            target_names=[name for name, _ in sorted(category_names.items(), key=lambda x: x[1])],
            output_dict=True
        )
        
        # Save reports
        with open(self.checkpoint_dir / 'classification_report.json', 'w') as f:
            json.dump(category_report, f, indent=2)
        
        # Save training summary
        summary = {
            'experiment_name': self.experiment_name,
            'total_epochs': len(self.train_losses),
            'best_val_accuracy': self.best_val_acc,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'final_train_accuracy': self.train_accuracies[-1] if self.train_accuracies else None,
            'final_val_accuracy': self.val_accuracies[-1] if self.val_accuracies else None,
            'config': self.config
        }
        
        with open(self.checkpoint_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training report saved to: {self.checkpoint_dir}")

class FashionTrainingConfig:
    """Configuration class for fashion training"""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default training configuration"""
        return {
            'epochs': 100,
            'batch_size': 32,
            'num_workers': 4,
            'target_size': (256, 256),
            'optimizer': {
                'type': 'adamw',
                'lr': 0.001,
                'weight_decay': 0.01
            },
            'scheduler': {
                'type': 'cosine',
                'min_lr': 1e-6
            },
            'early_stopping': True,
            'early_stopping_patience': 20,
            'checkpoint_dir': 'checkpoints',
            'save_every_n_epochs': 10
        }
    
    @staticmethod
    def get_fast_config() -> Dict[str, Any]:
        """Get configuration for fast training/testing"""
        config = FashionTrainingConfig.get_default_config()
        config.update({
            'epochs': 10,
            'batch_size': 16,
            'early_stopping_patience': 5
        })
        return config
    
    @staticmethod
    def get_production_config() -> Dict[str, Any]:
        """Get configuration for production training"""
        config = FashionTrainingConfig.get_default_config()
        config.update({
            'epochs': 200,
            'batch_size': 64,
            'optimizer': {
                'type': 'adamw',
                'lr': 0.0005,
                'weight_decay': 0.02
            },
            'scheduler': {
                'type': 'cosine',
                'min_lr': 1e-7
            },
            'early_stopping_patience': 30
        })
        return config

# Utility functions
def create_trainer(data_dir: str, 
                  config: Dict[str, Any] = None,
                  experiment_name: str = None) -> FashionTrainer:
    """Create a fashion trainer with default setup"""
    if config is None:
        config = FashionTrainingConfig.get_default_config()
    
    # Setup data module
    data_module = FashionDataModule(
        data_dir=data_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        target_size=config['target_size']
    )
    data_module.setup()
    
    # Get number of categories
    num_categories = len(data_module.get_category_mapping())
    
    # Create model
    model = FashionAIModel(num_classes=num_categories)
    
    # Create trainer
    trainer = FashionTrainer(
        model=model,
        data_module=data_module,
        config=config,
        experiment_name=experiment_name
    )
    
    return trainer

def train_fashion_model(data_dir: str,
                       config: Dict[str, Any] = None,
                       experiment_name: str = None) -> FashionTrainer:
    """Train a fashion model with the given configuration"""
    trainer = create_trainer(data_dir, config, experiment_name)
    trainer.train()
    return trainer