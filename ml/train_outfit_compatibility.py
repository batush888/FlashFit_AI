import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from pathlib import Path
import argparse
from datetime import datetime

from outfit_preprocessing import OutfitDataPreprocessor, OutfitCompatibilityDataset
from outfit_compatibility_model import OutfitCompatibilityModel, OutfitCompatibilityLoss, create_model

class OutfitCompatibilityTrainer:
    """Trainer class for outfit compatibility model"""
    
    def __init__(self, 
                 model: OutfitCompatibilityModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device: torch.device,
                 config: dict):
        """
        Initialize trainer
        
        Args:
            model: The outfit compatibility model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to train on
            config: Training configuration
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # Loss function
        self.criterion = OutfitCompatibilityLoss(
            bce_weight=config.get('bce_weight', 1.0),
            triplet_weight=config.get('triplet_weight', 0.1),
            margin=config.get('triplet_margin', 0.5)
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training history
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_name = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'cosine')
        
        if scheduler_name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.get('epochs', 100)
            )
        elif scheduler_name.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10
            )
        else:
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=1.0)  # No scheduling
    
    def train_epoch(self) -> tuple:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (items, targets) in enumerate(progress_bar):
            items = items.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, item_features = self.model(items)
            
            # Compute loss
            loss = self.criterion(predictions, targets, item_features)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
        accuracy = accuracy_score(all_targets, predictions_binary)
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> tuple:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for items, targets in tqdm(self.val_loader, desc="Validation"):
                items = items.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions, item_features = self.model(items)
                
                # Compute loss
                loss = self.criterion(predictions, targets, item_features)
                
                # Track metrics
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
        accuracy = accuracy_score(all_targets, predictions_binary)
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def train(self) -> dict:
        """Main training loop"""
        epochs = self.config.get('epochs', 100)
        save_dir = Path(self.config.get('save_dir', 'checkpoints'))
        save_dir.mkdir(exist_ok=True)
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_predictions, val_targets = self.validate_epoch()
            
            # Update learning rate
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()
            
            # Track history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'config': self.config,
                    'train_history': self.train_history
                }
                
                torch.save(checkpoint, save_dir / 'best_model.pth')
                print(f"New best model saved! Val Acc: {val_acc:.4f}")
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Early stopping
            if self.config.get('early_stopping', False):
                patience = self.config.get('patience', 20)
                if epoch - self.train_history['val_acc'].index(max(self.train_history['val_acc'])) >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.4f}")
        return self.train_history
    
    def evaluate(self) -> dict:
        """Evaluate model on test set"""
        print("Evaluating on test set...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for items, targets in tqdm(self.test_loader, desc="Testing"):
                items = items.to(self.device)
                targets = targets.to(self.device)
                
                predictions, _ = self.model(items)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate comprehensive metrics
        predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(all_targets, predictions_binary),
            'precision': precision_score(all_targets, predictions_binary),
            'recall': recall_score(all_targets, predictions_binary),
            'f1_score': f1_score(all_targets, predictions_binary),
            'auc_roc': roc_auc_score(all_targets, all_predictions)
        }
        
        print("\nTest Results:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax1.plot(self.train_history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(self.train_history['val_loss'], label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_history['train_acc'], label='Train Acc', color='blue')
        ax2.plot(self.train_history['val_acc'], label='Val Acc', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        ax3.plot(self.train_history['learning_rates'], color='green')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Loss difference
        loss_diff = np.array(self.train_history['val_loss']) - np.array(self.train_history['train_loss'])
        ax4.plot(loss_diff, color='purple')
        ax4.set_title('Validation - Training Loss (Overfitting Indicator)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()

def prepare_data(config: dict) -> tuple:
    """Prepare data loaders"""
    print("Preparing data...")
    
    # Initialize preprocessor
    preprocessor = OutfitDataPreprocessor(
        dataset_root=config['dataset_root'],
        train_images_path=config['train_images_path'],
        test_images_path=config['test_images_path'],
        image_size=tuple(config.get('image_size', [224, 224]))
    )
    
    # Process dataset
    items, combinations = preprocessor.process_dataset(config.get('processed_data_dir', 'processed_data'))
    
    # Create datasets
    train_transform = preprocessor.get_transform(is_training=True)
    val_transform = preprocessor.get_transform(is_training=False)
    
    # Create full dataset first
    full_dataset = OutfitCompatibilityDataset(combinations, train_transform)
    
    # Split data
    train_size = int(0.7 * len(combinations))
    val_size = int(0.15 * len(combinations))
    test_size = len(combinations) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Update transforms for validation and test sets
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    print(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")
    
    return train_loader, val_loader, test_loader

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Outfit Compatibility Model')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    args = parser.parse_args()
    
    # Default configuration
    default_config = {
        'dataset_root': '/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/data/datasets/outfit_items_dataset',
        'train_images_path': '/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/data/datasets/train_images',
        'test_images_path': '/Users/Batu/Downloads/SPSS_v20_MacOS/Administration/Lesson1/FlashFit_AI/data/test_images',
        'processed_data_dir': 'processed_data',
        'save_dir': 'checkpoints',
        'backbone': 'resnet50',
        'num_items': 4,
        'feature_dim': 512,
        'hidden_dim': 256,
        'dropout_rate': 0.3,
        'pretrained': True,
        'image_size': [224, 224],
        'batch_size': 16,
        'epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'bce_weight': 1.0,
        'triplet_weight': 0.1,
        'triplet_margin': 0.5,
        'grad_clip': 1.0,
        'early_stopping': True,
        'patience': 20,
        'num_workers': 4
    }
    
    # Load config if exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Update with defaults for missing keys
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
    else:
        config = default_config
        # Save default config
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Default config saved to {args.config}")
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project="outfit-compatibility",
            config=config,
            name=f"outfit-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(config)
    
    # Create model
    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = OutfitCompatibilityTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        config=config
    )
    
    # Train model
    history = trainer.train()
    
    # Evaluate model
    test_metrics = trainer.evaluate()
    
    # Plot training history
    save_dir = Path(config['save_dir'])
    trainer.plot_training_history(str(save_dir / 'training_history.png'))
    
    # Save final results
    results = {
        'config': config,
        'train_history': history,
        'test_metrics': test_metrics,
        'best_val_acc': trainer.best_val_acc
    }
    
    with open(save_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Training completed successfully!")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()