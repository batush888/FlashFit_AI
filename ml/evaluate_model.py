import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

from outfit_preprocessing import OutfitDataPreprocessor, OutfitCompatibilityDataset
from outfit_compatibility_model import OutfitCompatibilityModel, create_model
from train_outfit_compatibility import prepare_data

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model: OutfitCompatibilityModel, device: torch.device):
        """
        Initialize evaluator
        
        Args:
            model: Trained outfit compatibility model
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate_dataset(self, data_loader: DataLoader) -> Dict:
        """Evaluate model on a dataset"""
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_features = []
        
        print(f"Evaluating on {len(data_loader)} batches...")
        
        with torch.no_grad():
            for items, targets in tqdm(data_loader, desc="Evaluating"):
                items = items.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions, item_features = self.model(items)
                
                # Store results
                probabilities = torch.sigmoid(predictions).cpu().numpy()
                predictions_binary = (probabilities > 0.5).astype(int)
                
                all_predictions.extend(predictions_binary)
                all_probabilities.extend(probabilities)
                all_targets.extend(targets.cpu().numpy())
                all_features.append(item_features.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)
        all_features = np.concatenate(all_features, axis=0)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'targets': all_targets,
            'features': all_features
        }
    
    def _calculate_metrics(self, targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            # Basic classification metrics
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, zero_division=0),
            'recall': recall_score(targets, predictions, zero_division=0),
            'f1_score': f1_score(targets, predictions, zero_division=0),
            
            # Probability-based metrics
            'roc_auc': roc_auc_score(targets, probabilities) if len(np.unique(targets)) > 1 else 0.0,
            'average_precision': average_precision_score(targets, probabilities) if len(np.unique(targets)) > 1 else 0.0,
            
            # Additional metrics
            'specificity': self._calculate_specificity(targets, predictions),
            'balanced_accuracy': self._calculate_balanced_accuracy(targets, predictions),
            'mcc': self._calculate_mcc(targets, predictions),
        }
        
        return metrics
    
    def _calculate_specificity(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate specificity (true negative rate)"""
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_balanced_accuracy(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate balanced accuracy"""
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return (sensitivity + specificity) / 2
    
    def _calculate_mcc(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate Matthews Correlation Coefficient"""
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator if denominator > 0 else 0.0
    
    def plot_confusion_matrix(self, targets: np.ndarray, predictions: np.ndarray, 
                            save_path: Optional[str] = None, normalize: bool = True):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=['Incompatible', 'Compatible'],
                   yticklabels=['Incompatible', 'Compatible'])
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, targets: np.ndarray, probabilities: np.ndarray, save_path: Optional[str] = None):
        """Plot ROC curve"""
        if len(np.unique(targets)) <= 1:
            print("Cannot plot ROC curve: only one class present in targets")
            return
        
        fpr, tpr, thresholds = roc_curve(targets, probabilities)
        auc_score = roc_auc_score(targets, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, targets: np.ndarray, probabilities: np.ndarray, save_path: Optional[str] = None):
        """Plot Precision-Recall curve"""
        if len(np.unique(targets)) <= 1:
            print("Cannot plot PR curve: only one class present in targets")
            return
        
        precision, recall, thresholds = precision_recall_curve(targets, probabilities)
        ap_score = average_precision_score(targets, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {ap_score:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_probability_distribution(self, targets: np.ndarray, probabilities: np.ndarray, save_path: Optional[str] = None):
        """Plot probability distribution for each class"""
        plt.figure(figsize=(10, 6))
        
        # Separate probabilities by class
        compatible_probs = probabilities[targets == 1]
        incompatible_probs = probabilities[targets == 0]
        
        plt.hist(incompatible_probs, bins=50, alpha=0.7, label='Incompatible', color='red', density=True)
        plt.hist(compatible_probs, bins=50, alpha=0.7, label='Compatible', color='green', density=True)
        
        plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Probability Distribution by True Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_threshold_performance(self, targets: np.ndarray, probabilities: np.ndarray, save_path: Optional[str] = None):
        """Analyze performance across different thresholds"""
        thresholds = np.linspace(0, 1, 101)
        metrics = {
            'threshold': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'specificity': []
        }
        
        for threshold in thresholds:
            predictions = (probabilities > threshold).astype(int)
            
            metrics['threshold'].append(threshold)
            metrics['accuracy'].append(accuracy_score(targets, predictions))
            metrics['precision'].append(precision_score(targets, predictions, zero_division=0))
            metrics['recall'].append(recall_score(targets, predictions, zero_division=0))
            metrics['f1_score'].append(f1_score(targets, predictions, zero_division=0))
            metrics['specificity'].append(self._calculate_specificity(targets, predictions))
        
        # Plot threshold analysis
        plt.figure(figsize=(12, 8))
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']:
            plt.plot(metrics['threshold'], metrics[metric_name], label=metric_name.replace('_', ' ').title())
        
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Default Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Performance Metrics vs. Decision Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal threshold based on F1 score
        optimal_idx = np.argmax(metrics['f1_score'])
        optimal_threshold = metrics['threshold'][optimal_idx]
        optimal_f1 = metrics['f1_score'][optimal_idx]
        
        print(f"Optimal threshold (based on F1-score): {optimal_threshold:.3f}")
        print(f"F1-score at optimal threshold: {optimal_f1:.3f}")
        
        return metrics, optimal_threshold
    
    def generate_classification_report(self, targets: np.ndarray, predictions: np.ndarray) -> str:
        """Generate detailed classification report"""
        report = classification_report(
            targets, predictions,
            target_names=['Incompatible', 'Compatible'],
            digits=4
        )
        return report
    
    def analyze_feature_importance(self, features: np.ndarray, targets: np.ndarray, save_path: Optional[str] = None):
        """Analyze feature importance using correlation"""
        # Calculate correlation between features and targets
        correlations = []
        for i in range(features.shape[1]):
            corr = np.corrcoef(features[:, i], targets)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        feature_indices = range(len(correlations))
        plt.bar(feature_indices, correlations)
        plt.xlabel('Feature Index')
        plt.ylabel('Absolute Correlation with Target')
        plt.title('Feature Importance (Correlation with Compatibility)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Return top features
        top_features = sorted(enumerate(correlations), key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 most important features:")
        for idx, importance in top_features:
            print(f"Feature {idx}: {importance:.4f}")
        
        return correlations

def load_model(checkpoint_path: str, config: dict, device: torch.device) -> OutfitCompatibilityModel:
    """Load trained model from checkpoint"""
    model = create_model(config)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
        print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return model

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Outfit Compatibility Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory for results')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test', 'all'], default='test', 
                       help='Dataset to evaluate on')
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
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(config)
    
    # Select dataset
    if args.dataset == 'train':
        data_loader = train_loader
        dataset_name = 'Training'
    elif args.dataset == 'val':
        data_loader = val_loader
        dataset_name = 'Validation'
    elif args.dataset == 'test':
        data_loader = test_loader
        dataset_name = 'Test'
    else:  # all
        data_loaders = [('Training', train_loader), ('Validation', val_loader), ('Test', test_loader)]
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device)
    
    if args.dataset != 'all':
        # Evaluate single dataset
        print(f"\nEvaluating on {dataset_name} set...")
        results = evaluator.evaluate_dataset(data_loader)
        
        # Print metrics
        print(f"\n{dataset_name} Set Results:")
        print("=" * 50)
        for metric, value in results['metrics'].items():
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Generate classification report
        print("\nClassification Report:")
        print(evaluator.generate_classification_report(results['targets'], results['predictions']))
        
        # Create visualizations
        print("\nGenerating visualizations...")
        
        # Confusion Matrix
        evaluator.plot_confusion_matrix(
            results['targets'], results['predictions'],
            save_path=output_dir / f'{dataset_name.lower()}_confusion_matrix.png'
        )
        
        # ROC Curve
        evaluator.plot_roc_curve(
            results['targets'], results['probabilities'],
            save_path=output_dir / f'{dataset_name.lower()}_roc_curve.png'
        )
        
        # Precision-Recall Curve
        evaluator.plot_precision_recall_curve(
            results['targets'], results['probabilities'],
            save_path=output_dir / f'{dataset_name.lower()}_pr_curve.png'
        )
        
        # Probability Distribution
        evaluator.plot_probability_distribution(
            results['targets'], results['probabilities'],
            save_path=output_dir / f'{dataset_name.lower()}_prob_distribution.png'
        )
        
        # Threshold Analysis
        threshold_metrics, optimal_threshold = evaluator.analyze_threshold_performance(
            results['targets'], results['probabilities'],
            save_path=output_dir / f'{dataset_name.lower()}_threshold_analysis.png'
        )
        
        # Feature Importance
        evaluator.analyze_feature_importance(
            results['features'], results['targets'],
            save_path=output_dir / f'{dataset_name.lower()}_feature_importance.png'
        )
        
        # Save detailed results
        detailed_results = {
            'dataset': dataset_name,
            'metrics': results['metrics'],
            'optimal_threshold': optimal_threshold,
            'config': config
        }
        
        with open(output_dir / f'{dataset_name.lower()}_detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
    else:
        # Evaluate all datasets
        all_results = {}
        
        for dataset_name, data_loader in data_loaders:
            print(f"\nEvaluating on {dataset_name} set...")
            results = evaluator.evaluate_dataset(data_loader)
            all_results[dataset_name] = results['metrics']
            
            print(f"\n{dataset_name} Set Results:")
            print("=" * 50)
            for metric, value in results['metrics'].items():
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Create comparison table
        df = pd.DataFrame(all_results).T
        print("\nComparison Across Datasets:")
        print("=" * 80)
        print(df.round(4))
        
        # Save comparison results
        df.to_csv(output_dir / 'dataset_comparison.csv')
        
        with open(output_dir / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
    
    print(f"\nEvaluation completed! Results saved to {output_dir}")

if __name__ == "__main__":
    main()