import wandb
import os
import json
import ujson
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import warnings
warnings.filterwarnings('ignore')

# Initialize console for logging
console = Console()

@dataclass
class ExperimentConfig:
    """Configuration for W&B experiments"""
    project_name: str
    experiment_name: str
    tags: List[str]
    notes: str = ""
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}

@dataclass
class ModelMetrics:
    """Model performance metrics for tracking"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float
    epoch: int
    learning_rate: float
    batch_size: int
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class WandBIntegration:
    """
    Weights & Biases integration for FlashFit AI
    
    Features:
    - Experiment tracking and logging
    - Model performance monitoring
    - Hyperparameter optimization
    - Artifact management
    - Real-time visualization
    - Collaborative experiment sharing
    - Automated model versioning
    """
    
    def __init__(self, api_key: Optional[str] = None, entity: Optional[str] = None, 
                 project: str = "flashfit-ai", offline: bool = False):
        self.entity = entity
        self.project = project
        self.offline = offline
        self.current_run = None
        self.experiment_history = []
        
        # Initialize W&B
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
        
        # Set offline mode if specified
        if offline:
            os.environ['WANDB_MODE'] = 'offline'
            console.print("[yellow]⚠[/yellow] W&B running in offline mode")
        
        try:
            # Test W&B connection
            wandb.login()
            self.available = True
            console.print("[green]✓[/green] W&B integration initialized successfully")
        except Exception as e:
            self.available = False
            console.print(f"[red]✗[/red] W&B initialization failed: {e}")
            console.print("[yellow]⚠[/yellow] Continuing without W&B integration")
    
    def start_experiment(self, config: ExperimentConfig) -> Optional[wandb.Run]:
        """Start a new W&B experiment"""
        if not self.available:
            console.print("[yellow]W&B not available, skipping experiment start[/yellow]")
            return None
        
        try:
            # Initialize run
            self.current_run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=config.experiment_name,
                tags=config.tags,
                notes=config.notes,
                config=config.config,
                reinit=True
            )
            
            # Log system information
            self._log_system_info()
            
            console.print(f"[green]✓[/green] Started experiment: {config.experiment_name}")
            console.print(f"[blue]📊[/blue] W&B URL: {self.current_run.url}")
            
            return self.current_run
            
        except Exception as e:
            console.print(f"[red]Error starting experiment: {e}[/red]")
            return None
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None, 
                   commit: bool = True) -> bool:
        """Log metrics to W&B"""
        if not self.available or not self.current_run:
            return False
        
        try:
            wandb.log(metrics, step=step, commit=commit)
            return True
        except Exception as e:
            console.print(f"[red]Error logging metrics: {e}[/red]")
            return False
    
    def log_model_metrics(self, model_metrics: ModelMetrics, step: Optional[int] = None) -> bool:
        """Log model performance metrics"""
        metrics_dict = {
            'accuracy': model_metrics.accuracy,
            'precision': model_metrics.precision,
            'recall': model_metrics.recall,
            'f1_score': model_metrics.f1_score,
            'loss': model_metrics.loss,
            'epoch': model_metrics.epoch,
            'learning_rate': model_metrics.learning_rate,
            'batch_size': model_metrics.batch_size
        }
        
        return self.log_metrics(metrics_dict, step=step)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> bool:
        """Log hyperparameters to W&B config"""
        if not self.available or not self.current_run:
            return False
        
        try:
            wandb.config.update(hyperparams)
            return True
        except Exception as e:
            console.print(f"[red]Error logging hyperparameters: {e}[/red]")
            return False
    
    def log_image(self, image: Union[np.ndarray, Image.Image, plt.Figure], 
                 caption: str = "", key: str = "image") -> bool:
        """Log image to W&B"""
        if not self.available or not self.current_run:
            return False
        
        try:
            if isinstance(image, plt.Figure):
                wandb.log({key: wandb.Image(image, caption=caption)})
            else:
                wandb.log({key: wandb.Image(image, caption=caption)})
            return True
        except Exception as e:
            console.print(f"[red]Error logging image: {e}[/red]")
            return False
    
    def log_table(self, data: pd.DataFrame, key: str = "data_table") -> bool:
        """Log pandas DataFrame as W&B table"""
        if not self.available or not self.current_run:
            return False
        
        try:
            table = wandb.Table(dataframe=data)
            wandb.log({key: table})
            return True
        except Exception as e:
            console.print(f"[red]Error logging table: {e}[/red]")
            return False
    
    def log_artifact(self, file_path: Union[str, Path], artifact_name: str, 
                    artifact_type: str = "dataset", description: str = "") -> bool:
        """Log file as W&B artifact"""
        if not self.available or not self.current_run:
            return False
        
        try:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                description=description
            )
            artifact.add_file(str(file_path))
            wandb.log_artifact(artifact)
            return True
        except Exception as e:
            console.print(f"[red]Error logging artifact: {e}[/red]")
            return False
    
    def log_model(self, model: nn.Module, model_name: str = "model", 
                 metadata: Optional[Dict] = None) -> bool:
        """Log PyTorch model as W&B artifact"""
        if not self.available or not self.current_run:
            return False
        
        try:
            # Save model temporarily
            temp_path = f"temp_{model_name}.pth"
            torch.save(model.state_dict(), temp_path)
            
            # Create artifact
            artifact = wandb.Artifact(
                name=model_name,
                type="model",
                description=f"FlashFit AI model: {model_name}",
                metadata=metadata or {}
            )
            artifact.add_file(temp_path)
            wandb.log_artifact(artifact)
            
            # Clean up
            os.remove(temp_path)
            
            return True
        except Exception as e:
            console.print(f"[red]Error logging model: {e}[/red]")
            return False
    
    def create_sweep(self, sweep_config: Dict[str, Any]) -> Optional[str]:
        """Create hyperparameter sweep"""
        if not self.available:
            return None
        
        try:
            sweep_id = wandb.sweep(
                sweep=sweep_config,
                project=self.project,
                entity=self.entity
            )
            console.print(f"[green]✓[/green] Created sweep: {sweep_id}")
            return sweep_id
        except Exception as e:
            console.print(f"[red]Error creating sweep: {e}[/red]")
            return None
    
    def run_sweep_agent(self, sweep_id: str, train_function: Callable, count: int = 10) -> bool:
        """Run sweep agent for hyperparameter optimization"""
        if not self.available:
            return False
        
        try:
            wandb.agent(
                sweep_id=sweep_id,
                function=train_function,
                count=count,
                project=self.project,
                entity=self.entity
            )
            return True
        except Exception as e:
            console.print(f"[red]Error running sweep agent: {e}[/red]")
            return False
    
    def watch_model(self, model: nn.Module, log_freq: int = 100, 
                   log_graph: bool = True) -> bool:
        """Watch model for gradient and parameter tracking"""
        if not self.available or not self.current_run:
            return False
        
        try:
            wandb.watch(model, log_freq=log_freq, log_graph=log_graph)
            return True
        except Exception as e:
            console.print(f"[red]Error watching model: {e}[/red]")
            return False
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: Optional[List[str]] = None) -> bool:
        """Log confusion matrix visualization"""
        if not self.available or not self.current_run:
            return False
        
        try:
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_true, y_pred)
            
            # Create visualization
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Log to W&B
            wandb.log({"confusion_matrix": wandb.Image(plt)})
            plt.close()
            
            return True
        except Exception as e:
            console.print(f"[red]Error logging confusion matrix: {e}[/red]")
            return False
    
    def log_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray) -> bool:
        """Log feature importance visualization"""
        if not self.available or not self.current_run:
            return False
        
        try:
            # Create DataFrame
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df.head(20), x='importance', y='feature')
            plt.title('Top 20 Feature Importance')
            plt.xlabel('Importance Score')
            
            # Log to W&B
            wandb.log({"feature_importance": wandb.Image(plt)})
            plt.close()
            
            # Also log as table
            self.log_table(df, "feature_importance_table")
            
            return True
        except Exception as e:
            console.print(f"[red]Error logging feature importance: {e}[/red]")
            return False
    
    def log_learning_curve(self, train_scores: List[float], val_scores: List[float], 
                          metric_name: str = "accuracy") -> bool:
        """Log learning curve visualization"""
        if not self.available or not self.current_run:
            return False
        
        try:
            epochs = list(range(1, len(train_scores) + 1))
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_scores, label=f'Training {metric_name}', marker='o')
            plt.plot(epochs, val_scores, label=f'Validation {metric_name}', marker='s')
            plt.title(f'Learning Curve - {metric_name.title()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name.title())
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Log to W&B
            wandb.log({f"learning_curve_{metric_name}": wandb.Image(plt)})
            plt.close()
            
            return True
        except Exception as e:
            console.print(f"[red]Error logging learning curve: {e}[/red]")
            return False
    
    def log_system_performance(self) -> bool:
        """Log current system performance metrics"""
        if not self.available or not self.current_run:
            return False
        
        try:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'system/cpu_percent': cpu_percent,
                'system/memory_percent': memory.percent,
                'system/memory_available_gb': memory.available / (1024**3),
                'system/disk_percent': disk.percent,
                'system/disk_free_gb': disk.free / (1024**3)
            }
            
            # Add GPU metrics if available
            if torch.cuda.is_available():
                metrics['system/gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)
                metrics['system/gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024**3)
            
            return self.log_metrics(metrics)
            
        except Exception as e:
            console.print(f"[red]Error logging system performance: {e}[/red]")
            return False
    
    def finish_experiment(self, summary_metrics: Optional[Dict] = None) -> bool:
        """Finish current experiment"""
        if not self.available or not self.current_run:
            return False
        
        try:
            # Log final summary metrics
            if summary_metrics:
                for key, value in summary_metrics.items():
                    wandb.run.summary[key] = value
            
            # Log experiment to history
            self.experiment_history.append({
                'run_id': self.current_run.id,
                'name': self.current_run.name,
                'url': self.current_run.url,
                'finished_at': datetime.now().isoformat()
            })
            
            wandb.finish()
            console.print(f"[green]✓[/green] Experiment finished: {self.current_run.name}")
            
            self.current_run = None
            return True
            
        except Exception as e:
            console.print(f"[red]Error finishing experiment: {e}[/red]")
            return False
    
    def _log_system_info(self) -> bool:
        """Log system information at experiment start"""
        try:
            system_info = {
                'system/python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                'system/platform': os.sys.platform,
                'system/cpu_count': psutil.cpu_count(),
                'system/memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'system/gpu_available': torch.cuda.is_available()
            }
            
            if torch.cuda.is_available():
                system_info['system/gpu_name'] = torch.cuda.get_device_name(0)
                system_info['system/gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            return self.log_metrics(system_info)
            
        except Exception as e:
            console.print(f"[red]Error logging system info: {e}[/red]")
            return False
    
    def create_fashion_experiment_config(self, model_name: str, dataset_name: str, 
                                       hyperparams: Dict[str, Any]) -> ExperimentConfig:
        """Create experiment config for fashion AI models"""
        return ExperimentConfig(
            project_name=self.project,
            experiment_name=f"{model_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=["fashion-ai", model_name, dataset_name, "flashfit"],
            notes=f"Training {model_name} on {dataset_name} dataset for FlashFit AI",
            config={
                'model_name': model_name,
                'dataset': dataset_name,
                'framework': 'pytorch',
                'task': 'fashion_recommendation',
                **hyperparams
            }
        )
    
    def get_experiment_history(self) -> List[Dict]:
        """Get history of experiments"""
        return self.experiment_history
    
    def get_best_run(self, metric: str = "accuracy", project: Optional[str] = None) -> Optional[Dict]:
        """Get best run based on metric"""
        if not self.available:
            return None
        
        try:
            api = wandb.Api()
            runs = api.runs(project or self.project, filters={"state": "finished"})
            
            best_run = None
            best_score = float('-inf')
            
            for run in runs:
                if metric in run.summary:
                    score = run.summary[metric]
                    if score > best_score:
                        best_score = score
                        best_run = {
                            'id': run.id,
                            'name': run.name,
                            'url': run.url,
                            'score': score,
                            'config': dict(run.config),
                            'summary': dict(run.summary)
                        }
            
            return best_run
            
        except Exception as e:
            console.print(f"[red]Error getting best run: {e}[/red]")
            return None

# Context manager for W&B experiments
class WandBExperiment:
    """Context manager for W&B experiments"""
    
    def __init__(self, wandb_integration: WandBIntegration, config: ExperimentConfig):
        self.wandb_integration = wandb_integration
        self.config = config
        self.run = None
    
    def __enter__(self):
        self.run = self.wandb_integration.start_experiment(self.config)
        return self.wandb_integration
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            console.print(f"[red]Experiment failed with error: {exc_val}[/red]")
        
        self.wandb_integration.finish_experiment()

# Global instance
_wandb_integration = None

def get_wandb_integration(api_key: Optional[str] = None, entity: Optional[str] = None, 
                         project: str = "flashfit-ai", offline: bool = False) -> WandBIntegration:
    """Get or create W&B integration instance"""
    global _wandb_integration
    if _wandb_integration is None:
        _wandb_integration = WandBIntegration(api_key, entity, project, offline)
    return _wandb_integration

# Example training function for sweep
def example_train_function():
    """Example training function for W&B sweep"""
    # Initialize W&B run (this will be done automatically by sweep agent)
    config = wandb.config
    
    # Mock training loop
    for epoch in range(config.epochs):
        # Simulate training
        train_loss = np.random.exponential(1.0) * np.exp(-epoch * 0.1)
        val_loss = train_loss + np.random.normal(0, 0.1)
        accuracy = 1 - val_loss + np.random.normal(0, 0.05)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'learning_rate': config.learning_rate
        })
    
    # Log final metrics
    wandb.log({'final_accuracy': accuracy})

if __name__ == "__main__":
    # Demo usage
    console.print("[bold blue]W&B Integration Demo[/bold blue]")
    
    # Initialize W&B integration
    wandb_int = get_wandb_integration(offline=True)  # Use offline mode for demo
    
    if wandb_int.available:
        # Create experiment config
        config = wandb_int.create_fashion_experiment_config(
            model_name="fashion_encoder",
            dataset_name="fashion_dataset",
            hyperparams={
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 10,
                'optimizer': 'adam'
            }
        )
        
        # Run experiment
        with WandBExperiment(wandb_int, config) as wb:
            console.print("[green]Running demo experiment...[/green]")
            
            # Simulate training loop
            for epoch in range(5):
                # Mock metrics
                metrics = ModelMetrics(
                    accuracy=0.8 + epoch * 0.02 + np.random.normal(0, 0.01),
                    precision=0.75 + epoch * 0.03 + np.random.normal(0, 0.01),
                    recall=0.78 + epoch * 0.025 + np.random.normal(0, 0.01),
                    f1_score=0.76 + epoch * 0.028 + np.random.normal(0, 0.01),
                    loss=1.0 - epoch * 0.15 + np.random.normal(0, 0.05),
                    epoch=epoch,
                    learning_rate=0.001,
                    batch_size=32
                )
                
                wb.log_model_metrics(metrics, step=epoch)
                wb.log_system_performance()
            
            console.print("[green]✓ Demo experiment completed[/green]")
    
    else:
        console.print("[yellow]W&B not available, skipping demo[/yellow]")