from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from peft import AdaLoraConfig, PrefixTuningConfig, PromptTuningConfig, IA3Config
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import json
import ujson
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import warnings
warnings.filterwarnings('ignore')

# Initialize console for logging
console = Console()

@dataclass
class PEFTConfig:
    """Configuration for PEFT methods"""
    method: str  # 'lora', 'adalora', 'prefix', 'prompt', 'ia3'
    task_type: str  # 'SEQ_CLS', 'CAUSAL_LM', 'SEQ_2_SEQ_LM', etc.
    target_modules: List[str]
    r: int = 8  # Rank for LoRA
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"  # 'none', 'all', 'lora_only'
    
    # AdaLoRA specific
    target_r: int = 8
    init_r: int = 12
    tinit: int = 0
    tfinal: int = 1000
    deltaT: int = 10
    
    # Prefix tuning specific
    num_virtual_tokens: int = 20
    
    # Prompt tuning specific
    num_transformer_submodules: int = 1
    
    def to_peft_config(self):
        """Convert to appropriate PEFT config object"""
        if self.method.lower() == 'lora':
            return LoraConfig(
                task_type=getattr(TaskType, self.task_type),
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias=self.bias,
                target_modules=self.target_modules
            )
        elif self.method.lower() == 'adalora':
            return AdaLoraConfig(
                task_type=getattr(TaskType, self.task_type),
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                target_r=self.target_r,
                init_r=self.init_r,
                tinit=self.tinit,
                tfinal=self.tfinal,
                deltaT=self.deltaT
            )
        elif self.method.lower() == 'prefix':
            return PrefixTuningConfig(
                task_type=getattr(TaskType, self.task_type),
                num_virtual_tokens=self.num_virtual_tokens
            )
        elif self.method.lower() == 'prompt':
            return PromptTuningConfig(
                task_type=getattr(TaskType, self.task_type),
                num_virtual_tokens=self.num_virtual_tokens,
                num_transformer_submodules=self.num_transformer_submodules
            )
        elif self.method.lower() == 'ia3':
            return IA3Config(
                task_type=getattr(TaskType, self.task_type),
                target_modules=self.target_modules
            )
        else:
            raise ValueError(f"Unsupported PEFT method: {self.method}")

@dataclass
class OptimizationResults:
    """Results from PEFT optimization"""
    method: str
    original_params: int
    trainable_params: int
    param_reduction: float
    memory_usage_mb: float
    training_time_seconds: float
    final_accuracy: float
    final_loss: float
    convergence_epoch: int
    model_size_mb: float
    
    def efficiency_score(self) -> float:
        """Calculate efficiency score based on accuracy and parameter reduction"""
        return (self.final_accuracy * self.param_reduction) / (self.memory_usage_mb / 1000)

class FashionDataset(Dataset):
    """Custom dataset for fashion text classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class PEFTOptimizer:
    """
    Parameter Efficient Fine-Tuning (PEFT) optimizer for FlashFit AI
    
    Features:
    - LoRA (Low-Rank Adaptation)
    - AdaLoRA (Adaptive LoRA)
    - Prefix Tuning
    - Prompt Tuning
    - IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
    - Automatic hyperparameter optimization
    - Memory and compute efficiency analysis
    - Model compression and deployment optimization
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 cache_dir: str = "models/cache", device: str = "auto"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        console.print(f"[green]✓[/green] PEFT Optimizer initialized on {self.device}")
        
        # Initialize tokenizer and base model
        self.tokenizer = None
        self.base_model = None
        self.peft_models = {}
        self.optimization_history = []
        
        self._load_base_model()
    
    def _load_base_model(self):
        """Load base model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=10,  # Fashion categories
                cache_dir=self.cache_dir
            )
            
            console.print(f"[green]✓[/green] Loaded base model: {self.model_name}")
            
        except Exception as e:
            console.print(f"[red]Error loading base model: {e}[/red]")
            raise
    
    def create_peft_model(self, peft_config: PEFTConfig, model_id: str) -> nn.Module:
        """Create PEFT model with specified configuration"""
        try:
            # Convert config to PEFT config object
            config = peft_config.to_peft_config()
            
            # Create PEFT model
            peft_model = get_peft_model(self.base_model, config)
            peft_model.to(self.device)
            
            # Store model
            self.peft_models[model_id] = {
                'model': peft_model,
                'config': peft_config,
                'created_at': datetime.now().isoformat()
            }
            
            # Print model info
            peft_model.print_trainable_parameters()
            
            console.print(f"[green]✓[/green] Created PEFT model '{model_id}' using {peft_config.method}")
            
            return peft_model
            
        except Exception as e:
            console.print(f"[red]Error creating PEFT model: {e}[/red]")
            raise
    
    def optimize_model(self, peft_config: PEFTConfig, train_data: Dict[str, List], 
                      val_data: Dict[str, List], training_args: Dict[str, Any],
                      model_id: str = None) -> OptimizationResults:
        """Optimize model using PEFT method"""
        if model_id is None:
            model_id = f"{peft_config.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        console.print(f"[blue]🚀[/blue] Starting PEFT optimization: {peft_config.method}")
        
        start_time = datetime.now()
        
        try:
            # Create PEFT model
            peft_model = self.create_peft_model(peft_config, model_id)
            
            # Get parameter counts
            total_params = sum(p.numel() for p in peft_model.parameters())
            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            param_reduction = 1 - (trainable_params / total_params)
            
            # Create datasets
            train_dataset = FashionDataset(
                train_data['texts'], train_data['labels'], self.tokenizer
            )
            val_dataset = FashionDataset(
                val_data['texts'], val_data['labels'], self.tokenizer
            )
            
            # Setup training arguments
            training_arguments = TrainingArguments(
                output_dir=f"./results/{model_id}",
                num_train_epochs=training_args.get('epochs', 3),
                per_device_train_batch_size=training_args.get('batch_size', 8),
                per_device_eval_batch_size=training_args.get('batch_size', 8),
                learning_rate=training_args.get('learning_rate', 2e-4),
                weight_decay=training_args.get('weight_decay', 0.01),
                logging_dir=f'./logs/{model_id}',
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                report_to=None  # Disable wandb for now
            )
            
            # Create trainer
            trainer = Trainer(
                model=peft_model,
                args=training_arguments,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
                compute_metrics=self._compute_metrics
            )
            
            # Measure memory before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                memory_before = 0
            
            # Train model
            console.print("[yellow]📚[/yellow] Starting training...")
            train_result = trainer.train()
            
            # Measure memory after training
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_usage = memory_after - memory_before
            else:
                memory_usage = 0
            
            # Evaluate model
            eval_result = trainer.evaluate()
            
            # Calculate training time
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Get model size
            model_size = self._get_model_size(peft_model)
            
            # Create results
            results = OptimizationResults(
                method=peft_config.method,
                original_params=total_params,
                trainable_params=trainable_params,
                param_reduction=param_reduction,
                memory_usage_mb=memory_usage,
                training_time_seconds=training_time,
                final_accuracy=eval_result.get('eval_accuracy', 0.0),
                final_loss=eval_result.get('eval_loss', float('inf')),
                convergence_epoch=len(train_result.log_history),
                model_size_mb=model_size
            )
            
            # Store optimization history
            self.optimization_history.append({
                'model_id': model_id,
                'results': asdict(results),
                'config': asdict(peft_config),
                'timestamp': datetime.now().isoformat()
            })
            
            console.print(f"[green]✅[/green] Optimization completed: {peft_config.method}")
            console.print(f"[cyan]📊[/cyan] Trainable params: {trainable_params:,} ({param_reduction:.2%} reduction)")
            console.print(f"[cyan]📊[/cyan] Final accuracy: {results.final_accuracy:.4f}")
            console.print(f"[cyan]📊[/cyan] Training time: {training_time:.2f}s")
            
            return results
            
        except Exception as e:
            console.print(f"[red]Error during optimization: {e}[/red]")
            raise
    
    def compare_peft_methods(self, train_data: Dict[str, List], val_data: Dict[str, List],
                           training_args: Dict[str, Any], methods: List[str] = None) -> Dict[str, OptimizationResults]:
        """Compare different PEFT methods"""
        if methods is None:
            methods = ['lora', 'adalora', 'prefix', 'ia3']
        
        console.print(f"[bold blue]🔬 Comparing PEFT methods: {', '.join(methods)}[/bold blue]")
        
        results = {}
        
        # Define default configurations for each method
        default_configs = {
            'lora': PEFTConfig(
                method='lora',
                task_type='SEQ_CLS',
                target_modules=['query', 'value'],
                r=8,
                lora_alpha=32,
                lora_dropout=0.1
            ),
            'adalora': PEFTConfig(
                method='adalora',
                task_type='SEQ_CLS',
                target_modules=['query', 'value'],
                r=8,
                lora_alpha=32,
                target_r=4,
                init_r=12
            ),
            'prefix': PEFTConfig(
                method='prefix',
                task_type='SEQ_CLS',
                target_modules=[],
                num_virtual_tokens=20
            ),
            'ia3': PEFTConfig(
                method='ia3',
                task_type='SEQ_CLS',
                target_modules=['key', 'value', 'down_proj']
            )
        }
        
        for method in methods:
            if method in default_configs:
                try:
                    console.print(f"\n[yellow]🔧[/yellow] Testing {method.upper()}...")
                    config = default_configs[method]
                    result = self.optimize_model(config, train_data, val_data, training_args)
                    results[method] = result
                except Exception as e:
                    console.print(f"[red]Failed to test {method}: {e}[/red]")
                    continue
        
        # Display comparison
        self._display_comparison_results(results)
        
        return results
    
    def find_optimal_lora_config(self, train_data: Dict[str, List], val_data: Dict[str, List],
                                training_args: Dict[str, Any]) -> Tuple[PEFTConfig, OptimizationResults]:
        """Find optimal LoRA configuration through hyperparameter search"""
        console.print("[bold blue]🎯 Searching for optimal LoRA configuration[/bold blue]")
        
        # Define search space
        r_values = [4, 8, 16, 32]
        alpha_values = [16, 32, 64]
        dropout_values = [0.05, 0.1, 0.2]
        
        best_config = None
        best_results = None
        best_score = 0
        
        total_combinations = len(r_values) * len(alpha_values) * len(dropout_values)
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn()
        ) as progress:
            
            task = progress.add_task("Optimizing LoRA...", total=total_combinations)
            
            for r in r_values:
                for alpha in alpha_values:
                    for dropout in dropout_values:
                        try:
                            config = PEFTConfig(
                                method='lora',
                                task_type='SEQ_CLS',
                                target_modules=['query', 'value'],
                                r=r,
                                lora_alpha=alpha,
                                lora_dropout=dropout
                            )
                            
                            results = self.optimize_model(
                                config, train_data, val_data, training_args,
                                model_id=f"lora_r{r}_a{alpha}_d{dropout}"
                            )
                            
                            # Calculate composite score (accuracy + efficiency)
                            score = results.efficiency_score()
                            
                            if score > best_score:
                                best_score = score
                                best_config = config
                                best_results = results
                            
                            progress.advance(task)
                            
                        except Exception as e:
                            console.print(f"[red]Failed config r={r}, α={alpha}, dropout={dropout}: {e}[/red]")
                            progress.advance(task)
                            continue
        
        console.print(f"[green]✅[/green] Best LoRA config found:")
        console.print(f"[cyan]  • r={best_config.r}, α={best_config.lora_alpha}, dropout={best_config.lora_dropout}[/cyan]")
        console.print(f"[cyan]  • Efficiency score: {best_score:.4f}[/cyan]")
        
        return best_config, best_results
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def _display_comparison_results(self, results: Dict[str, OptimizationResults]):
        """Display comparison results in a table"""
        table = Table(title="PEFT Methods Comparison")
        
        table.add_column("Method", style="cyan")
        table.add_column("Accuracy", style="green")
        table.add_column("Param Reduction", style="yellow")
        table.add_column("Memory (MB)", style="blue")
        table.add_column("Time (s)", style="magenta")
        table.add_column("Efficiency Score", style="red")
        
        for method, result in results.items():
            table.add_row(
                method.upper(),
                f"{result.final_accuracy:.4f}",
                f"{result.param_reduction:.2%}",
                f"{result.memory_usage_mb:.1f}",
                f"{result.training_time_seconds:.1f}",
                f"{result.efficiency_score():.4f}"
            )
        
        console.print(table)
    
    def save_optimized_model(self, model_id: str, save_path: str) -> bool:
        """Save optimized PEFT model"""
        if model_id not in self.peft_models:
            console.print(f"[red]Model '{model_id}' not found[/red]")
            return False
        
        try:
            model_info = self.peft_models[model_id]
            model = model_info['model']
            
            # Save PEFT model
            model.save_pretrained(save_path)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(save_path)
            
            # Save metadata
            metadata = {
                'model_id': model_id,
                'base_model': self.model_name,
                'peft_config': asdict(model_info['config']),
                'created_at': model_info['created_at'],
                'saved_at': datetime.now().isoformat()
            }
            
            with open(Path(save_path) / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            console.print(f"[green]✅[/green] Model saved to: {save_path}")
            return True
            
        except Exception as e:
            console.print(f"[red]Error saving model: {e}[/red]")
            return False
    
    def load_optimized_model(self, model_path: str) -> Optional[nn.Module]:
        """Load optimized PEFT model"""
        try:
            # Load metadata
            with open(Path(model_path) / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            # Load PEFT model
            model = PeftModel.from_pretrained(self.base_model, model_path)
            
            console.print(f"[green]✅[/green] Loaded PEFT model from: {model_path}")
            return model
            
        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            return None
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        if not self.optimization_history:
            return {'message': 'No optimization history available'}
        
        report = {
            'summary': {
                'total_experiments': len(self.optimization_history),
                'methods_tested': list(set(exp['results']['method'] for exp in self.optimization_history)),
                'best_accuracy': max(exp['results']['final_accuracy'] for exp in self.optimization_history),
                'best_efficiency': max(exp['results']['param_reduction'] for exp in self.optimization_history)
            },
            'experiments': self.optimization_history,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on history"""
        if not self.optimization_history:
            return []
        
        recommendations = []
        
        # Find best performing method
        best_exp = max(self.optimization_history, key=lambda x: x['results']['final_accuracy'])
        recommendations.append(f"Best performing method: {best_exp['results']['method']} (accuracy: {best_exp['results']['final_accuracy']:.4f})")
        
        # Find most efficient method
        most_efficient = max(self.optimization_history, key=lambda x: x['results']['param_reduction'])
        recommendations.append(f"Most parameter efficient: {most_efficient['results']['method']} ({most_efficient['results']['param_reduction']:.2%} reduction)")
        
        # Memory recommendations
        avg_memory = np.mean([exp['results']['memory_usage_mb'] for exp in self.optimization_history])
        if avg_memory > 1000:
            recommendations.append("Consider using smaller batch sizes or gradient checkpointing for memory optimization")
        
        return recommendations

# Factory function
def create_peft_optimizer(model_name: str = "distilbert-base-uncased", 
                         cache_dir: str = "models/cache") -> PEFTOptimizer:
    """Create PEFT optimizer instance"""
    return PEFTOptimizer(model_name, cache_dir)

# Example usage function
def generate_sample_fashion_data(n_samples: int = 1000) -> Tuple[Dict[str, List], Dict[str, List]]:
    """Generate sample fashion data for testing"""
    categories = ['tops', 'bottoms', 'dresses', 'shoes', 'accessories', 'outerwear', 'activewear', 'formal', 'casual', 'vintage']
    
    # Generate sample texts and labels
    texts = []
    labels = []
    
    for i in range(n_samples):
        category = np.random.choice(categories)
        label = categories.index(category)
        
        # Generate sample fashion description
        colors = ['black', 'white', 'blue', 'red', 'green', 'yellow', 'pink', 'purple']
        materials = ['cotton', 'silk', 'denim', 'leather', 'wool', 'polyester', 'linen']
        styles = ['casual', 'formal', 'vintage', 'modern', 'classic', 'trendy']
        
        color = np.random.choice(colors)
        material = np.random.choice(materials)
        style = np.random.choice(styles)
        
        text = f"{style} {color} {material} {category} with comfortable fit and modern design"
        
        texts.append(text)
        labels.append(label)
    
    # Split into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_data = {'texts': train_texts, 'labels': train_labels}
    val_data = {'texts': val_texts, 'labels': val_labels}
    
    return train_data, val_data

if __name__ == "__main__":
    # Demo usage
    console.print("[bold blue]PEFT Optimizer Demo[/bold blue]")
    
    try:
        # Create optimizer
        optimizer = create_peft_optimizer()
        
        # Generate sample data
        console.print("[yellow]📊[/yellow] Generating sample fashion data...")
        train_data, val_data = generate_sample_fashion_data(500)  # Smaller dataset for demo
        
        # Training arguments
        training_args = {
            'epochs': 2,  # Reduced for demo
            'batch_size': 4,  # Smaller batch size
            'learning_rate': 2e-4,
            'weight_decay': 0.01
        }
        
        # Test LoRA optimization
        lora_config = PEFTConfig(
            method='lora',
            task_type='SEQ_CLS',
            target_modules=['query', 'value'],
            r=4,  # Smaller rank for demo
            lora_alpha=16,
            lora_dropout=0.1
        )
        
        console.print("\n[yellow]🔧[/yellow] Testing LoRA optimization...")
        results = optimizer.optimize_model(lora_config, train_data, val_data, training_args)
        
        # Generate report
        report = optimizer.generate_optimization_report()
        console.print("\n[green]📋[/green] Optimization Report:")
        console.print(Panel(json.dumps(report['summary'], indent=2), title="Summary"))
        
        console.print("\n[green]✅[/green] PEFT optimization demo completed!")
        
    except Exception as e:
        console.print(f"[red]Demo failed: {e}[/red]")
        console.print("[yellow]Note: This demo requires transformers and peft libraries[/yellow]")