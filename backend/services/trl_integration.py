from trl import (
    PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead,
    create_reference_model, DPOTrainer, DPOConfig,
    SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
)
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    BitsAndBytesConfig, pipeline
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
import json
import ujson
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset as HFDataset
import warnings
warnings.filterwarnings('ignore')

# Initialize console for logging
console = Console()

@dataclass
class RLHFConfig:
    """Configuration for Reinforcement Learning from Human Feedback"""
    model_name: str
    reward_model_name: str = None
    learning_rate: float = 1.41e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95
    whiten_rewards: bool = False
    kl_penalty: str = "kl"  # "kl" or "abs" or "mse" or "full"
    adap_kl_ctrl: bool = True
    init_kl_coef: float = 0.2
    use_score_scaling: bool = False
    use_score_norm: bool = False
    score_clip: Optional[float] = None

@dataclass
class DPOTrainingConfig:
    """Configuration for Direct Preference Optimization"""
    model_name: str
    beta: float = 0.1
    learning_rate: float = 5e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 150
    max_steps: int = 1000
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    dataloader_drop_last: bool = True
    eval_steps: int = 100
    save_steps: int = 1000
    logging_steps: int = 10
    remove_unused_columns: bool = False
    label_names: List[str] = None
    max_length: int = 512
    max_prompt_length: int = 256
    max_target_length: int = 256
    sanity_check: bool = False
    report_to: str = "none"
    
    def __post_init__(self):
        if self.label_names is None:
            self.label_names = []

@dataclass
class SFTTrainingConfig:
    """Configuration for Supervised Fine-Tuning"""
    model_name: str
    dataset_text_field: str = "text"
    max_seq_length: int = 512
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./sft_results"
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "constant"
    report_to: str = "none"
    packing: bool = False
    use_peft: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

class FashionRewardModel:
    """Reward model for fashion recommendations"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def compute_reward(self, query: str, response: str) -> float:
        """Compute reward for fashion recommendation response"""
        # Fashion-specific reward criteria
        reward = 0.0
        
        # Check for fashion keywords
        fashion_keywords = [
            'style', 'color', 'fit', 'material', 'brand', 'size', 'occasion',
            'trendy', 'classic', 'comfortable', 'elegant', 'casual', 'formal'
        ]
        
        response_lower = response.lower()
        keyword_count = sum(1 for keyword in fashion_keywords if keyword in response_lower)
        reward += keyword_count * 0.1
        
        # Check response length (not too short, not too long)
        response_length = len(response.split())
        if 10 <= response_length <= 100:
            reward += 0.5
        elif response_length < 5:
            reward -= 0.3
        elif response_length > 150:
            reward -= 0.2
        
        # Check for specific fashion advice
        advice_patterns = [
            'recommend', 'suggest', 'would look', 'pairs well', 'complements',
            'suitable for', 'perfect for', 'ideal choice'
        ]
        
        advice_count = sum(1 for pattern in advice_patterns if pattern in response_lower)
        reward += advice_count * 0.2
        
        # Penalize generic responses
        generic_phrases = ['i think', 'maybe', 'perhaps', 'not sure']
        generic_count = sum(1 for phrase in generic_phrases if phrase in response_lower)
        reward -= generic_count * 0.1
        
        # Ensure reward is in reasonable range
        return max(0.0, min(1.0, reward))
    
    def batch_compute_rewards(self, queries: List[str], responses: List[str]) -> List[float]:
        """Compute rewards for batch of query-response pairs"""
        return [self.compute_reward(q, r) for q, r in zip(queries, responses)]

class FashionDataset(Dataset):
    """Dataset for fashion text generation"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class TRLIntegration:
    """
    Transformer Reinforcement Learning (TRL) integration for FlashFit AI
    
    Features:
    - Proximal Policy Optimization (PPO) for RLHF
    - Direct Preference Optimization (DPO)
    - Supervised Fine-Tuning (SFT)
    - Custom reward models for fashion recommendations
    - Advanced training strategies
    - Model evaluation and comparison
    """
    
    def __init__(self, cache_dir: str = "models/cache", device: str = "auto"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        console.print(f"[green]✓[/green] TRL Integration initialized on {self.device}")
        
        # Initialize components
        self.models = {}
        self.trainers = {}
        self.reward_model = None
        self.training_history = []
        
        # Initialize reward model
        self._initialize_reward_model()
    
    def _initialize_reward_model(self):
        """Initialize fashion-specific reward model"""
        try:
            self.reward_model = FashionRewardModel()
            console.print("[green]✓[/green] Fashion reward model initialized")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize reward model: {e}[/yellow]")
    
    def setup_sft_training(self, config: SFTTrainingConfig, 
                          train_dataset: List[str], eval_dataset: List[str] = None) -> SFTTrainer:
        """Setup Supervised Fine-Tuning"""
        console.print(f"[blue]🚀[/blue] Setting up SFT training for {config.model_name}")
        
        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=self.cache_dir)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Setup PEFT if enabled
            if config.use_peft:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=config.target_modules,
                    bias="none"
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            
            # Prepare datasets
            train_data = [{config.dataset_text_field: text} for text in train_dataset]
            train_hf_dataset = HFDataset.from_list(train_data)
            
            eval_hf_dataset = None
            if eval_dataset:
                eval_data = [{config.dataset_text_field: text} for text in eval_dataset]
                eval_hf_dataset = HFDataset.from_list(eval_data)
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=config.output_dir,
                num_train_epochs=config.num_train_epochs,
                per_device_train_batch_size=config.per_device_train_batch_size,
                per_device_eval_batch_size=config.per_device_eval_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                gradient_checkpointing=config.gradient_checkpointing,
                learning_rate=config.learning_rate,
                lr_scheduler_type=config.lr_scheduler_type,
                warmup_steps=config.warmup_steps,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                eval_steps=config.eval_steps,
                evaluation_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                optim=config.optim,
                report_to=config.report_to,
                remove_unused_columns=False,
                dataloader_drop_last=True,
                bf16=torch.cuda.is_available(),
            )
            
            # Create SFT trainer
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_hf_dataset,
                eval_dataset=eval_hf_dataset,
                dataset_text_field=config.dataset_text_field,
                max_seq_length=config.max_seq_length,
                packing=config.packing,
            )
            
            # Store trainer
            trainer_id = f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.trainers[trainer_id] = {
                'trainer': trainer,
                'type': 'sft',
                'config': config,
                'created_at': datetime.now().isoformat()
            }
            
            console.print(f"[green]✅[/green] SFT trainer setup completed: {trainer_id}")
            return trainer
            
        except Exception as e:
            console.print(f"[red]Error setting up SFT training: {e}[/red]")
            raise
    
    def setup_ppo_training(self, config: RLHFConfig, 
                          model_name: str, tokenizer_name: str = None) -> PPOTrainer:
        """Setup PPO training for RLHF"""
        console.print(f"[blue]🚀[/blue] Setting up PPO training for {model_name}")
        
        try:
            if tokenizer_name is None:
                tokenizer_name = model_name
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=self.cache_dir)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with value head
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Create reference model
            ref_model = create_reference_model(model)
            
            # Setup PPO config
            ppo_config = PPOConfig(
                model_name=model_name,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                mini_batch_size=config.mini_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                ppo_epochs=config.ppo_epochs,
                max_grad_norm=config.max_grad_norm,
                target_kl=config.target_kl,
                cliprange=config.cliprange,
                cliprange_value=config.cliprange_value,
                vf_coef=config.vf_coef,
                gamma=config.gamma,
                lam=config.lam,
                whiten_rewards=config.whiten_rewards,
                kl_penalty=config.kl_penalty,
                adap_kl_ctrl=config.adap_kl_ctrl,
                init_kl_coef=config.init_kl_coef,
                use_score_scaling=config.use_score_scaling,
                use_score_norm=config.use_score_norm,
                score_clip=config.score_clip
            )
            
            # Create PPO trainer
            ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer
            )
            
            # Store trainer
            trainer_id = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.trainers[trainer_id] = {
                'trainer': ppo_trainer,
                'type': 'ppo',
                'config': config,
                'created_at': datetime.now().isoformat()
            }
            
            console.print(f"[green]✅[/green] PPO trainer setup completed: {trainer_id}")
            return ppo_trainer
            
        except Exception as e:
            console.print(f"[red]Error setting up PPO training: {e}[/red]")
            raise
    
    def setup_dpo_training(self, config: DPOTrainingConfig, 
                          train_dataset: List[Dict], eval_dataset: List[Dict] = None) -> DPOTrainer:
        """Setup Direct Preference Optimization training"""
        console.print(f"[blue]🚀[/blue] Setting up DPO training for {config.model_name}")
        
        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=self.cache_dir)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Prepare datasets
            train_hf_dataset = HFDataset.from_list(train_dataset)
            eval_hf_dataset = HFDataset.from_list(eval_dataset) if eval_dataset else None
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir="./dpo_results",
                learning_rate=config.learning_rate,
                lr_scheduler_type=config.lr_scheduler_type,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                per_device_train_batch_size=config.per_device_train_batch_size,
                per_device_eval_batch_size=config.per_device_eval_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                gradient_checkpointing=config.gradient_checkpointing,
                dataloader_drop_last=config.dataloader_drop_last,
                eval_steps=config.eval_steps,
                save_steps=config.save_steps,
                logging_steps=config.logging_steps,
                remove_unused_columns=config.remove_unused_columns,
                label_names=config.label_names,
                bf16=torch.cuda.is_available(),
                report_to=config.report_to,
                evaluation_strategy="steps" if eval_dataset else "no",
                save_strategy="steps"
            )
            
            # Create DPO trainer
            dpo_trainer = DPOTrainer(
                model=model,
                ref_model=None,  # Will create reference model automatically
                args=training_args,
                beta=config.beta,
                train_dataset=train_hf_dataset,
                eval_dataset=eval_hf_dataset,
                tokenizer=tokenizer,
                max_length=config.max_length,
                max_prompt_length=config.max_prompt_length,
                max_target_length=config.max_target_length,
                sanity_check=config.sanity_check
            )
            
            # Store trainer
            trainer_id = f"dpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.trainers[trainer_id] = {
                'trainer': dpo_trainer,
                'type': 'dpo',
                'config': config,
                'created_at': datetime.now().isoformat()
            }
            
            console.print(f"[green]✅[/green] DPO trainer setup completed: {trainer_id}")
            return dpo_trainer
            
        except Exception as e:
            console.print(f"[red]Error setting up DPO training: {e}[/red]")
            raise
    
    def train_with_ppo(self, trainer_id: str, queries: List[str], 
                      max_new_tokens: int = 50, num_steps: int = 100) -> Dict[str, Any]:
        """Train model using PPO with fashion-specific rewards"""
        if trainer_id not in self.trainers or self.trainers[trainer_id]['type'] != 'ppo':
            raise ValueError(f"PPO trainer '{trainer_id}' not found")
        
        console.print(f"[yellow]🎯[/yellow] Starting PPO training: {trainer_id}")
        
        trainer_info = self.trainers[trainer_id]
        ppo_trainer = trainer_info['trainer']
        
        training_stats = {
            'rewards': [],
            'kl_divergences': [],
            'losses': [],
            'step_times': []
        }
        
        try:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn()
            ) as progress:
                
                task = progress.add_task("PPO Training...", total=num_steps)
                
                for step in range(num_steps):
                    step_start = datetime.now()
                    
                    # Sample batch of queries
                    batch_queries = np.random.choice(queries, size=min(len(queries), 4), replace=False).tolist()
                    
                    # Generate responses
                    query_tensors = [ppo_trainer.tokenizer.encode(query, return_tensors="pt")[0] for query in batch_queries]
                    
                    response_tensors = ppo_trainer.generate(
                        query_tensors,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.7
                    )
                    
                    # Decode responses
                    responses = [ppo_trainer.tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
                    
                    # Compute rewards using fashion reward model
                    if self.reward_model:
                        rewards = [torch.tensor(self.reward_model.compute_reward(q, r)) for q, r in zip(batch_queries, responses)]
                    else:
                        # Fallback to simple length-based reward
                        rewards = [torch.tensor(min(1.0, len(r.split()) / 20.0)) for r in responses]
                    
                    # PPO step
                    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                    
                    # Record statistics
                    if stats:
                        training_stats['rewards'].append(np.mean([r.item() for r in rewards]))
                        training_stats['kl_divergences'].append(stats.get('objective/kl', 0))
                        training_stats['losses'].append(stats.get('ppo/loss/total', 0))
                    
                    step_time = (datetime.now() - step_start).total_seconds()
                    training_stats['step_times'].append(step_time)
                    
                    progress.advance(task)
                    
                    # Log progress every 10 steps
                    if step % 10 == 0 and stats:
                        console.print(f"Step {step}: Reward={np.mean([r.item() for r in rewards]):.3f}, KL={stats.get('objective/kl', 0):.3f}")
            
            # Store training history
            self.training_history.append({
                'trainer_id': trainer_id,
                'type': 'ppo',
                'stats': training_stats,
                'completed_at': datetime.now().isoformat()
            })
            
            console.print(f"[green]✅[/green] PPO training completed: {trainer_id}")
            return training_stats
            
        except Exception as e:
            console.print(f"[red]Error during PPO training: {e}[/red]")
            raise
    
    def generate_fashion_recommendations(self, model_name: str, queries: List[str], 
                                       max_new_tokens: int = 100) -> List[str]:
        """Generate fashion recommendations using trained model"""
        console.print(f"[blue]💡[/blue] Generating fashion recommendations...")
        
        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=self.cache_dir)
            
            # Create generation pipeline
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            recommendations = []
            
            for query in queries:
                # Generate response
                response = generator(
                    query,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )[0]['generated_text']
                
                # Extract only the new part (remove input query)
                recommendation = response[len(query):].strip()
                recommendations.append(recommendation)
            
            console.print(f"[green]✅[/green] Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            console.print(f"[red]Error generating recommendations: {e}[/red]")
            raise
    
    def evaluate_model_performance(self, model_name: str, test_queries: List[str], 
                                 ground_truth: List[str] = None) -> Dict[str, Any]:
        """Evaluate model performance on fashion recommendation task"""
        console.print(f"[blue]📊[/blue] Evaluating model performance...")
        
        try:
            # Generate recommendations
            recommendations = self.generate_fashion_recommendations(model_name, test_queries)
            
            # Compute rewards using reward model
            if self.reward_model:
                rewards = self.reward_model.batch_compute_rewards(test_queries, recommendations)
                avg_reward = np.mean(rewards)
                reward_std = np.std(rewards)
            else:
                rewards = []
                avg_reward = 0.0
                reward_std = 0.0
            
            # Compute text quality metrics
            avg_length = np.mean([len(rec.split()) for rec in recommendations])
            length_std = np.std([len(rec.split()) for rec in recommendations])
            
            # Fashion keyword coverage
            fashion_keywords = [
                'style', 'color', 'fit', 'material', 'brand', 'size', 'occasion',
                'trendy', 'classic', 'comfortable', 'elegant', 'casual', 'formal'
            ]
            
            keyword_coverage = []
            for rec in recommendations:
                rec_lower = rec.lower()
                coverage = sum(1 for keyword in fashion_keywords if keyword in rec_lower) / len(fashion_keywords)
                keyword_coverage.append(coverage)
            
            avg_keyword_coverage = np.mean(keyword_coverage)
            
            evaluation_results = {
                'model_name': model_name,
                'num_samples': len(test_queries),
                'avg_reward': avg_reward,
                'reward_std': reward_std,
                'avg_response_length': avg_length,
                'response_length_std': length_std,
                'avg_keyword_coverage': avg_keyword_coverage,
                'individual_rewards': rewards,
                'individual_keyword_coverage': keyword_coverage,
                'sample_recommendations': recommendations[:5],  # First 5 for inspection
                'evaluated_at': datetime.now().isoformat()
            }
            
            console.print(f"[green]✅[/green] Model evaluation completed")
            console.print(f"[cyan]📈[/cyan] Average reward: {avg_reward:.3f} ± {reward_std:.3f}")
            console.print(f"[cyan]📈[/cyan] Average response length: {avg_length:.1f} words")
            console.print(f"[cyan]📈[/cyan] Keyword coverage: {avg_keyword_coverage:.2%}")
            
            return evaluation_results
            
        except Exception as e:
            console.print(f"[red]Error during evaluation: {e}[/red]")
            raise
    
    def compare_training_methods(self, base_model: str, train_data: List[str], 
                               test_queries: List[str]) -> Dict[str, Any]:
        """Compare different training methods (SFT, PPO, DPO)"""
        console.print("[bold blue]🔬 Comparing TRL training methods[/bold blue]")
        
        comparison_results = {}
        
        try:
            # 1. Baseline evaluation (no fine-tuning)
            console.print("\n[yellow]📊[/yellow] Evaluating baseline model...")
            baseline_results = self.evaluate_model_performance(base_model, test_queries)
            comparison_results['baseline'] = baseline_results
            
            # 2. SFT training
            console.print("\n[yellow]🎓[/yellow] Testing SFT training...")
            sft_config = SFTTrainingConfig(
                model_name=base_model,
                num_train_epochs=1,  # Reduced for comparison
                per_device_train_batch_size=2,
                max_seq_length=256,
                output_dir="./sft_comparison"
            )
            
            sft_trainer = self.setup_sft_training(sft_config, train_data)
            sft_trainer.train()
            
            # Save and evaluate SFT model
            sft_model_path = "./sft_comparison/final"
            sft_trainer.save_model(sft_model_path)
            sft_results = self.evaluate_model_performance(sft_model_path, test_queries)
            comparison_results['sft'] = sft_results
            
            # 3. PPO training (simplified)
            console.print("\n[yellow]🎯[/yellow] Testing PPO training...")
            rlhf_config = RLHFConfig(
                model_name=base_model,
                batch_size=4,
                mini_batch_size=2,
                ppo_epochs=2
            )
            
            ppo_trainer = self.setup_ppo_training(rlhf_config, base_model)
            ppo_stats = self.train_with_ppo(list(self.trainers.keys())[-1], test_queries, num_steps=20)
            
            # Note: For full comparison, we'd need to save and evaluate the PPO model
            # This is simplified for demonstration
            comparison_results['ppo'] = {
                'training_stats': ppo_stats,
                'note': 'PPO evaluation requires model saving and reloading'
            }
            
            # Display comparison
            self._display_training_comparison(comparison_results)
            
            return comparison_results
            
        except Exception as e:
            console.print(f"[red]Error during method comparison: {e}[/red]")
            return comparison_results
    
    def _display_training_comparison(self, results: Dict[str, Any]):
        """Display training method comparison results"""
        table = Table(title="TRL Training Methods Comparison")
        
        table.add_column("Method", style="cyan")
        table.add_column("Avg Reward", style="green")
        table.add_column("Response Length", style="yellow")
        table.add_column("Keyword Coverage", style="blue")
        table.add_column("Notes", style="magenta")
        
        for method, result in results.items():
            if method == 'ppo' and 'training_stats' in result:
                # Special handling for PPO results
                avg_reward = np.mean(result['training_stats']['rewards']) if result['training_stats']['rewards'] else 0
                table.add_row(
                    method.upper(),
                    f"{avg_reward:.3f}",
                    "N/A",
                    "N/A",
                    result.get('note', '')
                )
            elif isinstance(result, dict) and 'avg_reward' in result:
                table.add_row(
                    method.upper(),
                    f"{result['avg_reward']:.3f}",
                    f"{result['avg_response_length']:.1f}",
                    f"{result['avg_keyword_coverage']:.2%}",
                    "Complete evaluation"
                )
        
        console.print(table)
    
    def save_training_results(self, output_path: str) -> bool:
        """Save all training results and history"""
        try:
            results_data = {
                'trainers': {k: {**v, 'trainer': None} for k, v in self.trainers.items()},  # Exclude trainer objects
                'training_history': self.training_history,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            console.print(f"[green]✅[/green] Training results saved to: {output_path}")
            return True
            
        except Exception as e:
            console.print(f"[red]Error saving results: {e}[/red]")
            return False

# Factory function
def create_trl_integration(cache_dir: str = "models/cache") -> TRLIntegration:
    """Create TRL integration instance"""
    return TRLIntegration(cache_dir)

# Example usage functions
def generate_sample_fashion_data(n_samples: int = 100) -> Tuple[List[str], List[str], List[Dict]]:
    """Generate sample fashion data for TRL training"""
    # SFT training data
    sft_data = []
    
    # Test queries
    test_queries = [
        "What should I wear to a business meeting?",
        "Recommend casual summer outfits",
        "What colors go well with navy blue?",
        "Suggest formal wear for a wedding",
        "What's trendy for fall fashion?"
    ]
    
    # DPO preference data (simplified)
    dpo_data = []
    
    # Generate SFT data
    fashion_contexts = [
        "For a business meeting, I recommend",
        "For casual summer wear, consider",
        "Colors that complement navy blue include",
        "For formal wedding attire, choose",
        "This fall's trending styles feature"
    ]
    
    fashion_responses = [
        "a well-tailored blazer with dress pants and leather shoes for a professional appearance.",
        "lightweight cotton shirts, linen shorts, and comfortable sandals in breathable fabrics.",
        "white, cream, light gray, and soft pastels that create elegant color combinations.",
        "a classic suit in charcoal or navy, paired with a crisp white shirt and silk tie.",
        "oversized sweaters, wide-leg trousers, and rich autumn colors like burgundy and olive."
    ]
    
    for i in range(n_samples):
        context = np.random.choice(fashion_contexts)
        response = np.random.choice(fashion_responses)
        sft_data.append(f"{context} {response}")
    
    # Generate DPO data (simplified preference pairs)
    for i in range(min(20, n_samples // 5)):
        query = np.random.choice(test_queries)
        chosen = f"I recommend {np.random.choice(fashion_responses)}"
        rejected = "I'm not sure what to suggest."
        
        dpo_data.append({
            'prompt': query,
            'chosen': chosen,
            'rejected': rejected
        })
    
    return sft_data, test_queries, dpo_data

if __name__ == "__main__":
    # Demo usage
    console.print("[bold blue]TRL Integration Demo[/bold blue]")
    
    try:
        # Create TRL integration
        trl = create_trl_integration()
        
        # Generate sample data
        console.print("[yellow]📊[/yellow] Generating sample fashion data...")
        sft_data, test_queries, dpo_data = generate_sample_fashion_data(50)
        
        # Test SFT training
        console.print("\n[yellow]🎓[/yellow] Testing SFT training...")
        sft_config = SFTTrainingConfig(
            model_name="distilgpt2",  # Smaller model for demo
            num_train_epochs=1,
            per_device_train_batch_size=2,
            max_seq_length=128,
            output_dir="./demo_sft"
        )
        
        sft_trainer = trl.setup_sft_training(sft_config, sft_data[:20])  # Smaller dataset
        
        # Test reward model
        if trl.reward_model:
            console.print("\n[yellow]🏆[/yellow] Testing reward model...")
            sample_query = "What should I wear to work?"
            sample_response = "I recommend a professional blazer with tailored pants and comfortable shoes."
            reward = trl.reward_model.compute_reward(sample_query, sample_response)
            console.print(f"Sample reward: {reward:.3f}")
        
        # Test generation
        console.print("\n[yellow]💡[/yellow] Testing fashion recommendation generation...")
        try:
            recommendations = trl.generate_fashion_recommendations(
                "distilgpt2", 
                test_queries[:2],  # Just first 2 queries
                max_new_tokens=30
            )
            
            for query, rec in zip(test_queries[:2], recommendations):
                console.print(f"[cyan]Q:[/cyan] {query}")
                console.print(f"[green]A:[/green] {rec}\n")
        except Exception as e:
            console.print(f"[yellow]Generation test skipped: {e}[/yellow]")
        
        console.print("[green]✅[/green] TRL integration demo completed!")
        
    except Exception as e:
        console.print(f"[red]Demo failed: {e}[/red]")
        console.print("[yellow]Note: This demo requires transformers, trl, and peft libraries[/yellow]")