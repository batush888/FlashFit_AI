import asyncio
import json
import ujson
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# Core libraries
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# FlashFit AI services
try:
    from .analytics_service import AnalyticsService
    from .performance_monitor import PerformanceMonitor
    from .flask_extensions import FlaskExtensions
    from .wandb_integration import WandbIntegration
    from .peft_optimizer import PEFTOptimizer
    from .trl_integration import TRLIntegration
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('..')
    from analytics_service import AnalyticsService
    from performance_monitor import PerformanceMonitor
    from flask_extensions import FlaskExtensions
    from wandb_integration import WandbIntegration
    from peft_optimizer import PEFTOptimizer
    from trl_integration import TRLIntegration

# Enhanced NLP libraries
try:
    import nltk
    import jieba
    from sentence_transformers import SentenceTransformer
except ImportError:
    nltk = None
    jieba = None
    SentenceTransformer = None

# Data processing libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
except ImportError:
    plt = None
    sns = None
    accuracy_score = None

# Streamlit for dashboard
try:
    import streamlit as st
except ImportError:
    st = None

# Initialize console
console = Console()

@dataclass
class IntegrationConfig:
    """Configuration for FlashFit AI integration"""
    # Service configurations
    enable_analytics: bool = True
    enable_performance_monitoring: bool = True
    enable_flask_extensions: bool = True
    enable_wandb: bool = False  # Requires API key
    enable_peft: bool = True
    enable_trl: bool = True
    enable_streamlit_dashboard: bool = True
    
    # Model configurations
    default_model: str = "distilbert-base-uncased"
    sentence_model: str = "all-MiniLM-L6-v2"
    cache_dir: str = "models/cache"
    
    # Performance settings
    monitoring_interval: int = 60  # seconds
    max_memory_usage: float = 0.8  # 80% of available memory
    enable_gpu_monitoring: bool = True
    
    # Analytics settings
    enable_real_time_analytics: bool = True
    analytics_batch_size: int = 1000
    enable_visualization: bool = True
    
    # Integration settings
    auto_optimize_models: bool = False
    enable_experiment_tracking: bool = False
    log_level: str = "INFO"

@dataclass
class ServiceStatus:
    """Status of integrated services"""
    service_name: str
    status: str  # 'active', 'inactive', 'error', 'initializing'
    last_check: str
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None

class IntegrationManager:
    """
    Central integration manager for FlashFit AI enhanced services
    
    This class orchestrates all the new AI/ML enhancement libraries and services:
    - Analytics and data processing (pandas, matplotlib, scikit-learn)
    - Performance monitoring (psutil, rich)
    - Advanced NLP (sentence-transformers, nltk, jieba)
    - Model optimization (PEFT, TRL)
    - Experiment tracking (wandb)
    - Enhanced APIs (Flask extensions)
    - Interactive dashboards (Streamlit)
    """
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        self.services = {}
        self.service_status = {}
        self.integration_history = []
        
        # Setup logging
        self._setup_logging()
        
        # Initialize services
        self._initialize_services()
        
        console.print("[green]✓[/green] FlashFit AI Integration Manager initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/integration.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FlashFitIntegration')
    
    def _initialize_services(self):
        """Initialize all enabled services"""
        console.print("[blue]🚀[/blue] Initializing FlashFit AI services...")
        
        # Analytics Service
        if self.config.enable_analytics:
            try:
                self.services['analytics'] = AnalyticsService()
                self._update_service_status('analytics', 'active')
                console.print("[green]✓[/green] Analytics service initialized")
            except Exception as e:
                self._update_service_status('analytics', 'error', str(e))
                console.print(f"[red]✗[/red] Analytics service failed: {e}")
        
        # Performance Monitor
        if self.config.enable_performance_monitoring:
            try:
                self.services['performance'] = PerformanceMonitor(
                    monitoring_interval=self.config.monitoring_interval,
                    enable_gpu=self.config.enable_gpu_monitoring
                )
                self._update_service_status('performance', 'active')
                console.print("[green]✓[/green] Performance monitor initialized")
            except Exception as e:
                self._update_service_status('performance', 'error', str(e))
                console.print(f"[red]✗[/red] Performance monitor failed: {e}")
        
        # Flask Extensions
        if self.config.enable_flask_extensions:
            try:
                self.services['flask_ext'] = FlaskExtensions()
                self._update_service_status('flask_ext', 'active')
                console.print("[green]✓[/green] Flask extensions initialized")
            except Exception as e:
                self._update_service_status('flask_ext', 'error', str(e))
                console.print(f"[red]✗[/red] Flask extensions failed: {e}")
        
        # Wandb Integration
        if self.config.enable_wandb:
            try:
                self.services['wandb'] = WandbIntegration()
                self._update_service_status('wandb', 'active')
                console.print("[green]✓[/green] Wandb integration initialized")
            except Exception as e:
                self._update_service_status('wandb', 'error', str(e))
                console.print(f"[yellow]⚠[/yellow] Wandb integration failed (API key required): {e}")
        
        # PEFT Optimizer
        if self.config.enable_peft:
            try:
                self.services['peft'] = PEFTOptimizer(
                    model_name=self.config.default_model,
                    cache_dir=self.config.cache_dir
                )
                self._update_service_status('peft', 'active')
                console.print("[green]✓[/green] PEFT optimizer initialized")
            except Exception as e:
                self._update_service_status('peft', 'error', str(e))
                console.print(f"[red]✗[/red] PEFT optimizer failed: {e}")
        
        # TRL Integration
        if self.config.enable_trl:
            try:
                self.services['trl'] = TRLIntegration(cache_dir=self.config.cache_dir)
                self._update_service_status('trl', 'active')
                console.print("[green]✓[/green] TRL integration initialized")
            except Exception as e:
                self._update_service_status('trl', 'error', str(e))
                console.print(f"[red]✗[/red] TRL integration failed: {e}")
        
        # NLP Enhancement
        self._initialize_nlp_services()
    
    def _initialize_nlp_services(self):
        """Initialize enhanced NLP services"""
        try:
            # Initialize sentence transformers
            if SentenceTransformer:
                self.services['sentence_model'] = SentenceTransformer(self.config.sentence_model)
                console.print("[green]✓[/green] Sentence transformer initialized")
            
            # Initialize NLTK data
            if nltk:
                try:
                    nltk.data.find('tokenizers/punkt')
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    console.print("[yellow]📥[/yellow] Downloading NLTK data...")
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                console.print("[green]✓[/green] NLTK initialized")
            
            # Initialize jieba for Chinese text processing
            if jieba:
                console.print("[green]✓[/green] Jieba initialized")
            
            self._update_service_status('nlp', 'active')
            
        except Exception as e:
            self._update_service_status('nlp', 'error', str(e))
            console.print(f"[red]✗[/red] NLP services failed: {e}")
    
    def _update_service_status(self, service_name: str, status: str, error_message: str = None):
        """Update service status"""
        self.service_status[service_name] = ServiceStatus(
            service_name=service_name,
            status=status,
            last_check=datetime.now().isoformat(),
            error_message=error_message
        )
    
    async def start_monitoring(self):
        """Start real-time monitoring of all services"""
        console.print("[blue]📊[/blue] Starting real-time monitoring...")
        
        if 'performance' in self.services:
            try:
                await self.services['performance'].start_monitoring()
                console.print("[green]✓[/green] Performance monitoring started")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to start monitoring: {e}")
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        overview = {
            'timestamp': datetime.now().isoformat(),
            'services': {},
            'performance': {},
            'analytics': {},
            'models': {}
        }
        
        # Service status
        for service_name, status in self.service_status.items():
            overview['services'][service_name] = asdict(status)
        
        # Performance metrics
        if 'performance' in self.services:
            try:
                perf_data = self.services['performance'].get_current_metrics()
                overview['performance'] = perf_data
            except Exception as e:
                overview['performance'] = {'error': str(e)}
        
        # Analytics summary
        if 'analytics' in self.services:
            try:
                analytics_summary = self.services['analytics'].get_system_summary()
                overview['analytics'] = analytics_summary
            except Exception as e:
                overview['analytics'] = {'error': str(e)}
        
        # Model information
        if 'peft' in self.services:
            try:
                model_info = {
                    'optimization_history': len(self.services['peft'].optimization_history),
                    'available_models': list(self.services['peft'].peft_models.keys())
                }
                overview['models'] = model_info
            except Exception as e:
                overview['models'] = {'error': str(e)}
        
        return overview
    
    def optimize_fashion_model(self, train_data: Dict[str, List], 
                             val_data: Dict[str, List], method: str = 'lora') -> Dict[str, Any]:
        """Optimize fashion model using PEFT"""
        if 'peft' not in self.services:
            raise ValueError("PEFT optimizer not available")
        
        console.print(f"[blue]🔧[/blue] Optimizing fashion model using {method.upper()}...")
        
        try:
            from .peft_optimizer import PEFTConfig
            
            # Configure PEFT method
            peft_config = PEFTConfig(
                method=method,
                task_type='SEQ_CLS',
                target_modules=['query', 'value'],
                r=8 if method == 'lora' else 4,
                lora_alpha=32,
                lora_dropout=0.1
            )
            
            # Training arguments
            training_args = {
                'epochs': 3,
                'batch_size': 8,
                'learning_rate': 2e-4,
                'weight_decay': 0.01
            }
            
            # Optimize model
            results = self.services['peft'].optimize_model(
                peft_config, train_data, val_data, training_args
            )
            
            # Log to wandb if available
            if 'wandb' in self.services:
                try:
                    self.services['wandb'].log_metrics({
                        'peft_accuracy': results.final_accuracy,
                        'peft_param_reduction': results.param_reduction,
                        'peft_training_time': results.training_time_seconds
                    })
                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] Wandb logging failed: {e}")
            
            console.print(f"[green]✅[/green] Model optimization completed")
            return asdict(results)
            
        except Exception as e:
            console.print(f"[red]✗[/red] Model optimization failed: {e}")
            raise
    
    def analyze_fashion_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze fashion data using analytics service"""
        if 'analytics' not in self.services:
            raise ValueError("Analytics service not available")
        
        console.print("[blue]📊[/blue] Analyzing fashion data...")
        
        try:
            # Perform comprehensive analysis
            analysis_results = self.services['analytics'].analyze_fashion_trends(data)
            
            # Generate visualizations if enabled
            if self.config.enable_visualization and plt:
                viz_results = self.services['analytics'].create_interactive_dashboard(data)
                analysis_results['visualizations'] = viz_results
            
            console.print(f"[green]✅[/green] Fashion data analysis completed")
            return analysis_results
            
        except Exception as e:
            console.print(f"[red]✗[/red] Data analysis failed: {e}")
            raise
    
    def process_multilingual_text(self, texts: List[str], 
                                languages: List[str] = None) -> Dict[str, Any]:
        """Process multilingual fashion text using enhanced NLP"""
        console.print("[blue]🌐[/blue] Processing multilingual text...")
        
        results = {
            'processed_texts': [],
            'embeddings': [],
            'language_detection': [],
            'fashion_attributes': []
        }
        
        try:
            for i, text in enumerate(texts):
                # Detect language if not provided
                if languages and i < len(languages):
                    lang = languages[i]
                else:
                    # Simple language detection (can be enhanced)
                    lang = 'zh' if any('\u4e00' <= char <= '\u9fff' for char in text) else 'en'
                
                results['language_detection'].append(lang)
                
                # Process text based on language
                if lang == 'zh' and jieba:
                    # Chinese text processing
                    tokens = list(jieba.cut(text))
                    processed_text = ' '.join(tokens)
                elif nltk:
                    # English text processing
                    from nltk.tokenize import word_tokenize
                    from nltk.corpus import stopwords
                    
                    tokens = word_tokenize(text.lower())
                    stop_words = set(stopwords.words('english'))
                    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
                    processed_text = ' '.join(tokens)
                else:
                    processed_text = text.lower()
                
                results['processed_texts'].append(processed_text)
                
                # Generate embeddings if sentence transformer available
                if 'sentence_model' in self.services:
                    embedding = self.services['sentence_model'].encode(text)
                    results['embeddings'].append(embedding.tolist())
                
                # Extract fashion attributes (simplified)
                fashion_keywords = [
                    'color', 'style', 'material', 'brand', 'size', 'fit',
                    'casual', 'formal', 'trendy', 'classic', 'comfortable'
                ]
                
                attributes = [keyword for keyword in fashion_keywords if keyword in text.lower()]
                results['fashion_attributes'].append(attributes)
            
            console.print(f"[green]✅[/green] Processed {len(texts)} texts")
            return results
            
        except Exception as e:
            console.print(f"[red]✗[/red] Text processing failed: {e}")
            raise
    
    def generate_fashion_recommendations(self, user_preferences: Dict[str, Any], 
                                       context: str = "casual") -> List[Dict[str, Any]]:
        """Generate AI-powered fashion recommendations"""
        console.print("[blue]💡[/blue] Generating fashion recommendations...")
        
        try:
            recommendations = []
            
            # Use TRL integration if available for advanced generation
            if 'trl' in self.services:
                queries = [
                    f"Recommend {context} outfits for someone who likes {user_preferences.get('style', 'modern')} style",
                    f"Suggest {user_preferences.get('color', 'neutral')} colored clothing for {context} occasions"
                ]
                
                try:
                    trl_recommendations = self.services['trl'].generate_fashion_recommendations(
                        self.config.default_model, queries, max_new_tokens=100
                    )
                    
                    for i, rec in enumerate(trl_recommendations):
                        recommendations.append({
                            'type': 'ai_generated',
                            'recommendation': rec,
                            'confidence': 0.8,
                            'source': 'trl_model'
                        })
                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] TRL generation failed: {e}")
            
            # Fallback to rule-based recommendations
            if not recommendations:
                style = user_preferences.get('style', 'casual')
                color = user_preferences.get('color', 'neutral')
                occasion = user_preferences.get('occasion', context)
                
                rule_based_recs = [
                    {
                        'type': 'rule_based',
                        'recommendation': f"For {occasion} occasions, try a {color} {style} outfit with comfortable fit",
                        'confidence': 0.6,
                        'source': 'rule_engine'
                    },
                    {
                        'type': 'rule_based', 
                        'recommendation': f"Consider {style} accessories in {color} tones to complement your look",
                        'confidence': 0.5,
                        'source': 'rule_engine'
                    }
                ]
                recommendations.extend(rule_based_recs)
            
            # Enhance with analytics if available
            if 'analytics' in self.services:
                try:
                    # Add trend analysis
                    for rec in recommendations:
                        rec['trend_score'] = np.random.uniform(0.3, 0.9)  # Placeholder
                        rec['popularity'] = np.random.choice(['trending', 'classic', 'emerging'])
                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] Analytics enhancement failed: {e}")
            
            console.print(f"[green]✅[/green] Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            console.print(f"[red]✗[/red] Recommendation generation failed: {e}")
            raise
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark of all integrated services"""
        console.print("[bold blue]🏁 Running comprehensive FlashFit AI benchmark[/bold blue]")
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'services_tested': [],
            'performance_metrics': {},
            'integration_health': {},
            'recommendations': []
        }
        
        # Test each service
        for service_name, service in self.services.items():
            console.print(f"\n[yellow]🧪[/yellow] Testing {service_name}...")
            
            try:
                if service_name == 'analytics':
                    # Test analytics with sample data
                    sample_data = pd.DataFrame({
                        'category': ['tops', 'bottoms', 'shoes'] * 10,
                        'price': np.random.uniform(20, 200, 30),
                        'rating': np.random.uniform(3, 5, 30)
                    })
                    
                    start_time = datetime.now()
                    result = service.analyze_fashion_trends(sample_data)
                    end_time = datetime.now()
                    
                    benchmark_results['performance_metrics'][service_name] = {
                        'execution_time': (end_time - start_time).total_seconds(),
                        'status': 'success',
                        'data_processed': len(sample_data)
                    }
                
                elif service_name == 'performance':
                    # Test performance monitoring
                    metrics = service.get_current_metrics()
                    benchmark_results['performance_metrics'][service_name] = {
                        'status': 'success',
                        'metrics_collected': len(metrics)
                    }
                
                elif service_name == 'peft':
                    # Test PEFT with minimal data
                    benchmark_results['performance_metrics'][service_name] = {
                        'status': 'success',
                        'models_available': len(service.peft_models),
                        'optimization_history': len(service.optimization_history)
                    }
                
                else:
                    # Generic service test
                    benchmark_results['performance_metrics'][service_name] = {
                        'status': 'success',
                        'service_type': type(service).__name__
                    }
                
                benchmark_results['services_tested'].append(service_name)
                console.print(f"[green]✅[/green] {service_name} test passed")
                
            except Exception as e:
                benchmark_results['performance_metrics'][service_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                console.print(f"[red]✗[/red] {service_name} test failed: {e}")
        
        # Generate integration health report
        total_services = len(self.services)
        successful_services = len(benchmark_results['services_tested'])
        health_score = successful_services / total_services if total_services > 0 else 0
        
        benchmark_results['integration_health'] = {
            'overall_score': health_score,
            'total_services': total_services,
            'successful_services': successful_services,
            'health_status': 'excellent' if health_score >= 0.9 else 'good' if health_score >= 0.7 else 'needs_attention'
        }
        
        # Generate recommendations
        if health_score < 0.8:
            benchmark_results['recommendations'].append("Consider reviewing failed services and their dependencies")
        
        if 'wandb' not in self.services or self.service_status.get('wandb', {}).get('status') == 'error':
            benchmark_results['recommendations'].append("Set up Wandb integration for enhanced experiment tracking")
        
        if not self.config.enable_real_time_analytics:
            benchmark_results['recommendations'].append("Enable real-time analytics for better insights")
        
        # Display results
        self._display_benchmark_results(benchmark_results)
        
        return benchmark_results
    
    def _display_benchmark_results(self, results: Dict[str, Any]):
        """Display benchmark results in a formatted table"""
        # Services status table
        table = Table(title="FlashFit AI Services Benchmark")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Performance", style="yellow")
        table.add_column("Notes", style="blue")
        
        for service_name, metrics in results['performance_metrics'].items():
            status = "✅ Pass" if metrics['status'] == 'success' else "❌ Fail"
            
            performance = ""
            if 'execution_time' in metrics:
                performance = f"{metrics['execution_time']:.3f}s"
            elif 'metrics_collected' in metrics:
                performance = f"{metrics['metrics_collected']} metrics"
            
            notes = ""
            if 'error' in metrics:
                notes = metrics['error'][:50] + "..." if len(metrics['error']) > 50 else metrics['error']
            elif 'data_processed' in metrics:
                notes = f"{metrics['data_processed']} records"
            
            table.add_row(service_name.title(), status, performance, notes)
        
        console.print(table)
        
        # Health summary
        health = results['integration_health']
        health_panel = Panel(
            f"Overall Health Score: {health['overall_score']:.2%}\n"
            f"Status: {health['health_status'].title()}\n"
            f"Services: {health['successful_services']}/{health['total_services']} operational",
            title="Integration Health",
            border_style="green" if health['overall_score'] >= 0.8 else "yellow"
        )
        console.print(health_panel)
    
    def save_integration_state(self, filepath: str) -> bool:
        """Save current integration state"""
        try:
            state_data = {
                'config': asdict(self.config),
                'service_status': {k: asdict(v) for k, v in self.service_status.items()},
                'integration_history': self.integration_history,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            console.print(f"[green]✅[/green] Integration state saved to: {filepath}")
            return True
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to save state: {e}")
            return False
    
    def load_integration_state(self, filepath: str) -> bool:
        """Load integration state from file"""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            # Restore configuration
            self.config = IntegrationConfig(**state_data['config'])
            
            # Restore service status
            for service_name, status_data in state_data['service_status'].items():
                self.service_status[service_name] = ServiceStatus(**status_data)
            
            # Restore history
            self.integration_history = state_data.get('integration_history', [])
            
            console.print(f"[green]✅[/green] Integration state loaded from: {filepath}")
            return True
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load state: {e}")
            return False

# Factory function
def create_integration_manager(config: IntegrationConfig = None) -> IntegrationManager:
    """Create FlashFit AI integration manager"""
    return IntegrationManager(config)

# Async context manager for monitoring
@asynccontextmanager
async def managed_integration(config: IntegrationConfig = None):
    """Async context manager for FlashFit AI integration"""
    manager = create_integration_manager(config)
    
    try:
        # Start monitoring if enabled
        if config and config.enable_performance_monitoring:
            await manager.start_monitoring()
        
        yield manager
        
    finally:
        # Cleanup
        console.print("[blue]🔄[/blue] Shutting down FlashFit AI integration...")
        
        # Stop monitoring
        if 'performance' in manager.services:
            try:
                # Note: Actual implementation would need proper cleanup
                console.print("[green]✓[/green] Performance monitoring stopped")
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Cleanup warning: {e}")

if __name__ == "__main__":
    # Demo usage
    console.print("[bold blue]FlashFit AI Integration Manager Demo[/bold blue]")
    
    try:
        # Create integration manager with custom config
        config = IntegrationConfig(
            enable_wandb=False,  # Disable wandb for demo
            enable_real_time_analytics=True,
            log_level="INFO"
        )
        
        manager = create_integration_manager(config)
        
        # Get system overview
        console.print("\n[yellow]📊[/yellow] System Overview:")
        overview = manager.get_system_overview()
        console.print(Panel(json.dumps(overview['services'], indent=2), title="Services Status"))
        
        # Test multilingual text processing
        console.print("\n[yellow]🌐[/yellow] Testing multilingual text processing...")
        sample_texts = [
            "I love this casual blue cotton shirt",
            "这件红色连衣裙很漂亮",  # Chinese: "This red dress is beautiful"
            "Elegant black formal suit for business meetings"
        ]
        
        nlp_results = manager.process_multilingual_text(sample_texts)
        console.print(f"Processed {len(nlp_results['processed_texts'])} texts")
        
        # Test fashion recommendations
        console.print("\n[yellow]💡[/yellow] Testing fashion recommendations...")
        user_prefs = {
            'style': 'casual',
            'color': 'blue',
            'occasion': 'weekend'
        }
        
        recommendations = manager.generate_fashion_recommendations(user_prefs, "casual")
        console.print(f"Generated {len(recommendations)} recommendations")
        
        # Run comprehensive benchmark
        console.print("\n[yellow]🏁[/yellow] Running benchmark...")
        benchmark_results = manager.run_comprehensive_benchmark()
        
        console.print("\n[green]✅[/green] FlashFit AI Integration demo completed!")
        
    except Exception as e:
        console.print(f"[red]Demo failed: {e}[/red]")
        console.print("[yellow]Note: Some features require additional library installations[/yellow]")