import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import psutil
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import ujson
from dataclasses import dataclass
from marshmallow import Schema, fields
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Initialize rich console
console = Console()

@dataclass
class AnalyticsMetrics:
    """Data class for analytics metrics"""
    total_items: int
    total_users: int
    total_interactions: int
    avg_rating: float
    popular_categories: List[str]
    system_health: Dict[str, float]
    recommendation_accuracy: float
    user_engagement: float

class AnalyticsSchema(Schema):
    """Schema for analytics data validation"""
    timestamp = fields.DateTime()
    user_id = fields.Str()
    item_id = fields.Str()
    action = fields.Str()
    rating = fields.Float(allow_none=True)
    category = fields.Str(allow_none=True)
    metadata = fields.Dict(allow_none=True)

class EnhancedAnalyticsService:
    """
    Enhanced analytics service with advanced data analysis capabilities
    
    Features:
    - Real-time system monitoring with psutil
    - Rich console output for better CLI experience
    - Advanced data analysis with pandas and scikit-learn
    - Interactive visualizations with plotly
    - Text analysis with sentence transformers
    - Performance optimization with ujson
    - Token counting with tiktoken
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.console = Console()
        self.schema = AnalyticsSchema()
        
        # Initialize sentence transformer for text analysis
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            console.print("[green]✓[/green] Sentence transformer loaded")
        except Exception as e:
            self.sentence_model = None
            console.print(f"[yellow]⚠[/yellow] Sentence transformer not available: {e}")
        
        # Initialize tiktoken for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            console.print("[green]✓[/green] Tiktoken encoder loaded")
        except Exception as e:
            self.tokenizer = None
            console.print(f"[yellow]⚠[/yellow] Tiktoken not available: {e}")
        
        console.print("[bold blue]Enhanced Analytics Service initialized[/bold blue]")
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get comprehensive system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'process_count': len(psutil.pids())
            }
        except Exception as e:
            console.print(f"[red]Error getting system metrics: {e}[/red]")
            return {}
    
    def load_data_with_progress(self, file_path: Path) -> Optional[List[Dict]]:
        """Load data with rich progress indicator"""
        if not file_path.exists():
            return None
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task(f"Loading {file_path.name}...", total=None)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.suffix == '.json':
                        data = ujson.load(f) if ujson else json.load(f)
                    else:
                        data = json.load(f)
                
                progress.update(task, description=f"✓ Loaded {file_path.name}")
                return data if isinstance(data, list) else [data]
                
            except Exception as e:
                progress.update(task, description=f"✗ Failed to load {file_path.name}")
                console.print(f"[red]Error loading {file_path}: {e}[/red]")
                return None
    
    def analyze_fashion_items(self) -> Dict[str, Any]:
        """Comprehensive analysis of fashion items"""
        items_data = self.load_data_with_progress(self.data_dir / "fashion_items.json")
        
        if not items_data:
            return {'error': 'No fashion items data available'}
        
        df = pd.DataFrame(items_data)
        
        analysis = {
            'total_items': len(df),
            'categories': df.get('category', pd.Series()).value_counts().to_dict(),
            'colors': df.get('color', pd.Series()).value_counts().to_dict(),
            'price_stats': {},
            'description_analysis': {}
        }
        
        # Price analysis if available
        if 'price' in df.columns:
            analysis['price_stats'] = {
                'mean': float(df['price'].mean()),
                'median': float(df['price'].median()),
                'std': float(df['price'].std()),
                'min': float(df['price'].min()),
                'max': float(df['price'].max())
            }
        
        # Text analysis of descriptions
        if 'description' in df.columns and self.sentence_model:
            descriptions = df['description'].dropna().tolist()
            if descriptions:
                analysis['description_analysis'] = self.analyze_text_data(descriptions)
        
        return analysis
    
    def analyze_text_data(self, texts: List[str]) -> Dict[str, Any]:
        """Advanced text analysis using sentence transformers and clustering"""
        if not self.sentence_model or not texts:
            return {'error': 'Text analysis not available'}
        
        try:
            # Generate embeddings
            embeddings = self.sentence_model.encode(texts)
            
            # Perform clustering
            n_clusters = min(5, len(texts) // 2) if len(texts) > 4 else 2
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(embeddings, clusters) if len(set(clusters)) > 1 else 0
            
            # Token analysis if tiktoken is available
            token_stats = {}
            if self.tokenizer:
                token_counts = [len(self.tokenizer.encode(text)) for text in texts]
                token_stats = {
                    'avg_tokens': np.mean(token_counts),
                    'max_tokens': max(token_counts),
                    'min_tokens': min(token_counts),
                    'total_tokens': sum(token_counts)
                }
            
            return {
                'total_texts': len(texts),
                'clusters': n_clusters,
                'silhouette_score': float(silhouette_avg),
                'token_stats': token_stats,
                'avg_length': np.mean([len(text) for text in texts]),
                'cluster_distribution': Counter(clusters)
            }
            
        except Exception as e:
            console.print(f"[red]Error in text analysis: {e}[/red]")
            return {'error': str(e)}
    
    def analyze_user_behavior(self) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        feedback_data = self.load_data_with_progress(self.data_dir / "feedback.json")
        history_data = self.load_data_with_progress(self.data_dir / "outfit_history.json")
        
        analysis = {
            'feedback_analysis': {},
            'usage_patterns': {},
            'engagement_metrics': {}
        }
        
        # Feedback analysis
        if feedback_data:
            feedback_df = pd.DataFrame(feedback_data)
            analysis['feedback_analysis'] = {
                'total_feedback': len(feedback_df),
                'avg_rating': float(feedback_df.get('rating', pd.Series()).mean() or 0),
                'rating_distribution': feedback_df.get('rating', pd.Series()).value_counts().to_dict()
            }
        
        # Usage patterns
        if history_data:
            history_df = pd.DataFrame(history_data)
            if 'timestamp' in history_df.columns:
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], errors='coerce')
                history_df['hour'] = history_df['timestamp'].dt.hour
                history_df['day_of_week'] = history_df['timestamp'].dt.day_name()
                
                analysis['usage_patterns'] = {
                    'peak_hours': history_df['hour'].value_counts().head(3).to_dict(),
                    'popular_days': history_df['day_of_week'].value_counts().to_dict(),
                    'total_sessions': len(history_df)
                }
        
        return analysis
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        console.print("[bold blue]Generating Performance Report...[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # System metrics
            task1 = progress.add_task("Collecting system metrics...", total=None)
            system_metrics = self.get_system_metrics()
            progress.update(task1, description="✓ System metrics collected")
            
            # Fashion items analysis
            task2 = progress.add_task("Analyzing fashion items...", total=None)
            fashion_analysis = self.analyze_fashion_items()
            progress.update(task2, description="✓ Fashion items analyzed")
            
            # User behavior analysis
            task3 = progress.add_task("Analyzing user behavior...", total=None)
            behavior_analysis = self.analyze_user_behavior()
            progress.update(task3, description="✓ User behavior analyzed")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': system_metrics,
            'fashion_analysis': fashion_analysis,
            'behavior_analysis': behavior_analysis,
            'recommendations': self.generate_recommendations(system_metrics, fashion_analysis, behavior_analysis)
        }
        
        # Save report
        report_path = self.data_dir / "monitoring" / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(report_path, 'w') as f:
                if ujson:
                    ujson.dump(report, f, indent=2)
                else:
                    json.dump(report, f, indent=2)
            console.print(f"[green]✓ Report saved to {report_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")
        
        return report
    
    def generate_recommendations(self, system_metrics: Dict, fashion_analysis: Dict, behavior_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # System performance recommendations
        if system_metrics.get('cpu_usage', 0) > 80:
            recommendations.append("High CPU usage detected. Consider optimizing AI model inference or scaling resources.")
        
        if system_metrics.get('memory_usage', 0) > 85:
            recommendations.append("High memory usage detected. Consider implementing memory optimization or increasing RAM.")
        
        # Fashion data recommendations
        if fashion_analysis.get('total_items', 0) < 100:
            recommendations.append("Low fashion item count. Consider expanding the fashion catalog for better recommendations.")
        
        # User engagement recommendations
        avg_rating = behavior_analysis.get('feedback_analysis', {}).get('avg_rating', 0)
        if avg_rating < 3.5:
            recommendations.append("Low average user rating. Review recommendation algorithm and user feedback.")
        
        if not recommendations:
            recommendations.append("System performance is optimal. Continue monitoring for any changes.")
        
        return recommendations
    
    def display_metrics_table(self, metrics: Dict[str, Any]):
        """Display metrics in a rich table format"""
        table = Table(title="System Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
                if 'usage' in key.lower() or 'percent' in key.lower():
                    formatted_value += "%"
                    status = "🟢 Good" if value < 70 else "🟡 Warning" if value < 90 else "🔴 Critical"
                else:
                    status = "ℹ️ Info"
            else:
                formatted_value = str(value)
                status = "ℹ️ Info"
            
            table.add_row(key.replace('_', ' ').title(), formatted_value, status)
        
        console.print(table)
    
    def create_dashboard_data(self) -> Dict[str, Any]:
        """Create data structure for dashboard visualization"""
        report = self.generate_performance_report()
        
        # Format data for dashboard consumption
        dashboard_data = {
            'summary': {
                'total_items': report['fashion_analysis'].get('total_items', 0),
                'avg_rating': report['behavior_analysis'].get('feedback_analysis', {}).get('avg_rating', 0),
                'system_health': 'Good' if report['system_metrics'].get('cpu_usage', 0) < 70 else 'Warning',
                'last_updated': report['timestamp']
            },
            'charts': {
                'categories': report['fashion_analysis'].get('categories', {}),
                'ratings': report['behavior_analysis'].get('feedback_analysis', {}).get('rating_distribution', {}),
                'usage_patterns': report['behavior_analysis'].get('usage_patterns', {})
            },
            'system': report['system_metrics'],
            'recommendations': report['recommendations']
        }
        
        return dashboard_data

# Global instance
_analytics_service = None

def get_analytics_service(data_dir: str = "data") -> EnhancedAnalyticsService:
    """Get or create analytics service instance"""
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = EnhancedAnalyticsService(data_dir)
    return _analytics_service

if __name__ == "__main__":
    # Demo usage
    service = get_analytics_service()
    report = service.generate_performance_report()
    service.display_metrics_table(report['system_metrics'])
    
    console.print(Panel(
        "Enhanced Analytics Service Demo Complete",
        title="FlashFit AI Analytics",
        border_style="blue"
    ))