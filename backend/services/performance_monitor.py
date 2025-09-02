import psutil
import time
import json
import ujson
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
import threading
import queue
from collections import deque, defaultdict
import numpy as np
from marshmallow import Schema, fields
import tiktoken
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Initialize rich console
console = Console()

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    gpu_available: bool
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None

@dataclass
class AIModelMetrics:
    """AI model performance metrics"""
    timestamp: str
    model_name: str
    inference_time_ms: float
    memory_usage_mb: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    throughput_requests_per_sec: Optional[float] = None
    error_rate: Optional[float] = None

@dataclass
class ApplicationMetrics:
    """Application-level metrics"""
    timestamp: str
    active_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    recommendation_requests: int
    upload_requests: int
    cache_hit_rate: Optional[float] = None

class MetricsSchema(Schema):
    """Schema for metrics validation"""
    timestamp = fields.DateTime()
    cpu_percent = fields.Float()
    memory_percent = fields.Float()
    inference_time_ms = fields.Float()
    accuracy = fields.Float(allow_none=True)

class PerformanceMonitor:
    """
    Advanced performance monitoring system with real-time tracking
    
    Features:
    - Real-time system monitoring with psutil
    - GPU monitoring with torch
    - AI model performance tracking
    - Rich console dashboard
    - Automated alerting system
    - Historical data analysis
    - Performance optimization suggestions
    """
    
    def __init__(self, data_dir: str = "data/monitoring", alert_thresholds: Optional[Dict] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.console = Console()
        self.schema = MetricsSchema()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_queue = queue.Queue()
        
        # Historical data storage
        self.system_metrics_history = deque(maxlen=1000)
        self.ai_metrics_history = deque(maxlen=1000)
        self.app_metrics_history = deque(maxlen=1000)
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'inference_time_ms': 1000.0,
            'error_rate': 0.05,
            'response_time_ms': 2000.0
        }
        
        # Performance counters
        self.request_counter = defaultdict(int)
        self.error_counter = defaultdict(int)
        self.response_times = deque(maxlen=100)
        
        # Initialize AI model monitoring
        self.model_monitors = {}
        
        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            console.print("[green]✓[/green] GPU monitoring enabled")
        else:
            console.print("[yellow]⚠[/yellow] GPU not available, CPU monitoring only")
        
        # Initialize tiktoken for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            console.print("[green]✓[/green] Token counting enabled")
        except Exception as e:
            self.tokenizer = None
            console.print(f"[yellow]⚠[/yellow] Token counting not available: {e}")
        
        console.print("[bold blue]Performance Monitor initialized[/bold blue]")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # GPU metrics if available
            gpu_memory_used = None
            gpu_memory_total = None
            if self.gpu_available:
                try:
                    gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                except Exception:
                    pass
            
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / 1024**3,
                disk_percent=disk.percent,
                disk_free_gb=disk.free / 1024**3,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                process_count=len(psutil.pids()),
                gpu_available=self.gpu_available,
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total
            )
        except Exception as e:
            console.print(f"[red]Error collecting system metrics: {e}[/red]")
            return None
    
    def track_ai_model_performance(self, model_name: str, inference_time: float, 
                                 memory_usage: float, accuracy: Optional[float] = None) -> AIModelMetrics:
        """Track AI model performance metrics"""
        metrics = AIModelMetrics(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            inference_time_ms=inference_time * 1000,  # Convert to ms
            memory_usage_mb=memory_usage / 1024**2,  # Convert to MB
            accuracy=accuracy
        )
        
        self.ai_metrics_history.append(metrics)
        return metrics
    
    def track_application_metrics(self, active_users: int, total_requests: int, 
                                successful_requests: int, avg_response_time: float) -> ApplicationMetrics:
        """Track application-level performance metrics"""
        failed_requests = total_requests - successful_requests
        
        metrics = ApplicationMetrics(
            timestamp=datetime.now().isoformat(),
            active_users=active_users,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_response_time * 1000,
            recommendation_requests=self.request_counter.get('recommendations', 0),
            upload_requests=self.request_counter.get('uploads', 0)
        )
        
        self.app_metrics_history.append(metrics)
        return metrics
    
    def check_alerts(self, system_metrics: SystemMetrics, ai_metrics: List[AIModelMetrics], 
                    app_metrics: ApplicationMetrics) -> List[str]:
        """Check for performance alerts based on thresholds"""
        alerts = []
        
        # System alerts
        if system_metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"🔴 High CPU usage: {system_metrics.cpu_percent:.1f}%")
        
        if system_metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"🔴 High memory usage: {system_metrics.memory_percent:.1f}%")
        
        if system_metrics.disk_percent > self.alert_thresholds['disk_percent']:
            alerts.append(f"🔴 High disk usage: {system_metrics.disk_percent:.1f}%")
        
        # AI model alerts
        for ai_metric in ai_metrics:
            if ai_metric.inference_time_ms > self.alert_thresholds['inference_time_ms']:
                alerts.append(f"🔴 Slow inference ({ai_metric.model_name}): {ai_metric.inference_time_ms:.1f}ms")
        
        # Application alerts
        if app_metrics.avg_response_time_ms > self.alert_thresholds['response_time_ms']:
            alerts.append(f"🔴 Slow response time: {app_metrics.avg_response_time_ms:.1f}ms")
        
        error_rate = app_metrics.failed_requests / max(app_metrics.total_requests, 1)
        if error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"🔴 High error rate: {error_rate:.2%}")
        
        return alerts
    
    def create_dashboard_layout(self) -> Layout:
        """Create rich dashboard layout"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="system"),
            Layout(name="ai_models")
        )
        
        layout["right"].split_column(
            Layout(name="application"),
            Layout(name="alerts")
        )
        
        return layout
    
    def update_dashboard(self, layout: Layout):
        """Update dashboard with current metrics"""
        # Collect current metrics
        system_metrics = self.collect_system_metrics()
        if not system_metrics:
            return
        
        # Header
        layout["header"].update(
            Panel(
                Text(f"FlashFit AI Performance Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                     style="bold blue"),
                border_style="blue"
            )
        )
        
        # System metrics table
        system_table = Table(title="System Metrics")
        system_table.add_column("Metric", style="cyan")
        system_table.add_column("Value", style="magenta")
        system_table.add_column("Status", style="green")
        
        system_table.add_row(
            "CPU Usage", 
            f"{system_metrics.cpu_percent:.1f}%",
            "🟢" if system_metrics.cpu_percent < 70 else "🟡" if system_metrics.cpu_percent < 90 else "🔴"
        )
        system_table.add_row(
            "Memory Usage", 
            f"{system_metrics.memory_percent:.1f}%",
            "🟢" if system_metrics.memory_percent < 70 else "🟡" if system_metrics.memory_percent < 90 else "🔴"
        )
        system_table.add_row(
            "Disk Usage", 
            f"{system_metrics.disk_percent:.1f}%",
            "🟢" if system_metrics.disk_percent < 80 else "🟡" if system_metrics.disk_percent < 95 else "🔴"
        )
        
        if system_metrics.gpu_available and system_metrics.gpu_memory_used:
            gpu_percent = (system_metrics.gpu_memory_used / system_metrics.gpu_memory_total) * 100
            system_table.add_row(
                "GPU Memory", 
                f"{gpu_percent:.1f}%",
                "🟢" if gpu_percent < 70 else "🟡" if gpu_percent < 90 else "🔴"
            )
        
        layout["system"].update(Panel(system_table, title="System", border_style="green"))
        
        # AI Models metrics
        ai_table = Table(title="AI Models")
        ai_table.add_column("Model", style="cyan")
        ai_table.add_column("Avg Inference (ms)", style="magenta")
        ai_table.add_column("Memory (MB)", style="yellow")
        
        # Calculate averages for recent AI metrics
        recent_ai_metrics = list(self.ai_metrics_history)[-10:]  # Last 10 metrics
        model_stats = defaultdict(list)
        
        for metric in recent_ai_metrics:
            model_stats[metric.model_name].append(metric)
        
        for model_name, metrics in model_stats.items():
            avg_inference = np.mean([m.inference_time_ms for m in metrics])
            avg_memory = np.mean([m.memory_usage_mb for m in metrics])
            ai_table.add_row(model_name, f"{avg_inference:.1f}", f"{avg_memory:.1f}")
        
        layout["ai_models"].update(Panel(ai_table, title="AI Models", border_style="blue"))
        
        # Application metrics
        app_table = Table(title="Application")
        app_table.add_column("Metric", style="cyan")
        app_table.add_column("Value", style="magenta")
        
        total_requests = sum(self.request_counter.values())
        total_errors = sum(self.error_counter.values())
        error_rate = (total_errors / max(total_requests, 1)) * 100
        
        app_table.add_row("Total Requests", str(total_requests))
        app_table.add_row("Error Rate", f"{error_rate:.2f}%")
        app_table.add_row("Avg Response Time", f"{np.mean(self.response_times) if self.response_times else 0:.1f}ms")
        
        layout["application"].update(Panel(app_table, title="Application", border_style="yellow"))
        
        # Alerts
        recent_app_metrics = list(self.app_metrics_history)[-1:]
        alerts = self.check_alerts(
            system_metrics, 
            recent_ai_metrics, 
            recent_app_metrics[0] if recent_app_metrics else ApplicationMetrics(
                timestamp=datetime.now().isoformat(),
                active_users=0, total_requests=total_requests, 
                successful_requests=total_requests - total_errors,
                failed_requests=total_errors, avg_response_time_ms=0,
                recommendation_requests=0, upload_requests=0
            )
        )
        
        alerts_text = "\n".join(alerts) if alerts else "🟢 All systems normal"
        layout["alerts"].update(Panel(alerts_text, title="Alerts", border_style="red" if alerts else "green"))
        
        # Footer
        layout["footer"].update(
            Panel(
                "Press Ctrl+C to stop monitoring | Data saved to: " + str(self.data_dir),
                border_style="dim"
            )
        )
    
    def start_monitoring(self, interval: float = 5.0):
        """Start real-time monitoring with rich dashboard"""
        self.is_monitoring = True
        layout = self.create_dashboard_layout()
        
        try:
            with Live(layout, refresh_per_second=1, screen=True):
                while self.is_monitoring:
                    self.update_dashboard(layout)
                    time.sleep(interval)
        except KeyboardInterrupt:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring and save data"""
        self.is_monitoring = False
        self.save_metrics_to_file()
        console.print("[yellow]Monitoring stopped[/yellow]")
    
    def save_metrics_to_file(self):
        """Save collected metrics to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save system metrics
        if self.system_metrics_history:
            system_file = self.data_dir / f"system_metrics_{timestamp}.json"
            with open(system_file, 'w') as f:
                data = [asdict(metric) for metric in self.system_metrics_history]
                if ujson:
                    ujson.dump(data, f, indent=2)
                else:
                    json.dump(data, f, indent=2)
        
        # Save AI metrics
        if self.ai_metrics_history:
            ai_file = self.data_dir / f"ai_metrics_{timestamp}.json"
            with open(ai_file, 'w') as f:
                data = [asdict(metric) for metric in self.ai_metrics_history]
                if ujson:
                    ujson.dump(data, f, indent=2)
                else:
                    json.dump(data, f, indent=2)
        
        console.print(f"[green]✓ Metrics saved to {self.data_dir}[/green]")
    
    def generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary report"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_duration': len(self.system_metrics_history) * 5,  # Assuming 5s intervals
            'system_summary': {},
            'ai_summary': {},
            'application_summary': {},
            'recommendations': []
        }
        
        # System summary
        if self.system_metrics_history:
            cpu_values = [m.cpu_percent for m in self.system_metrics_history]
            memory_values = [m.memory_percent for m in self.system_metrics_history]
            
            summary['system_summary'] = {
                'avg_cpu': np.mean(cpu_values),
                'max_cpu': max(cpu_values),
                'avg_memory': np.mean(memory_values),
                'max_memory': max(memory_values)
            }
        
        # AI summary
        if self.ai_metrics_history:
            inference_times = [m.inference_time_ms for m in self.ai_metrics_history]
            summary['ai_summary'] = {
                'avg_inference_time': np.mean(inference_times),
                'max_inference_time': max(inference_times),
                'total_inferences': len(self.ai_metrics_history)
            }
        
        # Generate recommendations
        if summary['system_summary'].get('avg_cpu', 0) > 70:
            summary['recommendations'].append("Consider CPU optimization or scaling")
        
        if summary['ai_summary'].get('avg_inference_time', 0) > 500:
            summary['recommendations'].append("AI model inference optimization needed")
        
        return summary

# Context manager for tracking AI model performance
class ModelPerformanceTracker:
    """Context manager for tracking AI model performance"""
    
    def __init__(self, monitor: PerformanceMonitor, model_name: str):
        self.monitor = monitor
        self.model_name = model_name
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            self.start_memory = torch.cuda.memory_allocated()
        else:
            self.start_memory = psutil.Process().memory_info().rss
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        inference_time = end_time - self.start_time
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() - self.start_memory
        else:
            memory_used = psutil.Process().memory_info().rss - self.start_memory
        
        self.monitor.track_ai_model_performance(
            self.model_name, inference_time, memory_used
        )

# Global instance
_performance_monitor = None

def get_performance_monitor(data_dir: str = "data/monitoring") -> PerformanceMonitor:
    """Get or create performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(data_dir)
    return _performance_monitor

if __name__ == "__main__":
    # Demo usage
    monitor = get_performance_monitor()
    
    console.print("[bold blue]Starting Performance Monitor Demo[/bold blue]")
    console.print("Press Ctrl+C to stop")
    
    try:
        monitor.start_monitoring(interval=2.0)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        summary = monitor.generate_performance_summary()
        console.print(Panel(json.dumps(summary, indent=2), title="Performance Summary"))