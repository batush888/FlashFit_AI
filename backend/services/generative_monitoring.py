#!/usr/bin/env python3
"""
Generative AI Monitoring Service
Provides comprehensive monitoring, metrics, and health checks for generative components.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import threading

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for generative components."""
    timestamp: float
    component: str
    operation: str
    duration_ms: float
    success: bool
    error_message: Optional[str] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    model_version: str = "v1.0"
    batch_size: int = 1
    embedding_dimension: int = 512

@dataclass
class HealthStatus:
    """Health status for a component."""
    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    last_check: float
    response_time_ms: float
    error_rate: float
    uptime_seconds: float
    details: Dict[str, Any]

@dataclass
class SystemMetrics:
    """System-wide metrics."""
    timestamp: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    active_users: int
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_usage_percent: float
    model_accuracy: float
    feedback_score: float

class GenerativeMonitoringService:
    """Monitoring service for generative AI components."""
    
    def __init__(self, metrics_retention_hours: int = 24):
        self.metrics_retention_hours = metrics_retention_hours
        self.metrics_file = Path("data/monitoring/generative_metrics.json")
        self.health_file = Path("data/monitoring/generative_health.json")
        self.alerts_file = Path("data/monitoring/generative_alerts.json")
        
        # Create directories
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for recent metrics
        self.performance_metrics: deque = deque(maxlen=10000)
        self.health_statuses: Dict[str, HealthStatus] = {}
        self.system_metrics: deque = deque(maxlen=1000)
        self.alerts: deque = deque(maxlen=1000)
        
        # Component tracking
        self.component_start_times: Dict[str, float] = {}
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Generative monitoring service initialized")
    
    def record_performance_metric(
        self,
        component: str,
        operation: str,
        duration_ms: float,
        success: bool,
        error_message: Optional[str] = None,
        **kwargs
    ):
        """Record a performance metric."""
        try:
            # Get system metrics
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            cpu_usage = process.cpu_percent()
            
            metric = PerformanceMetrics(
                timestamp=time.time(),
                component=component,
                operation=operation,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                **kwargs
            )
            
            self.performance_metrics.append(metric)
            
            # Update component tracking
            self.request_counts[component] += 1
            if not success:
                self.error_counts[component] += 1
            
            self.response_times[component].append(duration_ms)
            
            # Check for alerts
            self._check_performance_alerts(metric)
            
        except Exception as e:
            logger.error(f"Error recording performance metric: {e}")
    
    def update_health_status(
        self,
        component: str,
        status: str,
        response_time_ms: float,
        details: Optional[Dict[str, Any]] = None
    ):
        """Update health status for a component."""
        try:
            current_time = time.time()
            
            # Calculate error rate
            total_requests = self.request_counts.get(component, 0)
            error_requests = self.error_counts.get(component, 0)
            error_rate = error_requests / max(total_requests, 1)
            
            # Calculate uptime
            start_time = self.component_start_times.get(component, current_time)
            uptime = current_time - start_time
            
            health_status = HealthStatus(
                component=component,
                status=status,
                last_check=current_time,
                response_time_ms=response_time_ms,
                error_rate=error_rate,
                uptime_seconds=uptime,
                details=details or {}
            )
            
            self.health_statuses[component] = health_status
            
            # Check for health alerts
            self._check_health_alerts(health_status)
            
        except Exception as e:
            logger.error(f"Error updating health status: {e}")
    
    def register_component(self, component: str):
        """Register a new component for monitoring."""
        self.component_start_times[component] = time.time()
        logger.info(f"Registered component for monitoring: {component}")
    
    def get_component_metrics(self, component: str, hours: int = 1) -> List[PerformanceMetrics]:
        """Get metrics for a specific component."""
        cutoff_time = time.time() - (hours * 3600)
        return [
            metric for metric in self.performance_metrics
            if metric.component == component and metric.timestamp >= cutoff_time
        ]
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system overview metrics."""
        try:
            current_time = time.time()
            cutoff_time = current_time - 3600  # Last hour
            
            # Filter recent metrics
            recent_metrics = [
                metric for metric in self.performance_metrics
                if metric.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {
                    "status": "no_data",
                    "message": "No recent metrics available"
                }
            
            # Calculate aggregated metrics
            total_requests = len(recent_metrics)
            successful_requests = sum(1 for m in recent_metrics if m.success)
            failed_requests = total_requests - successful_requests
            
            response_times = [m.duration_ms for m in recent_metrics]
            avg_response_time = sum(response_times) / len(response_times)
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            p95_response_time = sorted_times[p95_index] if sorted_times else 0
            p99_response_time = sorted_times[p99_index] if sorted_times else 0
            
            # System resources
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            disk_usage = psutil.disk_usage('/').percent
            
            return {
                "timestamp": current_time,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": successful_requests / max(total_requests, 1),
                "average_response_time_ms": avg_response_time,
                "p95_response_time_ms": p95_response_time,
                "p99_response_time_ms": p99_response_time,
                "memory_usage_percent": memory_usage,
                "cpu_usage_percent": cpu_usage,
                "disk_usage_percent": disk_usage,
                "active_components": len(self.health_statuses),
                "healthy_components": sum(1 for h in self.health_statuses.values() if h.status == 'healthy'),
                "component_health": {name: status.status for name, status in self.health_statuses.items()}
            }
            
        except Exception as e:
            logger.error(f"Error getting system overview: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_component_health(self, component: str) -> Optional[Dict[str, Any]]:
        """Get health status for a specific component."""
        health_status = self.health_statuses.get(component)
        if not health_status:
            return None
        
        return {
            "component": health_status.component,
            "status": health_status.status,
            "last_check": health_status.last_check,
            "response_time_ms": health_status.response_time_ms,
            "error_rate": health_status.error_rate,
            "uptime_seconds": health_status.uptime_seconds,
            "uptime_hours": health_status.uptime_seconds / 3600,
            "details": health_status.details,
            "total_requests": self.request_counts.get(component, 0),
            "error_requests": self.error_counts.get(component, 0)
        }
    
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        cutoff_time = time.time() - (hours * 3600)
        return [
            alert for alert in self.alerts
            if alert.get('timestamp', 0) >= cutoff_time
        ]
    
    def _check_performance_alerts(self, metric: PerformanceMetrics):
        """Check for performance-based alerts."""
        alerts = []
        
        # High response time alert
        if metric.duration_ms > 5000:  # 5 seconds
            alerts.append({
                "type": "high_response_time",
                "severity": "warning",
                "component": metric.component,
                "operation": metric.operation,
                "value": metric.duration_ms,
                "threshold": 5000,
                "message": f"High response time detected: {metric.duration_ms:.2f}ms"
            })
        
        # High memory usage alert
        if metric.memory_usage_mb > 1000:  # 1GB
            alerts.append({
                "type": "high_memory_usage",
                "severity": "warning",
                "component": metric.component,
                "value": metric.memory_usage_mb,
                "threshold": 1000,
                "message": f"High memory usage detected: {metric.memory_usage_mb:.2f}MB"
            })
        
        # Error alert
        if not metric.success:
            alerts.append({
                "type": "operation_error",
                "severity": "error",
                "component": metric.component,
                "operation": metric.operation,
                "error": metric.error_message,
                "message": f"Operation failed: {metric.error_message}"
            })
        
        # Add alerts with timestamp
        for alert in alerts:
            alert["timestamp"] = metric.timestamp
            alert["id"] = f"{alert['type']}_{metric.component}_{int(metric.timestamp)}"
            self.alerts.append(alert)
    
    def _check_health_alerts(self, health_status: HealthStatus):
        """Check for health-based alerts."""
        alerts = []
        
        # Unhealthy component alert
        if health_status.status == 'unhealthy':
            alerts.append({
                "type": "component_unhealthy",
                "severity": "critical",
                "component": health_status.component,
                "message": f"Component {health_status.component} is unhealthy"
            })
        
        # High error rate alert
        if health_status.error_rate > 0.1:  # 10% error rate
            alerts.append({
                "type": "high_error_rate",
                "severity": "warning",
                "component": health_status.component,
                "value": health_status.error_rate,
                "threshold": 0.1,
                "message": f"High error rate detected: {health_status.error_rate:.2%}"
            })
        
        # Add alerts with timestamp
        for alert in alerts:
            alert["timestamp"] = health_status.last_check
            alert["id"] = f"{alert['type']}_{health_status.component}_{int(health_status.last_check)}"
            self.alerts.append(alert)
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics every 60 seconds
                self._collect_system_metrics()
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                # Save metrics to disk
                self._save_metrics_to_disk()
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _collect_system_metrics(self):
        """Collect system-wide metrics."""
        try:
            current_time = time.time()
            
            # Calculate metrics from recent performance data
            cutoff_time = current_time - 3600  # Last hour
            recent_metrics = [
                metric for metric in self.performance_metrics
                if metric.timestamp >= cutoff_time
            ]
            
            if recent_metrics:
                total_requests = len(recent_metrics)
                successful_requests = sum(1 for m in recent_metrics if m.success)
                failed_requests = total_requests - successful_requests
                
                response_times = [m.duration_ms for m in recent_metrics]
                avg_response_time = sum(response_times) / len(response_times)
                
                sorted_times = sorted(response_times)
                p95_index = int(0.95 * len(sorted_times))
                p99_index = int(0.99 * len(sorted_times))
                p95_response_time = sorted_times[p95_index] if sorted_times else 0
                p99_response_time = sorted_times[p99_index] if sorted_times else 0
            else:
                total_requests = successful_requests = failed_requests = 0
                avg_response_time = p95_response_time = p99_response_time = 0
            
            # System resources
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            disk_usage = psutil.disk_usage('/').percent
            
            system_metric = SystemMetrics(
                timestamp=current_time,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                average_response_time_ms=avg_response_time,
                p95_response_time_ms=p95_response_time,
                p99_response_time_ms=p99_response_time,
                active_users=0,  # Would need session tracking
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                disk_usage_percent=disk_usage,
                model_accuracy=0.0,  # Would need model evaluation
                feedback_score=0.0   # Would need feedback analysis
            )
            
            self.system_metrics.append(system_metric)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory bloat."""
        try:
            cutoff_time = time.time() - (self.metrics_retention_hours * 3600)
            
            # Clean performance metrics
            while (self.performance_metrics and 
                   self.performance_metrics[0].timestamp < cutoff_time):
                self.performance_metrics.popleft()
            
            # Clean system metrics
            while (self.system_metrics and 
                   self.system_metrics[0].timestamp < cutoff_time):
                self.system_metrics.popleft()
            
            # Clean alerts
            while (self.alerts and 
                   self.alerts[0].get('timestamp', 0) < cutoff_time):
                self.alerts.popleft()
                
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")
    
    def _save_metrics_to_disk(self):
        """Save metrics to disk for persistence."""
        try:
            # Save recent performance metrics
            recent_metrics = list(self.performance_metrics)[-1000:]  # Last 1000 metrics
            metrics_data = {
                "performance_metrics": [asdict(m) for m in recent_metrics],
                "system_metrics": [asdict(m) for m in list(self.system_metrics)[-100:]],
                "health_statuses": {k: asdict(v) for k, v in self.health_statuses.items()},
                "alerts": list(self.alerts)[-100:],
                "last_updated": time.time()
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics to disk: {e}")
    
    def stop_monitoring(self):
        """Stop the monitoring service."""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("Generative monitoring service stopped")

# Global monitoring service instance
_monitoring_service = None

def get_generative_monitoring_service() -> GenerativeMonitoringService:
    """Get or create the global monitoring service instance."""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = GenerativeMonitoringService()
    return _monitoring_service

# Context manager for performance monitoring
class monitor_performance:
    """Context manager for monitoring operation performance."""
    
    def __init__(self, component: str, operation: str, **kwargs):
        self.component = component
        self.operation = operation
        self.kwargs = kwargs
        self.start_time = None
        self.monitoring_service = get_generative_monitoring_service()
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        success = exc_type is None
        error_message = str(exc_val) if exc_val else None
        
        self.monitoring_service.record_performance_metric(
            component=self.component,
            operation=self.operation,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            **self.kwargs
        )