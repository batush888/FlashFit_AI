#!/usr/bin/env python3
"""
FlashFit AI Metrics Server
Separated HTTP server for Prometheus metrics exposition
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

# Optional imports with graceful fallback
try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info, REGISTRY
    from prometheus_client.core import CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    start_http_server = None
    Gauge = Counter = Histogram = Info = REGISTRY = CollectorRegistry = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from metrics_collector import MetricsCollector, SystemMetric
from alerting_engine import AlertingEngine

class PrometheusMetricsRegistry:
    """Registry for Prometheus metrics with dynamic creation"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        self.metrics: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus client not available")
            return
        
        # Initialize core FlashFit metrics
        self._initialize_core_metrics()
    
    def _initialize_core_metrics(self):
        """Initialize core system metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # System metrics
        self.metrics['cpu_usage'] = Gauge(
            'flashfit_cpu_usage_percent',
            'CPU Usage Percentage',
            registry=self.registry
        )
        
        self.metrics['memory_usage'] = Gauge(
            'flashfit_memory_usage_percent',
            'Memory Usage Percentage',
            registry=self.registry
        )
        
        self.metrics['memory_bytes'] = Gauge(
            'flashfit_memory_usage_bytes',
            'Memory Usage in Bytes',
            registry=self.registry
        )
        
        self.metrics['disk_usage'] = Gauge(
            'flashfit_disk_usage_percent',
            'Disk Usage Percentage',
            registry=self.registry
        )
        
        self.metrics['network_bytes_sent'] = Gauge(
            'flashfit_network_bytes_sent',
            'Network Bytes Sent',
            registry=self.registry
        )
        
        self.metrics['network_bytes_recv'] = Gauge(
            'flashfit_network_bytes_recv',
            'Network Bytes Received',
            registry=self.registry
        )
        
        self.metrics['process_count'] = Gauge(
            'flashfit_process_count',
            'Number of Running Processes',
            registry=self.registry
        )
        
        # Service health metrics
        self.metrics['service_health'] = Gauge(
            'flashfit_service_health_status',
            'Service Health Status (1=healthy, 0=unhealthy)',
            ['service'],
            registry=self.registry
        )
        
        self.metrics['service_response_time'] = Gauge(
            'flashfit_service_response_time_seconds',
            'Service Response Time in Seconds',
            ['service'],
            registry=self.registry
        )
        
        # Redis metrics
        self.metrics['redis_memory'] = Gauge(
            'flashfit_redis_memory_usage_bytes',
            'Redis Memory Usage in Bytes',
            registry=self.registry
        )
        
        self.metrics['redis_clients'] = Gauge(
            'flashfit_redis_connected_clients',
            'Redis Connected Clients',
            registry=self.registry
        )
        
        self.metrics['redis_commands'] = Gauge(
            'flashfit_redis_total_commands_processed',
            'Redis Total Commands Processed',
            registry=self.registry
        )
        
        self.metrics['redis_hits'] = Gauge(
            'flashfit_redis_keyspace_hits',
            'Redis Keyspace Hits',
            registry=self.registry
        )
        
        self.metrics['redis_misses'] = Gauge(
            'flashfit_redis_keyspace_misses',
            'Redis Keyspace Misses',
            registry=self.registry
        )
        
        # Alert metrics
        self.metrics['active_alerts'] = Gauge(
            'flashfit_active_alerts_total',
            'Number of Active Alerts',
            ['severity'],
            registry=self.registry
        )
        
        self.metrics['alerts_fired'] = Counter(
            'flashfit_alerts_fired_total',
            'Total Number of Alerts Fired',
            ['severity', 'rule_name'],
            registry=self.registry
        )
        
        # Monitoring system metrics
        self.metrics['metrics_collected'] = Counter(
            'flashfit_metrics_collected_total',
            'Total Number of Metrics Collected',
            registry=self.registry
        )
        
        self.metrics['collection_duration'] = Histogram(
            'flashfit_metrics_collection_duration_seconds',
            'Time spent collecting metrics',
            registry=self.registry
        )
        
        # System info
        self.metrics['system_info'] = Info(
            'flashfit_system_info',
            'FlashFit AI System Information',
            registry=self.registry
        )
        
        # Set system info
        self.metrics['system_info'].info({
            'version': '2.0.0',
            'component': 'monitoring',
            'build_date': datetime.utcnow().strftime('%Y-%m-%d'),
            'python_version': '3.13'
        })
    
    def update_metric(self, metric: SystemMetric):
        """Update a Prometheus metric from SystemMetric"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        metric_name = metric.metric_name.replace('flashfit_', '')
        
        if metric_name in self.metrics:
            prometheus_metric = self.metrics[metric_name]
            
            # Handle metrics with labels
            if metric.labels:
                if hasattr(prometheus_metric, 'labels'):
                    prometheus_metric.labels(**metric.labels).set(metric.value)
                else:
                    prometheus_metric.set(metric.value)
            else:
                prometheus_metric.set(metric.value)
        else:
            self.logger.debug(f"Unknown metric: {metric.metric_name}")
    
    def update_alert_metrics(self, alerting_engine: AlertingEngine):
        """Update alert-related metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Count active alerts by severity
        active_alerts = alerting_engine.get_active_alerts()
        severity_counts = {'critical': 0, 'warning': 0, 'info': 0}
        
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        # Update Prometheus metrics
        for severity, count in severity_counts.items():
            self.metrics['active_alerts'].labels(severity=severity).set(count)
    
    def get_metric_families(self):
        """Get all metric families for exposition"""
        if not PROMETHEUS_AVAILABLE:
            return []
        
        return self.registry.collect()

class MetricsServer:
    """HTTP server for metrics exposition with integrated collection and alerting"""
    
    def __init__(self, 
                 port: int = 9090,
                 host: str = '0.0.0.0',
                 redis_url: Optional[str] = None,
                 db_url: Optional[str] = None,
                 config: Dict[str, Any] = None):
        
        self.port = port
        self.host = host
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # Initialize components
        self.metrics_collector = MetricsCollector(redis_url=redis_url, db_url=db_url)
        self.alerting_engine = AlertingEngine(redis_url=redis_url, config=config)
        self.prometheus_registry = PrometheusMetricsRegistry()
        
        # Collection thread
        self.collection_thread = None
        self.collection_interval = self.config.get('collection_interval', 30)
        
        # Service endpoints for health checks
        self.service_endpoints = self.config.get('services', {
            'backend': 'http://localhost:8080',
            'frontend': 'http://localhost:3000'
        })
    
    def start(self):
        """Start the metrics server and collection loop"""
        if not PROMETHEUS_AVAILABLE:
            self.logger.error("Cannot start metrics server: prometheus_client not available")
            return False
        
        try:
            # Start Prometheus HTTP server
            start_http_server(self.port, addr=self.host, registry=self.prometheus_registry.registry)
            self.logger.info(f"Metrics server started on {self.host}:{self.port}")
            
            # Start metrics collection thread
            self.running = True
            self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self.collection_thread.start()
            
            self.logger.info(f"Metrics collection started (interval: {self.collection_interval}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
            return False
    
    def stop(self):
        """Stop the metrics server and collection"""
        self.running = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        # Close connections
        self.metrics_collector.close()
        self.alerting_engine.close()
        
        self.logger.info("Metrics server stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                start_time = time.time()
                
                # Collect all metrics
                all_metrics = []
                
                # System metrics
                system_metrics = self.metrics_collector.collect_system_metrics()
                all_metrics.extend(system_metrics)
                
                # Service health metrics
                service_metrics = self.metrics_collector.collect_service_health_metrics(self.service_endpoints)
                all_metrics.extend(service_metrics)
                
                # Redis metrics
                redis_metrics = self.metrics_collector.collect_redis_metrics()
                all_metrics.extend(redis_metrics)
                
                # Store metrics
                self.metrics_collector.store_metrics(all_metrics)
                
                # Update Prometheus metrics
                for metric in all_metrics:
                    self.prometheus_registry.update_metric(metric)
                    
                    # Evaluate for alerts
                    self.alerting_engine.evaluate_metric(
                        metric.metric_name,
                        metric.value,
                        metric.labels
                    )
                
                # Update alert metrics
                self.prometheus_registry.update_alert_metrics(self.alerting_engine)
                
                # Update collection metrics
                collection_time = time.time() - start_time
                if PROMETHEUS_AVAILABLE:
                    self.prometheus_registry.metrics['metrics_collected'].inc(len(all_metrics))
                    self.prometheus_registry.metrics['collection_duration'].observe(collection_time)
                
                self.logger.debug(f"Collected {len(all_metrics)} metrics in {collection_time:.3f}s")
                
                # Cleanup old data periodically
                if int(time.time()) % 3600 == 0:  # Every hour
                    self.metrics_collector.cleanup_old_metrics()
                    self.alerting_engine.cleanup_old_alerts()
                
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
            
            # Wait for next collection
            time.sleep(self.collection_interval)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        active_alerts = self.alerting_engine.get_active_alerts()
        latest_metrics = self.metrics_collector.get_latest_metrics()
        
        # Calculate health score based on active alerts
        critical_alerts = len([a for a in active_alerts if a.severity.value == 'critical'])
        warning_alerts = len([a for a in active_alerts if a.severity.value == 'warning'])
        
        health_score = max(0, 100 - (critical_alerts * 30) - (warning_alerts * 10))
        
        return {
            'status': 'healthy' if health_score >= 80 else 'degraded' if health_score >= 50 else 'unhealthy',
            'health_score': health_score,
            'active_alerts': {
                'total': len(active_alerts),
                'critical': critical_alerts,
                'warning': warning_alerts,
                'info': len(active_alerts) - critical_alerts - warning_alerts
            },
            'metrics': {
                'total_collected': len(latest_metrics),
                'last_collection': datetime.utcnow().isoformat()
            },
            'services': {
                name: 'healthy' if latest_metrics.get(f'flashfit_service_health_status', {}).get('value', 0) == 1 else 'unhealthy'
                for name in self.service_endpoints.keys()
            }
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for debugging"""
        latest_metrics = self.metrics_collector.get_latest_metrics()
        active_alerts = self.alerting_engine.get_active_alerts()
        
        return {
            'server': {
                'running': self.running,
                'port': self.port,
                'collection_interval': self.collection_interval
            },
            'metrics': {
                'latest_count': len(latest_metrics),
                'names': list(latest_metrics.keys())
            },
            'alerts': {
                'active_count': len(active_alerts),
                'rules_count': len(self.alerting_engine.alert_rules)
            },
            'components': {
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'redis_available': REDIS_AVAILABLE and self.metrics_collector.redis_client is not None,
                'postgres_available': self.metrics_collector.db_connection is not None
            }
        }

def create_metrics_server(port: int = 9090,
                         host: str = '0.0.0.0',
                         redis_url: str = None,
                         db_url: str = None,
                         config: Dict[str, Any] = None) -> MetricsServer:
    """Factory function to create a metrics server"""
    return MetricsServer(
        port=port,
        host=host,
        redis_url=redis_url,
        db_url=db_url,
        config=config
    )

if __name__ == "__main__":
    import signal
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Sample configuration
    config = {
        'collection_interval': 30,
        'services': {
            'backend': 'http://localhost:8080',
            'frontend': 'http://localhost:3000',
            'faiss-monitor': 'http://localhost:9091'
        },
        'notifications': {
            'log': {
                'enabled': True,
                'level': 'warning'
            }
        }
    }
    
    # Create and start server
    server = create_metrics_server(
        port=9090,
        redis_url="redis://localhost:6379/0",
        config=config
    )
    
    def signal_handler(signum, frame):
        print("\nShutting down metrics server...")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if server.start():
        print(f"✅ Metrics server running on http://localhost:{server.port}/metrics")
        print("Press Ctrl+C to stop")
        
        try:
            while server.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        print("❌ Failed to start metrics server")
        sys.exit(1)