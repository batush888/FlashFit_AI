#!/usr/bin/env python3
"""
FlashFit AI Metrics Collector
Separated metrics collection logic for production scalability
"""

import time
import logging
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# Optional imports with graceful fallback
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

@dataclass
class SystemMetric:
    """System-level metric data structure"""
    timestamp: str
    metric_name: str
    value: float
    unit: str
    labels: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_prometheus_format(self) -> str:
        """Convert to Prometheus exposition format"""
        labels_str = ""
        if self.labels:
            label_pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
            labels_str = "{" + ",".join(label_pairs) + "}"
        
        return f"{self.metric_name}{labels_str} {self.value}"

class MetricsCollector:
    """Centralized metrics collection service"""
    
    def __init__(self, redis_url: Optional[str] = None, db_url: Optional[str] = None):
        self.redis_client = None
        self.db_connection = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize Redis connection
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                self.logger.info("Connected to Redis for metrics storage")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {e}")
                self.redis_client = None
        
        # Initialize PostgreSQL connection
        if POSTGRES_AVAILABLE and db_url:
            try:
                self.db_connection = psycopg2.connect(db_url)
                self.logger.info("Connected to PostgreSQL for metrics storage")
            except Exception as e:
                self.logger.warning(f"Failed to connect to PostgreSQL: {e}")
                self.db_connection = None
    
    def collect_system_metrics(self) -> List[SystemMetric]:
        """Collect comprehensive system metrics"""
        metrics = []
        timestamp = datetime.utcnow().isoformat()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(SystemMetric(
                timestamp=timestamp,
                metric_name="flashfit_cpu_usage_percent",
                value=cpu_percent,
                unit="percent"
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(SystemMetric(
                timestamp=timestamp,
                metric_name="flashfit_memory_usage_percent",
                value=memory.percent,
                unit="percent"
            ))
            
            metrics.append(SystemMetric(
                timestamp=timestamp,
                metric_name="flashfit_memory_usage_bytes",
                value=memory.used,
                unit="bytes"
            ))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.append(SystemMetric(
                timestamp=timestamp,
                metric_name="flashfit_disk_usage_percent",
                value=(disk.used / disk.total) * 100,
                unit="percent"
            ))
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics.append(SystemMetric(
                timestamp=timestamp,
                metric_name="flashfit_network_bytes_sent",
                value=network.bytes_sent,
                unit="bytes"
            ))
            
            metrics.append(SystemMetric(
                timestamp=timestamp,
                metric_name="flashfit_network_bytes_recv",
                value=network.bytes_recv,
                unit="bytes"
            ))
            
            # Process metrics
            process_count = len(psutil.pids())
            metrics.append(SystemMetric(
                timestamp=timestamp,
                metric_name="flashfit_process_count",
                value=process_count,
                unit="count"
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def collect_service_health_metrics(self, services: Dict[str, str]) -> List[SystemMetric]:
        """Collect health metrics for configured services"""
        metrics = []
        timestamp = datetime.utcnow().isoformat()
        
        if not REQUESTS_AVAILABLE:
            self.logger.warning("requests library not available, skipping service health checks")
            return metrics
        
        for service_name, service_url in services.items():
            try:
                start_time = time.time()
                response = requests.get(f"{service_url}/health", timeout=5)
                response_time = time.time() - start_time
                
                # Response time metric
                metrics.append(SystemMetric(
                    timestamp=timestamp,
                    metric_name="flashfit_service_response_time_seconds",
                    value=response_time,
                    unit="seconds",
                    labels={"service": service_name}
                ))
                
                # Health status metric (1 for healthy, 0 for unhealthy)
                health_status = 1 if response.status_code == 200 else 0
                metrics.append(SystemMetric(
                    timestamp=timestamp,
                    metric_name="flashfit_service_health_status",
                    value=health_status,
                    unit="boolean",
                    labels={"service": service_name}
                ))
                
            except Exception as e:
                self.logger.error(f"Error checking health for {service_name}: {e}")
                # Mark service as unhealthy
                metrics.append(SystemMetric(
                    timestamp=timestamp,
                    metric_name="flashfit_service_health_status",
                    value=0,
                    unit="boolean",
                    labels={"service": service_name}
                ))
        
        return metrics
    
    def collect_redis_metrics(self) -> List[SystemMetric]:
        """Collect Redis-specific metrics"""
        metrics = []
        timestamp = datetime.utcnow().isoformat()
        
        if not self.redis_client:
            return metrics
        
        try:
            info = self.redis_client.info()
            
            # Memory usage
            metrics.append(SystemMetric(
                timestamp=timestamp,
                metric_name="flashfit_redis_memory_usage_bytes",
                value=info.get('used_memory', 0),
                unit="bytes"
            ))
            
            # Connected clients
            metrics.append(SystemMetric(
                timestamp=timestamp,
                metric_name="flashfit_redis_connected_clients",
                value=info.get('connected_clients', 0),
                unit="count"
            ))
            
            # Total commands processed
            metrics.append(SystemMetric(
                timestamp=timestamp,
                metric_name="flashfit_redis_total_commands_processed",
                value=info.get('total_commands_processed', 0),
                unit="count"
            ))
            
            # Keyspace hits/misses
            metrics.append(SystemMetric(
                timestamp=timestamp,
                metric_name="flashfit_redis_keyspace_hits",
                value=info.get('keyspace_hits', 0),
                unit="count"
            ))
            
            metrics.append(SystemMetric(
                timestamp=timestamp,
                metric_name="flashfit_redis_keyspace_misses",
                value=info.get('keyspace_misses', 0),
                unit="count"
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting Redis metrics: {e}")
        
        return metrics
    
    def store_metrics(self, metrics: List[SystemMetric]):
        """Store metrics in configured backends"""
        # Store in Redis (for real-time access)
        if self.redis_client:
            try:
                for metric in metrics:
                    key = f"metrics:{metric.metric_name}:{metric.timestamp}"
                    self.redis_client.setex(key, 3600, json.dumps(metric.to_dict()))  # 1 hour TTL
                    
                    # Also store latest value for quick access
                    latest_key = f"metrics:latest:{metric.metric_name}"
                    self.redis_client.set(latest_key, json.dumps(metric.to_dict()))
                    
            except Exception as e:
                self.logger.error(f"Error storing metrics in Redis: {e}")
        
        # Store in PostgreSQL (for historical analysis)
        if self.db_connection:
            try:
                cursor = self.db_connection.cursor()
                for metric in metrics:
                    cursor.execute(
                        """
                        INSERT INTO metrics (timestamp, metric_name, value, unit, labels)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            metric.timestamp,
                            metric.metric_name,
                            metric.value,
                            metric.unit,
                            json.dumps(metric.labels) if metric.labels else None
                        )
                    )
                self.db_connection.commit()
                cursor.close()
                
            except Exception as e:
                self.logger.error(f"Error storing metrics in PostgreSQL: {e}")
    
    def get_latest_metrics(self, metric_names: List[str] = None) -> Dict[str, SystemMetric]:
        """Retrieve latest metrics from storage"""
        latest_metrics = {}
        
        if not self.redis_client:
            return latest_metrics
        
        try:
            if metric_names:
                keys = [f"metrics:latest:{name}" for name in metric_names]
            else:
                keys = self.redis_client.keys("metrics:latest:*")
            
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    metric_data = json.loads(data)
                    metric = SystemMetric(**metric_data)
                    latest_metrics[metric.metric_name] = metric
                    
        except Exception as e:
            self.logger.error(f"Error retrieving latest metrics: {e}")
        
        return latest_metrics
    
    def cleanup_old_metrics(self, retention_hours: int = 24):
        """Clean up old metrics to prevent storage bloat"""
        if self.redis_client:
            try:
                # Clean up time-series metrics older than retention period
                cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)
                pattern = "metrics:*"
                
                for key in self.redis_client.scan_iter(match=pattern):
                    if ":latest:" not in key.decode():  # Don't delete latest metrics
                        # Extract timestamp from key and check if it's old
                        try:
                            key_parts = key.decode().split(":")
                            if len(key_parts) >= 3:
                                timestamp_str = key_parts[-1]
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                if timestamp < cutoff_time:
                                    self.redis_client.delete(key)
                        except Exception:
                            continue  # Skip malformed keys
                            
            except Exception as e:
                self.logger.error(f"Error cleaning up old metrics: {e}")
    
    def close(self):
        """Close database connections"""
        if self.redis_client:
            self.redis_client.close()
        
        if self.db_connection:
            self.db_connection.close()

def create_metrics_collector(redis_url: str = None, db_url: str = None) -> MetricsCollector:
    """Factory function to create a metrics collector with optional backends"""
    return MetricsCollector(redis_url=redis_url, db_url=db_url)

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    collector = create_metrics_collector(
        redis_url="redis://localhost:6379/0",
        db_url="postgresql://user:password@localhost:5432/flashfit"
    )
    
    # Collect and store metrics
    system_metrics = collector.collect_system_metrics()
    redis_metrics = collector.collect_redis_metrics()
    
    all_metrics = system_metrics + redis_metrics
    collector.store_metrics(all_metrics)
    
    print(f"Collected and stored {len(all_metrics)} metrics")
    
    # Retrieve latest metrics
    latest = collector.get_latest_metrics()
    print(f"Latest metrics: {len(latest)} available")
    
    collector.close()