#!/usr/bin/env python3
"""
FAISS Index Health Monitoring for FlashFit AI
Monitors FAISS vector store performance, health, and metrics
"""

import os
import sys
import time
import json
import logging
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: faiss-cpu not installed. Install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False
    faiss = None

try:
    from prometheus_client import Gauge, Counter, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    print("Warning: prometheus_client not installed. Install with: pip install prometheus_client")
    PROMETHEUS_AVAILABLE = False
    Gauge = Counter = Histogram = start_http_server = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    print("Warning: redis not installed. Install with: pip install redis")
    REDIS_AVAILABLE = False
    redis = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FAISSIndexStats:
    """Statistics for a FAISS index"""
    index_name: str
    total_vectors: int
    dimension: int
    index_size_bytes: int
    index_type: str
    is_trained: bool
    file_exists: bool
    last_modified: Optional[datetime]
    search_latency_ms: Optional[float]
    memory_usage_mb: Optional[float]
    
@dataclass
class FAISSHealthMetric:
    """Health metric for FAISS monitoring"""
    timestamp: datetime
    index_name: str
    metric_name: str
    metric_value: float
    status: str  # 'healthy', 'warning', 'critical'
    details: Dict[str, Any]

class FAISSMonitor:
    """
    Comprehensive FAISS index health monitoring system
    """
    
    def __init__(self, 
                 index_paths: Dict[str, str] = None,
                 redis_client: Optional[redis.Redis] = None,
                 prometheus_port: int = 9091):
        """
        Initialize FAISS monitor
        
        Args:
            index_paths: Dictionary mapping index names to file paths
            redis_client: Redis client for caching metrics
            prometheus_port: Port for Prometheus metrics server
        """
        self.index_paths = index_paths or self._get_default_index_paths()
        self.redis_client = redis_client
        self.prometheus_port = prometheus_port
        
        # Metrics storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.health_status = {}
        self.loaded_indices = {}
        
        # Performance tracking
        self.search_times = defaultdict(lambda: deque(maxlen=100))
        self.error_counts = defaultdict(int)
        
        # Initialize Prometheus metrics if available
        self._init_prometheus_metrics()
        
        logger.info(f"FAISS Monitor initialized for {len(self.index_paths)} indices")
        logger.info(f"Monitoring indices: {list(self.index_paths.keys())}")
    
    def _get_default_index_paths(self) -> Dict[str, str]:
        """Get default FAISS index paths"""
        base_path = Path(__file__).parent.parent
        return {
            'clip_fashion': str(base_path / 'data' / 'clip_fashion.index'),
            'blip_fashion': str(base_path / 'data' / 'blip_fashion.index'),
            'fashion_specific': str(base_path / 'data' / 'fashion_specific.index'),
            'user_embeddings': str(base_path / 'data' / 'user_embeddings.index')
        }
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, skipping metrics initialization")
            return
        
        # FAISS index metrics
        self.faiss_index_size = Gauge(
            'faiss_index_size_bytes', 
            'Size of FAISS index in bytes', 
            ['index_name']
        )
        
        self.faiss_vector_count = Gauge(
            'faiss_vector_count_total', 
            'Total number of vectors in FAISS index', 
            ['index_name']
        )
        
        self.faiss_search_latency = Histogram(
            'faiss_search_duration_seconds', 
            'FAISS search latency in seconds', 
            ['index_name'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.faiss_memory_usage = Gauge(
            'faiss_memory_usage_mb', 
            'Memory usage of FAISS index in MB', 
            ['index_name']
        )
        
        self.faiss_health_score = Gauge(
            'faiss_health_score', 
            'Health score of FAISS index (0-100)', 
            ['index_name']
        )
        
        self.faiss_error_count = Counter(
            'faiss_errors_total', 
            'Total number of FAISS errors', 
            ['index_name', 'error_type']
        )
        
        self.faiss_index_age = Gauge(
            'faiss_index_age_hours', 
            'Age of FAISS index file in hours', 
            ['index_name']
        )
        
        logger.info("Prometheus metrics initialized")
    
    def load_index(self, index_name: str) -> Optional[faiss.Index]:
        """Load FAISS index from disk"""
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available")
            return None
        
        if index_name not in self.index_paths:
            logger.error(f"Unknown index: {index_name}")
            return None
        
        index_path = self.index_paths[index_name]
        
        try:
            if not Path(index_path).exists():
                logger.warning(f"Index file does not exist: {index_path}")
                return None
            
            start_time = time.time()
            index = faiss.read_index(index_path)
            load_time = time.time() - start_time
            
            self.loaded_indices[index_name] = index
            logger.info(f"Loaded index {index_name} with {index.ntotal} vectors in {load_time:.3f}s")
            
            return index
            
        except Exception as e:
            logger.error(f"Failed to load index {index_name}: {e}")
            if PROMETHEUS_AVAILABLE:
                self.faiss_error_count.labels(index_name=index_name, error_type='load_error').inc()
            return None
    
    def get_index_stats(self, index_name: str) -> Optional[FAISSIndexStats]:
        """Get comprehensive statistics for a FAISS index"""
        if index_name not in self.index_paths:
            return None
        
        index_path = Path(self.index_paths[index_name])
        
        # File-based stats
        file_exists = index_path.exists()
        index_size_bytes = index_path.stat().st_size if file_exists else 0
        last_modified = datetime.fromtimestamp(index_path.stat().st_mtime) if file_exists else None
        
        # Load index for detailed stats
        index = self.loaded_indices.get(index_name) or self.load_index(index_name)
        
        if index is None:
            return FAISSIndexStats(
                index_name=index_name,
                total_vectors=0,
                dimension=0,
                index_size_bytes=index_size_bytes,
                index_type='unknown',
                is_trained=False,
                file_exists=file_exists,
                last_modified=last_modified,
                search_latency_ms=None,
                memory_usage_mb=None
            )
        
        # Index-based stats
        total_vectors = index.ntotal
        dimension = index.d
        index_type = type(index).__name__
        is_trained = index.is_trained
        
        # Calculate average search latency
        search_latency_ms = None
        if self.search_times[index_name]:
            search_latency_ms = np.mean(list(self.search_times[index_name])) * 1000
        
        # Estimate memory usage (rough approximation)
        memory_usage_mb = None
        if total_vectors > 0 and dimension > 0:
            # Rough estimate: vectors + overhead
            estimated_bytes = total_vectors * dimension * 4 + 1024 * 1024  # 4 bytes per float + 1MB overhead
            memory_usage_mb = estimated_bytes / (1024 * 1024)
        
        return FAISSIndexStats(
            index_name=index_name,
            total_vectors=total_vectors,
            dimension=dimension,
            index_size_bytes=index_size_bytes,
            index_type=index_type,
            is_trained=is_trained,
            file_exists=file_exists,
            last_modified=last_modified,
            search_latency_ms=search_latency_ms,
            memory_usage_mb=memory_usage_mb
        )
    
    def benchmark_search_performance(self, index_name: str, num_queries: int = 10, k: int = 10) -> Dict[str, float]:
        """Benchmark search performance for an index"""
        index = self.loaded_indices.get(index_name) or self.load_index(index_name)
        
        if index is None or index.ntotal == 0:
            return {'error': 'Index not available or empty'}
        
        try:
            # Generate random query vectors
            dimension = index.d
            query_vectors = np.random.random((num_queries, dimension)).astype(np.float32)
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
            query_vectors = query_vectors / norms
            
            # Benchmark search
            search_times = []
            
            for i in range(num_queries):
                start_time = time.time()
                distances, indices = index.search(query_vectors[i:i+1], k)
                search_time = time.time() - start_time
                search_times.append(search_time)
                
                # Store for metrics
                self.search_times[index_name].append(search_time)
            
            # Calculate statistics
            stats = {
                'mean_latency_ms': np.mean(search_times) * 1000,
                'median_latency_ms': np.median(search_times) * 1000,
                'p95_latency_ms': np.percentile(search_times, 95) * 1000,
                'p99_latency_ms': np.percentile(search_times, 99) * 1000,
                'min_latency_ms': np.min(search_times) * 1000,
                'max_latency_ms': np.max(search_times) * 1000,
                'queries_per_second': num_queries / np.sum(search_times)
            }
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                for search_time in search_times:
                    self.faiss_search_latency.labels(index_name=index_name).observe(search_time)
            
            logger.info(f"Benchmark completed for {index_name}: {stats['mean_latency_ms']:.2f}ms avg latency")
            
            return stats
            
        except Exception as e:
            logger.error(f"Benchmark failed for {index_name}: {e}")
            if PROMETHEUS_AVAILABLE:
                self.faiss_error_count.labels(index_name=index_name, error_type='benchmark_error').inc()
            return {'error': str(e)}
    
    def calculate_health_score(self, index_name: str) -> Tuple[float, Dict[str, Any]]:
        """Calculate health score for an index (0-100)"""
        stats = self.get_index_stats(index_name)
        
        if stats is None:
            return 0.0, {'error': 'Index not found'}
        
        score = 100.0
        details = {}
        
        # File existence (critical)
        if not stats.file_exists:
            score -= 50
            details['file_missing'] = True
        
        # Vector count (important)
        if stats.total_vectors == 0:
            score -= 30
            details['empty_index'] = True
        elif stats.total_vectors < 1000:
            score -= 10
            details['low_vector_count'] = stats.total_vectors
        
        # Search latency (performance)
        if stats.search_latency_ms is not None:
            if stats.search_latency_ms > 1000:  # > 1 second
                score -= 20
                details['high_latency'] = stats.search_latency_ms
            elif stats.search_latency_ms > 500:  # > 500ms
                score -= 10
                details['moderate_latency'] = stats.search_latency_ms
        
        # Index age (freshness)
        if stats.last_modified:
            age_hours = (datetime.now() - stats.last_modified).total_seconds() / 3600
            if age_hours > 168:  # > 1 week
                score -= 10
                details['stale_index'] = age_hours
            elif age_hours > 24:  # > 1 day
                score -= 5
                details['aging_index'] = age_hours
        
        # Memory usage (efficiency)
        if stats.memory_usage_mb is not None:
            if stats.memory_usage_mb > 1024:  # > 1GB
                score -= 5
                details['high_memory'] = stats.memory_usage_mb
        
        # Error rate
        error_count = self.error_counts.get(index_name, 0)
        if error_count > 10:
            score -= 15
            details['high_error_count'] = error_count
        elif error_count > 5:
            score -= 5
            details['moderate_error_count'] = error_count
        
        score = max(0.0, min(100.0, score))
        
        return score, details
    
    def update_prometheus_metrics(self):
        """Update all Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        for index_name in self.index_paths.keys():
            try:
                stats = self.get_index_stats(index_name)
                if stats is None:
                    continue
                
                # Update metrics
                self.faiss_index_size.labels(index_name=index_name).set(stats.index_size_bytes)
                self.faiss_vector_count.labels(index_name=index_name).set(stats.total_vectors)
                
                if stats.memory_usage_mb is not None:
                    self.faiss_memory_usage.labels(index_name=index_name).set(stats.memory_usage_mb)
                
                # Health score
                health_score, _ = self.calculate_health_score(index_name)
                self.faiss_health_score.labels(index_name=index_name).set(health_score)
                
                # Index age
                if stats.last_modified:
                    age_hours = (datetime.now() - stats.last_modified).total_seconds() / 3600
                    self.faiss_index_age.labels(index_name=index_name).set(age_hours)
                
            except Exception as e:
                logger.error(f"Failed to update metrics for {index_name}: {e}")
                self.faiss_error_count.labels(index_name=index_name, error_type='metrics_error').inc()
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics for all indices"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'indices': {},
            'summary': {
                'total_indices': len(self.index_paths),
                'healthy_indices': 0,
                'total_vectors': 0,
                'total_size_mb': 0,
                'average_health_score': 0
            }
        }
        
        health_scores = []
        
        for index_name in self.index_paths.keys():
            try:
                # Get stats
                stats = self.get_index_stats(index_name)
                if stats is None:
                    continue
                
                # Benchmark performance
                perf_stats = self.benchmark_search_performance(index_name, num_queries=5)
                
                # Calculate health
                health_score, health_details = self.calculate_health_score(index_name)
                
                # Store metrics
                metrics['indices'][index_name] = {
                    'stats': asdict(stats),
                    'performance': perf_stats,
                    'health_score': health_score,
                    'health_details': health_details
                }
                
                # Update summary
                if health_score >= 70:
                    metrics['summary']['healthy_indices'] += 1
                
                metrics['summary']['total_vectors'] += stats.total_vectors
                metrics['summary']['total_size_mb'] += stats.index_size_bytes / (1024 * 1024)
                health_scores.append(health_score)
                
            except Exception as e:
                logger.error(f"Failed to collect metrics for {index_name}: {e}")
                metrics['indices'][index_name] = {'error': str(e)}
        
        # Calculate average health score
        if health_scores:
            metrics['summary']['average_health_score'] = np.mean(health_scores)
        
        # Update Prometheus metrics
        self.update_prometheus_metrics()
        
        return metrics
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        logger.info("Generating FAISS health report...")
        
        metrics = self.collect_all_metrics()
        
        # Generate recommendations
        recommendations = []
        
        for index_name, index_metrics in metrics['indices'].items():
            if 'error' in index_metrics:
                recommendations.append(f"Fix error in {index_name}: {index_metrics['error']}")
                continue
            
            health_score = index_metrics.get('health_score', 0)
            health_details = index_metrics.get('health_details', {})
            
            if health_score < 50:
                recommendations.append(f"Critical: {index_name} health score is {health_score:.1f}")
            elif health_score < 70:
                recommendations.append(f"Warning: {index_name} health score is {health_score:.1f}")
            
            # Specific recommendations
            if health_details.get('file_missing'):
                recommendations.append(f"Rebuild missing index file for {index_name}")
            
            if health_details.get('empty_index'):
                recommendations.append(f"Populate empty index {index_name} with vectors")
            
            if health_details.get('high_latency'):
                recommendations.append(f"Optimize search performance for {index_name} (latency: {health_details['high_latency']:.1f}ms)")
            
            if health_details.get('stale_index'):
                recommendations.append(f"Update stale index {index_name} (age: {health_details['stale_index']:.1f} hours)")
        
        # System-level recommendations
        if metrics['summary']['average_health_score'] < 70:
            recommendations.append("Overall FAISS system health is below acceptable levels")
        
        if metrics['summary']['total_size_mb'] > 10240:  # > 10GB
            recommendations.append("Consider index optimization or cleanup - total size exceeds 10GB")
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'system_health': {
                'overall_score': metrics['summary']['average_health_score'],
                'status': 'healthy' if metrics['summary']['average_health_score'] >= 70 else 'degraded' if metrics['summary']['average_health_score'] >= 50 else 'critical'
            },
            'metrics': metrics,
            'recommendations': recommendations,
            'monitoring_info': {
                'faiss_available': FAISS_AVAILABLE,
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'redis_available': REDIS_AVAILABLE and self.redis_client is not None,
                'monitored_indices': list(self.index_paths.keys())
            }
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filepath: str = None):
        """Save health report to file"""
        if filepath is None:
            filepath = f"faiss_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Health report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def start_monitoring_server(self, port: int = None):
        """Start Prometheus metrics server"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available, cannot start metrics server")
            return False
        
        port = port or self.prometheus_port
        
        try:
            start_http_server(port)
            logger.info(f"FAISS monitoring server started on port {port}")
            logger.info(f"Metrics available at http://localhost:{port}/metrics")
            return True
        except Exception as e:
            logger.error(f"Failed to start monitoring server: {e}")
            return False

def create_faiss_monitor(redis_url: str = "redis://localhost:6379/0") -> FAISSMonitor:
    """Create FAISS monitor with Redis connection"""
    redis_client = None
    
    if REDIS_AVAILABLE:
        try:
            redis_client = redis.from_url(redis_url, decode_responses=False)
            redis_client.ping()
            logger.info("Connected to Redis for FAISS metrics caching")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
    
    return FAISSMonitor(redis_client=redis_client)

if __name__ == "__main__":
    print("🔍 FAISS Index Health Monitor")
    print("==============================\n")
    
    # Create monitor
    monitor = create_faiss_monitor()
    
    # Start metrics server
    if monitor.start_monitoring_server():
        print("✅ Metrics server started")
    
    # Generate health report
    print("📊 Generating health report...")
    report = monitor.generate_health_report()
    
    # Save report
    monitor.save_report(report)
    
    # Print summary
    print(f"\n📋 FAISS Health Summary:")
    print(f"   Overall Score: {report['system_health']['overall_score']:.1f}/100")
    print(f"   Status: {report['system_health']['status'].upper()}")
    print(f"   Monitored Indices: {len(report['metrics']['indices'])}")
    print(f"   Total Vectors: {report['metrics']['summary']['total_vectors']:,}")
    print(f"   Total Size: {report['metrics']['summary']['total_size_mb']:.1f} MB")
    
    if report['recommendations']:
        print(f"\n⚠️  Recommendations ({len(report['recommendations'])}):")
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"   {i}. {rec}")
        if len(report['recommendations']) > 5:
            print(f"   ... and {len(report['recommendations']) - 5} more")
    
    print("\n✅ FAISS monitoring completed!")
    print("\n📝 NEXT STEPS:")
    print("   1. Review detailed report in faiss_health_report_*.json")
    print("   2. Address any critical health issues")
    print("   3. Set up continuous monitoring with Prometheus")
    print("   4. Configure alerts for index health degradation")
    print("   5. Implement automated index rebuilding for critical failures")