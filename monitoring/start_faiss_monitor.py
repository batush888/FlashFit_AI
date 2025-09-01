#!/usr/bin/env python3
"""
FAISS Monitor Startup Script
Integrates FAISS monitoring with the FlashFit AI monitoring system
"""

import os
import sys
import time
import yaml
import logging
import argparse
import threading
from pathlib import Path
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

try:
    from faiss_monitor import FAISSMonitor, create_faiss_monitor
except ImportError as e:
    print(f"Error importing FAISS monitor: {e}")
    print("Make sure faiss_monitor.py is in the same directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FAISSMonitorService:
    """
    Service wrapper for FAISS monitoring integration
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize FAISS monitor service
        
        Args:
            config_path: Path to FAISS monitoring configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.monitor = None
        self.running = False
        self.metrics_thread = None
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        return os.path.join(os.path.dirname(__file__), 'faiss_config.yaml')
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded FAISS monitoring configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails"""
        return {
            'indices': {
                'clip_fashion': {'path': '../data/clip_fashion.index'},
                'blip_fashion': {'path': '../data/blip_fashion.index'},
                'fashion_specific': {'path': '../data/fashion_specific.index'},
                'user_embeddings': {'path': '../data/user_embeddings.index'}
            },
            'monitoring': {
                'collection_interval': 60,
                'benchmark': {'enabled': True, 'num_queries': 10}
            },
            'prometheus': {'enabled': True, 'port': 9091},
            'redis': {'enabled': True, 'url': 'redis://localhost:6379/0'}
        }
    
    def _setup_index_paths(self) -> Dict[str, str]:
        """Setup index paths from configuration"""
        base_path = Path(__file__).parent.parent
        index_paths = {}
        
        for index_name, index_config in self.config.get('indices', {}).items():
            path = index_config.get('path', '')
            if not os.path.isabs(path):
                path = str(base_path / path)
            index_paths[index_name] = path
        
        return index_paths
    
    def start(self):
        """Start FAISS monitoring service"""
        logger.info("Starting FAISS monitoring service...")
        
        try:
            # Setup index paths
            index_paths = self._setup_index_paths()
            
            # Create Redis client if enabled
            redis_client = None
            if self.config.get('redis', {}).get('enabled', False):
                redis_url = self.config['redis'].get('url', 'redis://localhost:6379/0')
                try:
                    import redis
                    redis_client = redis.from_url(redis_url, decode_responses=False)
                    redis_client.ping()
                    logger.info("Connected to Redis for FAISS metrics")
                except Exception as e:
                    logger.warning(f"Redis connection failed: {e}")
            
            # Create FAISS monitor
            prometheus_port = self.config.get('prometheus', {}).get('port', 9091)
            self.monitor = FAISSMonitor(
                index_paths=index_paths,
                redis_client=redis_client,
                prometheus_port=prometheus_port
            )
            
            # Start Prometheus metrics server
            if self.config.get('prometheus', {}).get('enabled', True):
                if self.monitor.start_monitoring_server():
                    logger.info(f"FAISS metrics server started on port {prometheus_port}")
                else:
                    logger.error("Failed to start metrics server")
            
            # Start metrics collection thread
            self.running = True
            self.metrics_thread = threading.Thread(target=self._metrics_collection_loop)
            self.metrics_thread.daemon = True
            self.metrics_thread.start()
            
            logger.info("FAISS monitoring service started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start FAISS monitoring service: {e}")
            raise
    
    def _metrics_collection_loop(self):
        """Background thread for continuous metrics collection"""
        collection_interval = self.config.get('monitoring', {}).get('collection_interval', 60)
        benchmark_enabled = self.config.get('monitoring', {}).get('benchmark', {}).get('enabled', True)
        
        logger.info(f"Starting metrics collection loop (interval: {collection_interval}s)")
        
        while self.running:
            try:
                # Update Prometheus metrics
                if self.monitor:
                    self.monitor.update_prometheus_metrics()
                    
                    # Run benchmarks periodically
                    if benchmark_enabled and int(time.time()) % 900 == 0:  # Every 15 minutes
                        logger.info("Running performance benchmarks...")
                        for index_name in self.monitor.index_paths.keys():
                            self.monitor.benchmark_search_performance(index_name, num_queries=5)
                
                time.sleep(collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(collection_interval)
    
    def stop(self):
        """Stop FAISS monitoring service"""
        logger.info("Stopping FAISS monitoring service...")
        
        self.running = False
        
        if self.metrics_thread and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=5)
        
        logger.info("FAISS monitoring service stopped")
    
    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        if not self.monitor:
            logger.error("Monitor not initialized")
            return {}
        
        logger.info("Generating FAISS health report...")
        report = self.monitor.generate_health_report()
        
        if save_to_file:
            self.monitor.save_report(report)
        
        return report
    
    def get_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return {
            'running': self.running,
            'monitor_initialized': self.monitor is not None,
            'config_loaded': bool(self.config),
            'metrics_thread_alive': self.metrics_thread.is_alive() if self.metrics_thread else False,
            'prometheus_port': self.config.get('prometheus', {}).get('port', 9091),
            'monitored_indices': list(self.config.get('indices', {}).keys())
        }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='FAISS Index Health Monitor')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--port', '-p', type=int, default=9091, help='Prometheus metrics port')
    parser.add_argument('--report-only', action='store_true', help='Generate report and exit')
    parser.add_argument('--daemon', '-d', action='store_true', help='Run as daemon')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🔍 FlashFit AI - FAISS Index Health Monitor")
    print("==========================================\n")
    
    try:
        # Create service
        service = FAISSMonitorService(config_path=args.config)
        
        # Override port if specified
        if args.port != 9091:
            service.config.setdefault('prometheus', {})['port'] = args.port
        
        if args.report_only:
            # Generate report only
            print("📊 Generating health report...")
            service.start()
            report = service.generate_report()
            
            # Print summary
            print(f"\n📋 FAISS Health Summary:")
            print(f"   Overall Score: {report['system_health']['overall_score']:.1f}/100")
            print(f"   Status: {report['system_health']['status'].upper()}")
            print(f"   Monitored Indices: {len(report['metrics']['indices'])}")
            
            if report['recommendations']:
                print(f"\n⚠️  Recommendations ({len(report['recommendations'])}):")
                for i, rec in enumerate(report['recommendations'][:3], 1):
                    print(f"   {i}. {rec}")
            
            service.stop()
            
        else:
            # Start monitoring service
            service.start()
            
            # Print status
            status = service.get_status()
            print(f"✅ Service Status:")
            print(f"   Running: {status['running']}")
            print(f"   Prometheus Port: {status['prometheus_port']}")
            print(f"   Monitored Indices: {len(status['monitored_indices'])}")
            print(f"   Metrics URL: http://localhost:{status['prometheus_port']}/metrics")
            
            if args.daemon:
                print("\n🔄 Running in daemon mode...")
                print("   Press Ctrl+C to stop")
                
                try:
                    while True:
                        time.sleep(60)
                        # Generate periodic reports
                        if int(time.time()) % 3600 == 0:  # Every hour
                            service.generate_report()
                            
                except KeyboardInterrupt:
                    print("\n🛑 Stopping service...")
                    service.stop()
            else:
                print("\n📊 Generating initial health report...")
                service.generate_report()
                print("\n✅ FAISS monitoring setup completed!")
                print("\n📝 NEXT STEPS:")
                print("   1. Check metrics at http://localhost:9091/metrics")
                print("   2. Configure Prometheus to scrape FAISS metrics")
                print("   3. Set up Grafana dashboards for visualization")
                print("   4. Configure alerting rules in AlertManager")
                print("   5. Run with --daemon flag for continuous monitoring")
                
                service.stop()
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()