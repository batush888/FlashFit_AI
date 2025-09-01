#!/usr/bin/env python3
"""
FlashFit AI Modular Monitoring System Setup
Production-ready monitoring with separated concerns
"""

import os
import sys
import logging
import signal
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Standard library imports
import json
import subprocess

# Optional imports with graceful fallback
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    print("Warning: PyYAML not installed. Install with: pip install PyYAML")
    YAML_AVAILABLE = False
    yaml = None

try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    DOTENV_AVAILABLE = False

# Import our modular components
try:
    from metrics_server import create_metrics_server
    from metrics_collector import create_metrics_collector
    from alerting_engine import create_alerting_engine
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import monitoring components: {e}")
    COMPONENTS_AVAILABLE = False

class ModularMonitoringSystem:
    """Modular monitoring system with separated concerns"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "monitoring_config.yaml"
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Components
        self.metrics_server = None
        self.running = False
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with fallback to defaults"""
        default_config = {
            'server': {
                'host': '0.0.0.0',
                'port': 9090,
                'collection_interval': 30
            },
            'storage': {
                'redis': {
                    'enabled': True,
                    'url': 'redis://localhost:6379/0'
                },
                'postgresql': {
                    'enabled': False
                }
            },
            'services': {
                'backend': {'url': 'http://localhost:8080'},
                'frontend': {'url': 'http://localhost:3000'}
            },
            'alerting': {'enabled': True},
            'notifications': {
                'log': {'enabled': True, 'level': 'warning'}
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        if not YAML_AVAILABLE:
            self.logger.warning("YAML not available, using default configuration")
            return default_config
        
        config_file = Path(self.config_path)
        if not config_file.exists():
            print(f"Warning: Config file {self.config_path} not found, using defaults")
            return default_config
        
        try:
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
                
            # Merge with defaults
            return self._merge_configs(default_config, loaded_config)
            
        except Exception as e:
            print(f"Error loading config file: {e}")
            return default_config
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # File logging if configured
        file_config = log_config.get('file', {})
        if file_config.get('enabled', False):
            try:
                from logging.handlers import RotatingFileHandler
                
                log_file = Path(file_config.get('path', '/tmp/flashfit_monitoring.log'))
                log_file.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=file_config.get('max_size_mb', 100) * 1024 * 1024,
                    backupCount=file_config.get('backup_count', 5)
                )
                file_handler.setFormatter(logging.Formatter(log_config.get('format')))
                
                root_logger = logging.getLogger()
                root_logger.addHandler(file_handler)
                
            except Exception as e:
                print(f"Warning: Could not setup file logging: {e}")
        
        return logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        missing_deps = []
        
        if not COMPONENTS_AVAILABLE:
            missing_deps.append("monitoring components (metrics_server, metrics_collector, alerting_engine)")
        
        # Check optional dependencies based on configuration
        storage_config = self.config.get('storage', {})
        
        if storage_config.get('redis', {}).get('enabled', False):
            try:
                import redis
            except ImportError:
                missing_deps.append("redis (pip install redis)")
        
        if storage_config.get('postgresql', {}).get('enabled', False):
            try:
                import psycopg2
            except ImportError:
                missing_deps.append("psycopg2 (pip install psycopg2-binary)")
        
        # Check for Prometheus client
        try:
            import prometheus_client
        except ImportError:
            missing_deps.append("prometheus_client (pip install prometheus_client)")
        
        # Check for system monitoring
        try:
            import psutil
        except ImportError:
            missing_deps.append("psutil (pip install psutil)")
        
        if missing_deps:
            self.logger.error("Missing dependencies:")
            for dep in missing_deps:
                self.logger.error(f"  - {dep}")
            return False
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install missing dependencies"""
        required_packages = [
            'prometheus_client',
            'psutil',
            'redis',
            'PyYAML',
            'python-dotenv'
        ]
        
        optional_packages = {
            'psycopg2-binary': 'PostgreSQL support',
            'requests': 'HTTP health checks',
            'rich': 'Enhanced console output'
        }
        
        try:
            # Install required packages
            for package in required_packages:
                self.logger.info(f"Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            
            # Install optional packages
            for package, description in optional_packages.items():
                try:
                    self.logger.info(f"Installing {package} ({description})...")
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                except subprocess.CalledProcessError:
                    self.logger.warning(f"Failed to install optional package {package}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def start(self) -> bool:
        """Start the modular monitoring system"""
        if not COMPONENTS_AVAILABLE:
            self.logger.error("Cannot start: monitoring components not available")
            return False
        
        try:
            # Extract configuration
            server_config = self.config.get('server', {})
            storage_config = self.config.get('storage', {})
            
            # Build connection URLs
            redis_url = None
            if storage_config.get('redis', {}).get('enabled', False):
                redis_url = storage_config['redis'].get('url')
            
            db_url = None
            if storage_config.get('postgresql', {}).get('enabled', False):
                db_url = storage_config['postgresql'].get('url')
            
            # Create metrics server with integrated components
            self.metrics_server = create_metrics_server(
                port=server_config.get('port', 9090),
                host=server_config.get('host', '0.0.0.0'),
                redis_url=redis_url,
                db_url=db_url,
                config=self.config
            )
            
            # Start the server
            if self.metrics_server.start():
                self.running = True
                self.logger.info("✅ Modular monitoring system started successfully")
                
                # Print status information
                self._print_status()
                return True
            else:
                self.logger.error("❌ Failed to start metrics server")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting monitoring system: {e}")
            return False
    
    def stop(self):
        """Stop the monitoring system"""
        self.running = False
        
        if self.metrics_server:
            self.metrics_server.stop()
            self.metrics_server = None
        
        self.logger.info("👋 Monitoring system stopped")
    
    def _print_status(self):
        """Print system status information"""
        server_config = self.config.get('server', {})
        port = server_config.get('port', 9090)
        
        print("\n" + "="*50)
        print("FlashFit AI Modular Monitoring System")
        print("="*50)
        print(f"📊 Metrics Server: http://localhost:{port}/metrics")
        print(f"🔧 Configuration: {self.config_path}")
        print(f"📝 Log Level: {self.config.get('logging', {}).get('level', 'INFO')}")
        
        # Component status
        if self.metrics_server:
            summary = self.metrics_server.get_metrics_summary()
            print(f"\n📈 Components:")
            print(f"  - Metrics Collection: ✅ (interval: {summary['server']['collection_interval']}s)")
            print(f"  - Alerting Engine: ✅ ({summary['alerts']['rules_count']} rules)")
            print(f"  - Prometheus: {'✅' if summary['components']['prometheus_available'] else '❌'}")
            print(f"  - Redis: {'✅' if summary['components']['redis_available'] else '❌'}")
            print(f"  - PostgreSQL: {'✅' if summary['components']['postgres_available'] else '❌'}")
        
        # Service monitoring
        services = self.config.get('services', {})
        if services:
            print(f"\n🔍 Monitored Services:")
            for name, config in services.items():
                print(f"  - {name}: {config.get('url', 'N/A')}")
        
        print(f"\n🚨 Notifications:")
        notifications = self.config.get('notifications', {})
        for channel, config in notifications.items():
            status = "✅" if config.get('enabled', False) else "❌"
            print(f"  - {channel}: {status}")
        
        print("\nPress Ctrl+C to stop")
        print("="*50)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        if not self.metrics_server:
            return {'status': 'stopped', 'message': 'Monitoring system not running'}
        
        return self.metrics_server.get_health_status()
    
    def run_daemon(self):
        """Run as daemon process"""
        if not self.start():
            sys.exit(1)
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='FlashFit AI Modular Monitoring System')
    parser.add_argument('--config', '-c', default='monitoring_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--port', '-p', type=int, default=None,
                       help='Override server port')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install missing dependencies')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies and exit')
    parser.add_argument('--daemon', '-d', action='store_true',
                       help='Run as daemon')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create monitoring system
    monitoring = ModularMonitoringSystem(config_path=args.config)
    
    # Override port if specified
    if args.port:
        monitoring.config['server']['port'] = args.port
    
    # Set verbose logging
    if args.verbose:
        monitoring.config['logging']['level'] = 'DEBUG'
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check dependencies
    if args.check_deps:
        if monitoring.check_dependencies():
            print("✅ All dependencies are available")
            sys.exit(0)
        else:
            print("❌ Some dependencies are missing")
            sys.exit(1)
    
    # Install dependencies
    if args.install_deps:
        if monitoring.install_dependencies():
            print("✅ Dependencies installed successfully")
        else:
            print("❌ Failed to install dependencies")
            sys.exit(1)
    
    # Check dependencies before starting
    if not monitoring.check_dependencies():
        print("❌ Missing dependencies. Run with --install-deps to install them.")
        sys.exit(1)
    
    # Run the monitoring system
    if args.daemon:
        monitoring.run_daemon()
    else:
        if monitoring.start():
            try:
                while monitoring.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                monitoring.stop()
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()