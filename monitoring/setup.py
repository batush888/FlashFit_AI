#!/usr/bin/env python3
"""
FlashFit AI Monitoring System Setup
Comprehensive monitoring with health checks and metrics collection
"""

import os
import sys
import logging
from typing import Optional, Dict, Any

# Standard library imports that are always available
import time
import json
import subprocess

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    DOTENV_AVAILABLE = False

# Core monitoring - with graceful fallback
try:
    from prometheus_client import start_http_server, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    print("Warning: prometheus_client not installed. Install with: pip install prometheus_client")
    PROMETHEUS_AVAILABLE = False
    start_http_server = None
    Gauge = None

# System monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("Warning: psutil not installed. Install with: pip install psutil")
    PSUTIL_AVAILABLE = False
    psutil = None

# HTTP requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    print("Warning: requests not installed. Install with: pip install requests")
    REQUESTS_AVAILABLE = False
    requests = None

# Database connections
try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    print("Warning: psycopg2 not installed. Install with: pip install psycopg2-binary")
    POSTGRES_AVAILABLE = False
    psycopg2 = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    print("Warning: redis not installed. Install with: pip install redis")
    REDIS_AVAILABLE = False
    redis = None

# Rich console for beautiful output
try:
    from rich.console import Console
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    print("Warning: rich not installed. Install with: pip install rich")
    console = None
    RICH_AVAILABLE = False

# Structured logging
try:
    import structlog
    logger = structlog.get_logger()
    STRUCTLOG_AVAILABLE = True
except ImportError:
    print("Warning: structlog not installed. Install with: pip install structlog")
    logger = None
    STRUCTLOG_AVAILABLE = False

# ===== Prometheus Metrics =====
if PROMETHEUS_AVAILABLE:
    CPU_USAGE = Gauge('flashfit_cpu_usage_percent', 'CPU Usage %')
    MEMORY_USAGE = Gauge('flashfit_memory_usage_percent', 'Memory Usage %')
else:
    CPU_USAGE = None
    MEMORY_USAGE = None

def start_monitoring_server(port: int = None):
    """Start Prometheus metrics HTTP server"""
    if port is None:
        port = int(os.getenv('PORT', 9090))
    
    if not PROMETHEUS_AVAILABLE or not start_http_server:
        print(f"Cannot start monitoring server: prometheus_client not available")
        return False
    
    try:
        start_http_server(int(port))
        if STRUCTLOG_AVAILABLE and logger:
            logger.info(f"Prometheus metrics server started on port {port}")
        else:
            print(f"✅ Prometheus metrics server started on port {port}")
        return True
    except Exception as e:
        if STRUCTLOG_AVAILABLE and logger:
            logger.error(f"Failed to start monitoring server on port {port}", exc_info=str(e))
        else:
            print(f"❌ Failed to start monitoring server: {e}")
        return False

def update_system_metrics():
    """Update CPU and memory metrics"""
    if not (PROMETHEUS_AVAILABLE and PSUTIL_AVAILABLE and psutil and CPU_USAGE and MEMORY_USAGE):
        print("Cannot update system metrics: required libraries not available")
        return False
    
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        
        CPU_USAGE.set(cpu_usage)
        MEMORY_USAGE.set(memory_usage)
        
        return True
    except Exception as e:
        if STRUCTLOG_AVAILABLE and logger:
            logger.error("Failed to update system metrics", exc_info=str(e))
        else:
            print(f"Error updating system metrics: {e}")
        return False

def check_postgresql_health():
    """Check PostgreSQL database connectivity"""
    try:
        # Get database connection parameters from environment
        db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'flashfit_ai'),
            'user': os.getenv('POSTGRES_USER', 'flashfit_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'flashfit_dev_password')
        }
        
        # Test connection
        if POSTGRES_AVAILABLE and psycopg2:
            conn = psycopg2.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password']
            )
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return result is not None
        return False
    except Exception as e:
        # PostgreSQL not available - this is expected in development
        if 'does not exist' in str(e) or 'connection' in str(e).lower():
            print(f"PostgreSQL not configured (expected in development): {e}")
        else:
            print(f"PostgreSQL health check failed: {e}")
        return False

def check_redis_health():
    """Check Redis connectivity"""
    try:
        # Get Redis connection parameters from environment
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        
        # Test connection
        if REDIS_AVAILABLE and redis:
            r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            r.ping()
            return True
        return False
    except Exception as e:
        if STRUCTLOG_AVAILABLE and logger:
            logger.error("Redis health check failed", exc_info=str(e))
        else:
            print(f"Redis health check failed: {e}")
        return False

def health_check(redis_url: Optional[str] = "redis://localhost:6379/0", 
                 db_url: Optional[str] = "postgresql://user:password@localhost:5432/flashfit") -> Dict[str, Any]:
    """Basic health check for monitoring system"""
    status = {
        "prometheus": False, 
        "redis": False, 
        "postgres": False,
        "system_metrics": False,
        "dependencies": {
            "prometheus_client": PROMETHEUS_AVAILABLE,
            "psutil": PSUTIL_AVAILABLE,
            "requests": REQUESTS_AVAILABLE,
            "psycopg2": POSTGRES_AVAILABLE,
            "redis": REDIS_AVAILABLE,
            "rich": RICH_AVAILABLE,
            "structlog": STRUCTLOG_AVAILABLE
        }
    }

    # Check Prometheus server by pinging metrics endpoint
    if REQUESTS_AVAILABLE and requests:
        try:
            resp = requests.get("http://localhost:9090", timeout=5)
            if resp.status_code == 200:
                status["prometheus"] = True
        except Exception:
            pass

    # Check Redis using dedicated function
    status["redis"] = check_redis_health()

    # Check Postgres using dedicated function
    status["postgres"] = check_postgresql_health()
    
    # Check system metrics availability
    if PSUTIL_AVAILABLE and psutil:
        try:
            psutil.cpu_percent()
            psutil.virtual_memory().percent
            status["system_metrics"] = True
        except Exception:
            pass

    # Display results
    if RICH_AVAILABLE and console:
        console.print("[bold green]Monitoring Health Status:[/bold green]", status)
    else:
        print(f"Monitoring Health Status: {json.dumps(status, indent=2)}")
    
    return status

def install_dependencies():
    """Helper function to install missing dependencies"""
    missing_deps = []
    
    if not PROMETHEUS_AVAILABLE:
        missing_deps.append("prometheus_client")
    if not PSUTIL_AVAILABLE:
        missing_deps.append("psutil")
    if not REQUESTS_AVAILABLE:
        missing_deps.append("requests")
    if not POSTGRES_AVAILABLE:
        missing_deps.append("psycopg2-binary")
    if not REDIS_AVAILABLE:
        missing_deps.append("redis")
    if not RICH_AVAILABLE:
        missing_deps.append("rich")
    if not STRUCTLOG_AVAILABLE:
        missing_deps.append("structlog")
    if not DOTENV_AVAILABLE:
        missing_deps.append("python-dotenv")
    
    if missing_deps:
        print("\nMissing dependencies detected:")
        print(f"Run: pip install {' '.join(missing_deps)}")
        print("Or install from requirements.txt: pip install -r requirements.txt")
        return False
    else:
        print("All dependencies are available!")
        return True

# ===== Startup Example =====
if __name__ == "__main__":
    print("FlashFit AI Monitoring System Setup")
    print("====================================")
    
    # Get configuration from environment
    port = int(os.getenv('PORT', 9090))
    host = os.getenv('HOST', '0.0.0.0')
    environment = os.getenv('ENVIRONMENT', 'development')
    
    print(f"Environment: {environment}")
    print(f"Starting monitoring server on {host}:{port}")
    
    # Check dependencies first
    if not install_dependencies():
        print("\nSome dependencies are missing. Please install them first.")
        sys.exit(1)
    
    # Start monitoring server
    if start_monitoring_server(port):
        print("✅ Monitoring server started successfully")
    else:
        print("❌ Failed to start monitoring server")
        if environment == 'production':
            sys.exit(1)
    
    # Update system metrics
    if update_system_metrics():
        print("✅ System metrics updated")
    else:
        print("❌ Failed to update system metrics")
    
    # Run health check
    print("\nRunning health check...")
    health_status = health_check()
    
    # Summary
    healthy_services = sum(1 for k, v in health_status.items() 
                          if k != "dependencies" and v)
    total_services = len([k for k in health_status.keys() if k != "dependencies"])
    
    print(f"\nHealth Check Summary: {healthy_services}/{total_services} services healthy")
    
    if healthy_services == total_services:
        print("🎉 All systems operational!")
    elif environment == 'production':
        print("⚠️  Some services need attention")
        sys.exit(1)
    else:
        print("⚠️  Some services need attention")
    
    # Keep server running in production
    if environment == 'production':
        print("\n🚀 Monitoring server running in production mode...")
        try:
            while True:
                time.sleep(60)
                # Update metrics every minute
                update_system_metrics()
        except KeyboardInterrupt:
            print("\n👋 Monitoring server stopped")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)