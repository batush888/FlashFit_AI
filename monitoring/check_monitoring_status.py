#!/usr/bin/env python3
"""
FlashFit AI Monitoring Status Checker

This script checks the status of all monitoring components and provides
a comprehensive overview of the monitoring system health.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

def check_endpoint(url: str, timeout: int = 5) -> Tuple[bool, str, float]:
    """Check if an endpoint is accessible and measure response time."""
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            return True, f"OK ({response.status_code})", response_time
        else:
            return False, f"HTTP {response.status_code}", response_time
    except requests.exceptions.ConnectionError:
        return False, "Connection refused", 0.0
    except requests.exceptions.Timeout:
        return False, "Timeout", timeout
    except Exception as e:
        return False, f"Error: {str(e)}", 0.0

def get_metrics_count(url: str) -> Tuple[int, List[str]]:
    """Get the number of metrics and sample metric names from an endpoint."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            metrics = []
            for line in lines:
                if line and not line.startswith('#') and '{' in line:
                    metric_name = line.split('{')[0]
                    if metric_name not in metrics:
                        metrics.append(metric_name)
            return len(metrics), metrics[:5]  # Return first 5 metrics as sample
        return 0, []
    except:
        return 0, []

def check_service_health() -> Dict:
    """Check the health of all FlashFit AI services."""
    services = {
        'Backend API': 'http://localhost:8080/health',
        'Frontend': 'http://localhost:3000',
        'Modular Monitor': 'http://localhost:9092/metrics',
        'FAISS Monitor': 'http://localhost:9091/metrics'
    }
    
    results = {}
    for service_name, url in services.items():
        is_up, status, response_time = check_endpoint(url)
        results[service_name] = {
            'status': 'UP' if is_up else 'DOWN',
            'url': url,
            'response_time': f"{response_time:.3f}s",
            'details': status
        }
    
    return results

def check_monitoring_stack() -> Dict:
    """Check the status of Prometheus/Grafana monitoring stack."""
    stack = {
        'Prometheus': 'http://localhost:9090/-/healthy',
        'Grafana': 'http://localhost:3001/api/health',
        'AlertManager': 'http://localhost:9093/-/healthy'
    }
    
    results = {}
    for component, url in stack.items():
        is_up, status, response_time = check_endpoint(url)
        results[component] = {
            'status': 'UP' if is_up else 'DOWN',
            'url': url,
            'response_time': f"{response_time:.3f}s",
            'details': status
        }
    
    return results

def get_metrics_summary() -> Dict:
    """Get a summary of available metrics from monitoring endpoints."""
    endpoints = {
        'Modular Monitor': 'http://localhost:9092/metrics',
        'FAISS Monitor': 'http://localhost:9091/metrics'
    }
    
    results = {}
    for name, url in endpoints.items():
        count, sample_metrics = get_metrics_count(url)
        results[name] = {
            'metrics_count': count,
            'sample_metrics': sample_metrics,
            'endpoint': url
        }
    
    return results

def print_status_report():
    """Print a comprehensive status report."""
    print("\n" + "="*60)
    print("🔍 FlashFit AI Monitoring Status Report")
    print("="*60)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check core services
    print("🚀 Core Services Status:")
    print("-" * 30)
    services = check_service_health()
    for service, info in services.items():
        status_icon = "✅" if info['status'] == 'UP' else "❌"
        print(f"{status_icon} {service:<20} {info['status']:<6} ({info['response_time']})")
        if info['status'] == 'DOWN':
            print(f"   └─ {info['details']}")
    print()
    
    # Check monitoring stack
    print("📊 Monitoring Stack Status:")
    print("-" * 30)
    stack = check_monitoring_stack()
    for component, info in stack.items():
        status_icon = "✅" if info['status'] == 'UP' else "❌"
        print(f"{status_icon} {component:<20} {info['status']:<6} ({info['response_time']})")
        if info['status'] == 'DOWN':
            print(f"   └─ {info['details']}")
    print()
    
    # Metrics summary
    print("📈 Metrics Summary:")
    print("-" * 30)
    metrics = get_metrics_summary()
    total_metrics = 0
    for name, info in metrics.items():
        count = info['metrics_count']
        total_metrics += count
        status_icon = "✅" if count > 0 else "❌"
        print(f"{status_icon} {name:<20} {count} metrics")
        if info['sample_metrics']:
            print(f"   └─ Sample: {', '.join(info['sample_metrics'][:3])}...")
    
    print(f"\n📊 Total Metrics Available: {total_metrics}")
    print()
    
    # Access URLs
    print("🌐 Access URLs:")
    print("-" * 30)
    urls = {
        'Modular Monitor Metrics': 'http://localhost:9092/metrics',
        'FAISS Monitor Metrics': 'http://localhost:9091/metrics',
        'Prometheus (if running)': 'http://localhost:9090',
        'Grafana (if running)': 'http://localhost:3001',
        'Backend API': 'http://localhost:8080',
        'Frontend App': 'http://localhost:3000'
    }
    
    for name, url in urls.items():
        print(f"🔗 {name:<25} {url}")
    
    print()
    print("💡 Next Steps:")
    print("-" * 30)
    
    # Check if monitoring stack is running
    stack_running = any(info['status'] == 'UP' for info in stack.values())
    if not stack_running:
        print("📋 To complete Phase 2 deployment:")
        print("   1. Install Docker: brew install --cask docker")
        print("   2. Start monitoring stack: docker compose -f docker-compose.monitoring.yml up -d")
        print("   3. Access Grafana at http://localhost:3001 (admin/flashfit_admin)")
        print("   4. Import dashboard from grafana/dashboards/flashfit-system-overview.json")
    else:
        print("✅ Monitoring stack is running!")
        print("   └─ Import dashboard: grafana/dashboards/flashfit-system-overview.json")
    
    print("\n📖 For detailed setup instructions, see: MONITORING_DEPLOYMENT.md")
    print("="*60)

if __name__ == "__main__":
    print_status_report()