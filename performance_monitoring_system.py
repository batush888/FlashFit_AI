#!/usr/bin/env python3
"""
FlashFit AI Performance Monitoring System
Sets up continuous monitoring and reliability metrics
"""

import asyncio
import aiohttp
import json
import time
import psutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import statistics

@dataclass
class PerformanceMetric:
    timestamp: str
    metric_type: str  # 'api_latency', 'memory_usage', 'cpu_usage', 'error_rate'
    value: float
    endpoint: Optional[str] = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None

@dataclass
class AlertRule:
    metric_type: str
    threshold: float
    comparison: str  # 'greater_than', 'less_than', 'equals'
    duration_minutes: int
    severity: str  # 'critical', 'warning', 'info'
    description: str

@dataclass
class Alert:
    timestamp: str
    rule: AlertRule
    current_value: float
    severity: str
    message: str
    resolved: bool = False
    resolved_at: Optional[str] = None

class PerformanceMonitor:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None
        self.metrics_buffer = deque(maxlen=1000)  # Keep last 1000 metrics
        self.alerts = []
        self.alert_rules = self._setup_default_alert_rules()
        self.monitoring_active = False
        self.report_file = Path("performance_monitoring_report.json")
        self.log_file = Path("performance_monitor.log")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_default_alert_rules(self) -> List[AlertRule]:
        """Setup default alert rules for monitoring"""
        return [
            AlertRule(
                metric_type="api_latency",
                threshold=3000.0,  # 3 seconds
                comparison="greater_than",
                duration_minutes=2,
                severity="warning",
                description="API response time exceeds 3 seconds"
            ),
            AlertRule(
                metric_type="api_latency",
                threshold=5000.0,  # 5 seconds
                comparison="greater_than",
                duration_minutes=1,
                severity="critical",
                description="API response time exceeds 5 seconds"
            ),
            AlertRule(
                metric_type="error_rate",
                threshold=5.0,  # 5% error rate
                comparison="greater_than",
                duration_minutes=5,
                severity="warning",
                description="Error rate exceeds 5%"
            ),
            AlertRule(
                metric_type="error_rate",
                threshold=10.0,  # 10% error rate
                comparison="greater_than",
                duration_minutes=2,
                severity="critical",
                description="Error rate exceeds 10%"
            ),
            AlertRule(
                metric_type="memory_usage",
                threshold=85.0,  # 85% memory usage
                comparison="greater_than",
                duration_minutes=10,
                severity="warning",
                description="Memory usage exceeds 85%"
            ),
            AlertRule(
                metric_type="cpu_usage",
                threshold=90.0,  # 90% CPU usage
                comparison="greater_than",
                duration_minutes=5,
                severity="warning",
                description="CPU usage exceeds 90%"
            )
        ]
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_api_metrics(self) -> List[PerformanceMetric]:
        """Collect API performance metrics"""
        if self.session is None:
            self.logger.warning("Session not initialized, skipping API metrics collection")
            return []
            
        endpoints = [
            "/health",
            "/api/user/profile",
            "/api/wardrobe",
            "/api/match",
            "/api/history/statistics"
        ]
        
        metrics = []
        
        for endpoint in endpoints:
            try:
                start_time = time.time()
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    metric = PerformanceMetric(
                        timestamp=datetime.now().isoformat(),
                        metric_type="api_latency",
                        value=latency_ms,
                        endpoint=endpoint,
                        status_code=response.status
                    )
                    metrics.append(metric)
                    
                    # Track error rates
                    if response.status >= 400:
                        error_metric = PerformanceMetric(
                            timestamp=datetime.now().isoformat(),
                            metric_type="error_rate",
                            value=1.0,  # Error occurred
                            endpoint=endpoint,
                            status_code=response.status,
                            error_message=f"HTTP {response.status}"
                        )
                        metrics.append(error_metric)
                    else:
                        success_metric = PerformanceMetric(
                            timestamp=datetime.now().isoformat(),
                            metric_type="error_rate",
                            value=0.0,  # Success
                            endpoint=endpoint,
                            status_code=response.status
                        )
                        metrics.append(success_metric)
            
            except Exception as e:
                error_metric = PerformanceMetric(
                    timestamp=datetime.now().isoformat(),
                    metric_type="error_rate",
                    value=1.0,
                    endpoint=endpoint,
                    error_message=str(e)
                )
                metrics.append(error_metric)
        
        return metrics
    
    def collect_system_metrics(self) -> List[PerformanceMetric]:
        """Collect system performance metrics"""
        metrics = []
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            metric_type="cpu_usage",
            value=cpu_percent
        ))
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        metrics.append(PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            metric_type="memory_usage",
            value=memory_percent
        ))
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        metrics.append(PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            metric_type="disk_usage",
            value=disk_percent
        ))
        
        return metrics
    
    def add_metric(self, metric: PerformanceMetric):
        """Add a metric to the buffer"""
        self.metrics_buffer.append(metric)
        self.check_alert_rules(metric)
    
    def check_alert_rules(self, metric: PerformanceMetric):
        """Check if metric triggers any alert rules"""
        for rule in self.alert_rules:
            if rule.metric_type == metric.metric_type:
                should_alert = False
                
                if rule.comparison == "greater_than" and metric.value > rule.threshold:
                    should_alert = True
                elif rule.comparison == "less_than" and metric.value < rule.threshold:
                    should_alert = True
                elif rule.comparison == "equals" and metric.value == rule.threshold:
                    should_alert = True
                
                if should_alert:
                    # Check if we already have an active alert for this rule
                    active_alerts = [a for a in self.alerts if not a.resolved and a.rule.metric_type == rule.metric_type]
                    
                    if not active_alerts:
                        alert = Alert(
                            timestamp=datetime.now().isoformat(),
                            rule=rule,
                            current_value=metric.value,
                            severity=rule.severity,
                            message=f"{rule.description}. Current value: {metric.value:.2f}"
                        )
                        self.alerts.append(alert)
                        self.logger.warning(f"ALERT: {alert.message}")
    
    def calculate_statistics(self, metric_type: str, time_window_minutes: int = 60) -> Dict[str, float]:
        """Calculate statistics for a specific metric type within a time window"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        relevant_metrics = [
            m for m in self.metrics_buffer 
            if m.metric_type == metric_type and 
            datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        if not relevant_metrics:
            return {}
        
        values = [m.value for m in relevant_metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99)
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def calculate_error_rate(self, time_window_minutes: int = 60) -> float:
        """Calculate error rate within time window"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        error_metrics = [
            m for m in self.metrics_buffer 
            if m.metric_type == "error_rate" and 
            datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        if not error_metrics:
            return 0.0
        
        total_requests = len(error_metrics)
        error_requests = sum(1 for m in error_metrics if m.value > 0)
        
        return (error_requests / total_requests) * 100
    
    def get_health_score(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        scores = {}
        weights = {
            "api_latency": 0.3,
            "error_rate": 0.3,
            "cpu_usage": 0.2,
            "memory_usage": 0.2
        }
        
        # API Latency Score (0-100, lower latency = higher score)
        latency_stats = self.calculate_statistics("api_latency", 30)
        if latency_stats:
            avg_latency = latency_stats["mean"]
            latency_score = max(0, 100 - (avg_latency / 50))  # 5000ms = 0 score
            scores["api_latency"] = min(100, latency_score)
        else:
            scores["api_latency"] = 100
        
        # Error Rate Score (0-100, lower error rate = higher score)
        error_rate = self.calculate_error_rate(30)
        scores["error_rate"] = max(0, 100 - (error_rate * 10))  # 10% error = 0 score
        
        # CPU Usage Score (0-100, lower usage = higher score)
        cpu_stats = self.calculate_statistics("cpu_usage", 30)
        if cpu_stats:
            avg_cpu = cpu_stats["mean"]
            scores["cpu_usage"] = max(0, 100 - avg_cpu)
        else:
            scores["cpu_usage"] = 100
        
        # Memory Usage Score (0-100, lower usage = higher score)
        memory_stats = self.calculate_statistics("memory_usage", 30)
        if memory_stats:
            avg_memory = memory_stats["mean"]
            scores["memory_usage"] = max(0, 100 - avg_memory)
        else:
            scores["memory_usage"] = 100
        
        # Calculate weighted overall score
        overall_score = sum(scores[metric] * weights[metric] for metric in weights.keys())
        
        return {
            "overall_score": round(overall_score, 1),
            "component_scores": scores,
            "health_status": self._get_health_status(overall_score)
        }
    
    def _get_health_status(self, score: float) -> str:
        """Get health status based on score"""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "fair"
        elif score >= 40:
            return "poor"
        else:
            return "critical"
    
    async def monitoring_loop(self, interval_seconds: int = 30, duration_minutes: int = 10):
        """Main monitoring loop"""
        self.monitoring_active = True
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        self.logger.info(f"Starting performance monitoring for {duration_minutes} minutes...")
        
        while self.monitoring_active and datetime.now() < end_time:
            try:
                # Collect API metrics
                api_metrics = await self.collect_api_metrics()
                for metric in api_metrics:
                    self.add_metric(metric)
                
                # Collect system metrics
                system_metrics = self.collect_system_metrics()
                for metric in system_metrics:
                    self.add_metric(metric)
                
                # Log current health score
                health = self.get_health_score()
                self.logger.info(f"Health Score: {health['overall_score']}/100 ({health['health_status']})")
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
        
        self.monitoring_active = False
        self.logger.info("Performance monitoring completed")
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        
        # Calculate statistics for different metrics
        api_latency_stats = self.calculate_statistics("api_latency", 60)
        cpu_stats = self.calculate_statistics("cpu_usage", 60)
        memory_stats = self.calculate_statistics("memory_usage", 60)
        error_rate = self.calculate_error_rate(60)
        
        # Get health score
        health = self.get_health_score()
        
        # Active alerts
        active_alerts = [a for a in self.alerts if not a.resolved]
        
        # Endpoint performance breakdown
        endpoint_performance = {}
        for endpoint in ["/health", "/api/user/profile", "/api/wardrobe", "/api/match", "/api/history/statistics"]:
            endpoint_metrics = [m for m in self.metrics_buffer if m.endpoint == endpoint and m.metric_type == "api_latency"]
            if endpoint_metrics:
                values = [m.value for m in endpoint_metrics]
                endpoint_performance[endpoint] = {
                    "avg_latency_ms": statistics.mean(values),
                    "max_latency_ms": max(values),
                    "min_latency_ms": min(values),
                    "request_count": len(values)
                }
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "Performance Monitoring Report",
                "monitoring_duration_minutes": 60,
                "total_metrics_collected": len(self.metrics_buffer)
            },
            "health_score": health,
            "performance_statistics": {
                "api_latency": api_latency_stats,
                "cpu_usage": cpu_stats,
                "memory_usage": memory_stats,
                "error_rate_percent": error_rate
            },
            "endpoint_performance": endpoint_performance,
            "active_alerts": [asdict(alert) for alert in active_alerts],
            "alert_history": [asdict(alert) for alert in self.alerts],
            "recommendations": self._generate_monitoring_recommendations(health, active_alerts)
        }
        
        return report
    
    def _generate_monitoring_recommendations(self, health: Dict[str, Any], active_alerts: List[Alert]) -> List[str]:
        """Generate recommendations based on monitoring data"""
        recommendations = []
        
        # Health-based recommendations
        if health["overall_score"] < 60:
            recommendations.append("System health is below acceptable levels - investigate performance issues")
        
        if health["component_scores"]["api_latency"] < 70:
            recommendations.append("API response times are slow - consider optimization or scaling")
        
        if health["component_scores"]["error_rate"] < 80:
            recommendations.append("High error rate detected - check application logs and fix issues")
        
        if health["component_scores"]["cpu_usage"] < 50:
            recommendations.append("High CPU usage - consider scaling or optimizing resource-intensive operations")
        
        if health["component_scores"]["memory_usage"] < 50:
            recommendations.append("High memory usage - check for memory leaks or consider increasing resources")
        
        # Alert-based recommendations
        if active_alerts:
            recommendations.append(f"Address {len(active_alerts)} active alerts to improve system stability")
        
        # General recommendations
        recommendations.extend([
            "Set up automated alerting for critical metrics",
            "Implement log aggregation for better debugging",
            "Consider implementing circuit breakers for external dependencies",
            "Set up regular performance testing and benchmarking",
            "Monitor database performance and query optimization"
        ])
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any]):
        """Save monitoring report to file"""
        with open(self.report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Monitoring report saved to: {self.report_file}")
    
    def print_monitoring_summary(self, report: Dict[str, Any]):
        """Print monitoring summary"""
        print("\n" + "="*70)
        print("📊 PERFORMANCE MONITORING SUMMARY")
        print("="*70)
        
        # Health Score
        health = report["health_score"]
        print(f"\n🏥 SYSTEM HEALTH:")
        print(f"   Overall Score: {health['overall_score']}/100 ({health['health_status'].upper()})")
        print(f"   API Latency: {health['component_scores']['api_latency']:.1f}/100")
        print(f"   Error Rate: {health['component_scores']['error_rate']:.1f}/100")
        print(f"   CPU Usage: {health['component_scores']['cpu_usage']:.1f}/100")
        print(f"   Memory Usage: {health['component_scores']['memory_usage']:.1f}/100")
        
        # Performance Stats
        stats = report["performance_statistics"]
        if stats["api_latency"]:
            print(f"\n⚡ API PERFORMANCE:")
            print(f"   Average Latency: {stats['api_latency']['mean']:.0f}ms")
            print(f"   95th Percentile: {stats['api_latency']['p95']:.0f}ms")
            print(f"   Max Latency: {stats['api_latency']['max']:.0f}ms")
        
        print(f"   Error Rate: {stats['error_rate_percent']:.1f}%")
        
        # System Resources
        if stats["cpu_usage"]:
            print(f"\n💻 SYSTEM RESOURCES:")
            print(f"   Average CPU: {stats['cpu_usage']['mean']:.1f}%")
            print(f"   Average Memory: {stats['memory_usage']['mean']:.1f}%")
        
        # Alerts
        active_alerts = report["active_alerts"]
        if active_alerts:
            print(f"\n🚨 ACTIVE ALERTS ({len(active_alerts)}):")
            for alert in active_alerts[:3]:  # Show top 3
                print(f"   • {alert['severity'].upper()}: {alert['message']}")
        
        # Top Recommendations
        print(f"\n💡 TOP RECOMMENDATIONS:")
        for rec in report["recommendations"][:3]:
            print(f"   • {rec}")

async def main():
    """Main function to run performance monitoring"""
    print("📊 FlashFit AI Performance Monitoring System")
    print("============================================\n")
    
    async with PerformanceMonitor() as monitor:
        # Run monitoring for 10 minutes
        await monitor.monitoring_loop(interval_seconds=30, duration_minutes=10)
        
        # Generate report
        print("\n📋 Generating monitoring report...")
        report = monitor.generate_monitoring_report()
        
        # Save report
        monitor.save_report(report)
        
        # Print summary
        monitor.print_monitoring_summary(report)
        
        print("\n✅ Performance monitoring completed!")
        print("\n📝 NEXT STEPS:")
        print("   1. Review detailed report in performance_monitoring_report.json")
        print("   2. Address any active alerts or performance issues")
        print("   3. Set up continuous monitoring in production")
        print("   4. Configure automated alerting and notifications")
        print("   5. Establish performance baselines and SLAs")

if __name__ == "__main__":
    asyncio.run(main())