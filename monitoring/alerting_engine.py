#!/usr/bin/env python3
"""
FlashFit AI Alerting Engine
Separated alerting logic for production scalability
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

# Optional imports with graceful fallback
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

class ComparisonOperator(Enum):
    """Comparison operators for alert rules"""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_EQUAL = "ge"
    LESS_EQUAL = "le"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    id: str
    name: str
    metric_name: str
    operator: ComparisonOperator
    threshold: float
    duration_minutes: int
    severity: AlertSeverity
    description: str
    labels: Dict[str, str] = None
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['operator'] = self.operator.value
        data['severity'] = self.severity.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertRule':
        data['operator'] = ComparisonOperator(data['operator'])
        data['severity'] = AlertSeverity(data['severity'])
        return cls(**data)

@dataclass
class Alert:
    """Active alert instance"""
    id: str
    rule_id: str
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    message: str
    labels: Dict[str, str]
    started_at: str
    last_seen: str
    resolved: bool = False
    resolved_at: Optional[str] = None
    notification_sent: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['severity'] = self.severity.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        data['severity'] = AlertSeverity(data['severity'])
        return cls(**data)

class AlertingEngine:
    """Centralized alerting engine"""
    
    def __init__(self, redis_url: Optional[str] = None, config: Dict[str, Any] = None):
        self.redis_client = None
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Alert state tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.metric_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Alert rules
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Notification handlers
        self.notification_handlers: Dict[str, Callable] = {
            'email': self._send_email_notification,
            'webhook': self._send_webhook_notification,
            'log': self._log_notification
        }
        
        # Initialize Redis connection
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                self.logger.info("Connected to Redis for alert storage")
                self._load_alert_state()
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {e}")
                self.redis_client = None
        
        # Load default alert rules
        self._load_default_alert_rules()
    
    def _load_default_alert_rules(self):
        """Load default alert rules for FlashFit AI"""
        default_rules = [
            AlertRule(
                id="high_cpu_usage",
                name="High CPU Usage",
                metric_name="flashfit_cpu_usage_percent",
                operator=ComparisonOperator.GREATER_THAN,
                threshold=80.0,
                duration_minutes=5,
                severity=AlertSeverity.WARNING,
                description="CPU usage is above 80% for more than 5 minutes"
            ),
            AlertRule(
                id="critical_cpu_usage",
                name="Critical CPU Usage",
                metric_name="flashfit_cpu_usage_percent",
                operator=ComparisonOperator.GREATER_THAN,
                threshold=95.0,
                duration_minutes=2,
                severity=AlertSeverity.CRITICAL,
                description="CPU usage is above 95% for more than 2 minutes"
            ),
            AlertRule(
                id="high_memory_usage",
                name="High Memory Usage",
                metric_name="flashfit_memory_usage_percent",
                operator=ComparisonOperator.GREATER_THAN,
                threshold=85.0,
                duration_minutes=5,
                severity=AlertSeverity.WARNING,
                description="Memory usage is above 85% for more than 5 minutes"
            ),
            AlertRule(
                id="critical_memory_usage",
                name="Critical Memory Usage",
                metric_name="flashfit_memory_usage_percent",
                operator=ComparisonOperator.GREATER_THAN,
                threshold=95.0,
                duration_minutes=2,
                severity=AlertSeverity.CRITICAL,
                description="Memory usage is above 95% for more than 2 minutes"
            ),
            AlertRule(
                id="high_disk_usage",
                name="High Disk Usage",
                metric_name="flashfit_disk_usage_percent",
                operator=ComparisonOperator.GREATER_THAN,
                threshold=90.0,
                duration_minutes=10,
                severity=AlertSeverity.WARNING,
                description="Disk usage is above 90% for more than 10 minutes"
            ),
            AlertRule(
                id="service_down",
                name="Service Down",
                metric_name="flashfit_service_health_status",
                operator=ComparisonOperator.EQUALS,
                threshold=0.0,
                duration_minutes=1,
                severity=AlertSeverity.CRITICAL,
                description="Service health check is failing"
            ),
            AlertRule(
                id="faiss_high_search_latency",
                name="FAISS High Search Latency",
                metric_name="faiss_search_duration_seconds",
                operator=ComparisonOperator.GREATER_THAN,
                threshold=1.0,
                duration_minutes=3,
                severity=AlertSeverity.WARNING,
                description="FAISS search latency is above 1 second"
            ),
            AlertRule(
                id="faiss_low_health_score",
                name="FAISS Low Health Score",
                metric_name="faiss_health_score",
                operator=ComparisonOperator.LESS_THAN,
                threshold=50.0,
                duration_minutes=5,
                severity=AlertSeverity.WARNING,
                description="FAISS index health score is below 50"
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.id] = rule
    
    def add_alert_rule(self, rule: AlertRule):
        """Add or update an alert rule"""
        self.alert_rules[rule.id] = rule
        self._save_alert_rules()
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self._save_alert_rules()
            self.logger.info(f"Removed alert rule: {rule_id}")
    
    def evaluate_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Evaluate a metric against all applicable alert rules"""
        timestamp = datetime.utcnow().isoformat()
        
        # Add metric to buffer for duration-based evaluation
        self.metric_buffer[metric_name].append({
            'timestamp': timestamp,
            'value': value,
            'labels': labels or {}
        })
        
        # Evaluate against all rules for this metric
        for rule in self.alert_rules.values():
            if rule.metric_name == metric_name and rule.enabled:
                self._evaluate_rule(rule, timestamp)
    
    def _evaluate_rule(self, rule: AlertRule, current_timestamp: str):
        """Evaluate a specific rule against metric buffer"""
        metric_data = list(self.metric_buffer[rule.metric_name])
        
        if not metric_data:
            return
        
        # Filter data within the duration window
        cutoff_time = datetime.fromisoformat(current_timestamp) - timedelta(minutes=rule.duration_minutes)
        recent_data = [
            data for data in metric_data
            if datetime.fromisoformat(data['timestamp']) >= cutoff_time
        ]
        
        if not recent_data:
            return
        
        # Check if condition is met for the entire duration
        condition_met = all(
            self._evaluate_condition(rule.operator, data['value'], rule.threshold)
            for data in recent_data
        )
        
        alert_id = f"{rule.id}_{hash(str(rule.labels or {}))}"
        
        if condition_met:
            if alert_id not in self.active_alerts:
                # Create new alert
                current_value = recent_data[-1]['value']
                alert = Alert(
                    id=alert_id,
                    rule_id=rule.id,
                    rule_name=rule.name,
                    metric_name=rule.metric_name,
                    current_value=current_value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    message=self._format_alert_message(rule, current_value),
                    labels=rule.labels or {},
                    started_at=current_timestamp,
                    last_seen=current_timestamp
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                self._save_alert_state()
                
                # Send notification
                self._send_notification(alert)
                
                self.logger.warning(f"Alert triggered: {alert.message}")
            else:
                # Update existing alert
                alert = self.active_alerts[alert_id]
                alert.last_seen = current_timestamp
                alert.current_value = recent_data[-1]['value']
                self._save_alert_state()
        else:
            # Resolve alert if it exists
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = current_timestamp
                
                del self.active_alerts[alert_id]
                self._save_alert_state()
                
                self.logger.info(f"Alert resolved: {alert.rule_name}")
    
    def _evaluate_condition(self, operator: ComparisonOperator, value: float, threshold: float) -> bool:
        """Evaluate a condition based on operator"""
        if operator == ComparisonOperator.GREATER_THAN:
            return value > threshold
        elif operator == ComparisonOperator.LESS_THAN:
            return value < threshold
        elif operator == ComparisonOperator.EQUALS:
            return abs(value - threshold) < 0.001  # Float comparison with tolerance
        elif operator == ComparisonOperator.NOT_EQUALS:
            return abs(value - threshold) >= 0.001
        elif operator == ComparisonOperator.GREATER_EQUAL:
            return value >= threshold
        elif operator == ComparisonOperator.LESS_EQUAL:
            return value <= threshold
        return False
    
    def _format_alert_message(self, rule: AlertRule, current_value: float) -> str:
        """Format alert message"""
        return f"{rule.description}. Current value: {current_value:.2f}, Threshold: {rule.threshold}"
    
    def _send_notification(self, alert: Alert):
        """Send alert notification through configured channels"""
        notification_config = self.config.get('notifications', {})
        
        for channel, config in notification_config.items():
            if config.get('enabled', False) and channel in self.notification_handlers:
                try:
                    self.notification_handlers[channel](alert, config)
                    alert.notification_sent = True
                except Exception as e:
                    self.logger.error(f"Failed to send {channel} notification: {e}")
    
    def _send_email_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send email notification"""
        if not EMAIL_AVAILABLE:
            self.logger.warning("Email notifications not available (smtplib not installed)")
            return
        
        smtp_server = config.get('smtp_server')
        smtp_port = config.get('smtp_port', 587)
        username = config.get('username')
        password = config.get('password')
        to_emails = config.get('to_emails', [])
        
        if not all([smtp_server, username, password, to_emails]):
            self.logger.error("Incomplete email configuration")
            return
        
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = ', '.join(to_emails)
        msg['Subject'] = f"FlashFit AI Alert: {alert.rule_name} [{alert.severity.value.upper()}]"
        
        body = f"""
        Alert: {alert.rule_name}
        Severity: {alert.severity.value.upper()}
        Metric: {alert.metric_name}
        Current Value: {alert.current_value:.2f}
        Threshold: {alert.threshold}
        Started At: {alert.started_at}
        
        Description: {alert.message}
        
        Labels: {json.dumps(alert.labels, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
    
    def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send webhook notification"""
        if not REQUESTS_AVAILABLE:
            self.logger.warning("Webhook notifications not available (requests not installed)")
            return
        
        webhook_url = config.get('url')
        if not webhook_url:
            self.logger.error("Webhook URL not configured")
            return
        
        payload = {
            'alert': alert.to_dict(),
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'flashfit-ai-monitoring'
        }
        
        headers = {'Content-Type': 'application/json'}
        headers.update(config.get('headers', {}))
        
        response = requests.post(
            webhook_url,
            json=payload,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
    
    def _log_notification(self, alert: Alert, config: Dict[str, Any]):
        """Log notification (always available fallback)"""
        log_level = config.get('level', 'warning').upper()
        logger_method = getattr(self.logger, log_level.lower(), self.logger.warning)
        
        logger_method(
            f"ALERT [{alert.severity.value.upper()}] {alert.rule_name}: "
            f"{alert.message} (Started: {alert.started_at})"
        )
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Alert]:
        """Get currently active alerts"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.started_at, reverse=True)
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        return list(self.alert_history)[-limit:]
    
    def _save_alert_state(self):
        """Save alert state to Redis"""
        if not self.redis_client:
            return
        
        try:
            # Save active alerts
            active_data = {aid: alert.to_dict() for aid, alert in self.active_alerts.items()}
            self.redis_client.set('alerts:active', json.dumps(active_data))
            
            # Save alert history
            history_data = [alert.to_dict() for alert in self.alert_history]
            self.redis_client.set('alerts:history', json.dumps(history_data))
            
        except Exception as e:
            self.logger.error(f"Error saving alert state: {e}")
    
    def _load_alert_state(self):
        """Load alert state from Redis"""
        if not self.redis_client:
            return
        
        try:
            # Load active alerts
            active_data = self.redis_client.get('alerts:active')
            if active_data:
                active_dict = json.loads(active_data)
                self.active_alerts = {
                    aid: Alert.from_dict(data) for aid, data in active_dict.items()
                }
            
            # Load alert history
            history_data = self.redis_client.get('alerts:history')
            if history_data:
                history_list = json.loads(history_data)
                self.alert_history = deque(
                    [Alert.from_dict(data) for data in history_list],
                    maxlen=1000
                )
                
        except Exception as e:
            self.logger.error(f"Error loading alert state: {e}")
    
    def _save_alert_rules(self):
        """Save alert rules to Redis"""
        if not self.redis_client:
            return
        
        try:
            rules_data = {rid: rule.to_dict() for rid, rule in self.alert_rules.items()}
            self.redis_client.set('alerts:rules', json.dumps(rules_data))
        except Exception as e:
            self.logger.error(f"Error saving alert rules: {e}")
    
    def cleanup_old_alerts(self, retention_hours: int = 168):  # 7 days default
        """Clean up old resolved alerts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)
        
        # Clean up alert history
        filtered_history = deque(maxlen=1000)
        for alert in self.alert_history:
            alert_time = datetime.fromisoformat(alert.started_at)
            if alert_time >= cutoff_time or not alert.resolved:
                filtered_history.append(alert)
        
        self.alert_history = filtered_history
        self._save_alert_state()
    
    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()

def create_alerting_engine(redis_url: str = None, config: Dict[str, Any] = None) -> AlertingEngine:
    """Factory function to create an alerting engine"""
    return AlertingEngine(redis_url=redis_url, config=config)

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Sample notification configuration
    config = {
        'notifications': {
            'log': {
                'enabled': True,
                'level': 'warning'
            },
            'email': {
                'enabled': False,  # Set to True and configure for email alerts
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'your-email@gmail.com',
                'password': 'your-app-password',
                'to_emails': ['admin@flashfit.ai']
            }
        }
    }
    
    engine = create_alerting_engine(
        redis_url="redis://localhost:6379/0",
        config=config
    )
    
    # Simulate metric evaluation
    engine.evaluate_metric('flashfit_cpu_usage_percent', 85.0)
    engine.evaluate_metric('flashfit_memory_usage_percent', 90.0)
    
    # Check active alerts
    active_alerts = engine.get_active_alerts()
    print(f"Active alerts: {len(active_alerts)}")
    
    for alert in active_alerts:
        print(f"- {alert.rule_name}: {alert.message}")
    
    engine.close()