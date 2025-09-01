#!/usr/bin/env python3
"""
Component Factory for FlashFit AI Monitoring System
Factory functions to create and configure monitoring components
"""

import logging
from typing import Dict, Any, Optional

# Import monitoring components
try:
    from metrics_collector import MetricsCollector
    from alerting_engine import AlertingEngine
    from metrics_server import MetricsServer, PrometheusMetricsRegistry
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import monitoring components: {e}")
    COMPONENTS_AVAILABLE = False

def create_metrics_collector(
    redis_url: Optional[str] = None,
    db_url: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Optional[MetricsCollector]:
    """
    Create and configure a MetricsCollector instance
    
    Args:
        redis_url: Redis connection URL
        db_url: PostgreSQL connection URL
        config: Configuration dictionary
    
    Returns:
        Configured MetricsCollector instance or None if creation fails
    """
    if not COMPONENTS_AVAILABLE:
        logging.error("Cannot create MetricsCollector: components not available")
        return None
    
    try:
        collector = MetricsCollector(
            redis_url=redis_url,
            db_url=db_url
        )
        
        services_config = config.get('services', {}) if config else {}
        logging.info(f"✅ MetricsCollector created with {len(services_config)} services configured")
        return collector
        
    except Exception as e:
        logging.error(f"Failed to create MetricsCollector: {e}")
        return None

def create_alerting_engine(
    redis_url: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Optional[AlertingEngine]:
    """
    Create and configure an AlertingEngine instance
    
    Args:
        redis_url: Redis connection URL for alert state persistence
        config: Configuration dictionary
    
    Returns:
        Configured AlertingEngine instance or None if creation fails
    """
    if not COMPONENTS_AVAILABLE:
        logging.error("Cannot create AlertingEngine: components not available")
        return None
    
    try:
        alerting_config = config.get('alerting', {}) if config else {}
        
        if not alerting_config.get('enabled', True):
            logging.info("Alerting disabled in configuration")
            return None
        
        engine = AlertingEngine(
            redis_url=redis_url,
            config=config or {}
        )
        
        # Get active alerts count for logging
        active_alerts = engine.get_active_alerts()
        logging.info(f"✅ AlertingEngine created with {len(active_alerts)} active alerts")
        return engine
        
    except Exception as e:
        logging.error(f"Failed to create AlertingEngine: {e}")
        return None

def create_prometheus_registry(
    config: Optional[Dict[str, Any]] = None
) -> Optional[PrometheusMetricsRegistry]:
    """
    Create and configure a PrometheusMetricsRegistry instance
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configured PrometheusMetricsRegistry instance or None if creation fails
    """
    if not COMPONENTS_AVAILABLE:
        logging.error("Cannot create PrometheusMetricsRegistry: components not available")
        return None
    
    try:
        registry = PrometheusMetricsRegistry()
        
        logging.info("✅ PrometheusMetricsRegistry created")
        return registry
        
    except Exception as e:
        logging.error(f"Failed to create PrometheusMetricsRegistry: {e}")
        return None

def create_metrics_server(
    port: int = 9090,
    host: str = '0.0.0.0',
    redis_url: Optional[str] = None,
    db_url: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Optional[MetricsServer]:
    """
    Create and configure a complete MetricsServer instance with all components
    
    Args:
        port: Server port
        host: Server host
        redis_url: Redis connection URL
        db_url: PostgreSQL connection URL
        config: Configuration dictionary
    
    Returns:
        Configured MetricsServer instance or None if creation fails
    """
    if not COMPONENTS_AVAILABLE:
        logging.error("Cannot create MetricsServer: components not available")
        return None
    
    try:
        # Create metrics server using the factory function from metrics_server module
        from metrics_server import create_metrics_server as _create_server
        
        server = _create_server(
            port=port,
            host=host,
            redis_url=redis_url or "redis://localhost:6379/0",
            db_url=db_url or "",
            config=config or {}
        )
        
        logging.info(f"✅ MetricsServer created on {host}:{port}")
        return server
        
    except Exception as e:
        logging.error(f"Failed to create MetricsServer: {e}")
        return None

def validate_configuration(config: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate monitoring system configuration
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Validate server configuration
    server_config = config.get('server', {})
    if not isinstance(server_config.get('port'), int):
        errors.append("server.port must be an integer")
    
    if server_config.get('port', 0) < 1 or server_config.get('port', 0) > 65535:
        errors.append("server.port must be between 1 and 65535")
    
    if not isinstance(server_config.get('collection_interval'), (int, float)):
        errors.append("server.collection_interval must be a number")
    
    # Validate storage configuration
    storage_config = config.get('storage', {})
    redis_config = storage_config.get('redis', {})
    if redis_config.get('enabled', False) and not redis_config.get('url'):
        errors.append("storage.redis.url is required when Redis is enabled")
    
    postgres_config = storage_config.get('postgresql', {})
    if postgres_config.get('enabled', False) and not postgres_config.get('url'):
        errors.append("storage.postgresql.url is required when PostgreSQL is enabled")
    
    # Validate services configuration
    services_config = config.get('services', {})
    for service_name, service_config in services_config.items():
        if not service_config.get('url'):
            errors.append(f"services.{service_name}.url is required")
    
    # Validate alerting configuration
    alerting_config = config.get('alerting', {})
    if alerting_config.get('enabled', True):
        rules_config = alerting_config.get('rules', {})
        
        # Validate alert rules structure
        for rule_type, rules in rules_config.items():
            if not isinstance(rules, list):
                errors.append(f"alerting.rules.{rule_type} must be a list")
                continue
            
            for i, rule in enumerate(rules):
                if not isinstance(rule, dict):
                    errors.append(f"alerting.rules.{rule_type}[{i}] must be a dictionary")
                    continue
                
                required_fields = ['name', 'metric', 'operator', 'threshold']
                for field in required_fields:
                    if field not in rule:
                        errors.append(f"alerting.rules.{rule_type}[{i}].{field} is required")
    
    # Validate notifications configuration
    notifications_config = config.get('notifications', {})
    email_config = notifications_config.get('email', {})
    if email_config.get('enabled', False):
        required_email_fields = ['smtp_server', 'username', 'password', 'from_email', 'to_emails']
        for field in required_email_fields:
            if not email_config.get(field):
                errors.append(f"notifications.email.{field} is required when email is enabled")
    
    webhook_config = notifications_config.get('webhook', {})
    if webhook_config.get('enabled', False):
        if not webhook_config.get('url'):
            errors.append("notifications.webhook.url is required when webhook is enabled")
    
    return len(errors) == 0, errors

def get_component_status() -> Dict[str, bool]:
    """
    Get the availability status of all monitoring components
    
    Returns:
        Dictionary mapping component names to availability status
    """
    status = {
        'components_available': COMPONENTS_AVAILABLE
    }
    
    # Check optional dependencies
    optional_deps = {
        'redis': 'redis',
        'postgresql': 'psycopg2',
        'prometheus': 'prometheus_client',
        'psutil': 'psutil',
        'yaml': 'yaml',
        'requests': 'requests',
        'rich': 'rich'
    }
    
    for name, module_name in optional_deps.items():
        try:
            __import__(module_name)
            status[f'{name}_available'] = True
        except ImportError:
            status[f'{name}_available'] = False
    
    return status