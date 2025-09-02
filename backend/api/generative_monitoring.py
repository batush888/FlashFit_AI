#!/usr/bin/env python3
"""
Generative AI Monitoring API Endpoints
Provides REST API endpoints for monitoring generative components.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from services.generative_monitoring import (
    get_generative_monitoring_service,
    GenerativeMonitoringService
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

@router.get("/health")
async def get_system_health():
    """Get overall system health status."""
    try:
        monitoring_service = get_generative_monitoring_service()
        overview = monitoring_service.get_system_overview()
        
        # Determine overall health status
        if overview.get("status") == "error":
            health_status = "unhealthy"
        elif overview.get("success_rate", 0) < 0.95:  # Less than 95% success rate
            health_status = "degraded"
        elif overview.get("average_response_time_ms", 0) > 2000:  # Slower than 2s
            health_status = "degraded"
        else:
            health_status = "healthy"
        
        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "overview": overview
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/{component}")
async def get_component_health(component: str):
    """Get health status for a specific component."""
    try:
        monitoring_service = get_generative_monitoring_service()
        health = monitoring_service.get_component_health(component)
        
        if not health:
            raise HTTPException(status_code=404, detail=f"Component '{component}' not found")
        
        return {
            "component": component,
            "health": health,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting component health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/overview")
async def get_metrics_overview(
    hours: int = Query(default=1, ge=1, le=24, description="Hours of data to include")
):
    """Get system metrics overview."""
    try:
        monitoring_service = get_generative_monitoring_service()
        overview = monitoring_service.get_system_overview()
        
        return {
            "metrics": overview,
            "period_hours": hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/{component}")
async def get_component_metrics(
    component: str,
    hours: int = Query(default=1, ge=1, le=24, description="Hours of data to include")
):
    """Get performance metrics for a specific component."""
    try:
        monitoring_service = get_generative_monitoring_service()
        metrics = monitoring_service.get_component_metrics(component, hours)
        
        if not metrics:
            return {
                "component": component,
                "metrics": [],
                "period_hours": hours,
                "message": "No metrics found for this component and time period",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate summary statistics
        response_times = [m.duration_ms for m in metrics]
        success_count = sum(1 for m in metrics if m.success)
        total_count = len(metrics)
        
        summary = {
            "total_requests": total_count,
            "successful_requests": success_count,
            "failed_requests": total_count - success_count,
            "success_rate": success_count / max(total_count, 1),
            "average_response_time_ms": sum(response_times) / len(response_times),
            "min_response_time_ms": min(response_times),
            "max_response_time_ms": max(response_times),
            "median_response_time_ms": sorted(response_times)[len(response_times) // 2]
        }
        
        return {
            "component": component,
            "summary": summary,
            "metrics": [{
                "timestamp": m.timestamp,
                "operation": m.operation,
                "duration_ms": m.duration_ms,
                "success": m.success,
                "error_message": m.error_message,
                "memory_usage_mb": m.memory_usage_mb,
                "cpu_usage_percent": m.cpu_usage_percent,
                "model_version": m.model_version,
                "batch_size": m.batch_size,
                "embedding_dimension": m.embedding_dimension
            } for m in metrics],
            "period_hours": hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting component metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_alerts(
    hours: int = Query(default=24, ge=1, le=168, description="Hours of alerts to include"),
    severity: Optional[str] = Query(default=None, description="Filter by severity: info, warning, error, critical")
):
    """Get system alerts."""
    try:
        monitoring_service = get_generative_monitoring_service()
        alerts = monitoring_service.get_alerts(hours)
        
        # Filter by severity if specified
        if severity:
            alerts = [alert for alert in alerts if alert.get('severity') == severity]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "period_hours": hours,
            "severity_filter": severity,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/components")
async def list_monitored_components():
    """List all monitored components."""
    try:
        monitoring_service = get_generative_monitoring_service()
        
        components = []
        for component_name, health_status in monitoring_service.health_statuses.items():
            component_info = {
                "name": component_name,
                "status": health_status.status,
                "last_check": health_status.last_check,
                "uptime_hours": health_status.uptime_seconds / 3600,
                "error_rate": health_status.error_rate,
                "response_time_ms": health_status.response_time_ms,
                "total_requests": monitoring_service.request_counts.get(component_name, 0)
            }
            components.append(component_info)
        
        # Sort by name
        components.sort(key=lambda x: x['name'])
        
        return {
            "components": components,
            "total_count": len(components),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing components: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register/{component}")
async def register_component(component: str):
    """Register a new component for monitoring."""
    try:
        monitoring_service = get_generative_monitoring_service()
        monitoring_service.register_component(component)
        
        return {
            "message": f"Component '{component}' registered for monitoring",
            "component": component,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error registering component: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/health/{component}")
async def update_component_health(
    component: str,
    status: str,
    response_time_ms: float,
    details: Optional[Dict[str, Any]] = None
):
    """Update health status for a component."""
    try:
        if status not in ['healthy', 'degraded', 'unhealthy']:
            raise HTTPException(
                status_code=400, 
                detail="Status must be one of: healthy, degraded, unhealthy"
            )
        
        monitoring_service = get_generative_monitoring_service()
        monitoring_service.update_health_status(
            component=component,
            status=status,
            response_time_ms=response_time_ms,
            details=details
        )
        
        return {
            "message": f"Health status updated for component '{component}'",
            "component": component,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating component health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard")
async def get_monitoring_dashboard():
    """Get comprehensive monitoring dashboard data."""
    try:
        monitoring_service = get_generative_monitoring_service()
        
        # Get system overview
        overview = monitoring_service.get_system_overview()
        
        # Get recent alerts
        alerts = monitoring_service.get_alerts(hours=24)
        
        # Get component health
        components = []
        for component_name, health_status in monitoring_service.health_statuses.items():
            component_info = {
                "name": component_name,
                "status": health_status.status,
                "error_rate": health_status.error_rate,
                "response_time_ms": health_status.response_time_ms,
                "uptime_hours": health_status.uptime_seconds / 3600
            }
            components.append(component_info)
        
        # Calculate alert summary
        alert_summary = {
            "total": len(alerts),
            "critical": len([a for a in alerts if a.get('severity') == 'critical']),
            "warning": len([a for a in alerts if a.get('severity') == 'warning']),
            "error": len([a for a in alerts if a.get('severity') == 'error'])
        }
        
        return {
            "dashboard": {
                "system_overview": overview,
                "components": components,
                "recent_alerts": alerts[:10],  # Last 10 alerts
                "alert_summary": alert_summary,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint for load balancers
@router.get("/ping")
async def ping():
    """Simple ping endpoint for health checks."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}