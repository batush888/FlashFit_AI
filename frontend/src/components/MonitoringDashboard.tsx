import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { RefreshCw, Activity, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

interface SystemOverview {
  timestamp: number;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  success_rate: number;
  average_response_time_ms: number;
  p95_response_time_ms: number;
  p99_response_time_ms: number;
  memory_usage_percent: number;
  cpu_usage_percent: number;
  disk_usage_percent: number;
  active_components: number;
  healthy_components: number;
  component_health: Record<string, string>;
}

interface ComponentHealth {
  name: string;
  status: string;
  last_check: number;
  uptime_hours: number;
  error_rate: number;
  response_time_ms: number;
  total_requests: number;
}

interface Alert {
  id: string;
  type: string;
  severity: string;
  component: string;
  message: string;
  timestamp: number;
  value?: number;
  threshold?: number;
}

interface MonitoringData {
  system_overview: SystemOverview;
  components: ComponentHealth[];
  recent_alerts: Alert[];
  alert_summary: {
    total: number;
    critical: number;
    warning: number;
    error: number;
  };
  timestamp: string;
}

const MonitoringDashboard: React.FC = () => {
  const [monitoringData, setMonitoringData] = useState<MonitoringData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchMonitoringData = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/monitoring/dashboard');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setMonitoringData(data.dashboard);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch monitoring data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMonitoringData();
  }, []);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(fetchMonitoringData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
        return 'text-green-600';
      case 'degraded':
        return 'text-yellow-600';
      case 'unhealthy':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'degraded':
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'unhealthy':
        return <XCircle className="h-4 w-4 text-red-600" />;
      default:
        return <Activity className="h-4 w-4 text-gray-600" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical':
        return 'destructive';
      case 'error':
        return 'destructive';
      case 'warning':
        return 'secondary';
      default:
        return 'outline';
    }
  };

  const formatUptime = (hours: number) => {
    if (hours < 1) {
      return `${Math.round(hours * 60)}m`;
    } else if (hours < 24) {
      return `${Math.round(hours)}h`;
    } else {
      return `${Math.round(hours / 24)}d`;
    }
  };

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  if (loading && !monitoringData) {
    return (
      <div className="flex items-center justify-center p-8">
        <RefreshCw className="h-6 w-6 animate-spin" />
        <span className="ml-2">Loading monitoring data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert>
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          Error loading monitoring data: {error}
          <Button
            variant="outline"
            size="sm"
            onClick={fetchMonitoringData}
            className="ml-2"
          >
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  if (!monitoringData) {
    return (
      <Alert>
        <AlertDescription>No monitoring data available</AlertDescription>
      </Alert>
    );
  }

  const { system_overview, components, recent_alerts, alert_summary } = monitoringData;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">System Monitoring</h1>
          <p className="text-muted-foreground">
            Real-time monitoring of generative AI components
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant={autoRefresh ? 'default' : 'outline'}
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            <Activity className="h-4 w-4 mr-2" />
            Auto Refresh
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={fetchMonitoringData}
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* System Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <CheckCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(system_overview.success_rate * 100).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              {system_overview.successful_requests} / {system_overview.total_requests} requests
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Response Time</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {Math.round(system_overview.average_response_time_ms)}ms
            </div>
            <p className="text-xs text-muted-foreground">
              P95: {Math.round(system_overview.p95_response_time_ms)}ms
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Resources</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {Math.round(system_overview.cpu_usage_percent)}%
            </div>
            <p className="text-xs text-muted-foreground">
              CPU • {Math.round(system_overview.memory_usage_percent)}% Memory
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Components</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {system_overview.healthy_components} / {system_overview.active_components}
            </div>
            <p className="text-xs text-muted-foreground">Healthy components</p>
          </CardContent>
        </Card>
      </div>

      {/* Tabs for detailed views */}
      <Tabs defaultValue="components" className="space-y-4">
        <TabsList>
          <TabsTrigger value="components">Components</TabsTrigger>
          <TabsTrigger value="alerts">
            Alerts
            {alert_summary.total > 0 && (
              <Badge variant="destructive" className="ml-2">
                {alert_summary.total}
              </Badge>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="components" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Component Health</CardTitle>
              <CardDescription>
                Status and performance metrics for all monitored components
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {components.map((component) => (
                  <div
                    key={component.name}
                    className="flex items-center justify-between p-4 border rounded-lg"
                  >
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(component.status)}
                      <div>
                        <h3 className="font-medium">{component.name}</h3>
                        <p className="text-sm text-muted-foreground">
                          Uptime: {formatUptime(component.uptime_hours)}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <Badge variant={component.status === 'healthy' ? 'default' : 'destructive'}>
                        {component.status}
                      </Badge>
                      <p className="text-sm text-muted-foreground mt-1">
                        {Math.round(component.response_time_ms)}ms • {(component.error_rate * 100).toFixed(1)}% errors
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Alerts</CardTitle>
              <CardDescription>
                Latest system alerts and warnings
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {recent_alerts.length === 0 ? (
                  <p className="text-muted-foreground text-center py-4">
                    No recent alerts
                  </p>
                ) : (
                  recent_alerts.map((alert) => (
                    <div
                      key={alert.id}
                      className="flex items-start justify-between p-3 border rounded-lg"
                    >
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <Badge variant={getSeverityColor(alert.severity)}>
                            {alert.severity}
                          </Badge>
                          <span className="text-sm font-medium">{alert.component}</span>
                        </div>
                        <p className="text-sm text-muted-foreground mb-1">
                          {alert.message}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {formatTimestamp(alert.timestamp)}
                        </p>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MonitoringDashboard;