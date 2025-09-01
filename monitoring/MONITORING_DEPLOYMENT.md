# FlashFit AI Monitoring Deployment Guide

## FlashFit AI Phase 2 Architecture Overview

![FlashFit AI Phase 2 Architecture](../docs/flashfit-ai-phase2-architecture.svg)
*FlashFit AI Phase 2 Architecture: Service layers, ports, data flow, and monitoring.*

### System Architecture

FlashFit AI Phase 2 implements a comprehensive microservices architecture with advanced monitoring and observability:

**Service Layers & Ports:**
- **Frontend Layer** (Port 3000): Vite development server with React UI, production-ready builds
- **Backend API Layer** (Port 8080): FastAPI server with endpoints `/api/recommend`, `/api/feedback`, `/api/faiss`
- **AI Model Layer**: Tri-model ensemble (CLIP Encoder, BLIP Captioner, Fashion Encoder) with fusion reranker
- **Monitoring Layer**: Modular monitoring (Port 9092) and FAISS monitoring (Port 9091)
- **Observability Layer**: Prometheus (Port 9090) metrics collection, Grafana (Port 3001) dashboards
- **Reverse Proxy Layer**: Nginx/Traefik routing `/app` → Frontend, `/api` → Backend, `/metrics` → Monitoring

**Data Flow:**
```
User → Frontend (3000) → Backend (8080) → AI Models → Redis/PostgreSQL
                                                    ↓
Metrics → Prometheus (9090) → Grafana (3001) ← Monitoring Services (9091/9092)
```

**Key Features:**
- **Per-user embeddings** with feedback loop integration for personalization
- **FAISS index monitoring** with health metrics and search latency tracking
- **Real-time alerting** engine with configurable rules
- **Containerized deployment** with Docker orchestration
- **Production-ready** reverse proxy with SSL, CORS, and rate limiting

**Metrics Endpoints:**
- Modular Monitoring: [http://localhost:9092/metrics](http://localhost:9092/metrics)
- FAISS Monitoring: [http://localhost:9091/metrics](http://localhost:9091/metrics)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana Dashboard: [http://localhost:3001](http://localhost:3001)

## Current Status

✅ **Active Monitoring Services:**
- **Modular Monitoring System**: Running on port 9092
  - Metrics endpoint: http://localhost:9092/metrics
  - Collecting system metrics, service health, Redis metrics
  - Alerting engine with configurable rules

- **FAISS Monitor**: Running on port 9091
  - Metrics endpoint: http://localhost:9091/metrics
  - Monitoring FAISS index health and performance
  - 3 indices monitored: clip_fashion, blip_fashion, fashion_specific

## Phase 2 Deployment Options

### Option 1: Docker-based Deployment (Recommended for Production)

**Prerequisites:**
```bash
# Install Docker Desktop for macOS
brew install --cask docker
# Or download from https://www.docker.com/products/docker-desktop

# Start Docker Desktop and verify installation
docker --version
docker compose --version
```

**Deploy Monitoring Stack:**
```bash
cd monitoring

# Start Prometheus, Grafana, and AlertManager
docker compose -f docker-compose.monitoring.yml up -d prometheus grafana alertmanager

# Verify services are running
docker compose -f docker-compose.monitoring.yml ps
```

**Access Points:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/flashfit_admin)
- AlertManager: http://localhost:9093

### Option 2: Native Installation (Development)

**Install Prometheus:**
```bash
# Using Homebrew
brew install prometheus

# Start Prometheus with our config
prometheus --config.file=prometheus.yml --storage.tsdb.path=./prometheus_data
```

**Install Grafana:**
```bash
# Using Homebrew
brew install grafana

# Start Grafana
brew services start grafana
# Or run directly: grafana-server --config=/usr/local/etc/grafana/grafana.ini
```

**Access Points:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Option 3: Cloud-based Monitoring

**Grafana Cloud:**
1. Sign up at https://grafana.com/products/cloud/
2. Configure Prometheus remote write in `prometheus.yml`:
```yaml
remote_write:
  - url: https://prometheus-prod-01-eu-west-0.grafana.net/api/prom/push
    basic_auth:
      username: YOUR_INSTANCE_ID
      password: YOUR_API_KEY
```

## Current Metrics Available

### System Metrics (Port 9092)
- `flashfit_cpu_usage_percent`: CPU utilization
- `flashfit_memory_usage_percent`: Memory utilization
- `flashfit_memory_usage_bytes`: Memory usage in bytes
- `flashfit_disk_usage_percent`: Disk utilization
- `flashfit_service_health_status`: Service health (0=down, 1=up)
- `flashfit_redis_*`: Redis performance metrics
- `flashfit_active_alerts_total`: Active alert count
- `flashfit_metrics_collected_total`: Metrics collection counter

### FAISS Metrics (Port 9091)
- `faiss_index_health_score`: Index health score (0-1)
- `faiss_index_vector_count`: Number of vectors per index
- `faiss_search_duration_seconds`: Search latency histogram
- `faiss_index_size_bytes`: Index file size
- `faiss_memory_usage_bytes`: Memory usage per index

## Grafana Dashboard

A pre-configured dashboard is available at:
`grafana/dashboards/flashfit-system-overview.json`

**Features:**
- System resource monitoring (CPU, Memory, Disk)
- Service health status
- FAISS index performance
- Active alerts table
- Real-time metrics with 5-second refresh

## Alert Configuration

**Alert Rules** (`alert_rules.yml`):
- High CPU usage (>90%)
- High memory usage (>95%)
- Service down alerts
- FAISS index health degradation
- Search latency thresholds

**AlertManager** (`alertmanager.yml`):
- Email notifications for critical alerts
- Webhook integration for custom handlers
- Alert grouping and deduplication

## Testing the Setup

```bash
# Check if monitoring services are running
curl -s http://localhost:9092/metrics | grep flashfit
curl -s http://localhost:9091/metrics | grep faiss

# Test Prometheus targets (after Prometheus is running)
curl -s http://localhost:9090/api/v1/targets

# Check Grafana health (after Grafana is running)
curl -s http://localhost:3001/api/health
```

## Production Considerations

1. **Security:**
   - Change default Grafana admin password
   - Configure HTTPS/TLS for all services
   - Set up proper authentication and authorization

2. **Scalability:**
   - Use external storage for Prometheus (e.g., Thanos)
   - Configure Grafana with external database
   - Set up Prometheus federation for multi-instance setups

3. **Backup:**
   - Regular backup of Grafana dashboards and datasources
   - Prometheus data retention policies
   - Alert rule version control

4. **Monitoring the Monitors:**
   - Set up monitoring for Prometheus itself
   - Configure dead man's switch alerts
   - Monitor disk space for metrics storage

## Troubleshooting

**Common Issues:**

1. **Metrics not appearing:**
   - Check if monitoring services are running
   - Verify Prometheus can reach targets
   - Check firewall/network connectivity

2. **Grafana connection issues:**
   - Verify Prometheus datasource configuration
   - Check Prometheus URL accessibility
   - Review Grafana logs for errors

3. **Alerts not firing:**
   - Verify alert rules syntax
   - Check AlertManager configuration
   - Test notification channels

**Log Locations:**
- Modular Monitor: Check terminal output or configure file logging
- FAISS Monitor: Check terminal output
- Prometheus: `--log.level=debug` for verbose logging
- Grafana: `/usr/local/var/log/grafana/` (Homebrew) or container logs

## Next Steps

1. **Install Docker** (recommended) or **native Prometheus/Grafana**
2. **Deploy monitoring stack** using chosen method
3. **Import Grafana dashboard** from `grafana/dashboards/`
4. **Configure alerts** based on your requirements
5. **Set up notifications** (email, Slack, etc.)
6. **Monitor and tune** alert thresholds based on baseline metrics

The monitoring infrastructure is ready and collecting metrics. Choose your deployment method and follow the corresponding setup instructions above.