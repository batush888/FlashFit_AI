# FlashFit AI DevOps Documentation

## Overview

This document provides comprehensive DevOps guidance for deploying, scaling, and maintaining FlashFit AI in production environments.

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores, 2.5GHz
- **RAM**: 8GB (4GB for models, 2GB for vector indices, 2GB for system)
- **Storage**: 20GB SSD (models: 5GB, indices: 2GB, logs: 1GB, buffer: 12GB)
- **Network**: 100Mbps bandwidth

### Recommended Production
- **CPU**: 8 cores, 3.0GHz+ (Intel Xeon or AMD EPYC)
- **RAM**: 16GB+ (better for concurrent users)
- **Storage**: 50GB+ NVMe SSD
- **Network**: 1Gbps+ bandwidth
- **GPU**: Optional NVIDIA GPU for faster inference (RTX 3080+ or Tesla T4+)

## Port Configuration

### Default Ports
- **Backend API**: 8080
- **Frontend**: 3000
- **Nginx Proxy**: 80, 443
- **Redis** (optional): 6379
- **Prometheus** (monitoring): 9090
- **Grafana** (dashboards): 3001

### Port Mapping
```yaml
# docker-compose.yml
services:
  backend:
    ports:
      - "8080:8080"
  frontend:
    ports:
      - "3000:3000"
  nginx:
    ports:
      - "80:80"
      - "443:443"
```

## Containerization

### Docker Setup

#### Backend Dockerfile
```dockerfile
# backend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 flashfit && chown -R flashfit:flashfit /app
USER flashfit

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Frontend Dockerfile
```dockerfile
# frontend/Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source and build
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose Configurations

#### Development Environment
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    volumes:
      - ./backend:/app
      - ./data:/app/data
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
      - LOG_LEVEL=debug
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    volumes:
      - ./frontend/src:/app/src
    environment:
      - NODE_ENV=development
      - VITE_API_URL=http://localhost:8080
    restart: unless-stopped
    depends_on:
      - backend

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend
      - frontend
    restart: unless-stopped

volumes:
  model_cache:
  vector_indices:
```

#### Production Environment
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    volumes:
      - model_cache:/app/models
      - vector_indices:/app/data
      - ./logs:/app/logs
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
      - LOG_LEVEL=info
      - WORKERS=4
    restart: always
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    environment:
      - NODE_ENV=production
      - VITE_API_URL=https://api.flashfit.ai
    restart: always
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '1'

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - backend
      - frontend
    restart: always

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: always
    command: redis-server --appendonly yes

volumes:
  model_cache:
  vector_indices:
  redis_data:
```

## Deployment Strategies

### 1. Single Server Deployment

```bash
#!/bin/bash
# deploy-single.sh

set -e

echo "🚀 Deploying FlashFit AI (Single Server)"

# Pull latest code
git pull origin main

# Build and deploy
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 30

# Health checks
echo "🔍 Running health checks..."
curl -f http://localhost:8080/health || exit 1
curl -f http://localhost:3000 || exit 1

echo "✅ Deployment successful!"
```

### 2. Blue-Green Deployment

```bash
#!/bin/bash
# deploy-blue-green.sh

set -e

CURRENT_ENV=$(docker-compose ps | grep "flashfit" | head -1 | awk '{print $1}' | grep -o "blue\|green" || echo "blue")
NEW_ENV=$([ "$CURRENT_ENV" = "blue" ] && echo "green" || echo "blue")

echo "🔄 Blue-Green Deployment: $CURRENT_ENV -> $NEW_ENV"

# Deploy to new environment
docker-compose -f docker-compose.$NEW_ENV.yml build
docker-compose -f docker-compose.$NEW_ENV.yml up -d

# Health check new environment
echo "🔍 Health checking $NEW_ENV environment..."
sleep 30
curl -f http://localhost:808$([ "$NEW_ENV" = "blue" ] && echo "1" || echo "2")/health

# Switch traffic
echo "🔀 Switching traffic to $NEW_ENV"
# Update load balancer configuration
./scripts/switch-traffic.sh $NEW_ENV

# Cleanup old environment
echo "🧹 Cleaning up $CURRENT_ENV environment"
sleep 60  # Grace period
docker-compose -f docker-compose.$CURRENT_ENV.yml down

echo "✅ Blue-Green deployment complete!"
```

### 3. Rolling Deployment

```bash
#!/bin/bash
# deploy-rolling.sh

set -e

REPLICAS=3

echo "🔄 Rolling Deployment (${REPLICAS} replicas)"

for i in $(seq 1 $REPLICAS); do
    echo "📦 Updating replica $i/$REPLICAS"
    
    # Stop one replica
    docker-compose -f docker-compose.prod.yml stop backend_$i
    
    # Update and restart
    docker-compose -f docker-compose.prod.yml build backend
    docker-compose -f docker-compose.prod.yml up -d backend_$i
    
    # Health check
    sleep 30
    curl -f http://localhost:808$i/health || exit 1
    
    echo "✅ Replica $i updated successfully"
done

echo "✅ Rolling deployment complete!"
```

## Scaling Strategies

### Horizontal Scaling

#### Load Balancer Configuration (Nginx)
```nginx
# nginx-lb.conf
upstream backend_servers {
    least_conn;
    server backend_1:8080 max_fails=3 fail_timeout=30s;
    server backend_2:8080 max_fails=3 fail_timeout=30s;
    server backend_3:8080 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.flashfit.ai;
    
    location / {
        proxy_pass http://backend_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Health checks
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
    }
    
    location /health {
        access_log off;
        proxy_pass http://backend_servers/health;
    }
}
```

#### Auto-Scaling with Docker Swarm
```yaml
# docker-stack.yml
version: '3.8'

services:
  backend:
    image: flashfit/backend:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 30s
        order: start-first
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'
    networks:
      - flashfit_network
    volumes:
      - model_cache:/app/models
      - vector_indices:/app/data

  frontend:
    image: flashfit/frontend:latest
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
    networks:
      - flashfit_network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.role == manager
    networks:
      - flashfit_network
    configs:
      - source: nginx_config
        target: /etc/nginx/nginx.conf

networks:
  flashfit_network:
    driver: overlay
    attachable: true

volumes:
  model_cache:
  vector_indices:

configs:
  nginx_config:
    file: ./nginx-lb.conf
```

### Vertical Scaling

#### Resource Optimization
```yaml
# High-performance configuration
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8'
        reservations:
          memory: 8G
          cpus: '4'
    environment:
      - WORKERS=8
      - WORKER_CLASS=uvicorn.workers.UvicornWorker
      - MAX_REQUESTS=1000
      - MAX_REQUESTS_JITTER=100
      - PRELOAD_APP=true
    volumes:
      - type: tmpfs
        target: /tmp
        tmpfs:
          size: 2G
```

## Monitoring and Observability

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'flashfit-backend'
    static_configs:
      - targets: ['backend:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'flashfit-frontend'
    static_configs:
      - targets: ['frontend:3000']
    metrics_path: '/metrics'
    
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "FlashFit AI Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Model Inference Time",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(model_inference_duration_seconds_sum[5m]) / rate(model_inference_duration_seconds_count[5m])",
            "legendFormat": "{{model_name}}"
          }
        ]
      }
    ]
  }
}
```

### Health Checks

#### Application Health Endpoint
```python
# backend/health.py
from fastapi import APIRouter
from datetime import datetime
import psutil
import os

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        },
        "models": {
            "clip_loaded": check_model_loaded("clip"),
            "blip_loaded": check_model_loaded("blip"),
            "fashion_loaded": check_model_loaded("fashion")
        },
        "vector_stores": {
            "clip_index_size": get_index_size("clip"),
            "blip_index_size": get_index_size("blip"),
            "fashion_index_size": get_index_size("fashion")
        }
    }

@router.get("/ready")
async def readiness_check():
    # Check if all critical components are ready
    checks = {
        "models_loaded": all([
            check_model_loaded("clip"),
            check_model_loaded("blip"),
            check_model_loaded("fashion")
        ]),
        "vector_stores_ready": all([
            check_vector_store("clip"),
            check_vector_store("blip"),
            check_vector_store("fashion")
        ]),
        "database_connected": check_database_connection()
    }
    
    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        return {"status": "not_ready", "checks": checks}, 503
```

## Security Configuration

### SSL/TLS Setup
```nginx
# nginx-ssl.conf
server {
    listen 443 ssl http2;
    server_name flashfit.ai www.flashfit.ai;
    
    ssl_certificate /etc/nginx/ssl/flashfit.ai.crt;
    ssl_certificate_key /etc/nginx/ssl/flashfit.ai.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    location / {
        proxy_pass http://frontend:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /api/ {
        proxy_pass http://backend:8080/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name flashfit.ai www.flashfit.ai;
    return 301 https://$server_name$request_uri;
}

# Rate limiting
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
}
```

### Environment Variables
```bash
# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# Security
SECRET_KEY=your-super-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
ALLOWED_HOSTS=flashfit.ai,www.flashfit.ai
CORS_ORIGINS=https://flashfit.ai,https://www.flashfit.ai

# Database
DATABASE_URL=postgresql://user:password@db:5432/flashfit
REDIS_URL=redis://redis:6379/0

# ML Models
MODEL_CACHE_DIR=/app/models
VECTOR_STORE_DIR=/app/data
MAX_BATCH_SIZE=32
INFERENCE_TIMEOUT=30

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
LOG_FORMAT=json
```

## Backup and Recovery

### Backup Strategy
```bash
#!/bin/bash
# backup.sh

set -e

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

echo "📦 Creating backup: $BACKUP_DIR"

# Backup vector indices
echo "💾 Backing up vector indices..."
cp -r ./data/*.index $BACKUP_DIR/
cp -r ./data/*.json $BACKUP_DIR/

# Backup user data
echo "👥 Backing up user data..."
docker exec flashfit_backend_1 python -c "from services.backup import create_backup; create_backup('$BACKUP_DIR')"

# Backup configuration
echo "⚙️ Backing up configuration..."
cp docker-compose.prod.yml $BACKUP_DIR/
cp nginx.conf $BACKUP_DIR/
cp .env.production $BACKUP_DIR/

# Create archive
echo "🗜️ Creating archive..."
tar -czf "$BACKUP_DIR.tar.gz" -C /backups $(basename $BACKUP_DIR)
rm -rf $BACKUP_DIR

# Upload to cloud storage (optional)
if [ "$CLOUD_BACKUP" = "true" ]; then
    echo "☁️ Uploading to cloud storage..."
    aws s3 cp "$BACKUP_DIR.tar.gz" s3://flashfit-backups/
fi

echo "✅ Backup complete: $BACKUP_DIR.tar.gz"
```

### Recovery Procedure
```bash
#!/bin/bash
# restore.sh

set -e

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

echo "🔄 Restoring from backup: $BACKUP_FILE"

# Stop services
echo "⏹️ Stopping services..."
docker-compose -f docker-compose.prod.yml down

# Extract backup
echo "📦 Extracting backup..."
RESTORE_DIR="/tmp/restore_$(date +%s)"
mkdir -p $RESTORE_DIR
tar -xzf $BACKUP_FILE -C $RESTORE_DIR

# Restore vector indices
echo "💾 Restoring vector indices..."
cp $RESTORE_DIR/*.index ./data/
cp $RESTORE_DIR/*.json ./data/

# Restore configuration
echo "⚙️ Restoring configuration..."
cp $RESTORE_DIR/docker-compose.prod.yml ./
cp $RESTORE_DIR/nginx.conf ./
cp $RESTORE_DIR/.env.production ./

# Start services
echo "🚀 Starting services..."
docker-compose -f docker-compose.prod.yml up -d

# Cleanup
rm -rf $RESTORE_DIR

echo "✅ Restore complete!"
```

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage by service
docker stats --no-stream

# Restart memory-heavy services
docker-compose restart backend

# Clear model cache
docker exec flashfit_backend_1 python -c "from models import clear_cache; clear_cache()"
```

#### Slow Response Times
```bash
# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8080/api/fusion/recommend

# Monitor model inference times
docker logs flashfit_backend_1 | grep "inference_time"

# Check vector store performance
docker exec flashfit_backend_1 python -c "from models.vector_store import benchmark; benchmark()"
```

#### Service Dependencies
```bash
# Check service health
docker-compose ps
docker-compose logs backend
docker-compose logs frontend

# Restart in dependency order
docker-compose restart backend
sleep 30
docker-compose restart frontend
docker-compose restart nginx
```

### Log Analysis
```bash
# Centralized logging with ELK stack
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  elasticsearch:7.14.0

docker run -d \
  --name kibana \
  -p 5601:5601 \
  --link elasticsearch:elasticsearch \
  kibana:7.14.0

# Configure log shipping
# Add to docker-compose.yml:
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

This DevOps documentation provides a comprehensive foundation for deploying and maintaining FlashFit AI in production environments.