# FlashFit AI Deployment Guide

This guide covers deploying FlashFit AI in various environments, from local development to production cloud deployments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Production Deployment](#production-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Environment Configuration](#environment-configuration)
6. [Security Considerations](#security-considerations)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB+ available space
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Required Software

- Docker 20.10+
- Docker Compose 2.0+
- Git
- Make (optional, for convenience commands)

### Hardware Recommendations

#### Development Environment
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 20GB

#### Production Environment
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 100GB+ SSD
- **GPU**: Optional, for faster ML inference

## Local Development

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/flashfit-ai.git
   cd flashfit-ai
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start development environment:**
   ```bash
   make dev-up
   # or
   docker-compose -f docker-compose.dev.yml up -d
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Redis Commander: http://localhost:8081

### Development Workflow

```bash
# Start services
make dev-up

# View logs
make logs

# Run tests
make test

# Stop services
make dev-down

# Rebuild after changes
make dev-rebuild
```

## Production Deployment

### Using Docker Compose

1. **Prepare the server:**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. **Clone and configure:**
   ```bash
   git clone https://github.com/your-org/flashfit-ai.git
   cd flashfit-ai
   
   # Set up production environment
   cp .env.example .env
   # Edit .env with production values
   ```

3. **Deploy:**
   ```bash
   make prod-up
   # or
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Production Environment Variables

Essential production settings in `.env`:

```bash
# Application
ENVIRONMENT=production
DEBUG=false

# Security
SECRET_KEY=your-super-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key
ALLOWED_HOSTS=your-domain.com,www.your-domain.com

# Database
REDIS_PASSWORD=secure-redis-password

# SSL/TLS
SSL_ENABLED=true
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem

# External Services
CLOUDINARY_CLOUD_NAME=your-cloudinary-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret
```

## Cloud Deployment

### AWS Deployment

#### Using AWS ECS

1. **Create ECS Cluster:**
   ```bash
   aws ecs create-cluster --cluster-name flashfit-ai
   ```

2. **Build and push images:**
   ```bash
   # Build images
   docker build -t flashfit-backend ./backend
   docker build -t flashfit-frontend ./frontend
   
   # Tag for ECR
   docker tag flashfit-backend:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/flashfit-backend:latest
   docker tag flashfit-frontend:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/flashfit-frontend:latest
   
   # Push to ECR
   docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/flashfit-backend:latest
   docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/flashfit-frontend:latest
   ```

3. **Create task definitions and services using the AWS Console or CLI**

#### Using AWS App Runner

1. **Create apprunner.yaml:**
   ```yaml
   version: 1.0
   runtime: docker
   build:
     commands:
       build:
         - echo "Build started on `date`"
         - docker build -t flashfit-backend ./backend
   run:
     runtime-version: latest
     command: uvicorn main:app --host 0.0.0.0 --port 8000
     network:
       port: 8000
       env:
         - name: ENVIRONMENT
           value: production
   ```

### Google Cloud Platform

#### Using Cloud Run

1. **Build and deploy backend:**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/flashfit-backend ./backend
   gcloud run deploy flashfit-backend \
     --image gcr.io/PROJECT-ID/flashfit-backend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

2. **Build and deploy frontend:**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/flashfit-frontend ./frontend
   gcloud run deploy flashfit-frontend \
     --image gcr.io/PROJECT-ID/flashfit-frontend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

### Azure Deployment

#### Using Azure Container Instances

1. **Create resource group:**
   ```bash
   az group create --name flashfit-rg --location eastus
   ```

2. **Deploy containers:**
   ```bash
   az container create \
     --resource-group flashfit-rg \
     --name flashfit-backend \
     --image your-registry/flashfit-backend:latest \
     --dns-name-label flashfit-api \
     --ports 8000
   ```

### Kubernetes Deployment

1. **Create namespace:**
   ```yaml
   apiVersion: v1
   kind: Namespace
   metadata:
     name: flashfit-ai
   ```

2. **Deploy backend:**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: flashfit-backend
     namespace: flashfit-ai
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: flashfit-backend
     template:
       metadata:
         labels:
           app: flashfit-backend
       spec:
         containers:
         - name: backend
           image: flashfit-backend:latest
           ports:
           - containerPort: 8000
           env:
           - name: ENVIRONMENT
             value: "production"
   ```

3. **Create service:**
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: flashfit-backend-service
     namespace: flashfit-ai
   spec:
     selector:
       app: flashfit-backend
     ports:
     - protocol: TCP
       port: 80
       targetPort: 8000
     type: LoadBalancer
   ```

## Environment Configuration

### SSL/TLS Setup

#### Using Let's Encrypt with Nginx

1. **Install Certbot:**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   ```

2. **Obtain certificate:**
   ```bash
   sudo certbot --nginx -d your-domain.com -d www.your-domain.com
   ```

3. **Auto-renewal:**
   ```bash
   sudo crontab -e
   # Add: 0 12 * * * /usr/bin/certbot renew --quiet
   ```

#### Using Custom SSL Certificates

1. **Place certificates:**
   ```bash
   sudo mkdir -p /etc/ssl/flashfit
   sudo cp your-cert.pem /etc/ssl/flashfit/
   sudo cp your-key.pem /etc/ssl/flashfit/
   sudo chmod 600 /etc/ssl/flashfit/*
   ```

2. **Update nginx configuration:**
   ```nginx
   server {
       listen 443 ssl http2;
       server_name your-domain.com;
       
       ssl_certificate /etc/ssl/flashfit/your-cert.pem;
       ssl_certificate_key /etc/ssl/flashfit/your-key.pem;
       
       # SSL configuration
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
   }
   ```

### Database Configuration

#### Redis Cluster Setup

1. **Create Redis cluster configuration:**
   ```bash
   # redis-cluster.conf
   port 7000
   cluster-enabled yes
   cluster-config-file nodes.conf
   cluster-node-timeout 5000
   appendonly yes
   ```

2. **Start cluster nodes:**
   ```bash
   redis-server redis-cluster.conf
   ```

3. **Create cluster:**
   ```bash
   redis-cli --cluster create 127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 \
     127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 --cluster-replicas 1
   ```

### Load Balancing

#### Nginx Load Balancer

```nginx
upstream flashfit_backend {
    least_conn;
    server backend1:8000 weight=3;
    server backend2:8000 weight=2;
    server backend3:8000 weight=1;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://flashfit_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Security Considerations

### Network Security

1. **Firewall configuration:**
   ```bash
   # Allow only necessary ports
   sudo ufw allow 22/tcp   # SSH
   sudo ufw allow 80/tcp   # HTTP
   sudo ufw allow 443/tcp  # HTTPS
   sudo ufw enable
   ```

2. **Docker network isolation:**
   ```yaml
   networks:
     frontend:
       driver: bridge
     backend:
       driver: bridge
       internal: true
   ```

### Application Security

1. **Environment variables:**
   ```bash
   # Use strong, unique secrets
   SECRET_KEY=$(openssl rand -hex 32)
   JWT_SECRET_KEY=$(openssl rand -hex 32)
   REDIS_PASSWORD=$(openssl rand -hex 16)
   ```

2. **CORS configuration:**
   ```python
   ALLOWED_ORIGINS = [
       "https://your-domain.com",
       "https://www.your-domain.com"
   ]
   ```

3. **Rate limiting:**
   ```nginx
   limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
   
   location /api/ {
       limit_req zone=api burst=20 nodelay;
   }
   ```

### Container Security

1. **Non-root user:**
   ```dockerfile
   RUN adduser --disabled-password --gecos '' appuser
   USER appuser
   ```

2. **Security scanning:**
   ```bash
   # Scan images for vulnerabilities
   docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
     aquasec/trivy image flashfit-backend:latest
   ```

## Monitoring and Logging

### Prometheus Monitoring

1. **Configure Prometheus:**
   ```yaml
   global:
     scrape_interval: 15s
   
   scrape_configs:
     - job_name: 'flashfit-backend'
       static_configs:
         - targets: ['backend:8000']
   ```

2. **Add custom metrics:**
   ```python
   from prometheus_client import Counter, Histogram
   
   REQUEST_COUNT = Counter('requests_total', 'Total requests')
   REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')
   ```

### Grafana Dashboards

1. **Import dashboard:**
   ```bash
   curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
     -H "Content-Type: application/json" \
     -d @grafana-dashboard.json
   ```

### Centralized Logging

#### ELK Stack

1. **Logstash configuration:**
   ```ruby
   input {
     beats {
       port => 5044
     }
   }
   
   filter {
     if [fields][service] == "flashfit-backend" {
       json {
         source => "message"
       }
     }
   }
   
   output {
     elasticsearch {
       hosts => ["elasticsearch:9200"]
       index => "flashfit-%{+YYYY.MM.dd}"
     }
   }
   ```

2. **Filebeat configuration:**
   ```yaml
   filebeat.inputs:
   - type: container
     paths:
       - '/var/lib/docker/containers/*/*.log'
     
   output.logstash:
     hosts: ["logstash:5044"]
   ```

## Troubleshooting

### Common Issues

#### Container Won't Start

1. **Check logs:**
   ```bash
   docker-compose logs backend
   docker-compose logs frontend
   ```

2. **Check resource usage:**
   ```bash
   docker stats
   df -h
   free -m
   ```

#### Database Connection Issues

1. **Test Redis connection:**
   ```bash
   docker-compose exec redis redis-cli ping
   ```

2. **Check network connectivity:**
   ```bash
   docker-compose exec backend ping redis
   ```

#### Performance Issues

1. **Monitor resource usage:**
   ```bash
   htop
   iotop
   nethogs
   ```

2. **Optimize Docker:**
   ```bash
   # Clean up unused resources
   docker system prune -a
   
   # Increase memory limits
   docker-compose up -d --scale backend=3
   ```

### Health Checks

1. **Application health:**
   ```bash
   curl -f http://localhost:8000/health || exit 1
   ```

2. **Database health:**
   ```bash
   redis-cli ping | grep PONG || exit 1
   ```

3. **Automated monitoring:**
   ```bash
   #!/bin/bash
   # health-check.sh
   
   services=("backend:8000/health" "frontend:3000")
   
   for service in "${services[@]}"; do
       if ! curl -f "http://$service" > /dev/null 2>&1; then
           echo "Service $service is down!"
           # Send alert
       fi
   done
   ```

### Backup and Recovery

#### Redis Backup

1. **Create backup:**
   ```bash
   docker-compose exec redis redis-cli BGSAVE
   docker cp $(docker-compose ps -q redis):/data/dump.rdb ./backup/
   ```

2. **Restore backup:**
   ```bash
   docker-compose down
   docker cp ./backup/dump.rdb $(docker-compose ps -q redis):/data/
   docker-compose up -d
   ```

#### Application Data Backup

1. **Backup uploads:**
   ```bash
   tar -czf uploads-backup-$(date +%Y%m%d).tar.gz data/uploads/
   ```

2. **Automated backup script:**
   ```bash
   #!/bin/bash
   # backup.sh
   
   BACKUP_DIR="/backups/$(date +%Y%m%d)"
   mkdir -p "$BACKUP_DIR"
   
   # Backup Redis
   docker-compose exec redis redis-cli BGSAVE
   docker cp $(docker-compose ps -q redis):/data/dump.rdb "$BACKUP_DIR/"
   
   # Backup uploads
   tar -czf "$BACKUP_DIR/uploads.tar.gz" data/uploads/
   
   # Upload to cloud storage
   aws s3 sync "$BACKUP_DIR" s3://your-backup-bucket/flashfit-ai/
   ```

## Performance Optimization

### Backend Optimization

1. **Increase worker processes:**
   ```bash
   uvicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
   ```

2. **Enable caching:**
   ```python
   from fastapi_cache import FastAPICache
   from fastapi_cache.backends.redis import RedisBackend
   
   FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
   ```

### Frontend Optimization

1. **Build optimization:**
   ```javascript
   // vite.config.js
   export default {
     build: {
       rollupOptions: {
         output: {
           manualChunks: {
             vendor: ['react', 'react-dom'],
             ui: ['@mui/material']
           }
         }
       }
     }
   }
   ```

2. **CDN configuration:**
   ```nginx
   location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
       expires 1y;
       add_header Cache-Control "public, immutable";
   }
   ```

This deployment guide provides comprehensive instructions for deploying FlashFit AI in various environments. Choose the deployment method that best fits your infrastructure and requirements.