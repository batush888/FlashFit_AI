# FlashFit AI - Production Deployment Guide

This guide covers the containerized production deployment of FlashFit AI with Docker Compose.

## Architecture Overview

FlashFit AI uses a microservices architecture with the following components:

- **Backend API** (Port 8080): FastAPI application with ML models
- **Frontend UI** (Port 3000): React application with Vite
- **Monitoring Service** (Port 9090): Prometheus metrics and health checks
- **PostgreSQL Database** (Port 5432): Primary data storage
- **Redis Cache** (Port 6379): Session and caching layer
- **Nginx Reverse Proxy** (Port 80/443): Unified access point
- **Prometheus** (Port 9091): Metrics collection
- **Grafana** (Port 3001): Monitoring dashboards

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space

## Environment Configuration

### 1. Environment Variables

Copy the `.env` file and customize for your environment:

```bash
cp .env .env.production
```

Key variables to configure:

```env
# Service Ports
BACKEND_PORT=8080
FRONTEND_PORT=3000
MONITOR_PORT=9090

# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=flashfit_ai
POSTGRES_USER=flashfit_user
POSTGRES_PASSWORD=your_secure_password_here

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# Production Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info
```

## Deployment Options

### Option 1: Development Deployment

For development with hot reload and debugging:

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop services
docker-compose -f docker-compose.dev.yml down
```

### Option 2: Production Deployment

For production with optimized builds and reverse proxy:

```bash
# Build and start all services
docker-compose -f docker-compose.production.yml up -d --build

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Stop services
docker-compose -f docker-compose.production.yml down
```

## Service Access

### Development Mode
- Frontend: http://localhost:3000
- Backend API: http://localhost:8080
- Monitoring: http://localhost:9090
- Prometheus: http://localhost:9091
- Grafana: http://localhost:3001
- PostgreSQL: localhost:5432
- Redis: localhost:6379

### Production Mode (with Nginx)
- Application: http://localhost (or your domain)
- API: http://localhost/api
- Monitoring: http://localhost/metrics
- Grafana: http://localhost/grafana
- Prometheus: http://localhost/prometheus

## Health Checks

All services include health checks. Monitor service health:

```bash
# Check all service status
docker-compose ps

# Check specific service health
docker-compose exec monitoring python test_db_connection.py

# View monitoring metrics
curl http://localhost:9090/metrics
```

## Database Setup

### Initial Setup

The PostgreSQL container automatically runs the initialization script:

```sql
-- Located at: postgres/init.sql
-- Creates tables for:
-- - User management
-- - Fashion items and wardrobe
-- - Outfit history and feedback
-- - System and performance metrics
-- - Health check logs
```

### Manual Database Operations

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U flashfit_user -d flashfit_ai

# Backup database
docker-compose exec postgres pg_dump -U flashfit_user flashfit_ai > backup.sql

# Restore database
docker-compose exec -T postgres psql -U flashfit_user flashfit_ai < backup.sql
```

## Monitoring and Observability

### Prometheus Metrics

The monitoring service exposes metrics at `/metrics`:

- System metrics (CPU, memory, disk)
- Application metrics (API response times, error rates)
- Database metrics (query performance, connection health)
- ML model metrics (inference time, accuracy)

### Grafana Dashboards

Access Grafana at http://localhost:3001 (admin/admin):

1. **System Overview**: CPU, memory, disk usage
2. **API Performance**: Response times, error rates, throughput
3. **Database Performance**: Query times, connection pools
4. **ML Model Performance**: Inference metrics, model accuracy

### Log Aggregation

View logs from all services:

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f monitoring
```

## Scaling and Performance

### Horizontal Scaling

Scale individual services:

```bash
# Scale backend instances
docker-compose up -d --scale backend=3

# Scale monitoring instances
docker-compose up -d --scale monitoring=2
```

### Resource Limits

Configure resource limits in docker-compose files:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

## Security Considerations

### Production Security Checklist

- [ ] Change default passwords in `.env.production`
- [ ] Use HTTPS with SSL certificates
- [ ] Configure firewall rules
- [ ] Enable authentication for Grafana/Prometheus
- [ ] Regular security updates
- [ ] Backup encryption

### SSL/TLS Setup

For HTTPS in production, update nginx configuration:

```nginx
server {
    listen 443 ssl http2;
    ssl_certificate /etc/ssl/certs/flashfit.crt;
    ssl_certificate_key /etc/ssl/private/flashfit.key;
    # ... rest of configuration
}
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8080
   
   # Change ports in .env file
   BACKEND_PORT=8081
   ```

2. **Database Connection Issues**
   ```bash
   # Test database connectivity
   docker-compose exec monitoring python test_db_connection.py
   
   # Check PostgreSQL logs
   docker-compose logs postgres
   ```

3. **Memory Issues**
   ```bash
   # Check container resource usage
   docker stats
   
   # Increase memory limits
   # Edit docker-compose.yml deploy.resources.limits.memory
   ```

4. **Build Issues**
   ```bash
   # Clean rebuild
   docker-compose down
   docker system prune -a
   docker-compose up -d --build
   ```

### Log Analysis

```bash
# Search for errors
docker-compose logs | grep -i error

# Monitor real-time logs
docker-compose logs -f --tail=100

# Export logs
docker-compose logs > deployment.log
```

## Backup and Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)

# Database backup
docker-compose exec postgres pg_dump -U flashfit_user flashfit_ai > "backup_${DATE}.sql"

# Redis backup
docker-compose exec redis redis-cli BGSAVE

# Application data backup
docker-compose exec backend tar -czf "/tmp/app_backup_${DATE}.tar.gz" /app/data
```

### Recovery Procedures

```bash
# Restore database
docker-compose exec -T postgres psql -U flashfit_user flashfit_ai < backup_20240831.sql

# Restore Redis
docker-compose exec redis redis-cli FLUSHALL
docker cp backup.rdb $(docker-compose ps -q redis):/data/dump.rdb
docker-compose restart redis
```

## Maintenance

### Regular Maintenance Tasks

1. **Update Dependencies**
   ```bash
   # Update base images
   docker-compose pull
   docker-compose up -d
   ```

2. **Clean Up**
   ```bash
   # Remove unused images
   docker image prune -a
   
   # Remove unused volumes
   docker volume prune
   ```

3. **Monitor Disk Usage**
   ```bash
   # Check Docker disk usage
   docker system df
   
   # Check container logs size
   docker-compose logs --tail=0 | wc -l
   ```

## Support and Documentation

- **Architecture Documentation**: See `docs/architecture.md`
- **API Documentation**: http://localhost:8080/docs (when running)
- **Monitoring Dashboards**: http://localhost:3001
- **Health Checks**: http://localhost:9090/health

For issues and support, check the troubleshooting section above or review service logs.