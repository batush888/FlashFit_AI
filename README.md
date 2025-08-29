# FlashFit AI - Smart Wardrobe Management System

## Overview

FlashFit AI is an intelligent wardrobe management system that uses computer vision and machine learning to help users organize their clothing, get outfit suggestions, and make better fashion decisions. The system analyzes clothing items using CLIP embeddings and provides personalized recommendations based on weather, occasion, and personal style preferences.

## Features

- **Smart Clothing Recognition**: Automatically categorize and tag clothing items using AI
- **Intelligent Outfit Suggestions**: Get personalized outfit recommendations
- **Weather Integration**: Outfit suggestions based on current weather conditions
- **Wardrobe Analytics**: Track your clothing usage and style preferences
- **User-Friendly Interface**: Modern, responsive web application
- **Secure Authentication**: JWT-based user authentication and authorization

## Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Python 3.10+**: Core programming language
- **CLIP**: OpenAI's vision-language model for clothing analysis
- **Redis**: Caching and session management
- **Uvicorn**: ASGI server for production deployment

### Frontend
- **React 18**: Modern UI library with hooks
- **TypeScript**: Type-safe JavaScript development
- **Vite**: Fast build tool and development server
- **TailwindCSS**: Utility-first CSS framework
- **Zustand**: Lightweight state management
- **React Query**: Data fetching and caching

### Infrastructure
- **Docker**: Containerization for consistent deployments
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Reverse proxy and load balancer
- **Prometheus**: Monitoring and metrics collection
- **Grafana**: Visualization and dashboards

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.10+ (for local development)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd flashfit-ai
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start Development Environment**
   ```bash
   # Using Docker Compose (Recommended)
   make dev-up
   
   # Or manually
   docker-compose -f docker-compose.dev.yml up --build
   ```

4. **Access the Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Redis Commander: http://localhost:8081

### Production Deployment

1. **Configure Environment**
   ```bash
   cp .env.example .env.prod
   # Update production settings in .env.prod
   ```

2. **Deploy with Docker Compose**
   ```bash
   make prod-up
   
   # Or manually
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Enable Monitoring (Optional)**
   ```bash
   docker-compose -f docker-compose.prod.yml --profile monitoring up -d
   ```

## Development

### Local Development (Without Docker)

#### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Available Make Commands

```bash
# Development
make dev-up          # Start development environment
make dev-down        # Stop development environment
make dev-logs        # View development logs

# Production
make prod-up         # Start production environment
make prod-down       # Stop production environment
make prod-logs       # View production logs

# Building
make build           # Build all images
make build-backend   # Build backend image only
make build-frontend  # Build frontend image only

# Testing
make test            # Run all tests
make test-backend    # Run backend tests
make test-frontend   # Run frontend tests

# Utilities
make clean           # Clean up containers and images
make shell-backend   # Access backend container shell
make shell-frontend  # Access frontend container shell
make redis-cli       # Access Redis CLI
```

## API Documentation

The API documentation is automatically generated and available at:
- Development: http://localhost:8000/docs
- Production: https://your-domain.com/docs

### Key Endpoints

- `POST /auth/register` - User registration
- `POST /auth/login` - User authentication
- `POST /clothing/upload` - Upload clothing item
- `GET /clothing/` - Get user's clothing items
- `POST /suggestions/generate` - Generate outfit suggestions
- `GET /wardrobe/analytics` - Get wardrobe analytics

## Configuration

### Environment Variables

Key configuration options (see `.env.example` for full list):

```bash
# Application
ENVIRONMENT=development
DEBUG=true
API_PORT=8000
FRONTEND_PORT=3000

# Security
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret

# Redis
REDIS_URL=redis://redis:6379/0

# ML Models
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
MODEL_PATH=./models

# File Upload
MAX_FILE_SIZE=10485760  # 10MB
UPLOAD_DIR=./uploads
```

## Architecture

### System Architecture
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Frontend  │────│    Nginx    │────│   Backend   │
│   (React)   │    │ (Proxy/LB)  │    │  (FastAPI)  │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                       ┌─────────────┐
                                       │    Redis    │
                                       │  (Cache)    │
                                       └─────────────┘
```

### Data Flow
1. User uploads clothing image via React frontend
2. Image is processed by FastAPI backend
3. CLIP model generates embeddings for the clothing item
4. Item is classified and stored with metadata
5. Recommendations are generated using similarity matching
6. Results are cached in Redis for faster subsequent requests

## Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend
npm test
```

### End-to-End Tests
```bash
# Using Docker
make test
```

## Monitoring

### Prometheus Metrics
- Application performance metrics
- API response times and error rates
- System resource usage
- Custom business metrics

### Grafana Dashboards
- System overview dashboard
- API performance dashboard
- User activity dashboard
- ML model performance dashboard

### Health Checks
- Backend: `GET /health`
- Frontend: `GET /health`
- Redis: Built-in health check

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check if ports are in use
   lsof -i :3000
   lsof -i :8000
   ```

2. **Docker Issues**
   ```bash
   # Clean up Docker resources
   make clean
   docker system prune -a
   ```

3. **Permission Issues**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   ```

4. **Model Loading Issues**
   - Ensure sufficient disk space for CLIP model
   - Check internet connection for model download
   - Verify CUDA availability for GPU acceleration

### Logs

```bash
# View all logs
make dev-logs

# View specific service logs
docker-compose logs backend
docker-compose logs frontend
docker-compose logs redis
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use TypeScript for all new frontend code
- Write tests for new features
- Update documentation as needed
- Use conventional commit messages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review existing issues and discussions

## Roadmap

- [ ] Mobile application (React Native)
- [ ] Advanced style recommendations
- [ ] Social features and outfit sharing
- [ ] Integration with fashion retailers
- [ ] AR try-on functionality
- [ ] Sustainability tracking
- [ ] Multi-language support