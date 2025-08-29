# FlashFit AI - Docker Management
.PHONY: help build up down logs clean dev prod test

# Default target
help:
	@echo "FlashFit AI - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start development environment"
	@echo "  make dev-up       - Start development environment"
	@echo "  make dev-build    - Build and start development environment"
	@echo "  make dev-logs     - Show development logs"
	@echo "  make dev-down     - Stop development environment"
	@echo ""
	@echo "Production:"
	@echo "  make prod         - Start production environment"
	@echo "  make prod-up      - Start production environment"
	@echo "  make prod-build   - Build and start production environment"
	@echo "  make prod-logs    - Show production logs"
	@echo "  make prod-down    - Stop production environment"
	@echo ""
	@echo "Utilities:"
	@echo "  make build        - Build all images"
	@echo "  make clean        - Clean up containers, images, and volumes"
	@echo "  make test         - Run tests"
	@echo "  make shell-backend - Open shell in backend container"
	@echo "  make shell-frontend - Open shell in frontend container"
	@echo "  make redis-cli    - Open Redis CLI"
	@echo ""

# Development Environment
dev:
	@echo "Starting development environment..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "Development environment started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend API: http://localhost:8000"
	@echo "Redis: localhost:6379"

dev-up:
	@echo "Starting development environment..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "Development environment started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend API: http://localhost:8000"
	@echo "Redis: localhost:6379"

dev-build:
	@echo "Building and starting development environment..."
	docker-compose -f docker-compose.dev.yml up -d --build

dev-logs:
	docker-compose -f docker-compose.dev.yml logs -f

dev-down:
	@echo "Stopping development environment..."
	docker-compose -f docker-compose.dev.yml down

# Production Environment
prod:
	@echo "Starting production environment..."
	docker-compose up -d
	@echo "Production environment started!"
	@echo "Application: http://localhost:80"

prod-up:
	@echo "Starting production environment..."
	docker-compose -f docker-compose.prod.yml up -d
	@echo "Production environment started!"
	@echo "Application: http://localhost:80"

prod-build:
	@echo "Building and starting production environment..."
	docker-compose up -d --build

prod-logs:
	docker-compose logs -f

prod-down:
	@echo "Stopping production environment..."
	docker-compose down

# Build Commands
build:
	@echo "Building all Docker images..."
	docker-compose build
	docker-compose -f docker-compose.dev.yml build

build-backend:
	@echo "Building backend image..."
	docker-compose build backend

build-frontend:
	@echo "Building frontend image..."
	docker-compose build frontend

# Utility Commands
clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker-compose -f docker-compose.dev.yml down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

clean-all:
	@echo "Cleaning up all Docker resources (including images)..."
	make clean
	docker image prune -a -f

# Testing
test: ## Run all tests
	@echo "Running comprehensive test suite..."
	node test-runner.js

test-backend: ## Run backend tests only
	@echo "Running backend tests..."
	cd backend && python -m pytest tests/ -v

test-frontend: ## Run frontend tests only
	@echo "Running frontend tests..."
	cd frontend && npm test

test-e2e: ## Run end-to-end tests
	@echo "Running E2E tests..."
	npx playwright test

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	node test-runner.js

test-install: ## Install test dependencies
	@echo "Installing test dependencies..."
	cd backend && pip install -r requirements-test.txt
	cd frontend && npm install --save-dev @testing-library/react @testing-library/jest-dom @testing-library/user-event jest ts-jest @types/jest
	npm install -g @playwright/test

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	cd backend && python -m pytest tests/ --cov=. --cov-report=html
	cd frontend && npm test -- --coverage

# Shell Access
shell-backend:
	@echo "Opening shell in backend container..."
	docker-compose -f docker-compose.dev.yml exec backend /bin/bash

shell-frontend:
	@echo "Opening shell in frontend container..."
	docker-compose -f docker-compose.dev.yml exec frontend /bin/sh

redis-cli:
	@echo "Opening Redis CLI..."
	docker-compose -f docker-compose.dev.yml exec redis redis-cli

# Database and Cache
redis-flush:
	@echo "Flushing Redis cache..."
	docker-compose -f docker-compose.dev.yml exec redis redis-cli FLUSHALL

# Monitoring
status:
	@echo "Docker containers status:"
	docker-compose ps
	@echo ""
	@echo "Development containers status:"
	docker-compose -f docker-compose.dev.yml ps

# Installation and Setup
setup:
	@echo "Setting up FlashFit AI development environment..."
	@echo "1. Installing frontend dependencies..."
	cd frontend && npm install
	@echo "2. Installing backend dependencies..."
	cd backend && pip install -r requirements.txt
	@echo "3. Building Docker images..."
	make build
	@echo "Setup complete! Run 'make dev' to start development environment."

# Quick start
quick-start:
	@echo "Quick starting FlashFit AI..."
	make dev-build
	@echo "Waiting for services to be ready..."
	sleep 10
	@echo "Opening application in browser..."
	open http://localhost:3000