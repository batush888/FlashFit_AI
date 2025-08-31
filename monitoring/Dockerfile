FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy monitoring application
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash monitor && \
    chown -R monitor:monitor /app
USER monitor

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=9090
ENV HOST=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT}/metrics', timeout=5)"

# Expose monitoring port
EXPOSE ${PORT}

# Start monitoring server
CMD ["python", "setup.py"]