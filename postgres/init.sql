-- FlashFit AI Database Initialization
-- This script sets up the database schema for monitoring and application data

-- Create database if not exists (handled by Docker environment variables)
-- CREATE DATABASE IF NOT EXISTS flashfit_ai;

-- Use the database
\c flashfit_ai;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    preferences JSONB DEFAULT '{}'
);

-- Fashion items table
CREATE TABLE IF NOT EXISTS fashion_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),
    brand VARCHAR(100),
    color VARCHAR(50),
    size VARCHAR(20),
    price DECIMAL(10,2),
    description TEXT,
    image_url VARCHAR(500),
    metadata JSONB DEFAULT '{}',
    vector_embedding FLOAT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User wardrobe table
CREATE TABLE IF NOT EXISTS user_wardrobe (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    item_id UUID REFERENCES fashion_items(id) ON DELETE CASCADE,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tags TEXT[],
    notes TEXT,
    UNIQUE(user_id, item_id)
);

-- Outfit history table
CREATE TABLE IF NOT EXISTS outfit_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    outfit_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback TEXT,
    occasion VARCHAR(100),
    weather_conditions JSONB
);

-- User feedback table
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    item_id UUID REFERENCES fashion_items(id) ON DELETE CASCADE,
    feedback_type VARCHAR(50) NOT NULL, -- 'like', 'dislike', 'rating', 'comment'
    feedback_value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    context JSONB DEFAULT '{}' -- Additional context like occasion, weather, etc.
);

-- Social interactions table
CREATE TABLE IF NOT EXISTS social_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    target_user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    interaction_type VARCHAR(50) NOT NULL, -- 'follow', 'like_outfit', 'share', 'comment'
    content JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Monitoring tables

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_type VARCHAR(50) NOT NULL, -- 'gauge', 'counter', 'histogram'
    labels JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Query performance metrics
CREATE TABLE IF NOT EXISTS query_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_hash VARCHAR(64) NOT NULL,
    query_text TEXT,
    execution_time_ms FLOAT NOT NULL,
    rows_returned INTEGER,
    database_name VARCHAR(100),
    user_name VARCHAR(100),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API endpoint metrics
CREATE TABLE IF NOT EXISTS api_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms FLOAT NOT NULL,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    user_id UUID REFERENCES users(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ML model performance metrics
CREATE TABLE IF NOT EXISTS ml_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    metric_name VARCHAR(100) NOT NULL, -- 'accuracy', 'precision', 'recall', 'f1_score', 'inference_time'
    metric_value FLOAT NOT NULL,
    dataset_size INTEGER,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Health check logs
CREATE TABLE IF NOT EXISTS health_checks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL, -- 'healthy', 'unhealthy', 'degraded'
    response_time_ms FLOAT,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance

-- User indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- Fashion items indexes
CREATE INDEX IF NOT EXISTS idx_fashion_items_category ON fashion_items(category);
CREATE INDEX IF NOT EXISTS idx_fashion_items_brand ON fashion_items(brand);
CREATE INDEX IF NOT EXISTS idx_fashion_items_created_at ON fashion_items(created_at);

-- User wardrobe indexes
CREATE INDEX IF NOT EXISTS idx_user_wardrobe_user_id ON user_wardrobe(user_id);
CREATE INDEX IF NOT EXISTS idx_user_wardrobe_item_id ON user_wardrobe(item_id);

-- Outfit history indexes
CREATE INDEX IF NOT EXISTS idx_outfit_history_user_id ON outfit_history(user_id);
CREATE INDEX IF NOT EXISTS idx_outfit_history_created_at ON outfit_history(created_at);
CREATE INDEX IF NOT EXISTS idx_outfit_history_rating ON outfit_history(rating);

-- Feedback indexes
CREATE INDEX IF NOT EXISTS idx_user_feedback_user_id ON user_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_item_id ON user_feedback(item_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_type ON user_feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_user_feedback_created_at ON user_feedback(created_at);

-- Monitoring indexes
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_timestamp ON system_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_query_metrics_timestamp ON query_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_metrics_execution_time ON query_metrics(execution_time_ms);
CREATE INDEX IF NOT EXISTS idx_api_metrics_endpoint_timestamp ON api_metrics(endpoint, timestamp);
CREATE INDEX IF NOT EXISTS idx_api_metrics_status_code ON api_metrics(status_code);
CREATE INDEX IF NOT EXISTS idx_ml_metrics_model_timestamp ON ml_metrics(model_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_health_checks_service_timestamp ON health_checks(service_name, timestamp);

-- Create views for monitoring dashboards

-- System performance overview
CREATE OR REPLACE VIEW system_performance_overview AS
SELECT 
    metric_name,
    AVG(metric_value) as avg_value,
    MIN(metric_value) as min_value,
    MAX(metric_value) as max_value,
    COUNT(*) as sample_count,
    DATE_TRUNC('hour', timestamp) as hour
FROM system_metrics 
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY metric_name, DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;

-- API performance summary
CREATE OR REPLACE VIEW api_performance_summary AS
SELECT 
    endpoint,
    method,
    COUNT(*) as request_count,
    AVG(response_time_ms) as avg_response_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time,
    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count,
    COUNT(CASE WHEN status_code >= 400 THEN 1 END)::FLOAT / COUNT(*) * 100 as error_rate
FROM api_metrics 
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY endpoint, method
ORDER BY request_count DESC;

-- ML model performance trends
CREATE OR REPLACE VIEW ml_performance_trends AS
SELECT 
    model_name,
    model_version,
    metric_name,
    AVG(metric_value) as avg_value,
    STDDEV(metric_value) as stddev_value,
    COUNT(*) as sample_count,
    DATE_TRUNC('day', timestamp) as day
FROM ml_metrics 
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY model_name, model_version, metric_name, DATE_TRUNC('day', timestamp)
ORDER BY day DESC, model_name, metric_name;

-- Service health status
CREATE OR REPLACE VIEW service_health_status AS
SELECT DISTINCT ON (service_name)
    service_name,
    status,
    response_time_ms,
    error_message,
    timestamp
FROM health_checks 
ORDER BY service_name, timestamp DESC;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO flashfit_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO flashfit_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO flashfit_user;

-- Insert sample data for testing
INSERT INTO users (username, email, password_hash) VALUES 
('testuser', 'test@flashfit.ai', '$2b$12$sample_hash_here')
ON CONFLICT (email) DO NOTHING;

-- Insert sample system metrics
INSERT INTO system_metrics (metric_name, metric_value, metric_type) VALUES 
('cpu_usage_percent', 25.5, 'gauge'),
('memory_usage_percent', 60.2, 'gauge'),
('disk_usage_percent', 45.8, 'gauge')
ON CONFLICT DO NOTHING;

COMMIT;