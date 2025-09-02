import pytest
import asyncio
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient
from httpx import AsyncClient
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from main import app
# Remove non-existent imports for now
# from core.config import get_settings
# from services.auth_service import AuthService
# from services.clothing_service import ClothingService
# from services.ml_service import MLService

# Add project directories to Python path for generative component imports
project_root = Path(__file__).parent.parent
ml_path = project_root.parent / "ml"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(ml_path))


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI app."""
    async with AsyncClient(base_url="http://test") as ac:
        yield ac


@pytest.fixture
def temp_upload_dir() -> Generator[str, None, None]:
    """Create a temporary directory for file uploads during testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_settings():
    """Mock application settings for testing."""
    class MockSettings:
        UPLOAD_DIR = "/tmp/test_uploads"
        SECRET_KEY = "test_secret_key"
        DATABASE_URL = "sqlite:///test.db"
    
    return MockSettings()


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = Mock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = True
    mock_redis.exists.return_value = False
    mock_redis.ping.return_value = True
    return mock_redis


@pytest.fixture
def mock_ml_service():
    """Mock ML service for testing."""
    mock_service = Mock()
    mock_service.generate_recommendations.return_value = {
        'recommendations': [],
        'query_analysis': {'caption': 'test image'},
        'performance_stats': {'total_requests': 1}
    }
    return mock_service


@pytest.fixture
def mock_auth_service():
    """Mock authentication service for testing."""
    mock_service = Mock()
    mock_service.verify_token.return_value = 'test_user'
    mock_service.create_token.return_value = 'test_token'
    return mock_service


@pytest.fixture
def mock_clothing_service():
    """Mock clothing service for testing."""
    mock_service = Mock()
    mock_service.get_user_wardrobe.return_value = {
        'items': [
            {'id': 1, 'name': 'Test Shirt', 'category': 'tops'},
            {'id': 2, 'name': 'Test Pants', 'category': 'bottoms'}
        ]
    }
    return mock_service


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User"
    }


@pytest.fixture
def sample_clothing_data():
    """Sample clothing data for testing."""
    return {
        "category": "shirt",
        "subcategory": "t-shirt",
        "color": "blue",
        "brand": "Test Brand",
        "size": "M",
        "tags": ["casual", "summer"]
    }


@pytest.fixture
def sample_image_file():
    """Create a sample image file for testing uploads."""
    # Create a simple 1x1 pixel PNG image
    image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xdd\xcc\xdb\x1d\x00\x00\x00\x00IEND\xaeB`\x82'
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_data)
        f.flush()
        yield f.name
    
    # Clean up
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass


@pytest.fixture
def auth_headers(mock_auth_service):
    """Generate authentication headers for testing."""
    token = mock_auth_service.create_access_token()
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def setup_test_environment(mock_settings, mock_redis):
    """Setup test environment with mocked dependencies."""
    # Simple test environment setup without complex patching
    yield


# Generative Component Fixtures
@pytest.fixture
def mock_generator_config():
    """Mock configuration for embedding generator."""
    config = Mock()
    config.embedding_dim = 512
    config.hidden_dims = [256, 256]
    config.device = 'cpu'
    config.learning_rate = 1e-3
    config.batch_size = 32
    config.dropout_rate = 0.1
    return config

@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    return np.random.randn(10, 512).astype(np.float32)

@pytest.fixture
def sample_metadata():
    """Generate sample metadata for testing."""
    return [
        {
            "id": f"item_{i}",
            "name": f"Test Item {i}",
            "category": "fashion",
            "brand": f"Brand {i % 3}",
            "price": 50.0 + i * 10,
            "tags": ["casual", "summer"] if i % 2 == 0 else ["formal", "winter"]
        }
        for i in range(10)
    ]

@pytest.fixture
def mock_vector_store(sample_embeddings, sample_metadata):
    """Mock vector store with sample data."""
    mock_store = Mock()
    mock_store.dim = 512
    mock_store.items = sample_metadata
    mock_store.add.return_value = None
    mock_store.search.return_value = [
        (sample_metadata[0], 0.95),
        (sample_metadata[1], 0.90),
        (sample_metadata[2], 0.85)
    ]
    mock_store.save.return_value = None
    return mock_store

@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator."""
    mock_gen = Mock()
    mock_gen.config.embedding_dim = 512
    mock_gen.generate_embeddings.return_value = np.random.randn(5, 512).astype(np.float32)
    mock_gen.save_model.return_value = None
    mock_gen.load_model.return_value = None
    mock_gen.fine_tune.return_value = 0.5
    mock_gen.get_model_info.return_value = {
        "parameters": 1000000,
        "embedding_dim": 512,
        "hidden_dims": [256, 256]
    }
    return mock_gen

@pytest.fixture
def mock_generative_monitoring_service():
    """Mock generative monitoring service."""
    mock_service = Mock()
    mock_service.record_performance_metric.return_value = None
    mock_service.update_health_status.return_value = None
    mock_service.get_system_overview.return_value = {
        "total_requests": 100,
        "successful_requests": 95,
        "failed_requests": 5,
        "success_rate": 0.95,
        "average_response_time_ms": 150.0
    }
    mock_service.get_component_metrics.return_value = []
    mock_service.get_recent_alerts.return_value = []
    return mock_service

@pytest.fixture
def test_feedback_data():
    """Generate test feedback data."""
    return {
        "user_id": "test_user_123",
        "query_embedding": [0.1] * 512,
        "generated_embeddings": [[0.2] * 512, [0.3] * 512],
        "user_rating": 4,
        "feedback_type": "positive",
        "comments": "Great recommendations!",
        "selected_items": ["item_1", "item_3"]
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "auth: mark test as requiring authentication"
    )
    config.addinivalue_line(
        "markers", "ml: mark test as requiring ML models"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "generative: mark test as generative component test"
    )