import pytest
import asyncio
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient
from httpx import AsyncClient
import tempfile
import os
from unittest.mock import Mock, patch

from main import app
from core.config import get_settings
from services.auth_service import AuthService
from services.clothing_service import ClothingService
from services.ml_service import MLService


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
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def temp_upload_dir() -> Generator[str, None, None]:
    """Create a temporary directory for file uploads during testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_settings():
    """Mock application settings for testing."""
    settings = get_settings()
    settings.UPLOAD_DIR = "/tmp/test_uploads"
    settings.MAX_FILE_SIZE = 1024 * 1024  # 1MB for testing
    settings.SECRET_KEY = "test-secret-key"
    settings.JWT_SECRET_KEY = "test-jwt-secret"
    settings.ENVIRONMENT = "testing"
    settings.DEBUG = True
    return settings


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
    mock_service = Mock(spec=MLService)
    mock_service.generate_embedding.return_value = [0.1] * 512  # Mock embedding
    mock_service.classify_clothing.return_value = {
        "category": "shirt",
        "subcategory": "t-shirt",
        "color": "blue",
        "confidence": 0.95
    }
    mock_service.find_similar_items.return_value = [
        {"id": "item1", "similarity": 0.9},
        {"id": "item2", "similarity": 0.8}
    ]
    return mock_service


@pytest.fixture
def mock_auth_service():
    """Mock authentication service for testing."""
    mock_service = Mock(spec=AuthService)
    mock_service.create_access_token.return_value = "mock-jwt-token"
    mock_service.verify_token.return_value = {"user_id": "test-user-123"}
    mock_service.hash_password.return_value = "hashed-password"
    mock_service.verify_password.return_value = True
    return mock_service


@pytest.fixture
def mock_clothing_service():
    """Mock clothing service for testing."""
    mock_service = Mock(spec=ClothingService)
    mock_service.save_clothing_item.return_value = {
        "id": "clothing-123",
        "user_id": "user-123",
        "category": "shirt",
        "image_url": "/uploads/test-image.jpg"
    }
    mock_service.get_user_clothing.return_value = [
        {
            "id": "clothing-123",
            "category": "shirt",
            "color": "blue",
            "image_url": "/uploads/test-image.jpg"
        }
    ]
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
    """Set up test environment with mocked dependencies."""
    with patch('core.config.get_settings', return_value=mock_settings), \
         patch('core.redis_client.redis_client', mock_redis):
        yield


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "auth: mark test as authentication related"
    )
    config.addinivalue_line(
        "markers", "ml: mark test as machine learning related"
    )