# FlashFit AI API Reference

## Base URL
```
http://localhost:8080
```

## Authentication
All API endpoints require JWT authentication via the `Authorization` header:
```
Authorization: Bearer <jwt_token>
```

## Core Endpoints

### 1. Multi-Model Fusion Recommendations

#### `POST /api/fusion/recommend`

Generate outfit recommendations using the tri-model ensemble (CLIP + BLIP + Fashion Encoder).

**Request Body:**
```json
{
  "user_id": "string",
  "image_data": "base64_encoded_image",
  "preferences": {
    "style": "casual|formal|sporty|trendy",
    "occasion": "work|party|casual|date",
    "season": "spring|summer|fall|winter"
  },
  "limit": 10
}
```

**Response:**
```json
{
  "status": "success",
  "recommendations": [
    {
      "item_id": "item_001",
      "name": "Blue Denim Jacket",
      "category": "outerwear",
      "image_url": "/static/items/item_001.jpg",
      "scores": {
        "clip_score": 0.85,
        "blip_score": 0.78,
        "fashion_score": 0.82,
        "fusion_score": 0.82
      },
      "confidence": 0.89,
      "tags": ["casual", "denim", "versatile"],
      "description": "Classic blue denim jacket perfect for casual layering"
    }
  ],
  "metadata": {
    "processing_time_ms": 245,
    "models_used": ["clip", "blip", "fashion"],
    "total_candidates": 156,
    "fusion_weights": {
      "clip": 0.4,
      "blip": 0.3,
      "fashion": 0.3
    }
  }
}
```

**Example Usage:**
```bash
curl -X POST "http://localhost:8080/api/fusion/recommend" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "image_data": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",
    "preferences": {
      "style": "casual",
      "occasion": "work",
      "season": "fall"
    },
    "limit": 5
  }'
```

### 2. User Feedback Collection

#### `POST /api/fusion/feedback`

Collect user feedback to improve future recommendations through online learning.

**Request Body:**
```json
{
  "user_id": "string",
  "item_id": "string",
  "feedback_type": "like|dislike|save|purchase|view",
  "rating": 1-5,
  "context": {
    "recommendation_id": "string",
    "position": 1,
    "session_id": "string"
  },
  "metadata": {
    "interaction_time_ms": 1500,
    "device_type": "mobile|desktop"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Feedback recorded successfully",
  "learning_impact": {
    "weight_adjustments": {
      "clip": 0.02,
      "blip": -0.01,
      "fashion": 0.01
    },
    "personalization_score": 0.73
  },
  "feedback_id": "fb_001"
}
```

**Example Usage:**
```bash
curl -X POST "http://localhost:8080/api/fusion/feedback" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "item_id": "item_001",
    "feedback_type": "like",
    "rating": 4,
    "context": {
      "recommendation_id": "rec_456",
      "position": 1,
      "session_id": "sess_789"
    }
  }'
```

### 3. System Statistics

#### `GET /api/fusion/stats`

Retrieve system performance metrics and model statistics.

**Response:**
```json
{
  "status": "success",
  "stats": {
    "vector_stores": {
      "clip_vectors": 1247,
      "blip_vectors": 1247,
      "fashion_vectors": 1247
    },
    "model_performance": {
      "avg_inference_time_ms": 234,
      "recommendations_served": 15678,
      "feedback_collected": 3421
    },
    "fusion_weights": {
      "current": {
        "clip": 0.42,
        "blip": 0.28,
        "fashion": 0.30
      },
      "default": {
        "clip": 0.40,
        "blip": 0.30,
        "fashion": 0.30
      }
    },
    "system_health": {
      "uptime_hours": 72.5,
      "memory_usage_mb": 2048,
      "cpu_usage_percent": 15.3,
      "last_index_update": "2024-01-15T10:30:00Z"
    }
  }
}
```

**Example Usage:**
```bash
curl -X GET "http://localhost:8080/api/fusion/stats" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
```

### 4. Image Upload

#### `POST /api/upload`

Upload and process fashion images for recommendation generation.

**Request:**
- Content-Type: `multipart/form-data`
- File field: `image`
- Supported formats: JPEG, PNG, WebP
- Max file size: 10MB

**Response:**
```json
{
  "status": "success",
  "image_id": "img_001",
  "processed_url": "/static/processed/img_001.jpg",
  "metadata": {
    "original_size": [1920, 1080],
    "processed_size": [512, 512],
    "file_size_bytes": 245760,
    "processing_time_ms": 156
  },
  "embeddings_generated": {
    "clip": true,
    "blip": true,
    "fashion": true
  }
}
```

**Example Usage:**
```bash
curl -X POST "http://localhost:8080/api/upload" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
  -F "image=@/path/to/fashion_item.jpg"
```

## Authentication Endpoints

### 5. User Login

#### `POST /api/auth/login`

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "status": "success",
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "user123",
    "username": "fashionista",
    "preferences": {
      "style": "casual",
      "size": "M"
    }
  }
}
```

### 6. User Registration

#### `POST /api/auth/register`

**Request Body:**
```json
{
  "username": "string",
  "email": "string",
  "password": "string",
  "preferences": {
    "style": "casual|formal|sporty|trendy",
    "size": "XS|S|M|L|XL|XXL"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "User registered successfully",
  "user_id": "user124"
}
```

## Wardrobe Management

### 7. Add Wardrobe Item

#### `POST /api/wardrobe`

**Request Body:**
```json
{
  "user_id": "string",
  "name": "string",
  "category": "tops|bottoms|outerwear|shoes|accessories",
  "image_data": "base64_encoded_image",
  "tags": ["string"],
  "metadata": {
    "brand": "string",
    "color": "string",
    "size": "string",
    "purchase_date": "2024-01-15"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "item_id": "item_002",
  "embeddings_generated": true,
  "added_to_index": true
}
```

### 8. Get User Wardrobe

#### `GET /api/wardrobe/{user_id}`

**Response:**
```json
{
  "status": "success",
  "items": [
    {
      "item_id": "item_002",
      "name": "White Cotton T-Shirt",
      "category": "tops",
      "image_url": "/static/wardrobe/item_002.jpg",
      "tags": ["casual", "basic", "white"],
      "added_date": "2024-01-15T09:00:00Z"
    }
  ],
  "total_items": 23
}
```

## Error Responses

### Standard Error Format
```json
{
  "status": "error",
  "error_code": "INVALID_INPUT",
  "message": "Human-readable error description",
  "details": {
    "field": "image_data",
    "issue": "Invalid base64 encoding"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_INPUT` | 400 | Request validation failed |
| `UNAUTHORIZED` | 401 | Invalid or missing authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Too many requests |
| `MODEL_ERROR` | 500 | AI model processing failed |
| `SYSTEM_ERROR` | 500 | Internal server error |

## Rate Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/api/fusion/recommend` | 100 requests | 1 hour |
| `/api/fusion/feedback` | 1000 requests | 1 hour |
| `/api/upload` | 50 requests | 1 hour |
| `/api/auth/*` | 10 requests | 1 minute |

## SDK Examples

### Python SDK
```python
import requests
import base64

class FlashFitClient:
    def __init__(self, base_url="http://localhost:8080", token=None):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    def get_recommendations(self, user_id, image_path, preferences=None, limit=10):
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        payload = {
            "user_id": user_id,
            "image_data": image_data,
            "preferences": preferences or {},
            "limit": limit
        }
        
        response = requests.post(
            f"{self.base_url}/api/fusion/recommend",
            json=payload,
            headers=self.headers
        )
        return response.json()
    
    def submit_feedback(self, user_id, item_id, feedback_type, rating=None):
        payload = {
            "user_id": user_id,
            "item_id": item_id,
            "feedback_type": feedback_type,
            "rating": rating
        }
        
        response = requests.post(
            f"{self.base_url}/api/fusion/feedback",
            json=payload,
            headers=self.headers
        )
        return response.json()

# Usage
client = FlashFitClient(token="your_jwt_token")
recommendations = client.get_recommendations(
    user_id="user123",
    image_path="outfit.jpg",
    preferences={"style": "casual", "occasion": "work"}
)
```

### JavaScript SDK
```javascript
class FlashFitClient {
  constructor(baseUrl = 'http://localhost:8080', token = null) {
    this.baseUrl = baseUrl;
    this.headers = token ? { 'Authorization': `Bearer ${token}` } : {};
  }

  async getRecommendations(userId, imageFile, preferences = {}, limit = 10) {
    const imageData = await this.fileToBase64(imageFile);
    
    const response = await fetch(`${this.baseUrl}/api/fusion/recommend`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.headers
      },
      body: JSON.stringify({
        user_id: userId,
        image_data: imageData,
        preferences,
        limit
      })
    });
    
    return response.json();
  }

  async submitFeedback(userId, itemId, feedbackType, rating = null) {
    const response = await fetch(`${this.baseUrl}/api/fusion/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.headers
      },
      body: JSON.stringify({
        user_id: userId,
        item_id: itemId,
        feedback_type: feedbackType,
        rating
      })
    });
    
    return response.json();
  }

  fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.onerror = error => reject(error);
    });
  }
}

// Usage
const client = new FlashFitClient('http://localhost:8080', 'your_jwt_token');
const recommendations = await client.getRecommendations(
  'user123',
  imageFile,
  { style: 'casual', occasion: 'work' }
);
```

## Testing

### Health Check
```bash
curl -X GET "http://localhost:8080/health"
```

### API Documentation
Interactive API documentation is available at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`