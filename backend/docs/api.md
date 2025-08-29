# FlashFit AI API Documentation

## Overview

FlashFit AI provides a RESTful API for AI-powered clothing management and outfit recommendations. The API is built with FastAPI and provides automatic interactive documentation.

## Base URL

- **Development**: `http://localhost:8080`
- **Production**: `https://your-domain.com`

## Interactive Documentation

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **OpenAPI Schema**: `/openapi.json`

## Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Endpoints

### Health Check

#### GET /health

Check the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:30:00Z",
  "version": "1.0.0"
}
```

### Authentication

#### POST /auth/register

Register a new user account.

**Request Body:**
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "secure_password123",
  "full_name": "John Doe"
}
```

**Response:**
```json
{
  "message": "User registered successfully",
  "user_id": "uuid-string",
  "access_token": "jwt-token",
  "token_type": "bearer"
}
```

**Status Codes:**
- `201`: User created successfully
- `400`: Invalid input data
- `409`: User already exists

#### POST /auth/login

Authenticate user and get access token.

**Request Body:**
```json
{
  "username": "john_doe",
  "password": "secure_password123"
}
```

**Response:**
```json
{
  "access_token": "jwt-token",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "uuid-string",
    "username": "john_doe",
    "email": "john@example.com",
    "full_name": "John Doe"
  }
}
```

**Status Codes:**
- `200`: Login successful
- `401`: Invalid credentials
- `400`: Invalid input data

#### GET /auth/me

Get current user information (requires authentication).

**Headers:**
```
Authorization: Bearer <jwt-token>
```

**Response:**
```json
{
  "id": "uuid-string",
  "username": "john_doe",
  "email": "john@example.com",
  "full_name": "John Doe",
  "created_at": "2024-01-20T10:30:00Z",
  "is_active": true
}
```

### Clothing Management

#### POST /clothing/upload

Upload a new clothing item with image.

**Headers:**
```
Authorization: Bearer <jwt-token>
Content-Type: multipart/form-data
```

**Request Body (Form Data):**
- `file`: Image file (JPEG, PNG, WebP)
- `name`: Clothing item name (optional)
- `category`: Clothing category (optional)
- `color`: Primary color (optional)
- `brand`: Brand name (optional)
- `size`: Size information (optional)
- `tags`: Comma-separated tags (optional)

**Response:**
```json
{
  "id": "uuid-string",
  "name": "Blue Denim Jacket",
  "category": "jacket",
  "color": "blue",
  "brand": "Levi's",
  "size": "M",
  "tags": ["casual", "denim", "outerwear"],
  "image_url": "/uploads/clothing/uuid-string.jpg",
  "embedding_id": "embedding-uuid",
  "created_at": "2024-01-20T10:30:00Z",
  "updated_at": "2024-01-20T10:30:00Z"
}
```

**Status Codes:**
- `201`: Clothing item created successfully
- `400`: Invalid file or data
- `401`: Authentication required
- `413`: File too large
- `415`: Unsupported file type

#### GET /clothing/

Get all clothing items for the authenticated user.

**Headers:**
```
Authorization: Bearer <jwt-token>
```

**Query Parameters:**
- `category`: Filter by category (optional)
- `color`: Filter by color (optional)
- `tags`: Filter by tags (optional)
- `limit`: Number of items to return (default: 50)
- `offset`: Number of items to skip (default: 0)

**Response:**
```json
{
  "items": [
    {
      "id": "uuid-string",
      "name": "Blue Denim Jacket",
      "category": "jacket",
      "color": "blue",
      "brand": "Levi's",
      "size": "M",
      "tags": ["casual", "denim", "outerwear"],
      "image_url": "/uploads/clothing/uuid-string.jpg",
      "created_at": "2024-01-20T10:30:00Z"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

#### GET /clothing/{item_id}

Get a specific clothing item by ID.

**Headers:**
```
Authorization: Bearer <jwt-token>
```

**Response:**
```json
{
  "id": "uuid-string",
  "name": "Blue Denim Jacket",
  "category": "jacket",
  "color": "blue",
  "brand": "Levi's",
  "size": "M",
  "tags": ["casual", "denim", "outerwear"],
  "image_url": "/uploads/clothing/uuid-string.jpg",
  "embedding_id": "embedding-uuid",
  "created_at": "2024-01-20T10:30:00Z",
  "updated_at": "2024-01-20T10:30:00Z"
}
```

**Status Codes:**
- `200`: Item found
- `404`: Item not found
- `401`: Authentication required

#### PUT /clothing/{item_id}

Update a clothing item.

**Headers:**
```
Authorization: Bearer <jwt-token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "Updated Blue Denim Jacket",
  "category": "jacket",
  "color": "blue",
  "brand": "Levi's",
  "size": "L",
  "tags": ["casual", "denim", "outerwear", "vintage"]
}
```

**Response:**
```json
{
  "id": "uuid-string",
  "name": "Updated Blue Denim Jacket",
  "category": "jacket",
  "color": "blue",
  "brand": "Levi's",
  "size": "L",
  "tags": ["casual", "denim", "outerwear", "vintage"],
  "image_url": "/uploads/clothing/uuid-string.jpg",
  "updated_at": "2024-01-20T11:00:00Z"
}
```

#### DELETE /clothing/{item_id}

Delete a clothing item.

**Headers:**
```
Authorization: Bearer <jwt-token>
```

**Response:**
```json
{
  "message": "Clothing item deleted successfully"
}
```

**Status Codes:**
- `200`: Item deleted successfully
- `404`: Item not found
- `401`: Authentication required

### Outfit Recommendations

#### POST /recommendations/generate

Generate outfit recommendations based on user preferences or specific items.

**Headers:**
```
Authorization: Bearer <jwt-token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "base_item_id": "uuid-string",
  "occasion": "casual",
  "weather": "mild",
  "color_preference": "neutral",
  "style_preference": "minimalist",
  "max_recommendations": 5
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "id": "recommendation-uuid",
      "score": 0.95,
      "items": [
        {
          "id": "item-uuid-1",
          "name": "White T-Shirt",
          "category": "top",
          "role": "base"
        },
        {
          "id": "item-uuid-2",
          "name": "Blue Jeans",
          "category": "bottom",
          "role": "complement"
        },
        {
          "id": "item-uuid-3",
          "name": "White Sneakers",
          "category": "shoes",
          "role": "accent"
        }
      ],
      "style_tags": ["casual", "comfortable", "everyday"],
      "occasion_match": "casual",
      "weather_suitability": "mild"
    }
  ],
  "total_recommendations": 1,
  "generation_time_ms": 150
}
```

#### GET /recommendations/history

Get user's recommendation history.

**Headers:**
```
Authorization: Bearer <jwt-token>
```

**Query Parameters:**
- `limit`: Number of recommendations to return (default: 20)
- `offset`: Number of recommendations to skip (default: 0)

**Response:**
```json
{
  "recommendations": [
    {
      "id": "recommendation-uuid",
      "created_at": "2024-01-20T10:30:00Z",
      "occasion": "casual",
      "items_count": 3,
      "score": 0.95,
      "was_used": true
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}
```

### Search and Similarity

#### POST /search/similar

Find similar clothing items based on visual similarity.

**Headers:**
```
Authorization: Bearer <jwt-token>
Content-Type: multipart/form-data
```

**Request Body (Form Data):**
- `file`: Image file to search for similar items
- `limit`: Maximum number of results (optional, default: 10)
- `threshold`: Similarity threshold (optional, default: 0.7)

**Response:**
```json
{
  "similar_items": [
    {
      "item": {
        "id": "uuid-string",
        "name": "Blue Denim Jacket",
        "category": "jacket",
        "image_url": "/uploads/clothing/uuid-string.jpg"
      },
      "similarity_score": 0.92,
      "match_type": "visual"
    }
  ],
  "query_time_ms": 45,
  "total_matches": 1
}
```

#### GET /search/text

Search clothing items by text query.

**Headers:**
```
Authorization: Bearer <jwt-token>
```

**Query Parameters:**
- `q`: Search query
- `limit`: Maximum number of results (default: 20)
- `category`: Filter by category (optional)

**Response:**
```json
{
  "results": [
    {
      "id": "uuid-string",
      "name": "Blue Denim Jacket",
      "category": "jacket",
      "relevance_score": 0.89,
      "image_url": "/uploads/clothing/uuid-string.jpg"
    }
  ],
  "query": "blue jacket",
  "total_results": 1,
  "search_time_ms": 25
}
```

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "detail": "Invalid input data",
  "errors": [
    {
      "field": "email",
      "message": "Invalid email format"
    }
  ]
}
```

### 401 Unauthorized
```json
{
  "detail": "Authentication required"
}
```

### 403 Forbidden
```json
{
  "detail": "Insufficient permissions"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "email"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error"
}
```

## Rate Limiting

API endpoints are rate-limited to prevent abuse:

- **Authentication endpoints**: 5 requests per minute
- **Upload endpoints**: 10 requests per minute
- **General endpoints**: 100 requests per minute
- **Search endpoints**: 50 requests per minute

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642680000
```

## File Upload Limits

- **Maximum file size**: 10MB
- **Supported formats**: JPEG, PNG, WebP
- **Minimum dimensions**: 224x224 pixels
- **Maximum dimensions**: 4096x4096 pixels

## Pagination

Endpoints that return lists support pagination:

**Query Parameters:**
- `limit`: Number of items per page (max: 100)
- `offset`: Number of items to skip

**Response includes pagination metadata:**
```json
{
  "items": [...],
  "total": 150,
  "limit": 20,
  "offset": 40,
  "has_next": true,
  "has_prev": true
}
```

## Webhooks

FlashFit AI supports webhooks for real-time notifications:

### Events
- `clothing.uploaded`: New clothing item uploaded
- `recommendation.generated`: New recommendation created
- `user.registered`: New user registered

### Webhook Payload
```json
{
  "event": "clothing.uploaded",
  "timestamp": "2024-01-20T10:30:00Z",
  "user_id": "uuid-string",
  "data": {
    "clothing_id": "uuid-string",
    "name": "Blue Denim Jacket"
  }
}
```

## SDK and Libraries

### Python SDK
```python
from flashfit_client import FlashFitClient

client = FlashFitClient(api_key="your-api-key")
clothing_items = client.clothing.list()
```

### JavaScript SDK
```javascript
import { FlashFitClient } from '@flashfit/client';

const client = new FlashFitClient({ apiKey: 'your-api-key' });
const items = await client.clothing.list();
```

## Support

For API support and questions:
- **Documentation**: [https://docs.flashfit.ai](https://docs.flashfit.ai)
- **Email**: api-support@flashfit.ai
- **GitHub Issues**: [https://github.com/flashfit-ai/api/issues](https://github.com/flashfit-ai/api/issues)