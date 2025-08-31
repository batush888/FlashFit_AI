# FlashFit AI Architecture Documentation

## System Overview

FlashFit AI is a tri-model ensemble recommendation system that combines three specialized AI models to provide intelligent outfit recommendations with real-time personalization.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLIP Encoder  │    │ BLIP Captioner  │    │ Fashion Encoder │
│                 │    │                 │    │                 │
│ Vision-Language │    │ Natural Language│    │ Fashion-Specific│
│   Alignment     │    │  Understanding  │    │  Compatibility  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Fusion Reranker     │
                    │                        │
                    │  Weighted Scoring &    │
                    │  Ensemble Intelligence │
                    └────────────┬───────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     Vector Store        │
                    │                        │
                    │   FAISS Similarity     │
                    │      Search            │
                    └────────────┬───────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      FastAPI           │
                    │                        │
                    │   RESTful Endpoints    │
                    │   (Port 8080)          │
                    └────────────┬───────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   React Frontend       │
                    │                        │
                    │    User Interface      │
                    │    (Port 3000)         │
                    └────────────────────────┘
```

## Component Details

### 1. CLIP Encoder (`backend/models/clip_encoder.py`)
- **Purpose**: General vision-language alignment for broad fashion matching
- **Model**: OpenAI CLIP (ViT-B/32)
- **Input**: Images and text descriptions
- **Output**: 512-dimensional embeddings
- **Use Case**: Initial similarity matching and cross-modal understanding

### 2. BLIP Captioner (`backend/models/blip_captioner.py`)
- **Purpose**: Natural language description and contextual understanding
- **Model**: Salesforce BLIP for conditional generation
- **Input**: Fashion images
- **Output**: Descriptive captions and contextual embeddings
- **Use Case**: Style description and semantic understanding

### 3. Fashion Encoder (`backend/models/fashion_encoder.py`)
- **Purpose**: Clothing-specific compatibility and style embeddings
- **Implementation**: Wrapper around CLIP with fashion-specific processing
- **Input**: Fashion item images
- **Output**: Fashion-optimized embeddings
- **Use Case**: Style compatibility and fashion-specific matching

### 4. Fusion Reranker (`backend/models/fusion_reranker.py`)
- **Purpose**: Intelligent weighted scoring across all models
- **Algorithm**: Weighted ensemble with adaptive scoring
- **Weights**: 
  - CLIP: 0.4 (general matching)
  - BLIP: 0.3 (contextual understanding)
  - Fashion: 0.3 (style compatibility)
- **Features**: 
  - Cross-model score normalization
  - Adaptive weight adjustment
  - Confidence-based reranking

### 5. Vector Store (`backend/models/vector_store.py`)
- **Purpose**: High-performance similarity search
- **Implementation**: FAISS (Facebook AI Similarity Search)
- **Index Types**: 
  - CLIP embeddings: `clip_fashion.index`
  - BLIP embeddings: `blip_fashion.index`
  - Fashion embeddings: `fashion_specific.index`
- **Features**: 
  - Real-time index updates
  - Batch embedding operations
  - Efficient nearest neighbor search

### 6. Recommendation Service (`backend/services/recommend_service.py`)
- **Purpose**: Orchestrates the entire recommendation pipeline
- **Features**:
  - Multi-model embedding generation
  - Fusion scoring and reranking
  - Feedback integration
  - Performance monitoring

## Data Flow

### Recommendation Pipeline

1. **Input Processing**
   ```
   User Image → [CLIP, BLIP, Fashion] Encoders → Embeddings
   ```

2. **Similarity Search**
   ```
   Embeddings → FAISS Vector Stores → Candidate Items
   ```

3. **Fusion Scoring**
   ```
   Candidates → Fusion Reranker → Weighted Scores → Ranked Results
   ```

4. **Response Generation**
   ```
   Ranked Results → API Response → Frontend Display
   ```

### Feedback Loop

1. **User Interaction**
   ```
   User Feedback → API Endpoint → Feedback Storage
   ```

2. **Model Adaptation**
   ```
   Feedback → Weight Adjustment → Reranker Update
   ```

3. **Vector Update**
   ```
   New Preferences → Embedding Adjustment → Index Update
   ```

## API Architecture

### Core Endpoints
- `POST /api/fusion/recommend` - Multi-model recommendations
- `POST /api/fusion/feedback` - User feedback collection
- `GET /api/fusion/stats` - System performance metrics
- `POST /api/upload` - Image upload and processing

### Authentication & Security
- JWT-based authentication
- Rate limiting
- Input validation
- CORS configuration

## Performance Characteristics

### Latency
- **Cold Start**: ~2-3 seconds (model loading)
- **Warm Inference**: ~200-500ms per recommendation
- **Batch Processing**: ~50-100ms per item

### Throughput
- **Concurrent Users**: 50-100 (single instance)
- **Recommendations/sec**: 10-20
- **Vector Search**: <50ms for 10K items

### Memory Usage
- **Model Loading**: ~2GB RAM
- **Vector Indices**: ~100MB per 10K items
- **Runtime**: ~500MB per active session

## Scalability Considerations

### Horizontal Scaling
- Stateless API design
- Shared vector store (Redis/external FAISS)
- Load balancer distribution

### Vertical Scaling
- GPU acceleration for model inference
- Larger vector indices
- Increased concurrent connections

### Optimization Opportunities
- Model quantization
- Embedding caching
- Batch inference
- Asynchronous processing

## Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.9+)
- **ML Libraries**: 
  - `transformers` (Hugging Face)
  - `open_clip_torch` (OpenCLIP)
  - `faiss-cpu` (Vector Search)
  - `scikit-learn` (ML utilities)
- **Data**: JSON file storage (development)

### Frontend
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **State Management**: Zustand

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Reverse Proxy**: Nginx
- **Development**: Hot reload, auto-restart

## Security & Privacy

### Data Protection
- No persistent user image storage
- Anonymized feedback collection
- Local processing (no external API calls)

### Model Security
- Input validation and sanitization
- Rate limiting on API endpoints
- Error handling without information leakage

## Monitoring & Observability

### Metrics Collection
- Model inference latency
- Recommendation accuracy
- User engagement rates
- System resource usage

### Logging
- Structured JSON logging
- Error tracking and alerting
- Performance monitoring
- User interaction analytics

## Future Architecture Evolution

### Phase 2: Fashion Fine-Tuning
- Custom fashion dataset integration
- Model fine-tuning pipeline
- A/B testing framework

### Phase 3: Advanced Arbitration
- Meta-learning model selection
- Per-user personalization
- Dynamic weight optimization

### Phase 4: Production Scale
- Microservices architecture
- Multi-tenant support
- Cloud-native deployment
- Real-time model updates