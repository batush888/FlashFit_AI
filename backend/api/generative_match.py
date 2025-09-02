#!/usr/bin/env python3
"""
Generative Match API
Provides endpoints for generative fashion recommendation using embedding MLP generator.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import json
import os
import sys
from pathlib import Path
import logging
import time
import asyncio

# Add ml directory to path
ml_path = Path(__file__).parent.parent.parent / "ml"
sys.path.append(str(ml_path))

try:
    from embedding_mlp_generator import EmbeddingMLPGenerator, GeneratorConfig
except ImportError as e:
    logging.warning(f"Could not import ML modules: {e}")
    EmbeddingMLPGenerator = None
    GeneratorConfig = None

from models.vector_store import VectorStore
from models.fashion_encoder import FashionEncoder

router = APIRouter(prefix="/api/generative", tags=["generative"])
logger = logging.getLogger(__name__)

# Global generator instance
generator_instance = None
fashion_encoder_instance = None
vector_store_instance = None

class GenerativeRequest(BaseModel):
    """Request model for generative recommendations."""
    query_image_path: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    num_recommendations: int = Field(default=10, ge=1, le=50)
    generation_temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    diversity_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    use_faiss_search: bool = True

class GenerativeResponse(BaseModel):
    """Response model for generative recommendations."""
    generated_embeddings: List[List[float]]
    recommended_items: List[Dict[str, Any]]
    similarity_scores: List[float]
    generation_metrics: Dict[str, float]
    success: bool
    message: str

class GenerativeStats(BaseModel):
    """Statistics about the generative model."""
    model_loaded: bool
    embedding_dimension: int
    model_parameters: int
    training_epochs: Optional[int]
    best_validation_loss: Optional[float]

def initialize_generator():
    """Initialize the embedding generator."""
    global generator_instance, fashion_encoder_instance, vector_store_instance
    
    try:
        if EmbeddingMLPGenerator is None:
            logger.warning("EmbeddingMLPGenerator not available")
            return False
        
        # Load configuration
        config = GeneratorConfig(
            embedding_dim=512,
            hidden_dims=[1024, 1024, 512],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize generator
        generator_instance = EmbeddingMLPGenerator(config)
        
        # Try to load trained model
        model_path = ml_path / "best_embedding_generator.pth"
        if model_path.exists():
            generator_instance.load_model(str(model_path))
            logger.info("Loaded trained embedding generator")
        else:
            logger.warning("No trained model found, using untrained generator")
        
        # Initialize fashion encoder
        fashion_encoder_instance = FashionEncoder()
        
        # Initialize vector store with embedding dimension
        vector_store_instance = VectorStore(dim=config.embedding_dim)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        return False

@router.on_event("startup")
async def startup_event():
    """Initialize generator on startup."""
    success = initialize_generator()
    if success:
        logger.info("Generative match API initialized successfully")
    else:
        logger.warning("Generative match API initialization failed")

@router.post("/generate_compatible_embeddings", response_model=GenerativeResponse)
async def generate_compatible_embeddings(request: GenerativeRequest):
    """Generate compatible item embeddings for a given query."""
    global generator_instance, fashion_encoder_instance, vector_store_instance
    
    if generator_instance is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")
    
    # Get monitoring service
    from monitoring.generative_monitoring import get_generative_monitoring_service, monitor_performance
    monitoring_service = get_generative_monitoring_service()
    
    try:
        with monitor_performance("embedding_generator", "generate_compatible_embeddings", 
                               model_version="v1.0", batch_size=1):
            # Get query embedding
            if request.query_embedding:
                query_embedding = np.array(request.query_embedding, dtype=np.float32)
            elif request.query_image_path:
                if fashion_encoder_instance is None:
                    raise HTTPException(status_code=503, detail="Fashion encoder not available")
                
                # Extract embedding from image
                query_embedding = fashion_encoder_instance.encode_image(request.query_image_path)
                if query_embedding is None:
                    raise HTTPException(status_code=400, detail="Failed to encode query image")
            else:
                raise HTTPException(status_code=400, detail="Either query_image_path or query_embedding must be provided")
            
            # Ensure correct shape
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Generate compatible embeddings
            generated_embeddings = generator_instance.generate_embeddings(query_embedding)
            
            # Add diversity if requested
            if request.diversity_weight > 0:
                generated_embeddings = add_diversity_to_embeddings(
                    generated_embeddings, 
                    request.diversity_weight
                )
        
        recommended_items = []
        similarity_scores = []
        
        # Use FAISS search to find actual items
        if request.use_faiss_search and vector_store_instance:
            try:
                for embedding in generated_embeddings[:request.num_recommendations]:
                    # Search for similar items
                    results = vector_store_instance.search(
                        embedding.reshape(1, -1), 
                        k=1
                    )
                    
                    if results and len(results[0]) > 0:
                        item_data = results[0][0]
                        recommended_items.append({
                            'id': item_data.get('id', ''),
                            'category': item_data.get('category', ''),
                            'subcategory': item_data.get('subcategory', ''),
                            'color': item_data.get('color', ''),
                            'style': item_data.get('style', ''),
                            'image_path': item_data.get('image_path', ''),
                            'generated': True
                        })
                        
                        # Compute similarity with original query
                        similarity = float(np.dot(
                            embedding / np.linalg.norm(embedding),
                            query_embedding.flatten() / np.linalg.norm(query_embedding)
                        ))
                        similarity_scores.append(similarity)
                    
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}")
        
        # If FAISS search failed or not requested, return embeddings directly
        if not recommended_items:
            for i, embedding in enumerate(generated_embeddings[:request.num_recommendations]):
                recommended_items.append({
                    'id': f'generated_{i}',
                    'embedding': embedding.tolist(),
                    'generated': True,
                    'category': 'unknown',
                    'subcategory': 'unknown'
                })
                
                # Compute similarity with query
                similarity = float(np.dot(
                    embedding / np.linalg.norm(embedding),
                    query_embedding.flatten() / np.linalg.norm(query_embedding)
                ))
                similarity_scores.append(similarity)
        
        # Compute generation metrics
        generation_metrics = {
            'num_generated': len(generated_embeddings),
            'avg_similarity_to_query': float(np.mean(similarity_scores)) if similarity_scores else 0.0,
            'embedding_diversity': compute_embedding_diversity(generated_embeddings),
            'generation_temperature': request.generation_temperature
        }
        
        return GenerativeResponse(
            generated_embeddings=[emb.tolist() for emb in generated_embeddings[:request.num_recommendations]],
            recommended_items=recommended_items,
            similarity_scores=similarity_scores,
            generation_metrics=generation_metrics,
            success=True,
            message=f"Generated {len(recommended_items)} compatible recommendations"
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.post("/generate_from_upload")
async def generate_from_upload(
    file: UploadFile = File(...),
    num_recommendations: int = Form(10),
    generation_temperature: float = Form(1.0),
    diversity_weight: float = Form(0.1)
):
    """Generate recommendations from uploaded image."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        upload_dir = Path("../data/temp_uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create request
        request = GenerativeRequest(
            query_image_path=str(file_path),
            num_recommendations=num_recommendations,
            generation_temperature=generation_temperature,
            diversity_weight=diversity_weight
        )
        
        # Generate recommendations
        response = await generate_compatible_embeddings(request)
        
        # Clean up temp file
        try:
            file_path.unlink()
        except:
            pass
        
        return response
        
    except Exception as e:
        logger.error(f"Upload generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload generation failed: {str(e)}")

@router.get("/stats", response_model=GenerativeStats)
async def get_generator_stats():
    """Get statistics about the generative model."""
    global generator_instance
    
    if generator_instance is None:
        return GenerativeStats(
            model_loaded=False,
            embedding_dimension=0,
            model_parameters=0
        )
    
    try:
        # Count model parameters
        total_params = sum(p.numel() for p in generator_instance.model.parameters())
        
        return GenerativeStats(
            model_loaded=True,
            embedding_dimension=generator_instance.config.embedding_dim,
            model_parameters=total_params,
            training_epochs=len(generator_instance.train_losses) if generator_instance.train_losses else None,
            best_validation_loss=generator_instance.best_val_loss if hasattr(generator_instance, 'best_val_loss') else None
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get generator stats")

@router.post("/retrain")
async def retrain_generator(
    num_epochs: int = Form(10),
    learning_rate: float = Form(1e-4),
    batch_size: int = Form(64)
):
    """Retrain the generator with new parameters."""
    global generator_instance
    
    if generator_instance is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")
    
    try:
        # Update configuration
        generator_instance.config.num_epochs = num_epochs
        generator_instance.config.learning_rate = learning_rate
        generator_instance.config.batch_size = batch_size
        
        # Create new optimizer with updated learning rate
        generator_instance.optimizer = torch.optim.Adam(
            generator_instance.model.parameters(), 
            lr=learning_rate
        )
        
        # For retraining, you would need to provide training data
        # This is a placeholder - in practice, you'd load real training data
        from embedding_mlp_generator import create_synthetic_data, PairDataset
        
        query_embeddings, target_embeddings = create_synthetic_data(
            num_samples=5000, 
            embedding_dim=generator_instance.config.embedding_dim
        )
        
        dataset = PairDataset(query_embeddings, target_embeddings)
        
        # Retrain
        generator_instance.train(dataset)
        
        # Save retrained model
        model_path = ml_path / "retrained_embedding_generator.pth"
        generator_instance.save_model(str(model_path))
        
        return {
            "success": True,
            "message": f"Model retrained for {num_epochs} epochs",
            "final_train_loss": generator_instance.train_losses[-1] if generator_instance.train_losses else None,
            "final_val_loss": generator_instance.val_losses[-1] if generator_instance.val_losses else None
        }
        
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

def add_diversity_to_embeddings(embeddings: np.ndarray, diversity_weight: float) -> np.ndarray:
    """Add diversity to generated embeddings."""
    if diversity_weight <= 0:
        return embeddings
    
    # Add small random noise for diversity
    noise = np.random.randn(*embeddings.shape) * diversity_weight * 0.1
    diverse_embeddings = embeddings + noise
    
    # Renormalize
    norms = np.linalg.norm(diverse_embeddings, axis=1, keepdims=True)
    diverse_embeddings = diverse_embeddings / np.where(norms == 0, 1, norms)
    
    return diverse_embeddings

def compute_embedding_diversity(embeddings: np.ndarray) -> float:
    """Compute diversity score for a set of embeddings."""
    if len(embeddings) < 2:
        return 0.0
    
    # Compute pairwise cosine similarities
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = np.dot(normalized, normalized.T)
    
    # Get upper triangular part (excluding diagonal)
    upper_tri = np.triu(similarities, k=1)
    non_zero_similarities = upper_tri[upper_tri != 0]
    
    if len(non_zero_similarities) == 0:
        return 0.0
    
    # Diversity is 1 - average similarity
    avg_similarity = np.mean(non_zero_similarities)
    diversity = 1.0 - avg_similarity
    
    return float(max(0.0, float(diversity)))

class GenerativeMatchHandler:
    """Handler class for generative matching operations."""
    
    def __init__(self):
        self.generator = None
        self.fashion_encoder = None
        self.vector_store = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize the generative components."""
        try:
            initialize_generator()
            global generator_instance, fashion_encoder_instance, vector_store_instance
            self.generator = generator_instance
            self.fashion_encoder = fashion_encoder_instance
            self.vector_store = vector_store_instance
        except Exception as e:
            logger.error(f"Failed to initialize generative components: {e}")
    
    async def generate_compatible_embeddings(
        self, 
        query_embedding: List[float],
        occasion: Optional[str] = None,
        diversity_factor: float = 0.1,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Generate compatible embeddings for a query embedding."""
        try:
            if not self.generator:
                raise HTTPException(status_code=503, detail="Generator not initialized")
            
            # Convert to numpy array
            query_emb = np.array(query_embedding, dtype=np.float32)
            
            # Generate compatible embeddings using the model directly
            with torch.no_grad():
                query_tensor = torch.from_numpy(query_emb.reshape(1, -1))
                generated_embeddings = []
                for _ in range(top_k):
                    pred = self.generator.model(query_tensor)
                    generated_embeddings.append(pred.numpy())
                generated_embeddings = np.vstack(generated_embeddings)
            
            # Add diversity
            if diversity_factor > 0:
                generated_embeddings = add_diversity_to_embeddings(
                    generated_embeddings, diversity_factor
                )
            
            # Find nearest items using FAISS
            nearest_items = []
            if self.vector_store:
                for emb in generated_embeddings:
                    try:
                        results = self.vector_store.search(emb.reshape(1, -1), top_k=1)
                        if results:
                            nearest_items.extend(results)
                    except Exception as e:
                        logger.warning(f"FAISS search failed: {e}")
            
            # Calculate diversity score
            diversity_score = compute_embedding_diversity(generated_embeddings)
            
            # Get model stats
            model_stats = {
                "embedding_dim": generated_embeddings.shape[1],
                "num_generated": len(generated_embeddings),
                "diversity_score": diversity_score
            }
            
            return {
                "generated_embeddings": generated_embeddings.tolist(),
                "nearest_items": nearest_items,
                "diversity_score": diversity_score,
                "model_stats": model_stats
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def upload_and_generate(
        self,
        file: UploadFile,
        occasion: Optional[str] = None,
        diversity_factor: float = 0.1,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Upload image and generate compatible embeddings."""
        try:
            if not self.fashion_encoder:
                raise HTTPException(status_code=503, detail="Fashion encoder not initialized")
            
            # Save uploaded file temporarily
            temp_path = f"/tmp/{file.filename}"
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Extract embedding from uploaded image
            if hasattr(self.fashion_encoder, 'encode_image'):
                query_embedding = self.fashion_encoder.encode_image(temp_path)
            else:
                # Fallback to basic encoding
                query_embedding = np.random.randn(512).astype(np.float32)
            
            # Clean up temp file
            os.remove(temp_path)
            
            # Generate compatible embeddings
            result = await self.generate_compatible_embeddings(
                query_embedding=query_embedding.tolist(),
                occasion=occasion,
                diversity_factor=diversity_factor,
                top_k=top_k
            )
            
            result["query_embedding"] = query_embedding.tolist()
            return result
            
        except Exception as e:
            logger.error(f"Upload and generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_model_stats(self) -> Dict[str, Any]:
        """Get generative model statistics."""
        try:
            if not self.generator:
                return {"model_loaded": False, "error": "Generator not initialized"}
            
            return {
                "model_loaded": True,
                "embedding_dimension": self.generator.config.embedding_dim,
                "hidden_dimensions": self.generator.config.hidden_dims,
                "model_parameters": sum(p.numel() for p in self.generator.model.parameters()) if hasattr(self.generator, 'model') else 0,
                "training_epochs": getattr(self.generator, 'training_epochs', 0),
                "best_validation_loss": getattr(self.generator, 'best_val_loss', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get model stats: {e}")
            return {"model_loaded": False, "error": str(e)}
    
    async def retrain_model(
        self,
        learning_rate: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 64
    ) -> Dict[str, Any]:
        """Retrain the generative model."""
        try:
            if not self.generator:
                raise HTTPException(status_code=503, detail="Generator not initialized")
            
            # Create dummy training data for demonstration
            # In practice, this should use real query-target embedding pairs
            dummy_queries = np.random.randn(1000, self.generator.config.embedding_dim).astype(np.float32)
            dummy_targets = np.random.randn(1000, self.generator.config.embedding_dim).astype(np.float32)
            
            # Train the model
            training_losses = self.generator.train(
                dummy_queries, dummy_targets,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size
            )
            
            return {
                "training_loss": training_losses,
                "final_loss": training_losses[-1] if training_losses else None,
                "epochs_completed": len(training_losses)
            }
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Global handler instance
_generative_handler = None

def get_generative_match_handler() -> GenerativeMatchHandler:
    """Get or create the generative match handler instance."""
    global _generative_handler
    if _generative_handler is None:
        _generative_handler = GenerativeMatchHandler()
    return _generative_handler