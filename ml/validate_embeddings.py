#!/usr/bin/env python3
"""
Validation script for the embedding generator model.
This script evaluates the quality of generated embeddings.
"""

import torch
import numpy as np
import argparse
import logging
from pathlib import Path
import json
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import seaborn as sns

from embedding_mlp_generator import EmbeddingMLPGenerator, GeneratorConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data from file.
    """
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        query_embeddings = np.array(data['query_embeddings'], dtype=np.float32)
        target_embeddings = np.array(data['target_embeddings'], dtype=np.float32)
        
        logger.info(f"Loaded {len(query_embeddings)} test pairs")
        return query_embeddings, target_embeddings
        
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise

def generate_test_data(num_samples: int = 200, embedding_dim: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic test data.
    """
    logger.info(f"Generating {num_samples} synthetic test pairs")
    
    # Generate random query embeddings
    query_embeddings = np.random.randn(num_samples, embedding_dim).astype(np.float32)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    
    # Generate compatible target embeddings
    noise = np.random.randn(num_samples, embedding_dim).astype(np.float32) * 0.3
    target_embeddings = query_embeddings + noise
    target_embeddings = target_embeddings / np.linalg.norm(target_embeddings, axis=1, keepdims=True)
    
    return query_embeddings, target_embeddings

def evaluate_embedding_quality(generated_embeddings: np.ndarray, 
                             target_embeddings: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the quality of generated embeddings.
    """
    metrics = {}
    
    # Cosine similarity
    cosine_sims = []
    for i in range(len(generated_embeddings)):
        sim = cosine_similarity(
            generated_embeddings[i:i+1], 
            target_embeddings[i:i+1]
        )[0, 0]
        cosine_sims.append(sim)
    
    metrics['mean_cosine_similarity'] = np.mean(cosine_sims)
    metrics['std_cosine_similarity'] = np.std(cosine_sims)
    metrics['min_cosine_similarity'] = np.min(cosine_sims)
    metrics['max_cosine_similarity'] = np.max(cosine_sims)
    
    # L2 distance
    l2_distances = np.linalg.norm(generated_embeddings - target_embeddings, axis=1)
    metrics['mean_l2_distance'] = np.mean(l2_distances)
    metrics['std_l2_distance'] = np.std(l2_distances)
    metrics['min_l2_distance'] = np.min(l2_distances)
    metrics['max_l2_distance'] = np.max(l2_distances)
    
    # Embedding diversity (average pairwise distance)
    pairwise_sims = cosine_similarity(generated_embeddings)
    # Exclude diagonal (self-similarity)
    mask = ~np.eye(pairwise_sims.shape[0], dtype=bool)
    diversity = 1 - np.mean(pairwise_sims[mask])
    metrics['embedding_diversity'] = diversity
    
    return metrics

def plot_embedding_analysis(query_embeddings: np.ndarray,
                          generated_embeddings: np.ndarray,
                          target_embeddings: np.ndarray,
                          save_path: str = None):
    """
    Create visualization plots for embedding analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Cosine similarity distribution
    cosine_sims = []
    for i in range(len(generated_embeddings)):
        sim = cosine_similarity(
            generated_embeddings[i:i+1], 
            target_embeddings[i:i+1]
        )[0, 0]
        cosine_sims.append(sim)
    
    axes[0, 0].hist(cosine_sims, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Cosine Similarity Distribution')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(cosine_sims), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(cosine_sims):.3f}')
    axes[0, 0].legend()
    
    # 2. L2 distance distribution
    l2_distances = np.linalg.norm(generated_embeddings - target_embeddings, axis=1)
    axes[0, 1].hist(l2_distances, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('L2 Distance Distribution')
    axes[0, 1].set_xlabel('L2 Distance')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(l2_distances), color='red', linestyle='--',
                       label=f'Mean: {np.mean(l2_distances):.3f}')
    axes[0, 1].legend()
    
    # 3. t-SNE visualization (sample subset for performance)
    sample_size = min(100, len(query_embeddings))
    indices = np.random.choice(len(query_embeddings), sample_size, replace=False)
    
    combined_embeddings = np.vstack([
        query_embeddings[indices],
        generated_embeddings[indices],
        target_embeddings[indices]
    ])
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size-1))
    embeddings_2d = tsne.fit_transform(combined_embeddings)
    
    # Split back into groups
    query_2d = embeddings_2d[:sample_size]
    generated_2d = embeddings_2d[sample_size:2*sample_size]
    target_2d = embeddings_2d[2*sample_size:]
    
    axes[1, 0].scatter(query_2d[:, 0], query_2d[:, 1], alpha=0.6, label='Query', s=30)
    axes[1, 0].scatter(generated_2d[:, 0], generated_2d[:, 1], alpha=0.6, label='Generated', s=30)
    axes[1, 0].scatter(target_2d[:, 0], target_2d[:, 1], alpha=0.6, label='Target', s=30)
    axes[1, 0].set_title('t-SNE Visualization')
    axes[1, 0].legend()
    
    # 4. Pairwise similarity heatmap (sample subset)
    sample_size_heatmap = min(20, len(generated_embeddings))
    indices_heatmap = np.random.choice(len(generated_embeddings), sample_size_heatmap, replace=False)
    
    similarity_matrix = cosine_similarity(generated_embeddings[indices_heatmap])
    sns.heatmap(similarity_matrix, ax=axes[1, 1], cmap='viridis', 
                cbar_kws={'label': 'Cosine Similarity'})
    axes[1, 1].set_title('Generated Embeddings Similarity Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Analysis plots saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Validate embedding generator model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config-path', type=str, help='Path to model configuration')
    parser.add_argument('--test-data-path', type=str, help='Path to test data JSON file')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic test data')
    parser.add_argument('--num-samples', type=int, default=200, help='Number of synthetic test samples')
    parser.add_argument('--embedding-dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--output-dir', type=str, default='./validation_results', help='Output directory')
    parser.add_argument('--plot', action='store_true', help='Generate analysis plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
        config = GeneratorConfig(**config_dict)
    else:
        config = GeneratorConfig(embedding_dim=args.embedding_dim)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    generator = EmbeddingMLPGenerator(config)
    generator.load_model(args.model_path)
    
    # Load or generate test data
    if args.test_data_path:
        query_embeddings, target_embeddings = load_test_data(args.test_data_path)
    elif args.synthetic:
        query_embeddings, target_embeddings = generate_test_data(
            args.num_samples, args.embedding_dim
        )
    else:
        logger.error("Either --test-data-path or --synthetic must be specified")
        return
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    generated_embeddings = generator.generate(query_embeddings)
    
    # Evaluate quality
    logger.info("Evaluating embedding quality...")
    metrics = evaluate_embedding_quality(generated_embeddings, target_embeddings)
    
    # Print results
    logger.info("\n=== Validation Results ===")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.6f}")
    
    # Save results
    results_path = output_dir / "validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Generate plots if requested
    if args.plot:
        plot_path = output_dir / "validation_analysis.png"
        plot_embedding_analysis(
            query_embeddings, generated_embeddings, target_embeddings, str(plot_path)
        )
    
    # Quality assessment
    logger.info("\n=== Quality Assessment ===")
    if metrics['mean_cosine_similarity'] > 0.8:
        logger.info("✅ Excellent embedding quality (cosine similarity > 0.8)")
    elif metrics['mean_cosine_similarity'] > 0.6:
        logger.info("✅ Good embedding quality (cosine similarity > 0.6)")
    elif metrics['mean_cosine_similarity'] > 0.4:
        logger.info("⚠️  Fair embedding quality (cosine similarity > 0.4)")
    else:
        logger.info("❌ Poor embedding quality (cosine similarity <= 0.4)")
    
    if metrics['embedding_diversity'] > 0.3:
        logger.info("✅ Good embedding diversity")
    else:
        logger.info("⚠️  Low embedding diversity - consider increasing diversity factor")

if __name__ == "__main__":
    main()