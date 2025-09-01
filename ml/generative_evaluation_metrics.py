#!/usr/bin/env python3
"""
Generative Evaluation Metrics for FlashFit AI

This module implements:
1. Comprehensive evaluation metrics for generative fashion AI
2. Novelty and diversity measurements
3. Quality assessment metrics
4. User preference correlation analysis
5. A/B testing utilities
6. Real-time monitoring dashboards
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
import pickle
from collections import defaultdict, Counter
from sklearn.metrics import ndcg_score, precision_score, recall_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for generative evaluation metrics"""
    # Diversity settings
    diversity_k: int = 10  # Top-k for diversity calculation
    cluster_k: int = 20    # Number of clusters for diversity analysis
    
    # Novelty settings
    novelty_threshold: float = 0.8  # Similarity threshold for novelty
    novelty_window_days: int = 30   # Time window for novelty calculation
    
    # Quality settings
    quality_sample_size: int = 1000  # Sample size for quality evaluation
    confidence_threshold: float = 0.7  # Minimum confidence for quality assessment
    
    # A/B testing
    ab_test_duration_days: int = 7
    ab_test_min_samples: int = 100
    
    # Visualization
    plot_save_dir: str = "plots/evaluation"
    
    def __post_init__(self):
        Path(self.plot_save_dir).mkdir(parents=True, exist_ok=True)

class NoveltyMetrics:
    """Metrics for measuring novelty of generated recommendations"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.historical_embeddings = []
        self.historical_timestamps = []
    
    def add_historical_data(self, embeddings: np.ndarray, timestamps: List[datetime]):
        """Add historical embeddings for novelty comparison"""
        self.historical_embeddings.extend(embeddings)
        self.historical_timestamps.extend(timestamps)
        
        # Keep only recent data within the novelty window
        cutoff_date = datetime.now() - timedelta(days=self.config.novelty_window_days)
        
        filtered_embeddings = []
        filtered_timestamps = []
        
        for emb, ts in zip(self.historical_embeddings, self.historical_timestamps):
            if ts >= cutoff_date:
                filtered_embeddings.append(emb)
                filtered_timestamps.append(ts)
        
        self.historical_embeddings = filtered_embeddings
        self.historical_timestamps = filtered_timestamps
    
    def compute_novelty_score(self, generated_embeddings: np.ndarray) -> Dict[str, float]:
        """
        Compute novelty score for generated embeddings
        
        Args:
            generated_embeddings: Generated item embeddings [num_samples, embedding_dim]
            
        Returns:
            Dictionary with novelty metrics
        """
        if len(self.historical_embeddings) == 0:
            return {'novelty_score': 1.0, 'novel_items_ratio': 1.0, 'avg_min_distance': 1.0}
        
        historical_matrix = np.array(self.historical_embeddings)
        
        novelty_scores = []
        novel_items = 0
        min_distances = []
        
        for gen_emb in generated_embeddings:
            # Compute cosine similarities with all historical embeddings
            similarities = np.dot(historical_matrix, gen_emb) / (
                np.linalg.norm(historical_matrix, axis=1) * np.linalg.norm(gen_emb)
            )
            
            max_similarity = np.max(similarities)
            min_distance = 1.0 - max_similarity
            min_distances.append(min_distance)
            
            # Item is novel if max similarity is below threshold
            if max_similarity < self.config.novelty_threshold:
                novel_items += 1
            
            # Novelty score is 1 - max_similarity
            novelty_scores.append(1.0 - max_similarity)
        
        return {
            'novelty_score': float(np.mean(novelty_scores)),
            'novel_items_ratio': float(novel_items / len(generated_embeddings)),
            'avg_min_distance': float(np.mean(min_distances)),
            'novelty_std': float(np.std(novelty_scores))
        }
    
    def compute_temporal_novelty(self, generated_embeddings: np.ndarray) -> Dict[str, float]:
        """
        Compute how novelty changes over time
        
        Returns:
            Temporal novelty analysis
        """
        if len(self.historical_embeddings) < 10:
            return {'temporal_novelty_trend': 0.0}
        
        # Group historical data by time periods
        time_periods = []
        current_time = datetime.now()
        
        for days_back in [1, 7, 14, 30]:
            cutoff = current_time - timedelta(days=days_back)
            period_embeddings = [
                emb for emb, ts in zip(self.historical_embeddings, self.historical_timestamps)
                if ts >= cutoff
            ]
            
            if len(period_embeddings) > 0:
                period_matrix = np.array(period_embeddings)
                
                # Compute novelty against this time period
                period_novelty = []
                for gen_emb in generated_embeddings:
                    similarities = np.dot(period_matrix, gen_emb) / (
                        np.linalg.norm(period_matrix, axis=1) * np.linalg.norm(gen_emb)
                    )
                    period_novelty.append(1.0 - np.max(similarities))
                
                time_periods.append({
                    'days_back': days_back,
                    'novelty_score': np.mean(period_novelty),
                    'num_items': len(period_embeddings)
                })
        
        # Compute trend (increasing novelty over time is positive)
        if len(time_periods) >= 2:
            novelty_values = [p['novelty_score'] for p in time_periods]
            trend = np.polyfit(range(len(novelty_values)), novelty_values, 1)[0]
        else:
            trend = 0.0
        
        return {
            'temporal_novelty_trend': float(trend),
            'time_periods': time_periods
        }

class DiversityMetrics:
    """Metrics for measuring diversity of generated recommendations"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def compute_intra_list_diversity(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Compute diversity within a single recommendation list
        
        Args:
            embeddings: Item embeddings [num_items, embedding_dim]
            
        Returns:
            Intra-list diversity metrics
        """
        if len(embeddings) < 2:
            return {'intra_list_diversity': 0.0, 'avg_pairwise_distance': 0.0}
        
        # Compute pairwise cosine distances
        similarities = np.dot(embeddings, embeddings.T) / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) * 
            np.linalg.norm(embeddings, axis=1, keepdims=True).T
        )
        
        # Convert to distances
        distances = 1.0 - similarities
        
        # Get upper triangular part (excluding diagonal)
        upper_tri_indices = np.triu_indices_from(distances, k=1)
        pairwise_distances = distances[upper_tri_indices]
        
        return {
            'intra_list_diversity': float(np.mean(pairwise_distances)),
            'avg_pairwise_distance': float(np.mean(pairwise_distances)),
            'min_pairwise_distance': float(np.min(pairwise_distances)),
            'max_pairwise_distance': float(np.max(pairwise_distances)),
            'diversity_std': float(np.std(pairwise_distances))
        }
    
    def compute_coverage_diversity(self, 
                                 generated_embeddings: np.ndarray,
                                 catalog_embeddings: np.ndarray) -> Dict[str, float]:
        """
        Compute how well generated items cover the catalog space
        
        Args:
            generated_embeddings: Generated item embeddings
            catalog_embeddings: Full catalog embeddings
            
        Returns:
            Coverage diversity metrics
        """
        # Use clustering to define regions of the catalog space
        kmeans = KMeans(n_clusters=self.config.cluster_k, random_state=42)
        catalog_clusters = kmeans.fit_predict(catalog_embeddings)
        
        # Assign generated items to clusters
        generated_clusters = kmeans.predict(generated_embeddings)
        
        # Compute coverage
        unique_generated_clusters = set(generated_clusters)
        unique_catalog_clusters = set(catalog_clusters)
        
        coverage_ratio = len(unique_generated_clusters) / len(unique_catalog_clusters)
        
        # Compute cluster distribution entropy
        cluster_counts = Counter(generated_clusters)
        total_generated = len(generated_embeddings)
        
        entropy = 0.0
        for count in cluster_counts.values():
            prob = count / total_generated
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        max_entropy = np.log2(len(unique_generated_clusters)) if len(unique_generated_clusters) > 0 else 0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            'coverage_ratio': float(coverage_ratio),
            'cluster_entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'num_covered_clusters': len(unique_generated_clusters),
            'total_clusters': len(unique_catalog_clusters)
        }
    
    def compute_categorical_diversity(self, 
                                    generated_items: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute diversity across categorical attributes
        
        Args:
            generated_items: List of generated items with attributes
            
        Returns:
            Categorical diversity metrics
        """
        if not generated_items:
            return {'categorical_diversity': 0.0}
        
        # Extract categorical attributes
        categories = ['category', 'color', 'brand', 'style', 'occasion']
        diversity_scores = {}
        
        for category in categories:
            if category in generated_items[0]:
                values = [item.get(category, 'unknown') for item in generated_items]
                unique_values = set(values)
                
                # Shannon entropy for this category
                value_counts = Counter(values)
                total_items = len(values)
                
                entropy = 0.0
                for count in value_counts.values():
                    prob = count / total_items
                    if prob > 0:
                        entropy -= prob * np.log2(prob)
                
                max_entropy = np.log2(len(unique_values)) if len(unique_values) > 0 else 0
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                diversity_scores[f'{category}_diversity'] = float(normalized_entropy)
                diversity_scores[f'{category}_unique_count'] = len(unique_values)
        
        # Overall categorical diversity (average)
        category_diversities = [v for k, v in diversity_scores.items() if k.endswith('_diversity')]
        diversity_scores['categorical_diversity'] = float(np.mean(category_diversities)) if category_diversities else 0.0
        
        return diversity_scores

class QualityMetrics:
    """Metrics for measuring quality of generated recommendations"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def compute_compatibility_quality(self, 
                                    query_embeddings: np.ndarray,
                                    generated_embeddings: np.ndarray,
                                    ground_truth_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute quality of compatibility predictions
        
        Args:
            query_embeddings: Query item embeddings
            generated_embeddings: Generated compatible item embeddings
            ground_truth_scores: Optional ground truth compatibility scores
            
        Returns:
            Quality metrics
        """
        # Compute cosine similarities as proxy for compatibility
        similarities = []
        for query_emb, gen_emb in zip(query_embeddings, generated_embeddings):
            sim = np.dot(query_emb, gen_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(gen_emb)
            )
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        quality_metrics = {
            'avg_compatibility': float(np.mean(similarities)),
            'compatibility_std': float(np.std(similarities)),
            'min_compatibility': float(np.min(similarities)),
            'max_compatibility': float(np.max(similarities))
        }
        
        # If ground truth is available, compute correlation
        if ground_truth_scores is not None:
            correlation, p_value = pearsonr(similarities, ground_truth_scores)
            spearman_corr, spearman_p = spearmanr(similarities, ground_truth_scores)
            
            quality_metrics.update({
                'pearson_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                'pearson_p_value': float(p_value) if not np.isnan(p_value) else 1.0,
                'spearman_correlation': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
                'spearman_p_value': float(spearman_p) if not np.isnan(spearman_p) else 1.0
            })
        
        return quality_metrics
    
    def compute_ranking_quality(self, 
                              predicted_scores: np.ndarray,
                              true_scores: np.ndarray,
                              k_values: List[int] = None) -> Dict[str, float]:
        """
        Compute ranking quality metrics
        
        Args:
            predicted_scores: Predicted compatibility scores
            true_scores: True compatibility scores
            k_values: List of k values for precision@k, recall@k
            
        Returns:
            Ranking quality metrics
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        # Convert to binary relevance (top 20% are relevant)
        threshold = np.percentile(true_scores, 80)
        true_binary = (true_scores >= threshold).astype(int)
        
        ranking_metrics = {}
        
        # NDCG
        try:
            ndcg = ndcg_score([true_scores], [predicted_scores])
            ranking_metrics['ndcg'] = float(ndcg)
        except:
            ranking_metrics['ndcg'] = 0.0
        
        # Precision@k and Recall@k
        for k in k_values:
            if k <= len(predicted_scores):
                # Get top-k predictions
                top_k_indices = np.argsort(predicted_scores)[-k:]
                top_k_binary = true_binary[top_k_indices]
                
                precision_k = np.sum(top_k_binary) / k
                recall_k = np.sum(top_k_binary) / np.sum(true_binary) if np.sum(true_binary) > 0 else 0
                
                ranking_metrics[f'precision@{k}'] = float(precision_k)
                ranking_metrics[f'recall@{k}'] = float(recall_k)
        
        # AUC if possible
        try:
            auc = roc_auc_score(true_binary, predicted_scores)
            ranking_metrics['auc'] = float(auc)
        except:
            ranking_metrics['auc'] = 0.5
        
        return ranking_metrics
    
    def compute_confidence_calibration(self, 
                                     predicted_scores: np.ndarray,
                                     confidence_scores: np.ndarray,
                                     true_scores: np.ndarray,
                                     num_bins: int = 10) -> Dict[str, float]:
        """
        Compute confidence calibration metrics
        
        Args:
            predicted_scores: Model predictions
            confidence_scores: Model confidence scores
            true_scores: Ground truth scores
            num_bins: Number of bins for calibration
            
        Returns:
            Calibration metrics
        """
        # Bin predictions by confidence
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_error = 0.0
        total_samples = len(predicted_scores)
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this confidence bin
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Compute accuracy in this bin (using correlation as proxy)
                bin_pred = predicted_scores[in_bin]
                bin_true = true_scores[in_bin]
                
                if len(bin_pred) > 1:
                    accuracy = np.corrcoef(bin_pred, bin_true)[0, 1]
                    if np.isnan(accuracy):
                        accuracy = 0.0
                else:
                    accuracy = 0.0
                
                # Average confidence in this bin
                avg_confidence = confidence_scores[in_bin].mean()
                
                # Contribution to calibration error
                calibration_error += np.abs(avg_confidence - accuracy) * prop_in_bin
                
                bin_accuracies.append(accuracy)
                bin_confidences.append(avg_confidence)
                bin_counts.append(int(in_bin.sum()))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)
                bin_counts.append(0)
        
        return {
            'expected_calibration_error': float(calibration_error),
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'avg_confidence': float(np.mean(confidence_scores)),
            'confidence_std': float(np.std(confidence_scores))
        }

class ABTestingMetrics:
    """Metrics for A/B testing generative vs traditional recommendations"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.test_data = defaultdict(list)
    
    def log_interaction(self, 
                       user_id: str,
                       variant: str,  # 'generative' or 'traditional'
                       query_item: str,
                       recommended_items: List[str],
                       user_actions: Dict[str, Any],
                       timestamp: datetime):
        """
        Log user interaction for A/B testing
        
        Args:
            user_id: User identifier
            variant: Test variant ('generative' or 'traditional')
            query_item: Query item ID
            recommended_items: List of recommended item IDs
            user_actions: User actions (clicks, likes, purchases, etc.)
            timestamp: Interaction timestamp
        """
        interaction = {
            'user_id': user_id,
            'variant': variant,
            'query_item': query_item,
            'recommended_items': recommended_items,
            'user_actions': user_actions,
            'timestamp': timestamp
        }
        
        self.test_data[variant].append(interaction)
    
    def compute_ab_test_metrics(self) -> Dict[str, Any]:
        """
        Compute A/B test metrics comparing generative vs traditional
        
        Returns:
            A/B test results with statistical significance
        """
        if len(self.test_data['generative']) < self.config.ab_test_min_samples or \
           len(self.test_data['traditional']) < self.config.ab_test_min_samples:
            return {'error': 'Insufficient data for A/B testing'}
        
        metrics = {}
        
        for metric_name in ['click_rate', 'like_rate', 'purchase_rate', 'engagement_time']:
            generative_values = []
            traditional_values = []
            
            # Extract metric values
            for interaction in self.test_data['generative']:
                if metric_name in interaction['user_actions']:
                    generative_values.append(interaction['user_actions'][metric_name])
            
            for interaction in self.test_data['traditional']:
                if metric_name in interaction['user_actions']:
                    traditional_values.append(interaction['user_actions'][metric_name])
            
            if len(generative_values) > 0 and len(traditional_values) > 0:
                # Compute means and confidence intervals
                gen_mean = np.mean(generative_values)
                trad_mean = np.mean(traditional_values)
                
                gen_std = np.std(generative_values)
                trad_std = np.std(traditional_values)
                
                # Statistical significance test (t-test)
                from scipy.stats import ttest_ind
                t_stat, p_value = ttest_ind(generative_values, traditional_values)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(generative_values) - 1) * gen_std**2 + 
                                    (len(traditional_values) - 1) * trad_std**2) / 
                                   (len(generative_values) + len(traditional_values) - 2))
                
                cohens_d = (gen_mean - trad_mean) / pooled_std if pooled_std > 0 else 0
                
                metrics[metric_name] = {
                    'generative_mean': float(gen_mean),
                    'traditional_mean': float(trad_mean),
                    'lift': float((gen_mean - trad_mean) / trad_mean * 100) if trad_mean > 0 else 0,
                    'p_value': float(p_value),
                    'cohens_d': float(cohens_d),
                    'significant': p_value < 0.05,
                    'generative_samples': len(generative_values),
                    'traditional_samples': len(traditional_values)
                }
        
        return metrics

class GenerativeEvaluator:
    """Main evaluator for generative fashion AI system"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.novelty_metrics = NoveltyMetrics(config)
        self.diversity_metrics = DiversityMetrics(config)
        self.quality_metrics = QualityMetrics(config)
        self.ab_testing = ABTestingMetrics(config)
        
        self.evaluation_history = []
    
    def evaluate_generation_batch(self, 
                                query_embeddings: np.ndarray,
                                generated_embeddings: np.ndarray,
                                catalog_embeddings: np.ndarray,
                                generated_items: List[Dict[str, Any]],
                                ground_truth_scores: Optional[np.ndarray] = None,
                                predicted_scores: Optional[np.ndarray] = None,
                                confidence_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a generation batch
        
        Args:
            query_embeddings: Query item embeddings
            generated_embeddings: Generated compatible item embeddings
            catalog_embeddings: Full catalog embeddings for coverage analysis
            generated_items: Generated items with metadata
            ground_truth_scores: Optional ground truth compatibility scores
            predicted_scores: Optional model prediction scores
            confidence_scores: Optional model confidence scores
            
        Returns:
            Comprehensive evaluation results
        """
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'batch_size': len(query_embeddings)
        }
        
        # Novelty metrics
        novelty_results = self.novelty_metrics.compute_novelty_score(generated_embeddings)
        temporal_novelty = self.novelty_metrics.compute_temporal_novelty(generated_embeddings)
        evaluation_results['novelty'] = {**novelty_results, **temporal_novelty}
        
        # Diversity metrics
        intra_diversity = self.diversity_metrics.compute_intra_list_diversity(generated_embeddings)
        coverage_diversity = self.diversity_metrics.compute_coverage_diversity(
            generated_embeddings, catalog_embeddings
        )
        categorical_diversity = self.diversity_metrics.compute_categorical_diversity(generated_items)
        
        evaluation_results['diversity'] = {
            **intra_diversity,
            **coverage_diversity,
            **categorical_diversity
        }
        
        # Quality metrics
        compatibility_quality = self.quality_metrics.compute_compatibility_quality(
            query_embeddings, generated_embeddings, ground_truth_scores
        )
        evaluation_results['quality'] = compatibility_quality
        
        # Ranking quality if scores available
        if predicted_scores is not None and ground_truth_scores is not None:
            ranking_quality = self.quality_metrics.compute_ranking_quality(
                predicted_scores, ground_truth_scores
            )
            evaluation_results['ranking'] = ranking_quality
        
        # Confidence calibration if available
        if (confidence_scores is not None and 
            predicted_scores is not None and 
            ground_truth_scores is not None):
            calibration = self.quality_metrics.compute_confidence_calibration(
                predicted_scores, confidence_scores, ground_truth_scores
            )
            evaluation_results['calibration'] = calibration
        
        # Add to history
        self.evaluation_history.append(evaluation_results)
        
        # Update historical data for novelty
        self.novelty_metrics.add_historical_data(
            generated_embeddings, 
            [datetime.now()] * len(generated_embeddings)
        )
        
        return evaluation_results
    
    def generate_evaluation_report(self, 
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Evaluation report
        """
        if not self.evaluation_history:
            return {'error': 'No evaluation data available'}
        
        # Aggregate metrics over time
        report = {
            'evaluation_period': {
                'start': self.evaluation_history[0]['timestamp'],
                'end': self.evaluation_history[-1]['timestamp'],
                'num_evaluations': len(self.evaluation_history)
            },
            'summary_metrics': {},
            'trends': {},
            'recommendations': []
        }
        
        # Aggregate metrics
        metric_categories = ['novelty', 'diversity', 'quality', 'ranking', 'calibration']
        
        for category in metric_categories:
            category_metrics = defaultdict(list)
            
            for eval_result in self.evaluation_history:
                if category in eval_result:
                    for metric_name, value in eval_result[category].items():
                        if isinstance(value, (int, float)):
                            category_metrics[metric_name].append(value)
            
            # Compute summary statistics
            category_summary = {}
            for metric_name, values in category_metrics.items():
                if values:
                    category_summary[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'trend': float(np.polyfit(range(len(values)), values, 1)[0]) if len(values) > 1 else 0.0
                    }
            
            if category_summary:
                report['summary_metrics'][category] = category_summary
        
        # Generate recommendations
        recommendations = []
        
        # Check novelty
        if 'novelty' in report['summary_metrics']:
            novelty_score = report['summary_metrics']['novelty'].get('novelty_score', {}).get('mean', 0)
            if novelty_score < 0.3:
                recommendations.append({
                    'type': 'novelty',
                    'priority': 'high',
                    'message': 'Low novelty detected. Consider increasing diversity in training data or adjusting generation parameters.'
                })
        
        # Check diversity
        if 'diversity' in report['summary_metrics']:
            diversity_score = report['summary_metrics']['diversity'].get('intra_list_diversity', {}).get('mean', 0)
            if diversity_score < 0.4:
                recommendations.append({
                    'type': 'diversity',
                    'priority': 'medium',
                    'message': 'Low intra-list diversity. Consider adjusting generation sampling parameters.'
                })
        
        # Check quality
        if 'quality' in report['summary_metrics']:
            compatibility_score = report['summary_metrics']['quality'].get('avg_compatibility', {}).get('mean', 0)
            if compatibility_score < 0.6:
                recommendations.append({
                    'type': 'quality',
                    'priority': 'high',
                    'message': 'Low compatibility quality. Consider retraining the model or improving training data quality.'
                })
        
        report['recommendations'] = recommendations
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report
    
    def create_evaluation_dashboard(self, save_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Create visualization dashboard for evaluation metrics
        
        Args:
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of saved plot paths
        """
        if not self.evaluation_history:
            return {'error': 'No evaluation data available'}
        
        if save_dir is None:
            save_dir = self.config.plot_save_dir
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        
        # Extract time series data
        timestamps = [datetime.fromisoformat(eval_result['timestamp']) for eval_result in self.evaluation_history]
        
        # Plot novelty trends
        novelty_scores = [eval_result.get('novelty', {}).get('novelty_score', 0) for eval_result in self.evaluation_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, novelty_scores, marker='o')
        plt.title('Novelty Score Over Time')
        plt.xlabel('Time')
        plt.ylabel('Novelty Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        novelty_path = Path(save_dir) / 'novelty_trends.png'
        plt.savefig(novelty_path)
        plt.close()
        plot_paths['novelty_trends'] = str(novelty_path)
        
        # Plot diversity trends
        diversity_scores = [eval_result.get('diversity', {}).get('intra_list_diversity', 0) for eval_result in self.evaluation_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, diversity_scores, marker='s', color='green')
        plt.title('Diversity Score Over Time')
        plt.xlabel('Time')
        plt.ylabel('Diversity Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        diversity_path = Path(save_dir) / 'diversity_trends.png'
        plt.savefig(diversity_path)
        plt.close()
        plot_paths['diversity_trends'] = str(diversity_path)
        
        # Plot quality trends
        quality_scores = [eval_result.get('quality', {}).get('avg_compatibility', 0) for eval_result in self.evaluation_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, quality_scores, marker='^', color='red')
        plt.title('Quality Score Over Time')
        plt.xlabel('Time')
        plt.ylabel('Compatibility Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        quality_path = Path(save_dir) / 'quality_trends.png'
        plt.savefig(quality_path)
        plt.close()
        plot_paths['quality_trends'] = str(quality_path)
        
        # Combined metrics plot
        plt.figure(figsize=(15, 8))
        plt.plot(timestamps, novelty_scores, marker='o', label='Novelty', alpha=0.7)
        plt.plot(timestamps, diversity_scores, marker='s', label='Diversity', alpha=0.7)
        plt.plot(timestamps, quality_scores, marker='^', label='Quality', alpha=0.7)
        
        plt.title('Generative AI Evaluation Metrics Over Time')
        plt.xlabel('Time')
        plt.ylabel('Score')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        combined_path = Path(save_dir) / 'combined_metrics.png'
        plt.savefig(combined_path)
        plt.close()
        plot_paths['combined_metrics'] = str(combined_path)
        
        logger.info(f"Evaluation dashboard created in {save_dir}")
        
        return plot_paths

def create_sample_evaluator() -> GenerativeEvaluator:
    """Create sample evaluator for testing"""
    config = EvaluationConfig()
    return GenerativeEvaluator(config)

if __name__ == "__main__":
    # Create sample evaluator
    evaluator = create_sample_evaluator()
    
    logger.info("Generative Evaluation Metrics ready for FlashFit AI")
    logger.info("Key features:")
    logger.info("- Novelty measurement with temporal analysis")
    logger.info("- Multi-dimensional diversity metrics")
    logger.info("- Quality assessment and ranking evaluation")
    logger.info("- Confidence calibration analysis")
    logger.info("- A/B testing utilities")
    logger.info("- Comprehensive reporting and visualization")
    
    # Test with synthetic data
    query_embeddings = np.random.randn(10, 512)
    generated_embeddings = np.random.randn(10, 512)
    catalog_embeddings = np.random.randn(1000, 512)
    
    generated_items = [
        {
            'category': np.random.choice(['shirt', 'pants', 'dress', 'jacket']),
            'color': np.random.choice(['red', 'blue', 'green', 'black', 'white']),
            'style': np.random.choice(['casual', 'formal', 'trendy', 'classic'])
        }
        for _ in range(10)
    ]
    
    # Run evaluation
    results = evaluator.evaluate_generation_batch(
        query_embeddings,
        generated_embeddings,
        catalog_embeddings,
        generated_items
    )
    
    logger.info(f"Sample evaluation completed: {len(results)} metrics computed")
    logger.info(f"Novelty score: {results['novelty']['novelty_score']:.3f}")
    logger.info(f"Diversity score: {results['diversity']['intra_list_diversity']:.3f}")
    logger.info(f"Quality score: {results['quality']['avg_compatibility']:.3f}")