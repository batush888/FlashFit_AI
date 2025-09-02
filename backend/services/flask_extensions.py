from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import json
import ujson
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import base64
import io
import zipfile
from dataclasses import dataclass, asdict
from marshmallow import Schema, fields, ValidationError
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import tiktoken
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import warnings
warnings.filterwarnings('ignore')

# Initialize console for logging
console = Console()

# Schemas for request validation
class AnalyticsRequestSchema(Schema):
    """Schema for analytics requests"""
    data_type = fields.Str(required=True, validate=lambda x: x in ['user', 'fashion', 'recommendation'])
    start_date = fields.DateTime(allow_none=True)
    end_date = fields.DateTime(allow_none=True)
    metrics = fields.List(fields.Str(), missing=['accuracy', 'precision', 'recall'])

class RecommendationRequestSchema(Schema):
    """Schema for recommendation requests"""
    user_id = fields.Str(required=True)
    item_features = fields.Dict(missing={})
    num_recommendations = fields.Int(missing=10, validate=lambda x: 1 <= x <= 100)
    algorithm = fields.Str(missing='collaborative', validate=lambda x: x in ['collaborative', 'content', 'hybrid'])

class ModelTrainingRequestSchema(Schema):
    """Schema for model training requests"""
    model_type = fields.Str(required=True, validate=lambda x: x in ['classification', 'clustering', 'recommendation'])
    data_source = fields.Str(required=True)
    parameters = fields.Dict(missing={})
    validation_split = fields.Float(missing=0.2, validate=lambda x: 0.1 <= x <= 0.5)

@dataclass
class APIResponse:
    """Standard API response format"""
    success: bool
    data: Any = None
    message: str = ""
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        return asdict(self)

class FlashFitFlaskExtensions:
    """
    Flask-based API extensions for FlashFit AI
    
    Provides additional endpoints for:
    - Advanced analytics and reporting
    - Data visualization
    - Model training and evaluation
    - System monitoring
    - Batch processing
    - Export/import functionality
    """
    
    def __init__(self, app: Optional[Flask] = None, data_dir: str = "data"):
        self.app = app
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize schemas
        self.analytics_schema = AnalyticsRequestSchema()
        self.recommendation_schema = RecommendationRequestSchema()
        self.model_training_schema = ModelTrainingRequestSchema()
        
        # Initialize ML models and tools
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Initialize tokenizer for text analysis
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
        
        # Initialize sentence transformer for embeddings
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            self.sentence_model = None
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_ttl = timedelta(minutes=30)
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize Flask app with extensions"""
        self.app = app
        
        # Enable CORS
        CORS(app, resources={
            r"/api/v2/*": {
                "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        })
        
        # Register routes
        self.register_routes()
        
        console.print("[green]✓[/green] Flask extensions initialized")
    
    def register_routes(self):
        """Register all API routes"""
        app = self.app
        
        @app.route('/api/v2/health', methods=['GET'])
        def health_check():
            """Enhanced health check with system metrics"""
            try:
                system_info = {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent,
                    'gpu_available': torch.cuda.is_available(),
                    'timestamp': datetime.now().isoformat()
                }
                
                if torch.cuda.is_available():
                    system_info['gpu_memory_used'] = torch.cuda.memory_allocated() / 1024**3
                    system_info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                return jsonify(APIResponse(
                    success=True,
                    data=system_info,
                    message="System healthy"
                ).to_dict())
            except Exception as e:
                return jsonify(APIResponse(
                    success=False,
                    message=f"Health check failed: {str(e)}"
                ).to_dict()), 500
        
        @app.route('/api/v2/analytics/generate', methods=['POST'])
        def generate_analytics():
            """Generate comprehensive analytics report"""
            try:
                # Validate request
                data = self.analytics_schema.load(request.json or {})
                
                # Generate analytics based on data type
                analytics_data = self._generate_analytics_data(data)
                
                return jsonify(APIResponse(
                    success=True,
                    data=analytics_data,
                    message="Analytics generated successfully"
                ).to_dict())
            
            except ValidationError as e:
                return jsonify(APIResponse(
                    success=False,
                    message=f"Validation error: {e.messages}"
                ).to_dict()), 400
            except Exception as e:
                return jsonify(APIResponse(
                    success=False,
                    message=f"Analytics generation failed: {str(e)}"
                ).to_dict()), 500
        
        @app.route('/api/v2/visualizations/create', methods=['POST'])
        def create_visualization():
            """Create data visualizations"""
            try:
                data = request.json or {}
                viz_type = data.get('type', 'line')
                chart_data = data.get('data', [])
                
                # Create visualization
                image_data = self._create_visualization(viz_type, chart_data, data.get('options', {}))
                
                return jsonify(APIResponse(
                    success=True,
                    data={'image': image_data, 'type': viz_type},
                    message="Visualization created successfully"
                ).to_dict())
            
            except Exception as e:
                return jsonify(APIResponse(
                    success=False,
                    message=f"Visualization creation failed: {str(e)}"
                ).to_dict()), 500
        
        @app.route('/api/v2/models/train', methods=['POST'])
        def train_model():
            """Train ML models with provided data"""
            try:
                # Validate request
                data = self.model_training_schema.load(request.json or {})
                
                # Train model
                training_results = self._train_model(data)
                
                return jsonify(APIResponse(
                    success=True,
                    data=training_results,
                    message="Model training completed"
                ).to_dict())
            
            except ValidationError as e:
                return jsonify(APIResponse(
                    success=False,
                    message=f"Validation error: {e.messages}"
                ).to_dict()), 400
            except Exception as e:
                return jsonify(APIResponse(
                    success=False,
                    message=f"Model training failed: {str(e)}"
                ).to_dict()), 500
        
        @app.route('/api/v2/recommendations/advanced', methods=['POST'])
        def advanced_recommendations():
            """Generate advanced recommendations using multiple algorithms"""
            try:
                # Validate request
                data = self.recommendation_schema.load(request.json or {})
                
                # Generate recommendations
                recommendations = self._generate_advanced_recommendations(data)
                
                return jsonify(APIResponse(
                    success=True,
                    data=recommendations,
                    message="Advanced recommendations generated"
                ).to_dict())
            
            except ValidationError as e:
                return jsonify(APIResponse(
                    success=False,
                    message=f"Validation error: {e.messages}"
                ).to_dict()), 400
            except Exception as e:
                return jsonify(APIResponse(
                    success=False,
                    message=f"Recommendation generation failed: {str(e)}"
                ).to_dict()), 500
        
        @app.route('/api/v2/text/analyze', methods=['POST'])
        def analyze_text():
            """Advanced text analysis using NLP tools"""
            try:
                data = request.json or {}
                text = data.get('text', '')
                
                if not text:
                    return jsonify(APIResponse(
                        success=False,
                        message="Text is required"
                    ).to_dict()), 400
                
                # Perform text analysis
                analysis_results = self._analyze_text(text, data.get('options', {}))
                
                return jsonify(APIResponse(
                    success=True,
                    data=analysis_results,
                    message="Text analysis completed"
                ).to_dict())
            
            except Exception as e:
                return jsonify(APIResponse(
                    success=False,
                    message=f"Text analysis failed: {str(e)}"
                ).to_dict()), 500
        
        @app.route('/api/v2/data/export', methods=['POST'])
        def export_data():
            """Export data in various formats"""
            try:
                data = request.json or {}
                export_type = data.get('type', 'json')
                data_source = data.get('source', 'all')
                
                # Export data
                exported_data = self._export_data(data_source, export_type, data.get('options', {}))
                
                if export_type in ['csv', 'xlsx']:
                    # Return file for download
                    return send_file(
                        exported_data,
                        as_attachment=True,
                        download_name=f"flashfit_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_type}"
                    )
                else:
                    return jsonify(APIResponse(
                        success=True,
                        data=exported_data,
                        message="Data exported successfully"
                    ).to_dict())
            
            except Exception as e:
                return jsonify(APIResponse(
                    success=False,
                    message=f"Data export failed: {str(e)}"
                ).to_dict()), 500
        
        @app.route('/api/v2/batch/process', methods=['POST'])
        def batch_process():
            """Process batch operations"""
            try:
                data = request.json or {}
                operation = data.get('operation', '')
                batch_data = data.get('data', [])
                
                # Process batch operation
                results = self._process_batch_operation(operation, batch_data, data.get('options', {}))
                
                return jsonify(APIResponse(
                    success=True,
                    data=results,
                    message=f"Batch {operation} completed"
                ).to_dict())
            
            except Exception as e:
                return jsonify(APIResponse(
                    success=False,
                    message=f"Batch processing failed: {str(e)}"
                ).to_dict()), 500
    
    def _generate_analytics_data(self, request_data: Dict) -> Dict:
        """Generate analytics data based on request"""
        data_type = request_data['data_type']
        
        # Mock analytics data - in real implementation, this would query actual data
        analytics = {
            'summary': {
                'total_records': np.random.randint(1000, 10000),
                'processed_today': np.random.randint(50, 500),
                'accuracy_score': np.random.uniform(0.85, 0.98),
                'performance_trend': 'improving'
            },
            'metrics': {},
            'trends': [],
            'insights': []
        }
        
        if data_type == 'user':
            analytics['metrics'] = {
                'active_users': np.random.randint(100, 1000),
                'new_registrations': np.random.randint(10, 100),
                'engagement_rate': np.random.uniform(0.6, 0.9),
                'retention_rate': np.random.uniform(0.7, 0.95)
            }
            analytics['insights'] = [
                "User engagement increased by 15% this week",
                "Mobile users show higher retention rates",
                "Peak usage hours: 7-9 PM"
            ]
        
        elif data_type == 'fashion':
            analytics['metrics'] = {
                'total_items': np.random.randint(5000, 50000),
                'categories_covered': np.random.randint(20, 100),
                'avg_rating': np.random.uniform(4.0, 4.8),
                'popular_styles': ['casual', 'formal', 'sporty']
            }
            analytics['insights'] = [
                "Casual wear dominates user preferences",
                "Sustainable fashion trending upward",
                "Color preferences vary by season"
            ]
        
        elif data_type == 'recommendation':
            analytics['metrics'] = {
                'recommendations_served': np.random.randint(10000, 100000),
                'click_through_rate': np.random.uniform(0.15, 0.35),
                'conversion_rate': np.random.uniform(0.05, 0.15),
                'avg_response_time': np.random.uniform(50, 200)
            }
            analytics['insights'] = [
                "Personalized recommendations show 25% higher CTR",
                "Image-based recommendations outperform text-based",
                "Users prefer 5-10 recommendations per session"
            ]
        
        return analytics
    
    def _create_visualization(self, viz_type: str, data: List, options: Dict) -> str:
        """Create data visualization and return as base64 encoded image"""
        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8')
        
        if not data:
            # Generate sample data for demonstration
            x = np.linspace(0, 10, 100)
            y = np.sin(x) + np.random.normal(0, 0.1, 100)
            data = [{'x': x[i], 'y': y[i]} for i in range(len(x))]
        
        if viz_type == 'line':
            x_vals = [item.get('x', i) for i, item in enumerate(data)]
            y_vals = [item.get('y', 0) for item in data]
            plt.plot(x_vals, y_vals, linewidth=2, color='#2E86AB')
            plt.title('Performance Trend', fontsize=14, fontweight='bold')
        
        elif viz_type == 'bar':
            categories = [item.get('category', f'Cat {i}') for i, item in enumerate(data[:10])]
            values = [item.get('value', np.random.randint(10, 100)) for item in data[:10]]
            plt.bar(categories, values, color='#A23B72')
            plt.title('Category Distribution', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
        
        elif viz_type == 'scatter':
            x_vals = [item.get('x', np.random.uniform(0, 10)) for item in data[:100]]
            y_vals = [item.get('y', np.random.uniform(0, 10)) for item in data[:100]]
            plt.scatter(x_vals, y_vals, alpha=0.6, color='#F18F01')
            plt.title('Data Correlation', fontsize=14, fontweight='bold')
        
        elif viz_type == 'heatmap':
            # Create sample correlation matrix
            matrix_data = np.random.rand(5, 5)
            sns.heatmap(matrix_data, annot=True, cmap='viridis', square=True)
            plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _train_model(self, request_data: Dict) -> Dict:
        """Train ML model with provided parameters"""
        model_type = request_data['model_type']
        
        # Generate sample training data
        n_samples = 1000
        n_features = 10
        
        if model_type == 'classification':
            X = np.random.randn(n_samples, n_features)
            y = np.random.randint(0, 3, n_samples)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            results = {
                'model_type': 'RandomForestClassifier',
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'feature_importance': model.feature_importances_.tolist(),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
        
        elif model_type == 'clustering':
            X = np.random.randn(n_samples, n_features)
            
            # Train K-Means
            n_clusters = request_data.get('parameters', {}).get('n_clusters', 3)
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X)
            
            results = {
                'model_type': 'KMeans',
                'n_clusters': n_clusters,
                'inertia': model.inertia_,
                'cluster_centers': model.cluster_centers_.tolist(),
                'samples_per_cluster': np.bincount(labels).tolist(),
                'total_samples': len(X)
            }
        
        else:
            results = {
                'model_type': 'recommendation',
                'message': 'Recommendation model training not implemented in demo'
            }
        
        return results
    
    def _generate_advanced_recommendations(self, request_data: Dict) -> Dict:
        """Generate advanced recommendations using multiple algorithms"""
        user_id = request_data['user_id']
        num_recommendations = request_data['num_recommendations']
        algorithm = request_data['algorithm']
        
        # Mock recommendation generation
        recommendations = []
        
        for i in range(num_recommendations):
            recommendations.append({
                'item_id': f'item_{np.random.randint(1000, 9999)}',
                'title': f'Fashion Item {i+1}',
                'category': np.random.choice(['tops', 'bottoms', 'dresses', 'shoes', 'accessories']),
                'confidence_score': np.random.uniform(0.7, 0.99),
                'price_range': np.random.choice(['$', '$$', '$$$']),
                'style_tags': np.random.choice([['casual', 'comfortable'], ['formal', 'elegant'], ['trendy', 'modern']], size=1)[0],
                'similarity_score': np.random.uniform(0.8, 0.95)
            })
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return {
            'user_id': user_id,
            'algorithm_used': algorithm,
            'recommendations': recommendations,
            'metadata': {
                'generation_time_ms': np.random.uniform(50, 200),
                'total_candidates': np.random.randint(1000, 5000),
                'filtering_applied': ['availability', 'user_preferences', 'quality_score']
            }
        }
    
    def _analyze_text(self, text: str, options: Dict) -> Dict:
        """Perform advanced text analysis"""
        results = {
            'basic_stats': {
                'character_count': len(text),
                'word_count': len(text.split()),
                'sentence_count': text.count('.') + text.count('!') + text.count('?'),
                'paragraph_count': text.count('\n\n') + 1
            },
            'language_detection': 'en',  # Simplified
            'sentiment': {
                'polarity': np.random.uniform(-1, 1),
                'subjectivity': np.random.uniform(0, 1)
            },
            'keywords': [],
            'entities': [],
            'embeddings': None
        }
        
        # Token counting if available
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text)
                results['token_count'] = len(tokens)
                results['estimated_cost'] = len(tokens) * 0.0001  # Mock cost calculation
            except Exception:
                pass
        
        # Generate embeddings if sentence transformer is available
        if self.sentence_model:
            try:
                embeddings = self.sentence_model.encode([text])
                results['embeddings'] = embeddings[0].tolist()[:10]  # First 10 dimensions for demo
                results['embedding_dimension'] = len(embeddings[0])
            except Exception:
                pass
        
        # Extract keywords (simplified)
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Filter short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        results['keywords'] = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return results
    
    def _export_data(self, source: str, export_type: str, options: Dict) -> Union[str, io.BytesIO]:
        """Export data in specified format"""
        # Generate sample data
        sample_data = {
            'users': pd.DataFrame({
                'user_id': [f'user_{i}' for i in range(100)],
                'age': np.random.randint(18, 65, 100),
                'gender': np.random.choice(['M', 'F', 'Other'], 100),
                'signup_date': pd.date_range('2023-01-01', periods=100, freq='D'),
                'active': np.random.choice([True, False], 100, p=[0.8, 0.2])
            }),
            'items': pd.DataFrame({
                'item_id': [f'item_{i}' for i in range(200)],
                'category': np.random.choice(['tops', 'bottoms', 'dresses', 'shoes'], 200),
                'price': np.random.uniform(20, 200, 200),
                'rating': np.random.uniform(3.0, 5.0, 200),
                'in_stock': np.random.choice([True, False], 200, p=[0.9, 0.1])
            })
        }
        
        if source == 'all':
            data = pd.concat(sample_data.values(), keys=sample_data.keys())
        else:
            data = sample_data.get(source, pd.DataFrame())
        
        if export_type == 'json':
            return data.to_json(orient='records', date_format='iso')
        
        elif export_type == 'csv':
            buffer = io.StringIO()
            data.to_csv(buffer, index=False)
            return buffer.getvalue()
        
        elif export_type == 'xlsx':
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                if source == 'all':
                    for name, df in sample_data.items():
                        df.to_excel(writer, sheet_name=name, index=False)
                else:
                    data.to_excel(writer, sheet_name=source, index=False)
            buffer.seek(0)
            return buffer
        
        return data.to_dict('records')
    
    def _process_batch_operation(self, operation: str, batch_data: List, options: Dict) -> Dict:
        """Process batch operations"""
        results = {
            'operation': operation,
            'total_items': len(batch_data),
            'processed': 0,
            'failed': 0,
            'results': [],
            'errors': []
        }
        
        for i, item in enumerate(batch_data):
            try:
                if operation == 'validate':
                    # Mock validation
                    is_valid = np.random.choice([True, False], p=[0.9, 0.1])
                    results['results'].append({
                        'item_id': item.get('id', i),
                        'valid': is_valid,
                        'score': np.random.uniform(0.7, 1.0) if is_valid else np.random.uniform(0.0, 0.6)
                    })
                    
                elif operation == 'classify':
                    # Mock classification
                    category = np.random.choice(['A', 'B', 'C'])
                    confidence = np.random.uniform(0.8, 0.99)
                    results['results'].append({
                        'item_id': item.get('id', i),
                        'predicted_category': category,
                        'confidence': confidence
                    })
                
                elif operation == 'embed':
                    # Mock embedding generation
                    embedding = np.random.randn(128).tolist()  # 128-dim embedding
                    results['results'].append({
                        'item_id': item.get('id', i),
                        'embedding': embedding[:5],  # First 5 dims for demo
                        'dimension': 128
                    })
                
                results['processed'] += 1
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'item_id': item.get('id', i),
                    'error': str(e)
                })
        
        results['success_rate'] = results['processed'] / results['total_items'] if results['total_items'] > 0 else 0
        
        return results

# Factory function to create Flask app with extensions
def create_flask_app_with_extensions(config: Optional[Dict] = None) -> Flask:
    """Create Flask app with FlashFit extensions"""
    app = Flask(__name__)
    
    if config:
        app.config.update(config)
    
    # Initialize extensions
    extensions = FlashFitFlaskExtensions()
    extensions.init_app(app)
    
    return app

if __name__ == "__main__":
    # Demo usage
    app = create_flask_app_with_extensions({
        'DEBUG': True,
        'TESTING': False
    })
    
    console.print("[bold blue]Starting FlashFit Flask Extensions Demo[/bold blue]")
    console.print("Available endpoints:")
    console.print("• GET  /api/v2/health - System health check")
    console.print("• POST /api/v2/analytics/generate - Generate analytics")
    console.print("• POST /api/v2/visualizations/create - Create visualizations")
    console.print("• POST /api/v2/models/train - Train ML models")
    console.print("• POST /api/v2/recommendations/advanced - Advanced recommendations")
    console.print("• POST /api/v2/text/analyze - Text analysis")
    console.print("• POST /api/v2/data/export - Export data")
    console.print("• POST /api/v2/batch/process - Batch processing")
    
    app.run(host='0.0.0.0', port=5001, debug=True)