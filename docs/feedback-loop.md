# FlashFit AI Feedback Loop Documentation

## Overview

The FlashFit AI feedback loop is a real-time learning system that continuously adapts recommendations based on user interactions. This document details how user feedback flows through the system to improve future recommendations.

## Feedback Loop Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Action   │───▶│  Feedback API   │───▶│ Feedback Store  │
│                 │    │                 │    │                 │
│ • Like/Dislike  │    │ POST /feedback  │    │   feedback.json │
│ • Save/Purchase │    │                 │    │                 │
│ • View Time     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Updated Weights │◀───│ Weight Adjuster │◀───│ Feedback Parser │
│                 │    │                 │    │                 │
│ CLIP: 0.42      │    │ • Score Analysis│    │ • Type Analysis │
│ BLIP: 0.28      │    │ • Weight Calc   │    │ • Context Eval  │
│ Fashion: 0.30   │    │ • Adaptation    │    │ • Pattern Detect│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Fusion Reranker │───▶│ Vector Updates  │───▶│ Next Recommend  │
│                 │    │                 │    │                 │
│ • New Weights   │    │ • Index Refresh │    │ • Improved      │
│ • Personalized  │    │ • Embedding Adj │    │   Accuracy      │
│ • Adaptive      │    │ • Similarity    │    │ • Personalized  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Feedback Types and Processing

### 1. Explicit Feedback

#### Like/Dislike Actions
```json
{
  "feedback_type": "like",
  "rating": 4,
  "impact": {
    "weight_adjustment": {
      "clip": +0.02,
      "blip": +0.01,
      "fashion": +0.03
    },
    "confidence_boost": 0.15
  }
}
```

**Processing Logic:**
- **Positive Feedback (Like/4-5 stars)**: Increases weights for models that contributed most to the recommendation
- **Negative Feedback (Dislike/1-2 stars)**: Decreases weights and adjusts similarity thresholds
- **Neutral Feedback (3 stars)**: Minimal weight adjustment, used for calibration

#### Save/Purchase Actions
```json
{
  "feedback_type": "purchase",
  "rating": 5,
  "impact": {
    "weight_adjustment": {
      "clip": +0.05,
      "blip": +0.03,
      "fashion": +0.07
    },
    "embedding_boost": 0.25,
    "similarity_threshold": -0.02
  }
}
```

**Processing Logic:**
- **High-Value Actions**: Stronger weight adjustments and embedding modifications
- **Purchase Intent**: Significant boost to fashion-specific weights
- **Save Actions**: Moderate adjustments with emphasis on style compatibility

### 2. Implicit Feedback

#### View Time Analysis
```json
{
  "feedback_type": "view",
  "interaction_time_ms": 3500,
  "impact": {
    "engagement_score": 0.7,
    "weight_adjustment": {
      "clip": +0.01,
      "blip": +0.005,
      "fashion": +0.01
    }
  }
}
```

**Processing Logic:**
- **Short Views (<1s)**: Negative signal, slight weight decrease
- **Medium Views (1-5s)**: Neutral to slightly positive
- **Long Views (>5s)**: Strong positive signal, weight increase

#### Click-Through Patterns
```json
{
  "feedback_type": "click_pattern",
  "position": 2,
  "total_clicks": 3,
  "impact": {
    "position_bias_adjustment": -0.1,
    "diversity_preference": 0.15
  }
}
```

## Weight Adjustment Algorithm

### Base Weight Calculation
```python
def adjust_weights(current_weights, feedback_data):
    """
    Adjust fusion weights based on user feedback
    """
    adjustment_factor = calculate_adjustment_factor(feedback_data)
    
    # Determine which models contributed most to the recommendation
    contributing_models = analyze_contribution(feedback_data['scores'])
    
    new_weights = {}
    for model in ['clip', 'blip', 'fashion']:
        if feedback_data['rating'] >= 4:  # Positive feedback
            if model in contributing_models:
                new_weights[model] = current_weights[model] + adjustment_factor
            else:
                new_weights[model] = current_weights[model] + (adjustment_factor * 0.3)
        else:  # Negative feedback
            if model in contributing_models:
                new_weights[model] = current_weights[model] - adjustment_factor
            else:
                new_weights[model] = current_weights[model] - (adjustment_factor * 0.1)
    
    # Normalize weights to sum to 1.0
    return normalize_weights(new_weights)
```

### Adjustment Factors

| Feedback Type | Base Factor | Multiplier | Max Change |
|---------------|-------------|------------|------------|
| Like | 0.02 | 1.0 | 0.05 |
| Dislike | 0.02 | -1.0 | -0.05 |
| Save | 0.03 | 1.5 | 0.08 |
| Purchase | 0.05 | 2.0 | 0.10 |
| Long View | 0.01 | 0.5 | 0.02 |
| Quick Exit | 0.01 | -0.5 | -0.02 |

## Personalization Mechanisms

### 1. User-Specific Weight Profiles

```json
{
  "user_id": "user123",
  "personalized_weights": {
    "clip": 0.45,
    "blip": 0.25,
    "fashion": 0.30
  },
  "confidence_level": 0.78,
  "feedback_count": 47,
  "last_updated": "2024-01-15T10:30:00Z"
}
```

### 2. Style Preference Learning

```json
{
  "user_id": "user123",
  "learned_preferences": {
    "style_vectors": {
      "casual": [0.8, 0.2, 0.6, ...],
      "formal": [0.3, 0.9, 0.4, ...],
      "sporty": [0.7, 0.1, 0.8, ...]
    },
    "color_preferences": {
      "preferred": ["blue", "black", "white"],
      "avoided": ["bright_yellow", "neon_green"]
    },
    "brand_affinity": {
      "nike": 0.9,
      "adidas": 0.7,
      "zara": 0.8
    }
  }
}
```

## Feedback Processing Pipeline

### Step 1: Feedback Ingestion
```python
def process_feedback(feedback_data):
    # Validate feedback data
    if not validate_feedback(feedback_data):
        raise ValueError("Invalid feedback data")
    
    # Store raw feedback
    store_feedback(feedback_data)
    
    # Extract features
    features = extract_feedback_features(feedback_data)
    
    return features
```

### Step 2: Impact Analysis
```python
def analyze_feedback_impact(features, user_profile):
    # Calculate model contribution scores
    contribution_scores = calculate_model_contributions(
        features['recommendation_scores']
    )
    
    # Determine adjustment magnitude
    adjustment_magnitude = calculate_adjustment_magnitude(
        features['feedback_type'],
        features['rating'],
        user_profile['confidence_level']
    )
    
    return {
        'contributions': contribution_scores,
        'magnitude': adjustment_magnitude
    }
```

### Step 3: Weight Updates
```python
def update_fusion_weights(user_id, impact_analysis):
    current_weights = get_user_weights(user_id)
    
    # Apply adjustments
    new_weights = apply_weight_adjustments(
        current_weights,
        impact_analysis['contributions'],
        impact_analysis['magnitude']
    )
    
    # Apply constraints (min/max bounds)
    constrained_weights = apply_weight_constraints(new_weights)
    
    # Store updated weights
    store_user_weights(user_id, constrained_weights)
    
    return constrained_weights
```

### Step 4: Vector Store Updates
```python
def update_vector_embeddings(user_id, feedback_data):
    # Get item embeddings
    item_embeddings = get_item_embeddings(feedback_data['item_id'])
    
    # Calculate embedding adjustments
    if feedback_data['rating'] >= 4:
        # Boost similar items
        boost_similar_embeddings(user_id, item_embeddings, factor=0.1)
    else:
        # Reduce similarity to disliked items
        reduce_similar_embeddings(user_id, item_embeddings, factor=0.1)
    
    # Update user preference vectors
    update_user_preference_vectors(user_id, item_embeddings, feedback_data)
```

## Real-Time Learning Metrics

### Performance Tracking
```json
{
  "learning_metrics": {
    "feedback_volume": {
      "daily_average": 156,
      "weekly_trend": "+12%",
      "feedback_types": {
        "likes": 45,
        "saves": 23,
        "purchases": 8,
        "dislikes": 12
      }
    },
    "adaptation_speed": {
      "weight_convergence_time": "3.2 days",
      "recommendation_improvement": "+18% accuracy",
      "user_satisfaction_delta": "+0.23"
    },
    "personalization_depth": {
      "users_with_profiles": 1247,
      "average_feedback_per_user": 23.4,
      "personalization_confidence": 0.76
    }
  }
}
```

### A/B Testing Integration
```json
{
  "ab_test_config": {
    "test_name": "feedback_sensitivity_v2",
    "variants": {
      "control": {
        "adjustment_factor": 0.02,
        "users": 500
      },
      "treatment": {
        "adjustment_factor": 0.05,
        "users": 500
      }
    },
    "metrics": {
      "recommendation_accuracy": {
        "control": 0.73,
        "treatment": 0.78
      },
      "user_engagement": {
        "control": 0.65,
        "treatment": 0.71
      }
    }
  }
}
```

## Cold Start Handling

### New User Bootstrap
```python
def initialize_new_user(user_id, initial_preferences=None):
    # Start with default weights
    default_weights = {
        'clip': 0.40,
        'blip': 0.30,
        'fashion': 0.30
    }
    
    # Apply preference-based adjustments
    if initial_preferences:
        adjusted_weights = adjust_weights_for_preferences(
            default_weights, 
            initial_preferences
        )
    else:
        adjusted_weights = default_weights
    
    # Store initial profile
    store_user_weights(user_id, adjusted_weights)
    
    return adjusted_weights
```

### Rapid Learning Phase
```python
def rapid_learning_adjustment(user_id, feedback_count):
    # Increase learning rate for new users
    if feedback_count < 10:
        learning_rate = 2.0  # Double the adjustment factor
    elif feedback_count < 25:
        learning_rate = 1.5  # 50% increase
    else:
        learning_rate = 1.0  # Normal rate
    
    return learning_rate
```

## Privacy and Data Handling

### Data Retention Policy
- **Raw Feedback**: 90 days
- **Aggregated Metrics**: 1 year
- **User Profiles**: Until account deletion
- **Anonymous Analytics**: Indefinite

### Privacy Protection
```python
def anonymize_feedback(feedback_data):
    # Remove personally identifiable information
    anonymized = {
        'user_hash': hash_user_id(feedback_data['user_id']),
        'item_category': feedback_data['item_category'],
        'feedback_type': feedback_data['feedback_type'],
        'rating': feedback_data['rating'],
        'timestamp': feedback_data['timestamp']
    }
    
    return anonymized
```

## Future Enhancements

### Phase 2: Advanced Learning
- **Multi-Armed Bandit**: Dynamic exploration vs exploitation
- **Contextual Learning**: Time, weather, occasion-aware adjustments
- **Social Learning**: Incorporate similar user preferences

### Phase 3: Meta-Learning
- **Learning to Learn**: Optimize learning rates per user
- **Transfer Learning**: Apply insights across user segments
- **Ensemble Meta-Models**: Multiple learning strategies

## Monitoring and Debugging

### Feedback Loop Health Checks
```bash
# Check feedback processing rate
curl -X GET "http://localhost:8080/api/fusion/stats" | jq '.stats.feedback_processing_rate'

# Monitor weight drift
curl -X GET "http://localhost:8080/api/fusion/stats" | jq '.stats.weight_stability'

# Validate learning effectiveness
curl -X GET "http://localhost:8080/api/fusion/stats" | jq '.stats.recommendation_improvement'
```

### Debug Tools
```python
# Trace feedback impact for specific user
def trace_feedback_impact(user_id, feedback_id):
    feedback = get_feedback(feedback_id)
    before_weights = get_historical_weights(user_id, feedback['timestamp'])
    after_weights = get_user_weights(user_id)
    
    return {
        'feedback': feedback,
        'weight_changes': calculate_weight_diff(before_weights, after_weights),
        'impact_score': calculate_impact_score(feedback)
    }
```

This feedback loop system ensures that FlashFit AI continuously improves its recommendations while maintaining user privacy and system stability.