# Day 4.1: Content-Based Recommendation Fundamentals

## Learning Objectives
By the end of this session, you will:
- Understand the principles and advantages of content-based recommendation systems
- Master the mathematical foundations of content-based filtering
- Learn feature extraction and representation techniques for different content types
- Implement content-based recommendation algorithms from scratch
- Handle various data types including text, numerical, and categorical features

## 1. Content-Based Recommendation Systems Overview

### Definition
Content-based recommendation systems suggest items to users based on the features/characteristics of items and user preferences for those features. The core principle: **"Recommend items similar to those the user has liked in the past."**

### Key Components
1. **Item Profiles**: Feature representations of items
2. **User Profiles**: Models of user preferences based on item features
3. **Similarity Computation**: Methods to match user preferences with item features
4. **Recommendation Generation**: Ranking and filtering mechanisms

### Mathematical Foundation
Given:
- I = set of items with feature vectors f_i
- U = user with preference profile p_u
- Similarity function sim(p_u, f_i)

Recommendation score:
```
score(u, i) = sim(p_u, f_i)
```

## 2. Advantages and Disadvantages

### Advantages
1. **No Cold Start for Items**: Can recommend new items immediately if features are available
2. **Transparency**: Recommendations can be easily explained
3. **User Independence**: No need for other users' data
4. **Domain Knowledge**: Can incorporate expert knowledge through feature engineering
5. **Overspecialization Control**: Can be designed to introduce diversity

### Disadvantages
1. **Feature Engineering Dependence**: Quality depends heavily on feature selection
2. **Limited Discovery**: Tends to recommend similar items (overspecialization)
3. **New User Cold Start**: Still struggles with users who have no history
4. **Content Analysis Complexity**: Difficult for multimedia content
5. **Static Recommendations**: May not adapt to changing user preferences

## 3. Implementation: Comprehensive Content-Based Recommendation System

```python
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cosine, euclidean
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import time
import re
from collections import defaultdict, Counter
import pickle

class ContentBasedRecommender:
    """
    Comprehensive content-based recommendation system supporting multiple feature types
    and various similarity computation methods.
    """
    
    def __init__(self, 
                 similarity_metrics: List[str] = ['cosine'],
                 feature_weights: Dict[str, float] = None,
                 normalization_method: str = 'standard',
                 dimensionality_reduction: str = None,
                 n_components: int = 100):
        """
        Initialize Content-Based Recommender.
        
        Args:
            similarity_metrics: List of similarity metrics to use
            feature_weights: Weights for different feature categories
            normalization_method: Method for feature normalization
            dimensionality_reduction: Method for reducing feature dimensions
            n_components: Number of components for dimensionality reduction
        """
        self.similarity_metrics = similarity_metrics
        self.feature_weights = feature_weights or {}
        self.normalization_method = normalization_method
        self.dimensionality_reduction = dimensionality_reduction
        self.n_components = n_components
        
        # Feature processing components
        self.text_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.numerical_scaler = StandardScaler()
        self.categorical_encoders = {}
        self.dimensionality_reducer = None
        
        # Data structures
        self.item_features = {}
        self.item_feature_matrix = None
        self.user_profiles = {}
        self.feature_names = []
        self.item_ids = []
        
        # Metadata
        self.feature_statistics = {}
        self.user_interaction_history = defaultdict(list)
        
    def fit_item_features(self, items_data: List[Dict]):
        """
        Process and fit item features from raw item data.
        
        Args:
            items_data: List of dictionaries containing item information
                Format: [{'item_id': 'id1', 'title': 'Title', 'genre': 'Action', 'year': 2020, ...}]
        """
        print("Processing item features...")
        start_time = time.time()
        
        # Convert to DataFrame for easier processing
        items_df = pd.DataFrame(items_data)
        if 'item_id' not in items_df.columns:
            raise ValueError("items_data must contain 'item_id' field")
        
        items_df = items_df.set_index('item_id')
        self.item_ids = items_df.index.tolist()
        
        # Separate features by type
        text_features = {}
        numerical_features = {}
        categorical_features = {}
        
        for column in items_df.columns:
            if items_df[column].dtype == 'object':
                # Check if it's text or categorical
                sample_values = items_df[column].dropna().head(10)
                avg_length = np.mean([len(str(val).split()) for val in sample_values])
                
                if avg_length > 3:  # Likely text
                    text_features[column] = items_df[column].fillna('')
                else:  # Likely categorical
                    categorical_features[column] = items_df[column].fillna('unknown')
            else:
                # Numerical features
                numerical_features[column] = items_df[column].fillna(items_df[column].mean())
        
        # Process each feature type
        processed_features = []
        feature_names = []
        
        # Process text features
        for feature_name, feature_data in text_features.items():
            print(f"Processing text feature: {feature_name}")
            
            # Create separate vectorizer for each text feature
            vectorizer = TfidfVectorizer(
                max_features=min(1000, len(feature_data)),
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            tfidf_matrix = vectorizer.fit_transform(feature_data)
            processed_features.append(tfidf_matrix.toarray())
            
            # Store vectorizer for later use
            self.categorical_encoders[f'{feature_name}_tfidf'] = vectorizer
            
            # Add feature names
            vocab_names = [f"{feature_name}_tfidf_{word}" for word in vectorizer.get_feature_names_out()]
            feature_names.extend(vocab_names)
        
        # Process numerical features
        if numerical_features:
            print(f"Processing {len(numerical_features)} numerical features")
            numerical_df = pd.DataFrame(numerical_features)
            
            if self.normalization_method == 'standard':
                scaled_features = self.numerical_scaler.fit_transform(numerical_df)
            elif self.normalization_method == 'minmax':
                scaler = MinMaxScaler()
                scaled_features = scaler.fit_transform(numerical_df)
                self.numerical_scaler = scaler
            else:
                scaled_features = numerical_df.values
            
            processed_features.append(scaled_features)
            feature_names.extend([f"numerical_{col}" for col in numerical_df.columns])
        
        # Process categorical features
        for feature_name, feature_data in categorical_features.items():
            print(f"Processing categorical feature: {feature_name}")
            
            # Label encoding + one-hot encoding
            label_encoder = LabelEncoder()
            encoded = label_encoder.fit_transform(feature_data)
            
            # One-hot encoding
            n_classes = len(label_encoder.classes_)
            one_hot = np.eye(n_classes)[encoded]
            
            processed_features.append(one_hot)
            self.categorical_encoders[feature_name] = label_encoder
            
            # Add feature names
            class_names = [f"{feature_name}_{cls}" for cls in label_encoder.classes_]
            feature_names.extend(class_names)
        
        # Combine all features
        if processed_features:
            self.item_feature_matrix = np.hstack(processed_features)
            self.feature_names = feature_names
            
            print(f"Combined feature matrix shape: {self.item_feature_matrix.shape}")
        else:
            raise ValueError("No features could be processed from the data")
        
        # Apply feature weights if specified
        self._apply_feature_weights()
        
        # Apply dimensionality reduction if specified
        if self.dimensionality_reduction:
            self._apply_dimensionality_reduction()
        
        # Store individual item features
        for i, item_id in enumerate(self.item_ids):
            self.item_features[item_id] = self.item_feature_matrix[i]
        
        # Calculate feature statistics
        self._calculate_feature_statistics()
        
        processing_time = time.time() - start_time
        print(f"Feature processing completed in {processing_time:.2f} seconds")
    
    def _apply_feature_weights(self):
        """Apply weights to different feature categories."""
        if not self.feature_weights:
            return
        
        print("Applying feature weights...")
        
        for weight_key, weight_value in self.feature_weights.items():
            # Find feature indices matching the weight key
            matching_indices = [i for i, name in enumerate(self.feature_names) 
                              if weight_key in name]
            
            if matching_indices:
                self.item_feature_matrix[:, matching_indices] *= weight_value
                print(f"Applied weight {weight_value} to {len(matching_indices)} features matching '{weight_key}'")
    
    def _apply_dimensionality_reduction(self):
        """Apply dimensionality reduction to feature matrix."""
        print(f"Applying {self.dimensionality_reduction} dimensionality reduction...")
        
        original_shape = self.item_feature_matrix.shape
        
        if self.dimensionality_reduction == 'pca':
            self.dimensionality_reducer = PCA(n_components=min(self.n_components, original_shape[1]))
            reduced_features = self.dimensionality_reducer.fit_transform(self.item_feature_matrix)
            
        elif self.dimensionality_reduction == 'svd':
            self.dimensionality_reducer = TruncatedSVD(n_components=min(self.n_components, original_shape[1]))
            reduced_features = self.dimensionality_reducer.fit_transform(self.item_feature_matrix)
            
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {self.dimensionality_reduction}")
        
        self.item_feature_matrix = reduced_features
        self.feature_names = [f"{self.dimensionality_reduction}_component_{i}" 
                             for i in range(reduced_features.shape[1])]
        
        print(f"Reduced features from {original_shape[1]} to {reduced_features.shape[1]} dimensions")
        
        if hasattr(self.dimensionality_reducer, 'explained_variance_ratio_'):
            explained_variance = np.sum(self.dimensionality_reducer.explained_variance_ratio_)
            print(f"Explained variance ratio: {explained_variance:.3f}")
    
    def _calculate_feature_statistics(self):
        """Calculate statistics about the feature matrix."""
        self.feature_statistics = {
            'n_items': len(self.item_ids),
            'n_features': self.item_feature_matrix.shape[1],
            'feature_density': np.count_nonzero(self.item_feature_matrix) / self.item_feature_matrix.size,
            'mean_feature_values': np.mean(self.item_feature_matrix, axis=0),
            'std_feature_values': np.std(self.item_feature_matrix, axis=0),
            'feature_ranges': {
                'min': np.min(self.item_feature_matrix, axis=0),
                'max': np.max(self.item_feature_matrix, axis=0)
            }
        }
    
    def create_user_profile(self, 
                           user_id: str, 
                           user_interactions: List[Tuple[str, float]],
                           profile_method: str = 'weighted_average',
                           learning_rate: float = 0.1) -> np.ndarray:
        """
        Create or update user profile based on interaction history.
        
        Args:
            user_id: User identifier
            user_interactions: List of (item_id, rating) tuples
            profile_method: Method for creating profile
            learning_rate: Learning rate for profile updates
            
        Returns:
            User profile vector
        """
        valid_interactions = []
        
        # Filter valid interactions
        for item_id, rating in user_interactions:
            if item_id in self.item_features:
                valid_interactions.append((item_id, rating))
        
        if not valid_interactions:
            # Return zero profile for users with no valid interactions
            return np.zeros(self.item_feature_matrix.shape[1])
        
        # Extract item features and ratings
        item_features = np.array([self.item_features[item_id] for item_id, _ in valid_interactions])
        ratings = np.array([rating for _, rating in valid_interactions])
        
        # Create profile based on method
        if profile_method == 'weighted_average':
            # Weight by ratings (normalized to positive values)
            min_rating = np.min(ratings)
            max_rating = np.max(ratings)
            
            if max_rating > min_rating:
                normalized_ratings = (ratings - min_rating) / (max_rating - min_rating)
            else:
                normalized_ratings = np.ones_like(ratings)
            
            weights = normalized_ratings + 0.1  # Add small epsilon to avoid zero weights
            weights = weights / np.sum(weights)
            
            user_profile = np.average(item_features, axis=0, weights=weights)
            
        elif profile_method == 'positive_negative':
            # Separate positive and negative feedback
            mean_rating = np.mean(ratings)
            positive_mask = ratings >= mean_rating
            negative_mask = ratings < mean_rating
            
            positive_profile = np.zeros(self.item_feature_matrix.shape[1])
            negative_profile = np.zeros(self.item_feature_matrix.shape[1])
            
            if np.any(positive_mask):
                positive_profile = np.mean(item_features[positive_mask], axis=0)
            
            if np.any(negative_mask):
                negative_profile = np.mean(item_features[negative_mask], axis=0)
            
            # Combine positive and negative profiles
            user_profile = positive_profile - 0.5 * negative_profile
            
        elif profile_method == 'simple_average':
            user_profile = np.mean(item_features, axis=0)
            
        elif profile_method == 'incremental':
            # Incremental learning approach
            if user_id in self.user_profiles:
                current_profile = self.user_profiles[user_id].copy()
            else:
                current_profile = np.zeros(self.item_feature_matrix.shape[1])
            
            # Update profile incrementally
            for item_id, rating in valid_interactions:
                item_feature = self.item_features[item_id]
                
                # Normalize rating to [-1, 1] range
                normalized_rating = (rating - 3.0) / 2.0  # Assuming 1-5 rating scale
                
                # Update profile
                current_profile += learning_rate * normalized_rating * (item_feature - current_profile)
            
            user_profile = current_profile
            
        else:
            raise ValueError(f"Unknown profile method: {profile_method}")
        
        # Store user profile and interaction history
        self.user_profiles[user_id] = user_profile
        self.user_interaction_history[user_id] = valid_interactions
        
        return user_profile
    
    def compute_item_similarities(self, 
                                item_id: str, 
                                n_similar: int = 10) -> List[Tuple[str, float]]:
        """
        Find items similar to a given item.
        
        Args:
            item_id: Target item ID
            n_similar: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if item_id not in self.item_features:
            return []
        
        target_features = self.item_features[item_id]
        similarities = []
        
        for other_item_id, other_features in self.item_features.items():
            if other_item_id != item_id:
                # Compute similarity using multiple metrics
                similarity_scores = []
                
                for metric in self.similarity_metrics:
                    if metric == 'cosine':
                        sim = 1 - cosine(target_features, other_features)
                    elif metric == 'euclidean':
                        sim = 1 / (1 + euclidean(target_features, other_features))
                    elif metric == 'dot_product':
                        sim = np.dot(target_features, other_features)
                    else:
                        sim = 0
                    
                    similarity_scores.append(sim)
                
                # Average similarity across metrics
                avg_similarity = np.mean(similarity_scores)
                similarities.append((other_item_id, avg_similarity))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]
    
    def recommend_items(self, 
                       user_id: str, 
                       n_recommendations: int = 10,
                       exclude_seen: bool = True,
                       diversity_factor: float = 0.0,
                       explanation: bool = False) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Generate content-based recommendations for a user.
        
        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to generate
            exclude_seen: Whether to exclude previously seen items
            diversity_factor: Factor to promote diversity (0.0 = no diversity, 1.0 = max diversity)
            explanation: Whether to include explanations
            
        Returns:
            List of (item_id, score, explanation) tuples
        """
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        seen_items = set()
        
        if exclude_seen:
            seen_items = {item_id for item_id, _ in self.user_interaction_history[user_id]}
        
        # Calculate scores for all items
        item_scores = []
        
        for item_id, item_features in self.item_features.items():
            if item_id in seen_items:
                continue
            
            # Compute content-based score
            content_scores = []
            
            for metric in self.similarity_metrics:
                if metric == 'cosine':
                    score = 1 - cosine(user_profile, item_features)
                elif metric == 'euclidean':
                    score = 1 / (1 + euclidean(user_profile, item_features))
                elif metric == 'dot_product':
                    score = np.dot(user_profile, item_features)
                else:
                    score = 0
                
                content_scores.append(score)
            
            avg_score = np.mean(content_scores)
            
            # Prepare explanation if requested
            explanation_dict = None
            if explanation:
                explanation_dict = self._generate_explanation(user_id, item_id, avg_score)
            
            item_scores.append((item_id, avg_score, explanation_dict))
        
        # Apply diversity if requested
        if diversity_factor > 0:
            item_scores = self._apply_diversity(item_scores, diversity_factor, n_recommendations)
        else:
            item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return item_scores[:n_recommendations]
    
    def _apply_diversity(self, 
                        item_scores: List[Tuple[str, float, Optional[Dict]]], 
                        diversity_factor: float,
                        n_recommendations: int) -> List[Tuple[str, float, Optional[Dict]]]:
        """Apply diversity to recommendation list using MMR-like approach."""
        if len(item_scores) <= n_recommendations:
            return item_scores
        
        # Sort by score initially
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        remaining = item_scores.copy()
        
        # Select first item (highest score)
        selected.append(remaining.pop(0))
        
        # Select remaining items using MMR
        while len(selected) < n_recommendations and remaining:
            best_item = None
            best_mmr_score = -float('inf')
            best_idx = -1
            
            for idx, (item_id, score, explanation) in enumerate(remaining):
                item_features = self.item_features[item_id]
                
                # Calculate maximum similarity to already selected items
                max_similarity = 0
                for selected_item_id, _, _ in selected:
                    selected_features = self.item_features[selected_item_id]
                    similarity = 1 - cosine(item_features, selected_features)
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score: balance relevance and diversity
                mmr_score = (1 - diversity_factor) * score - diversity_factor * max_similarity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_item = (item_id, score, explanation)
                    best_idx = idx
            
            if best_item:
                selected.append(best_item)
                remaining.pop(best_idx)
        
        return selected
    
    def _generate_explanation(self, user_id: str, item_id: str, score: float) -> Dict:
        """Generate explanation for why an item was recommended."""
        user_profile = self.user_profiles[user_id]
        item_features = self.item_features[item_id]
        
        # Find top contributing features
        feature_contributions = user_profile * item_features
        top_feature_indices = np.argsort(np.abs(feature_contributions))[-5:]
        
        top_features = []
        for idx in top_feature_indices:
            if idx < len(self.feature_names):
                feature_name = self.feature_names[idx]
                contribution = feature_contributions[idx]
                user_preference = user_profile[idx]
                item_value = item_features[idx]
                
                top_features.append({
                    'feature': feature_name,
                    'contribution': float(contribution),
                    'user_preference': float(user_preference),
                    'item_value': float(item_value)
                })
        
        explanation = {
            'overall_score': float(score),
            'top_features': top_features,
            'similarity_method': self.similarity_metrics,
            'based_on_interactions': len(self.user_interaction_history[user_id])
        }
        
        return explanation
    
    def get_feature_importance(self, user_id: str) -> Dict[str, float]:
        """Get feature importance for a user's profile."""
        if user_id not in self.user_profiles:
            return {}
        
        user_profile = self.user_profiles[user_id]
        
        # Calculate absolute importance
        importance_scores = np.abs(user_profile)
        
        # Normalize to sum to 1
        total_importance = np.sum(importance_scores)
        if total_importance > 0:
            importance_scores = importance_scores / total_importance
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            if i < len(importance_scores):
                feature_importance[feature_name] = float(importance_scores[i])
        
        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def analyze_user_preferences(self, user_id: str) -> Dict:
        """Analyze and summarize user preferences."""
        if user_id not in self.user_profiles:
            return {}
        
        user_profile = self.user_profiles[user_id]
        interactions = self.user_interaction_history[user_id]
        
        # Basic statistics
        ratings = [rating for _, rating in interactions]
        
        analysis = {
            'n_interactions': len(interactions),
            'avg_rating': np.mean(ratings) if ratings else 0,
            'rating_std': np.std(ratings) if len(ratings) > 1 else 0,
            'min_rating': np.min(ratings) if ratings else 0,
            'max_rating': np.max(ratings) if ratings else 0,
            'profile_strength': np.linalg.norm(user_profile),
            'top_features': self.get_feature_importance(user_id)
        }
        
        return analysis
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        model_data = {
            'item_features': self.item_features,
            'item_feature_matrix': self.item_feature_matrix,
            'user_profiles': self.user_profiles,
            'feature_names': self.feature_names,
            'item_ids': self.item_ids,
            'similarity_metrics': self.similarity_metrics,
            'feature_weights': self.feature_weights,
            'categorical_encoders': self.categorical_encoders,
            'numerical_scaler': self.numerical_scaler,
            'dimensionality_reducer': self.dimensionality_reducer,
            'feature_statistics': self.feature_statistics,
            'user_interaction_history': dict(self.user_interaction_history)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.item_features = model_data['item_features']
        self.item_feature_matrix = model_data['item_feature_matrix']
        self.user_profiles = model_data['user_profiles']
        self.feature_names = model_data['feature_names']
        self.item_ids = model_data['item_ids']
        self.similarity_metrics = model_data['similarity_metrics']
        self.feature_weights = model_data['feature_weights']
        self.categorical_encoders = model_data['categorical_encoders']
        self.numerical_scaler = model_data['numerical_scaler']
        self.dimensionality_reducer = model_data['dimensionality_reducer']
        self.feature_statistics = model_data['feature_statistics']
        self.user_interaction_history = defaultdict(list, model_data['user_interaction_history'])
        
        print(f"Model loaded from {filepath}")
    
    def visualize_feature_space(self, max_items: int = 100):
        """Visualize the feature space using dimensionality reduction."""
        if self.item_feature_matrix.shape[1] < 2:
            print("Need at least 2 features for visualization")
            return
        
        # Sample items if too many
        n_items = min(max_items, len(self.item_ids))
        sample_indices = np.random.choice(len(self.item_ids), n_items, replace=False)
        sample_features = self.item_feature_matrix[sample_indices]
        sample_item_ids = [self.item_ids[i] for i in sample_indices]
        
        # Apply PCA for visualization
        if self.item_feature_matrix.shape[1] > 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(sample_features)
            
            explained_variance = np.sum(pca.explained_variance_ratio_)
            title = f'Item Feature Space (PCA, {explained_variance:.1%} variance explained)'
        else:
            features_2d = sample_features
            title = 'Item Feature Space'
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=50)
        
        # Add labels for a few items
        for i in range(min(10, len(sample_item_ids))):
            plt.annotate(sample_item_ids[i], 
                        (features_2d[i, 0], features_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_model_statistics(self) -> Dict:
        """Get comprehensive model statistics."""
        stats = {
            'n_items': len(self.item_ids),
            'n_features': self.item_feature_matrix.shape[1] if self.item_feature_matrix is not None else 0,
            'n_users_with_profiles': len(self.user_profiles),
            'feature_density': self.feature_statistics.get('feature_density', 0),
            'similarity_metrics': self.similarity_metrics,
            'dimensionality_reduction': self.dimensionality_reduction,
            'avg_interactions_per_user': np.mean([len(interactions) 
                                                for interactions in self.user_interaction_history.values()]) 
                                               if self.user_interaction_history else 0
        }
        
        return stats

# Specialized content-based recommenders for different domains
class TextContentRecommender(ContentBasedRecommender):
    """Specialized recommender for text-heavy content like articles, books, etc."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Enhanced text processing
        self.text_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.9,
            analyzer='word',
            lowercase=True
        )
        
        # Additional text processing tools
        self.entity_extractor = None  # Could integrate NER
        self.topic_model = None       # Could integrate topic modeling
    
    def extract_text_features(self, text_content: str) -> Dict:
        """Extract advanced text features."""
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text_content)
        features['word_count'] = len(text_content.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text_content))
        features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
        
        # Readability metrics (simplified)
        words = text_content.split()
        if words:
            features['avg_word_length'] = np.mean([len(word) for word in words])
        else:
            features['avg_word_length'] = 0
        
        return features

class MultimediaContentRecommender(ContentBasedRecommender):
    """Specialized recommender for multimedia content with various feature types."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Specialized feature weights for multimedia
        self.feature_weights = {
            'visual': 0.3,
            'audio': 0.2,
            'text': 0.3,
            'metadata': 0.2
        }
    
    def process_multimedia_features(self, content_data: Dict) -> Dict:
        """Process multimedia-specific features."""
        features = {}
        
        # Visual features (placeholder - would integrate with computer vision)
        if 'image_features' in content_data:
            features['dominant_colors'] = content_data.get('dominant_colors', [])
            features['brightness'] = content_data.get('brightness', 0)
            features['contrast'] = content_data.get('contrast', 0)
        
        # Audio features (placeholder - would integrate with audio analysis)
        if 'audio_features' in content_data:
            features['tempo'] = content_data.get('tempo', 0)
            features['key'] = content_data.get('key', 'C')
            features['energy'] = content_data.get('energy', 0)
        
        # Duration and technical specs
        features['duration'] = content_data.get('duration', 0)
        features['file_size'] = content_data.get('file_size', 0)
        features['resolution'] = content_data.get('resolution', '720p')
        
        return features

# Example usage and comprehensive testing
def create_content_based_test_data():
    """Create comprehensive test data for content-based recommendations."""
    np.random.seed(42)
    
    # Create diverse movie dataset
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Documentary']
    directors = ['Director_A', 'Director_B', 'Director_C', 'Director_D', 'Director_E']
    actors = ['Actor_1', 'Actor_2', 'Actor_3', 'Actor_4', 'Actor_5', 'Actor_6']
    
    movies = []
    for i in range(500):
        # Generate movie features
        movie = {
            'item_id': f'movie_{i}',
            'title': f'Movie Title {i}',
            'genre': random.choice(genres),
            'director': random.choice(directors),
            'main_actor': random.choice(actors),
            'year': random.randint(1980, 2023),
            'runtime': random.randint(80, 180),
            'imdb_rating': round(random.uniform(3.0, 9.0), 1),
            'budget': random.randint(1, 200) * 1000000,  # In millions
            'plot_summary': f"This is an engaging {random.choice(genres).lower()} movie about various characters and their adventures. The story unfolds with drama and excitement.",
            'country': random.choice(['USA', 'UK', 'France', 'Germany', 'Japan']),
            'language': random.choice(['English', 'French', 'German', 'Japanese'])
        }
        movies.append(movie)
    
    # Create user interaction data
    users = [f'user_{i}' for i in range(100)]
    interactions = []
    
    # Create user preferences based on genre preferences
    user_genre_preferences = {}
    for user in users:
        # Each user prefers 1-3 genres
        n_preferred = random.randint(1, 3)
        preferred_genres = random.sample(genres, n_preferred)
        user_genre_preferences[user] = preferred_genres
    
    # Generate interactions based on preferences
    for user in users:
        preferred_genres = user_genre_preferences[user]
        n_interactions = random.randint(10, 50)
        
        for _ in range(n_interactions):
            # 70% chance to rate preferred genre, 30% chance random
            if random.random() < 0.7 and preferred_genres:
                # Find movies in preferred genres
                preferred_movies = [m for m in movies if m['genre'] in preferred_genres]
                if preferred_movies:
                    movie = random.choice(preferred_movies)
                    rating = random.uniform(3.5, 5.0)  # Higher rating for preferred
                else:
                    movie = random.choice(movies)
                    rating = random.uniform(2.0, 4.0)
            else:
                movie = random.choice(movies)
                rating = random.uniform(1.5, 4.5)  # Random rating
            
            interactions.append((user, movie['item_id'], round(rating, 1)))
    
    return movies, interactions, user_genre_preferences

if __name__ == "__main__":
    print("Creating content-based recommendation test scenario...")
    movies, interactions, user_preferences = create_content_based_test_data()
    
    print(f"Created {len(movies)} movies and {len(interactions)} interactions")
    
    # Initialize recommender
    recommender = ContentBasedRecommender(
        similarity_metrics=['cosine', 'euclidean'],
        feature_weights={'genre': 2.0, 'director': 1.5, 'plot_summary': 1.0},
        normalization_method='standard',
        dimensionality_reduction='pca',
        n_components=50
    )
    
    # Fit item features
    print("\nFitting item features...")
    recommender.fit_item_features(movies)
    
    # Create user profiles
    print("Creating user profiles...")
    user_interactions = defaultdict(list)
    for user_id, item_id, rating in interactions:
        user_interactions[user_id].append((item_id, rating))
    
    # Test different profile creation methods
    test_user = 'user_0'
    test_interactions = user_interactions[test_user]
    
    print(f"\nTesting profile creation for {test_user} with {len(test_interactions)} interactions")
    
    profile_methods = ['weighted_average', 'positive_negative', 'simple_average']
    
    for method in profile_methods:
        profile = recommender.create_user_profile(test_user, test_interactions, profile_method=method)
        print(f"{method}: Profile norm = {np.linalg.norm(profile):.3f}")
    
    # Generate recommendations
    print(f"\nGenerating recommendations for {test_user}...")
    recommendations = recommender.recommend_items(
        test_user, 
        n_recommendations=10,
        diversity_factor=0.2,
        explanation=True
    )
    
    print("Top 5 recommendations:")
    for i, (item_id, score, explanation) in enumerate(recommendations[:5]):
        print(f"{i+1}. {item_id}: Score = {score:.3f}")
        if explanation:
            print(f"   Based on {explanation['based_on_interactions']} interactions")
            print(f"   Top contributing feature: {explanation['top_features'][0]['feature']}")
    
    # Analyze user preferences
    print(f"\nAnalyzing preferences for {test_user}...")
    user_analysis = recommender.analyze_user_preferences(test_user)
    print(f"Average rating: {user_analysis['avg_rating']:.2f}")
    print(f"Profile strength: {user_analysis['profile_strength']:.3f}")
    print(f"Top 3 feature preferences:")
    
    top_features = list(user_analysis['top_features'].items())[:3]
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")
    
    # Test item similarity
    print(f"\nFinding items similar to movie_0...")
    similar_items = recommender.compute_item_similarities('movie_0', n_similar=5)
    
    print("Most similar items:")
    for item_id, similarity in similar_items:
        print(f"  {item_id}: {similarity:.3f}")
    
    # Visualize feature space
    print("\nVisualizing feature space...")
    recommender.visualize_feature_space(max_items=50)
    
    # Get model statistics
    stats = recommender.get_model_statistics()
    print(f"\nModel Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nContent-based recommendation system testing complete!")
```

## 4. Key Concepts and Techniques

### 4.1 Feature Extraction Methods
- **Text Features**: TF-IDF, word embeddings, topic models
- **Numerical Features**: Normalization, binning, statistical measures
- **Categorical Features**: One-hot encoding, embedding, frequency encoding
- **Multimedia Features**: Color histograms, SIFT, audio spectrograms

### 4.2 User Profile Learning
- **Explicit Profiling**: Direct user input, surveys, preferences
- **Implicit Profiling**: Learning from user behavior and interactions
- **Profile Evolution**: Adapting to changing user preferences over time
- **Profile Combination**: Merging multiple profile sources

### 4.3 Similarity Computation
- **Vector Space Model**: Cosine similarity, Euclidean distance
- **Probabilistic Models**: Naive Bayes, language models
- **Machine Learning**: Classification, regression approaches

## 5. Study Questions

### Basic Level
1. What are the main advantages of content-based recommendations over collaborative filtering?
2. How does feature selection impact recommendation quality?
3. What is the overspecialization problem and how can it be addressed?
4. How do you handle different types of features in a unified system?

### Intermediate Level
5. Implement a content-based system that learns user preferences incrementally.
6. Design a method to automatically extract features from unstructured content.
7. How would you handle temporal changes in item features?
8. Create a diversity-aware content-based recommendation algorithm.

### Advanced Level
9. Implement a multi-modal content-based system handling text, images, and audio.
10. Design a system that explains recommendations using feature contributions.
11. How would you scale content-based recommendations to millions of items?
12. Create a hybrid system combining content-based with collaborative filtering.

### Tricky Questions
13. How would you recommend items with completely new feature types not seen during training?
14. A user likes action movies but only from a specific decade. How would your system capture this nuanced preference?
15. Design a content-based system that can handle contradictory user preferences.
16. How would you create recommendations when item features are noisy or incomplete?

## 6. Key Takeaways

1. **Content-based systems excel at item cold start** but struggle with user cold start
2. **Feature engineering is critical** - quality of features determines system performance
3. **User profiles must balance** stability with adaptability to changing preferences
4. **Diversity mechanisms** are essential to avoid overspecialization
5. **Multiple similarity metrics** can improve robustness
6. **Explainability is a natural advantage** of content-based approaches

## Next Session Preview
In the next session, we'll dive deep into feature extraction and representation methods, exploring advanced techniques for processing text, images, audio, and other multimedia content for recommendation systems.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create day2_09_ranking_evaluation.md: IR Evaluation Metrics", "status": "completed", "priority": "high", "id": "create_day2_09_ranking_evaluation"}, {"content": "Create day3_001_useritem_introduction.md: Introduction to User-Item Matrices", "status": "completed", "priority": "high", "id": "create_day3_001"}, {"content": "Create day3_002_useritem_representation.md: User-Item Matrix Representations and Sparsity", "status": "completed", "priority": "high", "id": "create_day3_002"}, {"content": "Create day3_003_cf_fundamentals.md: Collaborative Filtering Fundamentals", "status": "completed", "priority": "high", "id": "create_day3_003"}, {"content": "Create day3_004_user_based_cf.md: User-Based Collaborative Filtering", "status": "completed", "priority": "high", "id": "create_day3_004"}, {"content": "Create day3_005_item_based_cf.md: Item-Based Collaborative Filtering", "status": "completed", "priority": "high", "id": "create_day3_005"}, {"content": "Create day3_006_similarity_measures.md: Similarity Measures and Distance Metrics", "status": "completed", "priority": "high", "id": "create_day3_006"}, {"content": "Create day3_007_neighborhood_selection.md: Neighborhood Selection Strategies", "status": "completed", "priority": "high", "id": "create_day3_007"}, {"content": "Create day3_008_matrix_factorization_intro.md: Introduction to Matrix Factorization", "status": "completed", "priority": "high", "id": "create_day3_008"}, {"content": "Create day3_009_svd_techniques.md: SVD and Advanced Factorization Techniques", "status": "completed", "priority": "high", "id": "create_day3_009"}, {"content": "Create day3_010_cold_start_problem.md: Cold Start Problem Analysis", "status": "completed", "priority": "high", "id": "create_day3_010"}, {"content": "Create day3_011_cold_start_solutions.md: Cold Start Solutions and Strategies", "status": "completed", "priority": "high", "id": "create_day3_011"}, {"content": "Create day4_001_content_based_fundamentals.md: Content-Based Recommendation Fundamentals", "status": "completed", "priority": "high", "id": "create_day4_001"}, {"content": "Create day4_002_feature_extraction_methods.md: Feature Extraction and Representation Methods", "status": "in_progress", "priority": "high", "id": "create_day4_002"}, {"content": "Create day4_003_text_processing_nlp.md: Text Processing and NLP for Recommendations", "status": "pending", "priority": "high", "id": "create_day4_003"}, {"content": "Create day4_004_content_similarity_matching.md: Content Similarity and Matching Algorithms", "status": "pending", "priority": "high", "id": "create_day4_004"}, {"content": "Create day4_005_knowledge_based_systems.md: Knowledge-Based Recommendation Systems", "status": "pending", "priority": "high", "id": "create_day4_005"}, {"content": "Create day4_006_constraint_based_recommendations.md: Constraint-Based and Rule-Based Systems", "status": "pending", "priority": "high", "id": "create_day4_006"}]