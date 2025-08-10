# Day 4.2: Feature Extraction and Representation Methods

## Learning Objectives
By the end of this session, you will:
- Master advanced feature extraction techniques for different content types
- Understand dimensionality reduction methods for high-dimensional features
- Implement feature selection and engineering strategies
- Learn domain-specific feature extraction approaches
- Build scalable feature processing pipelines

## 1. Advanced Text Feature Extraction

### TF-IDF with Advanced Normalization

```python
import numpy as np
import math
from collections import defaultdict, Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

class AdvancedTextFeatureExtractor:
    """
    Advanced text feature extraction with multiple techniques
    """
    
    def __init__(self, max_features=10000, min_df=2, max_df=0.95):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary = {}
        self.idf_scores = {}
        self.feature_names = []
        self.fitted = False
        
    def fit(self, documents):
        """
        Fit the feature extractor on document collection
        
        Args:
            documents: List of text documents
        """
        # Build vocabulary and compute IDF scores
        self._build_vocabulary(documents)
        self._compute_idf_scores(documents)
        self.fitted = True
        
    def transform(self, documents):
        """
        Transform documents to feature vectors
        
        Args:
            documents: List of text documents
            
        Returns:
            Feature matrix (n_docs x n_features)
        """
        if not self.fitted:
            raise ValueError("Extractor must be fitted before transform")
            
        feature_matrix = []
        
        for doc in documents:
            features = self._extract_document_features(doc)
            feature_matrix.append(features)
            
        return np.array(feature_matrix)
    
    def fit_transform(self, documents):
        """Fit and transform in one step"""
        self.fit(documents)
        return self.transform(documents)
    
    def _build_vocabulary(self, documents):
        """Build vocabulary from documents"""
        term_doc_freq = defaultdict(int)
        total_docs = len(documents)
        
        # Count document frequency for each term
        for doc in documents:
            terms = set(self._tokenize(doc))
            for term in terms:
                term_doc_freq[term] += 1
        
        # Filter terms by document frequency
        filtered_terms = {}
        for term, df in term_doc_freq.items():
            if (isinstance(self.min_df, int) and df >= self.min_df) or \
               (isinstance(self.min_df, float) and df/total_docs >= self.min_df):
                if (isinstance(self.max_df, int) and df <= self.max_df) or \
                   (isinstance(self.max_df, float) and df/total_docs <= self.max_df):
                    filtered_terms[term] = df
        
        # Select top features by document frequency
        sorted_terms = sorted(filtered_terms.items(), 
                             key=lambda x: x[1], reverse=True)
        
        self.vocabulary = {term: idx for idx, (term, _) in 
                          enumerate(sorted_terms[:self.max_features])}
        self.feature_names = list(self.vocabulary.keys())
    
    def _compute_idf_scores(self, documents):
        """Compute IDF scores for vocabulary terms"""
        total_docs = len(documents)
        
        for term in self.vocabulary:
            # Count documents containing term
            doc_freq = sum(1 for doc in documents if term in self._tokenize(doc))
            
            # Compute IDF with smoothing
            self.idf_scores[term] = math.log(total_docs / (1 + doc_freq))
    
    def _extract_document_features(self, document):
        """Extract TF-IDF features for a single document"""
        terms = self._tokenize(document)
        term_counts = Counter(terms)
        doc_length = len(terms)
        
        features = np.zeros(len(self.vocabulary))
        
        for term, count in term_counts.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                
                # TF with log normalization
                tf = 1 + math.log(count) if count > 0 else 0
                
                # Apply IDF
                idf = self.idf_scores[term]
                
                # TF-IDF score
                features[idx] = tf * idf
        
        # L2 normalization
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features
    
    def _tokenize(self, text):
        """Tokenize text with preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Remove stopwords (simple list)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                    'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were',
                    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'}
        
        tokens = [token for token in tokens if token not in stopwords and len(token) > 2]
        
        return tokens

# N-gram feature extraction
class NGramFeatureExtractor:
    """Extract n-gram features from text"""
    
    def __init__(self, n_range=(1, 3), max_features=5000):
        self.n_range = n_range
        self.max_features = max_features
        self.ngram_vocab = {}
        
    def fit_transform(self, documents):
        """Extract n-gram features from documents"""
        # Collect all n-grams
        all_ngrams = defaultdict(int)
        
        for doc in documents:
            tokens = self._tokenize(doc)
            for n in range(self.n_range[0], self.n_range[1] + 1):
                ngrams = self._extract_ngrams(tokens, n)
                for ngram in ngrams:
                    all_ngrams[ngram] += 1
        
        # Select top n-grams
        sorted_ngrams = sorted(all_ngrams.items(), 
                              key=lambda x: x[1], reverse=True)
        
        self.ngram_vocab = {ngram: idx for idx, (ngram, _) in 
                           enumerate(sorted_ngrams[:self.max_features])}
        
        # Create feature matrix
        feature_matrix = []
        for doc in documents:
            features = self._document_to_ngram_vector(doc)
            feature_matrix.append(features)
            
        return np.array(feature_matrix)
    
    def _extract_ngrams(self, tokens, n):
        """Extract n-grams from token list"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n])
            ngrams.append(ngram)
        return ngrams
    
    def _document_to_ngram_vector(self, document):
        """Convert document to n-gram feature vector"""
        tokens = self._tokenize(document)
        features = np.zeros(len(self.ngram_vocab))
        
        # Count n-grams in document
        ngram_counts = defaultdict(int)
        for n in range(self.n_range[0], self.n_range[1] + 1):
            ngrams = self._extract_ngrams(tokens, n)
            for ngram in ngrams:
                ngram_counts[ngram] += 1
        
        # Fill feature vector
        for ngram, count in ngram_counts.items():
            if ngram in self.ngram_vocab:
                idx = self.ngram_vocab[ngram]
                features[idx] = count
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features
    
    def _tokenize(self, text):
        """Simple tokenization"""
        return text.lower().split()
```

## 2. Numerical Feature Processing

### Advanced Numerical Feature Engineering

```python
class NumericalFeatureProcessor:
    """
    Advanced numerical feature processing and engineering
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_stats = {}
        self.engineered_features = {}
        
    def fit(self, numerical_data, feature_names):
        """
        Fit scalers and compute statistics
        
        Args:
            numerical_data: Array of shape (n_samples, n_features)
            feature_names: List of feature names
        """
        numerical_data = np.array(numerical_data)
        
        for i, feature_name in enumerate(feature_names):
            feature_values = numerical_data[:, i]
            
            # Compute statistics
            self.feature_stats[feature_name] = {
                'mean': np.mean(feature_values),
                'std': np.std(feature_values),
                'min': np.min(feature_values),
                'max': np.max(feature_values),
                'median': np.median(feature_values),
                'q25': np.percentile(feature_values, 25),
                'q75': np.percentile(feature_values, 75)
            }
            
            # Fit scaler
            scaler = StandardScaler()
            scaler.fit(feature_values.reshape(-1, 1))
            self.scalers[feature_name] = scaler
    
    def transform(self, numerical_data, feature_names, engineer_features=True):
        """
        Transform numerical features with scaling and engineering
        
        Args:
            numerical_data: Array of shape (n_samples, n_features)
            feature_names: List of feature names
            engineer_features: Whether to create engineered features
            
        Returns:
            Transformed feature matrix
        """
        numerical_data = np.array(numerical_data)
        transformed_features = []
        
        # Apply scaling
        for i, feature_name in enumerate(feature_names):
            if feature_name in self.scalers:
                feature_values = numerical_data[:, i]
                scaled_values = self.scalers[feature_name].transform(
                    feature_values.reshape(-1, 1)
                ).flatten()
                transformed_features.append(scaled_values)
            else:
                transformed_features.append(numerical_data[:, i])
        
        base_features = np.column_stack(transformed_features)
        
        if not engineer_features:
            return base_features
        
        # Feature engineering
        engineered = self._engineer_features(base_features, feature_names)
        
        # Combine base and engineered features
        return np.column_stack([base_features, engineered])
    
    def _engineer_features(self, features, feature_names):
        """Create engineered features"""
        n_samples = features.shape[0]
        engineered_features = []
        
        # Polynomial features (degree 2)
        for i in range(len(feature_names)):
            for j in range(i, len(feature_names)):
                if i == j:
                    # Squared features
                    engineered_features.append(features[:, i] ** 2)
                else:
                    # Interaction features
                    engineered_features.append(features[:, i] * features[:, j])
        
        # Log features (for positive features)
        for i, feature_name in enumerate(feature_names):
            if np.all(features[:, i] > 0):
                engineered_features.append(np.log(features[:, i] + 1))
        
        # Binned features
        for i, feature_name in enumerate(feature_names):
            binned = self._create_binned_features(features[:, i])
            engineered_features.extend(binned.T)
        
        return np.column_stack(engineered_features)
    
    def _create_binned_features(self, feature_values, n_bins=5):
        """Create binned categorical features from numerical values"""
        # Equal-frequency binning
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(feature_values, percentiles)
        bin_edges[-1] += 1e-6  # Avoid boundary issues
        
        # Create binary features for each bin
        binned_features = np.zeros((len(feature_values), n_bins))
        
        for i, value in enumerate(feature_values):
            bin_idx = np.digitize(value, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)  # Handle edge cases
            binned_features[i, bin_idx] = 1
        
        return binned_features

class TimeSeriesFeatureExtractor:
    """Extract features from time-series data for temporal recommendations"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_temporal_features(self, timestamps, values):
        """
        Extract time-series features
        
        Args:
            timestamps: Array of timestamps
            values: Array of corresponding values
            
        Returns:
            Dictionary of temporal features
        """
        timestamps = np.array(timestamps)
        values = np.array(values)
        
        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        values = values[sort_idx]
        
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(values)
        features['std'] = np.std(values)
        features['min'] = np.min(values)
        features['max'] = np.max(values)
        features['range'] = features['max'] - features['min']
        
        # Trend features
        if len(values) > 1:
            # Linear trend slope
            time_diffs = timestamps[1:] - timestamps[0]
            slope, _ = np.polyfit(time_diffs, values, 1)
            features['trend_slope'] = slope
            
            # Recent vs historical average
            recent_period = len(values) // 4  # Last 25%
            if recent_period > 0:
                recent_avg = np.mean(values[-recent_period:])
                historical_avg = np.mean(values[:-recent_period])
                features['recent_vs_historical'] = recent_avg - historical_avg
        
        # Frequency domain features (if enough data)
        if len(values) >= 10:
            fft_values = np.fft.fft(values)
            fft_magnitude = np.abs(fft_values)
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
            features['dominant_frequency'] = dominant_freq_idx
            
            # Spectral energy
            features['spectral_energy'] = np.sum(fft_magnitude ** 2)
        
        # Seasonality detection (simplified)
        if len(values) >= 7:  # At least a week
            # Weekly pattern (assuming daily data)
            weekly_pattern = self._detect_weekly_pattern(values)
            features['weekly_seasonality'] = weekly_pattern
        
        self.feature_names = list(features.keys())
        return features
    
    def _detect_weekly_pattern(self, values):
        """Detect weekly seasonality pattern"""
        if len(values) < 14:  # Need at least 2 weeks
            return 0.0
        
        # Compute autocorrelation at lag 7
        n = len(values)
        mean_val = np.mean(values)
        
        numerator = 0
        denominator = 0
        
        for i in range(n - 7):
            numerator += (values[i] - mean_val) * (values[i + 7] - mean_val)
            denominator += (values[i] - mean_val) ** 2
        
        if denominator > 0:
            return numerator / denominator
        return 0.0
```

## 3. Categorical Feature Processing

### Advanced Categorical Encoding

```python
class CategoricalFeatureProcessor:
    """
    Advanced categorical feature processing with multiple encoding strategies
    """
    
    def __init__(self):
        self.encoders = {}
        self.category_stats = {}
        
    def fit(self, categorical_data, method='auto'):
        """
        Fit categorical encoders
        
        Args:
            categorical_data: Dictionary of feature_name -> list_of_categories
            method: Encoding method ('onehot', 'target', 'embedding', 'auto')
        """
        for feature_name, categories in categorical_data.items():
            unique_categories = list(set(categories))
            category_counts = Counter(categories)
            
            # Store statistics
            self.category_stats[feature_name] = {
                'unique_count': len(unique_categories),
                'most_common': category_counts.most_common(5),
                'categories': unique_categories
            }
            
            # Choose encoding method
            if method == 'auto':
                encoding_method = self._choose_encoding_method(unique_categories)
            else:
                encoding_method = method
            
            # Create encoder
            if encoding_method == 'onehot':
                self.encoders[feature_name] = OneHotEncoder(unique_categories)
            elif encoding_method == 'target':
                self.encoders[feature_name] = TargetEncoder()
            elif encoding_method == 'embedding':
                self.encoders[feature_name] = EmbeddingEncoder(unique_categories)
            else:
                self.encoders[feature_name] = OneHotEncoder(unique_categories)
    
    def transform(self, categorical_data):
        """
        Transform categorical features
        
        Args:
            categorical_data: Dictionary of feature_name -> list_of_categories
            
        Returns:
            Dictionary of feature_name -> encoded_features
        """
        encoded_features = {}
        
        for feature_name, categories in categorical_data.items():
            if feature_name in self.encoders:
                encoder = self.encoders[feature_name]
                encoded = encoder.transform(categories)
                encoded_features[feature_name] = encoded
            else:
                # Handle unknown features with zero encoding
                encoded_features[feature_name] = np.zeros((len(categories), 1))
        
        return encoded_features
    
    def _choose_encoding_method(self, unique_categories):
        """Choose appropriate encoding method based on cardinality"""
        cardinality = len(unique_categories)
        
        if cardinality <= 10:
            return 'onehot'
        elif cardinality <= 50:
            return 'target'
        else:
            return 'embedding'

class OneHotEncoder:
    """One-hot encoding for categorical features"""
    
    def __init__(self, categories):
        self.categories = sorted(categories)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
    
    def transform(self, categories):
        """Transform categories to one-hot vectors"""
        n_samples = len(categories)
        n_categories = len(self.categories)
        
        encoded = np.zeros((n_samples, n_categories))
        
        for i, category in enumerate(categories):
            if category in self.category_to_idx:
                idx = self.category_to_idx[category]
                encoded[i, idx] = 1.0
        
        return encoded

class TargetEncoder:
    """Target encoding for high-cardinality categorical features"""
    
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.category_means = {}
        self.global_mean = 0.0
    
    def fit(self, categories, targets):
        """
        Fit target encoder
        
        Args:
            categories: List of category values
            targets: List of target values
        """
        self.global_mean = np.mean(targets)
        
        category_stats = defaultdict(list)
        for cat, target in zip(categories, targets):
            category_stats[cat].append(target)
        
        # Compute smoothed means
        for category, target_values in category_stats.items():
            n = len(target_values)
            category_mean = np.mean(target_values)
            
            # Apply smoothing
            smoothed_mean = (category_mean * n + self.global_mean * self.smoothing) / (n + self.smoothing)
            self.category_means[category] = smoothed_mean
    
    def transform(self, categories):
        """Transform categories to target-encoded values"""
        encoded = []
        for category in categories:
            if category in self.category_means:
                encoded.append(self.category_means[category])
            else:
                encoded.append(self.global_mean)
        
        return np.array(encoded).reshape(-1, 1)

class EmbeddingEncoder:
    """Embedding encoding for very high-cardinality categorical features"""
    
    def __init__(self, categories, embedding_dim=None):
        self.categories = sorted(categories)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # Choose embedding dimension
        if embedding_dim is None:
            self.embedding_dim = min(50, max(4, len(self.categories) // 2))
        else:
            self.embedding_dim = embedding_dim
        
        # Initialize random embeddings
        np.random.seed(42)
        self.embeddings = np.random.normal(0, 0.1, (len(self.categories), self.embedding_dim))
    
    def transform(self, categories):
        """Transform categories to embedding vectors"""
        n_samples = len(categories)
        encoded = np.zeros((n_samples, self.embedding_dim))
        
        for i, category in enumerate(categories):
            if category in self.category_to_idx:
                idx = self.category_to_idx[category]
                encoded[i] = self.embeddings[idx]
        
        return encoded
```

## 4. Dimensionality Reduction Techniques

### Feature Selection and Reduction

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

class FeatureDimensionalityReducer:
    """
    Feature selection and dimensionality reduction toolkit
    """
    
    def __init__(self):
        self.selectors = {}
        self.reducers = {}
        self.selected_features = {}
        
    def apply_feature_selection(self, X, y, feature_names, method='mutual_info', k=100):
        """
        Apply feature selection
        
        Args:
            X: Feature matrix
            y: Target values (for supervised selection)
            feature_names: List of feature names
            method: Selection method ('mutual_info', 'f_score', 'variance')
            k: Number of features to select
            
        Returns:
            Selected features and feature names
        """
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_score':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'variance':
            selector = self._create_variance_selector(k)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Fit and transform
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_indices]
        
        # Store selector
        self.selectors[method] = selector
        self.selected_features[method] = {
            'indices': selected_indices,
            'names': selected_feature_names
        }
        
        return X_selected, selected_feature_names
    
    def apply_dimensionality_reduction(self, X, method='pca', n_components=50):
        """
        Apply dimensionality reduction
        
        Args:
            X: Feature matrix
            method: Reduction method ('pca', 'svd', 'tsne')
            n_components: Number of components
            
        Returns:
            Reduced feature matrix
        """
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=min(n_components, 3), random_state=42)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        # Fit and transform
        X_reduced = reducer.fit_transform(X)
        
        # Store reducer
        self.reducers[method] = reducer
        
        return X_reduced
    
    def _create_variance_selector(self, k):
        """Create variance-based feature selector"""
        from sklearn.feature_selection import VarianceThreshold, SelectKBest
        from sklearn.pipeline import Pipeline
        
        # First remove low-variance features, then select top-k
        pipeline = Pipeline([
            ('variance', VarianceThreshold(threshold=0.01)),
            ('select_k', SelectKBest(k=k))
        ])
        
        return pipeline
    
    def get_feature_importance(self, method='mutual_info'):
        """Get feature importance scores"""
        if method in self.selectors:
            selector = self.selectors[method]
            if hasattr(selector, 'scores_'):
                return selector.scores_
        return None

class FeaturePipeline:
    """
    Complete feature processing pipeline
    """
    
    def __init__(self):
        self.text_extractor = AdvancedTextFeatureExtractor()
        self.numerical_processor = NumericalFeatureProcessor()
        self.categorical_processor = CategoricalFeatureProcessor()
        self.dimensionality_reducer = FeatureDimensionalityReducer()
        self.feature_names = []
        self.fitted = False
    
    def fit(self, data):
        """
        Fit the complete feature pipeline
        
        Args:
            data: Dictionary with 'text', 'numerical', 'categorical' keys
        """
        all_features = []
        feature_names = []
        
        # Process text features
        if 'text' in data and data['text']:
            self.text_extractor.fit(data['text'])
            text_features = self.text_extractor.transform(data['text'])
            all_features.append(text_features)
            
            text_feature_names = [f"text_{name}" for name in self.text_extractor.feature_names]
            feature_names.extend(text_feature_names)
        
        # Process numerical features
        if 'numerical' in data and data['numerical']:
            numerical_data = np.array(data['numerical']['values'])
            numerical_names = data['numerical']['names']
            
            self.numerical_processor.fit(numerical_data, numerical_names)
            numerical_features = self.numerical_processor.transform(
                numerical_data, numerical_names, engineer_features=True
            )
            all_features.append(numerical_features)
            
            # Add engineered feature names
            base_names = [f"num_{name}" for name in numerical_names]
            # Simplified - in practice, would track all engineered feature names
            feature_names.extend(base_names)
        
        # Process categorical features
        if 'categorical' in data and data['categorical']:
            self.categorical_processor.fit(data['categorical'])
            categorical_features = self.categorical_processor.transform(data['categorical'])
            
            for feature_name, features in categorical_features.items():
                all_features.append(features)
                
                # Add feature names based on encoding type
                if features.shape[1] > 1:  # One-hot or embedding
                    cat_names = [f"cat_{feature_name}_{i}" for i in range(features.shape[1])]
                else:  # Target encoding
                    cat_names = [f"cat_{feature_name}"]
                feature_names.extend(cat_names)
        
        # Combine all features
        if all_features:
            combined_features = np.hstack(all_features)
            self.feature_names = feature_names
            self.fitted = True
            
            return combined_features
        
        return np.array([])
    
    def transform(self, data):
        """Transform new data using fitted pipeline"""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        all_features = []
        
        # Transform text features
        if 'text' in data and data['text']:
            text_features = self.text_extractor.transform(data['text'])
            all_features.append(text_features)
        
        # Transform numerical features
        if 'numerical' in data and data['numerical']:
            numerical_data = np.array(data['numerical']['values'])
            numerical_names = data['numerical']['names']
            
            numerical_features = self.numerical_processor.transform(
                numerical_data, numerical_names, engineer_features=True
            )
            all_features.append(numerical_features)
        
        # Transform categorical features
        if 'categorical' in data and data['categorical']:
            categorical_features = self.categorical_processor.transform(data['categorical'])
            
            for feature_name, features in categorical_features.items():
                all_features.append(features)
        
        # Combine all features
        if all_features:
            return np.hstack(all_features)
        
        return np.array([])
```

## 5. Domain-Specific Feature Extraction

### E-commerce Product Features

```python
class EcommerceFeatureExtractor:
    """
    Domain-specific feature extraction for e-commerce products
    """
    
    def __init__(self):
        self.price_normalizer = None
        self.brand_encoder = None
        self.category_hierarchy = {}
        
    def extract_product_features(self, product_data):
        """
        Extract comprehensive features for e-commerce products
        
        Args:
            product_data: Dictionary with product information
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Price features
        if 'price' in product_data:
            features.update(self._extract_price_features(product_data))
        
        # Brand features
        if 'brand' in product_data:
            features.update(self._extract_brand_features(product_data['brand']))
        
        # Category features
        if 'category' in product_data:
            features.update(self._extract_category_features(product_data['category']))
        
        # Review features
        if 'reviews' in product_data:
            features.update(self._extract_review_features(product_data['reviews']))
        
        # Image features (simplified)
        if 'images' in product_data:
            features.update(self._extract_image_features(product_data['images']))
        
        # Availability features
        if 'availability' in product_data:
            features.update(self._extract_availability_features(product_data['availability']))
        
        return features
    
    def _extract_price_features(self, product_data):
        """Extract price-related features"""
        price = product_data['price']
        features = {'price': price}
        
        # Price range features
        if price < 10:
            features['price_range'] = 'budget'
        elif price < 50:
            features['price_range'] = 'economy'
        elif price < 200:
            features['price_range'] = 'mid_range'
        else:
            features['price_range'] = 'premium'
        
        # Discount features
        if 'original_price' in product_data:
            original_price = product_data['original_price']
            if original_price > price:
                discount_pct = (original_price - price) / original_price
                features['discount_percentage'] = discount_pct
                features['has_discount'] = 1
            else:
                features['discount_percentage'] = 0
                features['has_discount'] = 0
        
        return features
    
    def _extract_brand_features(self, brand):
        """Extract brand-related features"""
        features = {'brand': brand}
        
        # Brand popularity (simplified - would use real data)
        popular_brands = {'apple', 'samsung', 'nike', 'adidas', 'amazon'}
        features['is_popular_brand'] = 1 if brand.lower() in popular_brands else 0
        
        # Brand category (simplified)
        tech_brands = {'apple', 'samsung', 'google', 'microsoft'}
        fashion_brands = {'nike', 'adidas', 'zara', 'h&m'}
        
        if brand.lower() in tech_brands:
            features['brand_category'] = 'tech'
        elif brand.lower() in fashion_brands:
            features['brand_category'] = 'fashion'
        else:
            features['brand_category'] = 'other'
        
        return features
    
    def _extract_category_features(self, category):
        """Extract category hierarchy features"""
        features = {'category': category}
        
        # Category depth (number of levels)
        category_levels = category.split(' > ')
        features['category_depth'] = len(category_levels)
        
        # Extract each level
        for i, level in enumerate(category_levels):
            features[f'category_level_{i}'] = level
        
        # Main category
        features['main_category'] = category_levels[0] if category_levels else 'unknown'
        
        return features
    
    def _extract_review_features(self, reviews_data):
        """Extract review-based features"""
        features = {}
        
        if 'average_rating' in reviews_data:
            features['avg_rating'] = reviews_data['average_rating']
            
            # Rating categories
            rating = reviews_data['average_rating']
            if rating >= 4.5:
                features['rating_category'] = 'excellent'
            elif rating >= 4.0:
                features['rating_category'] = 'very_good'
            elif rating >= 3.5:
                features['rating_category'] = 'good'
            elif rating >= 3.0:
                features['rating_category'] = 'average'
            else:
                features['rating_category'] = 'poor'
        
        if 'review_count' in reviews_data:
            count = reviews_data['review_count']
            features['review_count'] = count
            features['has_reviews'] = 1 if count > 0 else 0
            
            # Review count categories
            if count >= 1000:
                features['review_volume'] = 'high'
            elif count >= 100:
                features['review_volume'] = 'medium'
            elif count > 0:
                features['review_volume'] = 'low'
            else:
                features['review_volume'] = 'none'
        
        return features
    
    def _extract_image_features(self, images_data):
        """Extract image-based features (simplified)"""
        features = {}
        
        if isinstance(images_data, list):
            features['image_count'] = len(images_data)
            features['has_images'] = 1 if len(images_data) > 0 else 0
        else:
            features['image_count'] = 1 if images_data else 0
            features['has_images'] = 1 if images_data else 0
        
        return features
    
    def _extract_availability_features(self, availability_data):
        """Extract availability features"""
        features = {}
        
        if 'in_stock' in availability_data:
            features['in_stock'] = 1 if availability_data['in_stock'] else 0
        
        if 'shipping_time' in availability_data:
            shipping_days = availability_data['shipping_time']
            features['shipping_days'] = shipping_days
            
            # Shipping speed categories
            if shipping_days <= 1:
                features['shipping_speed'] = 'same_day'
            elif shipping_days <= 2:
                features['shipping_speed'] = 'next_day'
            elif shipping_days <= 7:
                features['shipping_speed'] = 'standard'
            else:
                features['shipping_speed'] = 'slow'
        
        return features

# Media content feature extraction
class MediaContentFeatureExtractor:
    """Feature extraction for movies, music, books"""
    
    def extract_movie_features(self, movie_data):
        """Extract features for movies"""
        features = {}
        
        # Basic features
        features['title_length'] = len(movie_data.get('title', ''))
        features['year'] = movie_data.get('year', 0)
        features['duration'] = movie_data.get('duration_minutes', 0)
        
        # Age features
        current_year = 2024
        if 'year' in movie_data:
            movie_age = current_year - movie_data['year']
            features['movie_age'] = movie_age
            
            if movie_age <= 1:
                features['age_category'] = 'new'
            elif movie_age <= 5:
                features['age_category'] = 'recent'
            elif movie_age <= 20:
                features['age_category'] = 'classic'
            else:
                features['age_category'] = 'vintage'
        
        # Genre features
        if 'genres' in movie_data:
            genres = movie_data['genres']
            features['genre_count'] = len(genres)
            
            # Popular genre indicators
            popular_genres = {'action', 'comedy', 'drama', 'thriller', 'sci-fi'}
            for genre in popular_genres:
                features[f'is_{genre}'] = 1 if genre.lower() in [g.lower() for g in genres] else 0
        
        # Cast features
        if 'actors' in movie_data:
            features['cast_size'] = len(movie_data['actors'])
        
        # Rating features
        if 'rating' in movie_data:
            rating = movie_data['rating']
            features['rating'] = rating
            features['is_highly_rated'] = 1 if rating >= 8.0 else 0
        
        return features
```

## 6. Study Questions

### Beginner Level

1. What are the main types of features in content-based recommendation systems?
2. Why is feature normalization important in recommendation systems?
3. What is TF-IDF and how is it used in content-based recommendations?
4. Explain the difference between one-hot encoding and target encoding for categorical features.
5. What are n-gram features and when would you use them?

### Intermediate Level

6. Implement a feature extraction pipeline that combines text, numerical, and categorical features.
7. How would you handle high-cardinality categorical features in a recommendation system?
8. Design a feature engineering strategy for time-series data in recommendations.
9. What are the trade-offs between PCA and truncated SVD for dimensionality reduction?
10. How would you extract features from hierarchical categorical data (like product categories)?

### Advanced Level

11. Implement an automatic feature selection system that adapts based on recommendation performance.
12. Design a feature extraction system for multimedia content (images, videos, audio).
13. How would you handle missing features and feature drift in a production recommendation system?
14. Implement a feature learning system that can discover new features from user interactions.
15. Design a cross-domain feature transfer system for recommendations across different domains.

### Tricky Questions

16. How would you detect and handle feature leakage in a recommendation system?
17. Design a feature extraction system that can handle both structured and unstructured data efficiently.
18. How would you ensure feature consistency across different data sources and formats?
19. Implement a feature importance analysis system that can explain which features contribute most to recommendations.
20. How would you design a scalable feature processing pipeline for real-time recommendations with millions of items?

## Key Takeaways

1. **Feature quality** directly impacts recommendation system performance
2. **Domain expertise** is crucial for effective feature engineering
3. **Dimensionality reduction** helps manage computational complexity
4. **Feature selection** improves model performance and interpretability
5. **Automated pipelines** ensure consistency and scalability
6. **Multi-modal features** (text, numerical, categorical) provide richer representations
7. **Temporal features** capture dynamic user preferences and item characteristics

## Next Session Preview

In Day 4.3, we'll explore **Text Processing and NLP for Recommendations**, covering:
- Advanced NLP techniques for content analysis
- Semantic similarity and word embeddings
- Topic modeling for content understanding
- Sentiment analysis for review-based features
- Multi-language content processing