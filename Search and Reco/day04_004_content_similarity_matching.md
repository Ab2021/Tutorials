# Day 4.4: Content Similarity and Matching Algorithms

## Learning Objectives
By the end of this session, you will:
- Master advanced similarity metrics beyond basic cosine similarity
- Implement locality-sensitive hashing for scalable similarity search
- Apply graph-based similarity measures for content networks
- Build multi-modal content matching systems
- Design real-time similarity computation architectures

## 1. Advanced Similarity Metrics

### Beyond Cosine Similarity

```python
import numpy as np
import scipy.spatial.distance as distance
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import math
from collections import Counter

class AdvancedSimilarityMetrics:
    """
    Implementation of various similarity metrics for content matching
    """
    
    def __init__(self):
        self.metric_cache = {}
        
    def cosine_similarity(self, vec1, vec2):
        """Standard cosine similarity"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
        
        return dot_product / norm_product
    
    def pearson_correlation(self, vec1, vec2):
        """
        Pearson correlation coefficient similarity
        Better for handling user rating patterns
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        # Remove zero ratings for both vectors simultaneously
        mask = (vec1 != 0) & (vec2 != 0)
        if np.sum(mask) < 2:
            return 0.0
        
        vec1_filtered = vec1[mask]
        vec2_filtered = vec2[mask]
        
        correlation = np.corrcoef(vec1_filtered, vec2_filtered)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def jaccard_similarity(self, set1, set2):
        """
        Jaccard similarity for set-based features
        Useful for categorical features like genres, tags
        """
        if isinstance(set1, (list, tuple)):
            set1 = set(set1)
        if isinstance(set2, (list, tuple)):
            set2 = set(set2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def dice_coefficient(self, set1, set2):
        """
        Dice coefficient (Sørensen-Dice index)
        More sensitive to small sets than Jaccard
        """
        if isinstance(set1, (list, tuple)):
            set1 = set(set1)
        if isinstance(set2, (list, tuple)):
            set2 = set(set2)
        
        intersection = len(set1.intersection(set2))
        total_size = len(set1) + len(set2)
        
        return 2 * intersection / total_size if total_size > 0 else 0.0
    
    def manhattan_similarity(self, vec1, vec2):
        """
        Manhattan distance converted to similarity
        Good for high-dimensional sparse vectors
        """
        manhattan_dist = np.sum(np.abs(vec1 - vec2))
        # Convert distance to similarity (0-1 range)
        max_possible_dist = np.sum(np.abs(vec1) + np.abs(vec2))
        
        if max_possible_dist == 0:
            return 1.0
        
        return 1.0 - (manhattan_dist / max_possible_dist)
    
    def euclidean_similarity(self, vec1, vec2):
        """
        Euclidean distance converted to similarity
        """
        euclidean_dist = np.linalg.norm(vec1 - vec2)
        # Convert to similarity using Gaussian kernel
        return math.exp(-euclidean_dist**2 / (2 * 1.0**2))
    
    def hamming_similarity(self, vec1, vec2):
        """
        Hamming similarity for binary vectors
        Useful for one-hot encoded categorical features
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        matches = np.sum(vec1 == vec2)
        return matches / len(vec1)
    
    def tanimoto_coefficient(self, vec1, vec2):
        """
        Tanimoto coefficient (extended Jaccard for continuous values)
        """
        dot_product = np.dot(vec1, vec2)
        norm1_squared = np.dot(vec1, vec1)
        norm2_squared = np.dot(vec2, vec2)
        
        denominator = norm1_squared + norm2_squared - dot_product
        
        return dot_product / denominator if denominator > 0 else 0.0
    
    def bhattacharyya_similarity(self, vec1, vec2):
        """
        Bhattacharyya similarity for probability distributions
        Useful for topic distributions or rating distributions
        """
        # Normalize vectors to probability distributions
        vec1_norm = vec1 / (np.sum(vec1) + 1e-10)
        vec2_norm = vec2 / (np.sum(vec2) + 1e-10)
        
        # Compute Bhattacharyya coefficient
        bc = np.sum(np.sqrt(vec1_norm * vec2_norm))
        
        return bc
    
    def kl_divergence_similarity(self, vec1, vec2):
        """
        KL divergence converted to similarity
        Useful for comparing probability distributions
        """
        # Normalize to probability distributions
        vec1_norm = vec1 / (np.sum(vec1) + 1e-10)
        vec2_norm = vec2 / (np.sum(vec2) + 1e-10)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        vec1_norm += epsilon
        vec2_norm += epsilon
        
        # Compute symmetric KL divergence
        kl1 = np.sum(vec1_norm * np.log(vec1_norm / vec2_norm))
        kl2 = np.sum(vec2_norm * np.log(vec2_norm / vec1_norm))
        symmetric_kl = (kl1 + kl2) / 2
        
        # Convert to similarity
        return math.exp(-symmetric_kl)
    
    def compute_similarity_matrix(self, vectors, metric='cosine'):
        """
        Compute pairwise similarity matrix
        
        Args:
            vectors: List or array of feature vectors
            metric: Similarity metric to use
            
        Returns:
            Similarity matrix
        """
        n = len(vectors)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity = 1.0
                else:
                    similarity = self._compute_pairwise_similarity(
                        vectors[i], vectors[j], metric
                    )
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def _compute_pairwise_similarity(self, vec1, vec2, metric):
        """Compute similarity between two vectors"""
        if metric == 'cosine':
            return self.cosine_similarity(vec1, vec2)
        elif metric == 'pearson':
            return self.pearson_correlation(vec1, vec2)
        elif metric == 'manhattan':
            return self.manhattan_similarity(vec1, vec2)
        elif metric == 'euclidean':
            return self.euclidean_similarity(vec1, vec2)
        elif metric == 'hamming':
            return self.hamming_similarity(vec1, vec2)
        elif metric == 'tanimoto':
            return self.tanimoto_coefficient(vec1, vec2)
        elif metric == 'bhattacharyya':
            return self.bhattacharyya_similarity(vec1, vec2)
        elif metric == 'kl_divergence':
            return self.kl_divergence_similarity(vec1, vec2)
        else:
            return self.cosine_similarity(vec1, vec2)

class WeightedSimilarityCalculator:
    """
    Calculate similarity with different weights for different feature types
    """
    
    def __init__(self):
        self.feature_weights = {}
        self.similarity_calculator = AdvancedSimilarityMetrics()
    
    def set_feature_weights(self, weights):
        """
        Set weights for different feature types
        
        Args:
            weights: Dictionary of feature_type -> weight
        """
        self.feature_weights = weights.copy()
        # Normalize weights
        total_weight = sum(self.feature_weights.values())
        if total_weight > 0:
            for key in self.feature_weights:
                self.feature_weights[key] /= total_weight
    
    def compute_weighted_similarity(self, item1_features, item2_features):
        """
        Compute weighted similarity across different feature types
        
        Args:
            item1_features: Dictionary of feature_type -> feature_vector
            item2_features: Dictionary of feature_type -> feature_vector
            
        Returns:
            Weighted similarity score
        """
        total_similarity = 0.0
        total_weight = 0.0
        
        for feature_type in item1_features:
            if feature_type in item2_features and feature_type in self.feature_weights:
                weight = self.feature_weights[feature_type]
                
                # Choose appropriate similarity metric based on feature type
                if feature_type == 'text':
                    similarity = self.similarity_calculator.cosine_similarity(
                        item1_features[feature_type], 
                        item2_features[feature_type]
                    )
                elif feature_type == 'categorical':
                    similarity = self.similarity_calculator.jaccard_similarity(
                        item1_features[feature_type], 
                        item2_features[feature_type]
                    )
                elif feature_type == 'numerical':
                    similarity = self.similarity_calculator.euclidean_similarity(
                        item1_features[feature_type], 
                        item2_features[feature_type]
                    )
                elif feature_type == 'ratings':
                    similarity = self.similarity_calculator.pearson_correlation(
                        item1_features[feature_type], 
                        item2_features[feature_type]
                    )
                else:
                    similarity = self.similarity_calculator.cosine_similarity(
                        item1_features[feature_type], 
                        item2_features[feature_type]
                    )
                
                total_similarity += weight * similarity
                total_weight += weight
        
        return total_similarity / total_weight if total_weight > 0 else 0.0
```

## 2. Locality-Sensitive Hashing (LSH)

### Scalable Similarity Search

```python
import random
import hashlib
from collections import defaultdict

class LSHIndex:
    """
    Locality-Sensitive Hashing for approximate similarity search
    """
    
    def __init__(self, dimension, num_hashes=10, num_bands=5):
        self.dimension = dimension
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        
        # Generate random hash functions
        self.hash_functions = []
        random.seed(42)
        
        for _ in range(num_hashes):
            # Random hyperplane for cosine similarity LSH
            random_vector = np.random.normal(0, 1, dimension)
            random_vector = random_vector / np.linalg.norm(random_vector)
            self.hash_functions.append(random_vector)
        
        # Hash tables for each band
        self.hash_tables = [defaultdict(list) for _ in range(num_bands)]
        self.item_signatures = {}
        
    def _compute_signature(self, vector):
        """Compute LSH signature for a vector"""
        signature = []
        for hash_func in self.hash_functions:
            # Hash based on side of hyperplane
            hash_value = 1 if np.dot(vector, hash_func) >= 0 else 0
            signature.append(hash_value)
        return signature
    
    def _hash_signature_bands(self, signature):
        """Hash signature into bands"""
        band_hashes = []
        for band_idx in range(self.num_bands):
            start_idx = band_idx * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            band = tuple(signature[start_idx:end_idx])
            
            # Hash the band
            band_hash = hashlib.md5(str(band).encode()).hexdigest()
            band_hashes.append(band_hash)
        
        return band_hashes
    
    def add_item(self, item_id, vector):
        """Add item to LSH index"""
        # Compute signature
        signature = self._compute_signature(vector)
        self.item_signatures[item_id] = signature
        
        # Hash into bands and add to hash tables
        band_hashes = self._hash_signature_bands(signature)
        for band_idx, band_hash in enumerate(band_hashes):
            self.hash_tables[band_idx][band_hash].append(item_id)
    
    def find_candidates(self, query_vector, min_bands=1):
        """Find candidate similar items"""
        # Compute query signature
        query_signature = self._compute_signature(query_vector)
        query_band_hashes = self._hash_signature_bands(query_signature)
        
        # Find candidates from hash tables
        candidates = set()
        band_matches = defaultdict(int)
        
        for band_idx, band_hash in enumerate(query_band_hashes):
            if band_hash in self.hash_tables[band_idx]:
                for item_id in self.hash_tables[band_idx][band_hash]:
                    band_matches[item_id] += 1
        
        # Filter candidates by minimum band matches
        for item_id, match_count in band_matches.items():
            if match_count >= min_bands:
                candidates.add(item_id)
        
        return list(candidates)
    
    def estimate_similarity(self, item1_id, item2_id):
        """Estimate similarity between two items using signatures"""
        if item1_id not in self.item_signatures or item2_id not in self.item_signatures:
            return 0.0
        
        sig1 = self.item_signatures[item1_id]
        sig2 = self.item_signatures[item2_id]
        
        # Hamming similarity of signatures
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

class MinHashLSH:
    """
    MinHash LSH for Jaccard similarity of sets
    """
    
    def __init__(self, num_perm=128, num_bands=16):
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        
        # Generate hash functions
        self.hash_funcs = []
        random.seed(42)
        
        # Large prime numbers for hash functions
        self.p = 2**31 - 1
        
        for _ in range(num_perm):
            a = random.randint(1, self.p - 1)
            b = random.randint(0, self.p - 1)
            self.hash_funcs.append((a, b))
        
        self.hash_tables = [defaultdict(list) for _ in range(num_bands)]
        self.item_signatures = {}
    
    def _compute_minhash_signature(self, item_set):
        """Compute MinHash signature for a set"""
        signature = []
        
        for a, b in self.hash_funcs:
            min_hash = float('inf')
            
            for item in item_set:
                # Hash the item
                item_hash = hash(item)
                hash_value = (a * item_hash + b) % self.p
                min_hash = min(min_hash, hash_value)
            
            signature.append(min_hash)
        
        return signature
    
    def add_item(self, item_id, item_set):
        """Add set to MinHash LSH index"""
        signature = self._compute_minhash_signature(item_set)
        self.item_signatures[item_id] = signature
        
        # Hash signature into bands
        for band_idx in range(self.num_bands):
            start_idx = band_idx * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            band = tuple(signature[start_idx:end_idx])
            
            band_hash = hashlib.md5(str(band).encode()).hexdigest()
            self.hash_tables[band_idx][band_hash].append(item_id)
    
    def find_similar_items(self, query_set, threshold=0.5):
        """Find items similar to query set"""
        query_signature = self._compute_minhash_signature(query_set)
        
        # Find candidates
        candidates = set()
        for band_idx in range(self.num_bands):
            start_idx = band_idx * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            band = tuple(query_signature[start_idx:end_idx])
            
            band_hash = hashlib.md5(str(band).encode()).hexdigest()
            if band_hash in self.hash_tables[band_idx]:
                candidates.update(self.hash_tables[band_idx][band_hash])
        
        # Estimate Jaccard similarity for candidates
        similar_items = []
        for candidate_id in candidates:
            similarity = self._estimate_jaccard_similarity(
                query_signature, 
                self.item_signatures[candidate_id]
            )
            if similarity >= threshold:
                similar_items.append((candidate_id, similarity))
        
        # Sort by similarity
        similar_items.sort(key=lambda x: x[1], reverse=True)
        return similar_items
    
    def _estimate_jaccard_similarity(self, sig1, sig2):
        """Estimate Jaccard similarity from MinHash signatures"""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

class ScalableSimilaritySearch:
    """
    Scalable similarity search system combining multiple LSH techniques
    """
    
    def __init__(self, feature_types):
        self.feature_types = feature_types
        self.lsh_indexes = {}
        self.items = {}
        
        # Initialize appropriate LSH for each feature type
        for feature_type, config in feature_types.items():
            if config['type'] == 'dense_vector':
                self.lsh_indexes[feature_type] = LSHIndex(
                    dimension=config['dimension'],
                    num_hashes=config.get('num_hashes', 20),
                    num_bands=config.get('num_bands', 10)
                )
            elif config['type'] == 'set':
                self.lsh_indexes[feature_type] = MinHashLSH(
                    num_perm=config.get('num_perm', 128),
                    num_bands=config.get('num_bands', 16)
                )
    
    def add_item(self, item_id, features):
        """Add item with multiple feature types"""
        self.items[item_id] = features
        
        for feature_type, feature_value in features.items():
            if feature_type in self.lsh_indexes:
                if self.feature_types[feature_type]['type'] == 'dense_vector':
                    self.lsh_indexes[feature_type].add_item(item_id, feature_value)
                elif self.feature_types[feature_type]['type'] == 'set':
                    self.lsh_indexes[feature_type].add_item(item_id, feature_value)
    
    def find_similar_items(self, query_features, k=10, combine_method='union'):
        """Find similar items across multiple feature types"""
        all_candidates = {}
        
        # Get candidates from each feature type
        for feature_type, feature_value in query_features.items():
            if feature_type in self.lsh_indexes:
                if self.feature_types[feature_type]['type'] == 'dense_vector':
                    candidates = self.lsh_indexes[feature_type].find_candidates(feature_value)
                elif self.feature_types[feature_type]['type'] == 'set':
                    candidates = [item_id for item_id, _ in 
                                self.lsh_indexes[feature_type].find_similar_items(feature_value, threshold=0.3)]
                else:
                    candidates = []
                
                # Weight candidates by feature type
                weight = self.feature_types[feature_type].get('weight', 1.0)
                for candidate in candidates:
                    if candidate in all_candidates:
                        all_candidates[candidate] += weight
                    else:
                        all_candidates[candidate] = weight
        
        # Combine candidates based on method
        if combine_method == 'union':
            final_candidates = list(all_candidates.keys())
        elif combine_method == 'intersection':
            # Only items found in multiple feature types
            min_features = max(1, len(query_features) // 2)
            final_candidates = [item_id for item_id, count in all_candidates.items() 
                              if count >= min_features]
        else:
            final_candidates = list(all_candidates.keys())
        
        # Compute exact similarities for final ranking
        similarities = []
        similarity_calc = WeightedSimilarityCalculator()
        
        # Set weights based on feature types
        weights = {ft: config.get('weight', 1.0) for ft, config in self.feature_types.items()}
        similarity_calc.set_feature_weights(weights)
        
        for candidate_id in final_candidates:
            if candidate_id in self.items:
                similarity = similarity_calc.compute_weighted_similarity(
                    query_features, 
                    self.items[candidate_id]
                )
                similarities.append((candidate_id, similarity))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
```

## 3. Graph-Based Similarity

### Content Networks and Graph Similarity

```python
import networkx as nx
from collections import defaultdict, deque
import numpy as np

class ContentGraph:
    """
    Graph-based similarity for content networks
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.node_features = {}
        self.similarity_cache = {}
        
    def add_content_item(self, item_id, features=None):
        """Add content item as node"""
        self.graph.add_node(item_id)
        if features:
            self.node_features[item_id] = features
    
    def add_relationship(self, item1, item2, weight=1.0, relationship_type='similar'):
        """Add relationship between content items"""
        self.graph.add_edge(item1, item2, weight=weight, type=relationship_type)
    
    def build_content_graph(self, items, similarity_threshold=0.5):
        """
        Build content graph based on feature similarity
        
        Args:
            items: Dictionary of item_id -> features
            similarity_threshold: Minimum similarity to create edge
        """
        similarity_calc = AdvancedSimilarityMetrics()
        
        # Add all items as nodes
        for item_id, features in items.items():
            self.add_content_item(item_id, features)
        
        # Compute similarities and add edges
        item_ids = list(items.keys())
        for i, item1 in enumerate(item_ids):
            for j, item2 in enumerate(item_ids[i+1:], i+1):
                # Compute similarity between items
                similarity = similarity_calc.cosine_similarity(
                    items[item1], items[item2]
                )
                
                if similarity >= similarity_threshold:
                    self.add_relationship(item1, item2, weight=similarity)
    
    def compute_graph_similarity(self, item1, item2, method='path_based'):
        """
        Compute graph-based similarity between items
        
        Args:
            item1, item2: Item IDs
            method: 'path_based', 'random_walk', 'structural'
            
        Returns:
            Graph similarity score
        """
        cache_key = f"{item1}_{item2}_{method}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        if item1 not in self.graph or item2 not in self.graph:
            return 0.0
        
        if method == 'path_based':
            similarity = self._path_based_similarity(item1, item2)
        elif method == 'random_walk':
            similarity = self._random_walk_similarity(item1, item2)
        elif method == 'structural':
            similarity = self._structural_similarity(item1, item2)
        else:
            similarity = 0.0
        
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def _path_based_similarity(self, item1, item2):
        """Compute similarity based on shortest path"""
        try:
            shortest_path = nx.shortest_path_length(
                self.graph, item1, item2, weight='weight'
            )
            # Convert distance to similarity
            max_distance = 5.0  # Assume max meaningful distance
            similarity = max(0, 1.0 - (shortest_path / max_distance))
            return similarity
        except nx.NetworkXNoPath:
            return 0.0
    
    def _random_walk_similarity(self, item1, item2, walk_length=10, num_walks=100):
        """Compute similarity using random walks"""
        if item1 == item2:
            return 1.0
        
        # Count how often item2 is visited when starting from item1
        visit_count = 0
        
        for _ in range(num_walks):
            current_node = item1
            
            for step in range(walk_length):
                neighbors = list(self.graph.neighbors(current_node))
                if not neighbors:
                    break
                
                # Choose next node based on edge weights
                weights = [self.graph[current_node][neighbor].get('weight', 1.0) 
                          for neighbor in neighbors]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    probabilities = [w/total_weight for w in weights]
                    current_node = np.random.choice(neighbors, p=probabilities)
                else:
                    current_node = np.random.choice(neighbors)
                
                if current_node == item2:
                    visit_count += 1
                    break
        
        return visit_count / num_walks
    
    def _structural_similarity(self, item1, item2):
        """Compute structural similarity based on common neighbors"""
        neighbors1 = set(self.graph.neighbors(item1))
        neighbors2 = set(self.graph.neighbors(item2))
        
        if not neighbors1 and not neighbors2:
            return 0.0
        
        # Jaccard similarity of neighborhoods
        intersection = len(neighbors1.intersection(neighbors2))
        union = len(neighbors1.union(neighbors2))
        
        return intersection / union if union > 0 else 0.0
    
    def find_similar_items_graph(self, query_item, k=10, method='combined'):
        """Find similar items using graph-based methods"""
        if query_item not in self.graph:
            return []
        
        similarities = []
        
        for item in self.graph.nodes():
            if item != query_item:
                if method == 'combined':
                    # Combine multiple similarity measures
                    path_sim = self.compute_graph_similarity(query_item, item, 'path_based')
                    struct_sim = self.compute_graph_similarity(query_item, item, 'structural')
                    combined_sim = 0.6 * path_sim + 0.4 * struct_sim
                    similarities.append((item, combined_sim))
                else:
                    similarity = self.compute_graph_similarity(query_item, item, method)
                    similarities.append((item, similarity))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def compute_centrality_scores(self):
        """Compute various centrality scores for content items"""
        centrality_scores = {}
        
        # Degree centrality
        centrality_scores['degree'] = nx.degree_centrality(self.graph)
        
        # Betweenness centrality
        centrality_scores['betweenness'] = nx.betweenness_centrality(self.graph)
        
        # Closeness centrality
        centrality_scores['closeness'] = nx.closeness_centrality(self.graph)
        
        # PageRank
        centrality_scores['pagerank'] = nx.pagerank(self.graph)
        
        return centrality_scores
    
    def detect_content_communities(self, method='louvain'):
        """Detect communities in content graph"""
        if method == 'louvain':
            # Louvain community detection
            import community as community_louvain
            communities = community_louvain.best_partition(self.graph)
        else:
            # Simple connected components
            communities = {}
            for i, component in enumerate(nx.connected_components(self.graph)):
                for node in component:
                    communities[node] = i
        
        return communities

class MultiModalSimilarity:
    """
    Multi-modal content similarity handling different content types
    """
    
    def __init__(self):
        self.modality_weights = {}
        self.similarity_calculators = {}
        self.cross_modal_mappings = {}
        
    def register_modality(self, modality_name, similarity_calculator, weight=1.0):
        """Register a content modality with its similarity calculator"""
        self.similarity_calculators[modality_name] = similarity_calculator
        self.modality_weights[modality_name] = weight
    
    def add_cross_modal_mapping(self, modality1, modality2, mapping_function):
        """Add cross-modal mapping function"""
        key = f"{modality1}_{modality2}"
        self.cross_modal_mappings[key] = mapping_function
    
    def compute_multimodal_similarity(self, item1_features, item2_features):
        """
        Compute similarity across multiple modalities
        
        Args:
            item1_features: Dictionary of modality -> features
            item2_features: Dictionary of modality -> features
            
        Returns:
            Combined multimodal similarity
        """
        total_similarity = 0.0
        total_weight = 0.0
        
        # Within-modality similarities
        for modality in item1_features:
            if modality in item2_features and modality in self.similarity_calculators:
                calculator = self.similarity_calculators[modality]
                weight = self.modality_weights[modality]
                
                similarity = calculator.compute_similarity(
                    item1_features[modality], 
                    item2_features[modality]
                )
                
                total_similarity += weight * similarity
                total_weight += weight
        
        # Cross-modality similarities
        for modality1 in item1_features:
            for modality2 in item2_features:
                if modality1 != modality2:
                    mapping_key = f"{modality1}_{modality2}"
                    reverse_key = f"{modality2}_{modality1}"
                    
                    if mapping_key in self.cross_modal_mappings:
                        # Map modality1 to modality2 space and compare
                        mapped_features = self.cross_modal_mappings[mapping_key](
                            item1_features[modality1]
                        )
                        
                        if modality2 in self.similarity_calculators:
                            calculator = self.similarity_calculators[modality2]
                            cross_similarity = calculator.compute_similarity(
                                mapped_features, 
                                item2_features[modality2]
                            )
                            
                            # Use lower weight for cross-modal similarity
                            cross_weight = 0.3 * self.modality_weights.get(modality2, 1.0)
                            total_similarity += cross_weight * cross_similarity
                            total_weight += cross_weight
        
        return total_similarity / total_weight if total_weight > 0 else 0.0

class TextSimilarityCalculator:
    """Text-specific similarity calculator"""
    
    def __init__(self):
        self.similarity_metrics = AdvancedSimilarityMetrics()
    
    def compute_similarity(self, text_features1, text_features2):
        """Compute text similarity"""
        return self.similarity_metrics.cosine_similarity(text_features1, text_features2)

class ImageSimilarityCalculator:
    """Image-specific similarity calculator (placeholder)"""
    
    def compute_similarity(self, image_features1, image_features2):
        """Compute image similarity"""
        # Placeholder - would use CNN features, SIFT, etc.
        similarity_calc = AdvancedSimilarityMetrics()
        return similarity_calc.euclidean_similarity(image_features1, image_features2)
```

## 4. Real-Time Similarity Computation

### Efficient Real-Time Systems

```python
import threading
import time
from collections import OrderedDict
import heapq
from typing import List, Tuple, Dict, Any

class RealTimeSimilarityEngine:
    """
    Real-time similarity computation with caching and optimization
    """
    
    def __init__(self, cache_size=10000, update_interval=60):
        self.cache_size = cache_size
        self.update_interval = update_interval
        
        # Similarity cache with LRU eviction
        self.similarity_cache = OrderedDict()
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        # Item features and metadata
        self.item_features = {}
        self.item_metadata = {}
        self.last_updated = {}
        
        # Similarity calculators
        self.similarity_calculator = AdvancedSimilarityMetrics()
        self.lsh_index = None
        
        # Threading for background updates
        self.update_thread = None
        self.stop_updates = False
        self.lock = threading.RLock()
        
        # Start background update thread
        self.start_background_updates()
    
    def add_item(self, item_id, features, metadata=None):
        """Add or update item features"""
        with self.lock:
            self.item_features[item_id] = features
            self.item_metadata[item_id] = metadata or {}
            self.last_updated[item_id] = time.time()
            
            # Clear related cached similarities
            self._invalidate_cache_for_item(item_id)
    
    def remove_item(self, item_id):
        """Remove item from system"""
        with self.lock:
            if item_id in self.item_features:
                del self.item_features[item_id]
            if item_id in self.item_metadata:
                del self.item_metadata[item_id]
            if item_id in self.last_updated:
                del self.last_updated[item_id]
            
            self._invalidate_cache_for_item(item_id)
    
    def compute_similarity(self, item1_id, item2_id, use_cache=True):
        """
        Compute similarity between two items
        
        Args:
            item1_id, item2_id: Item identifiers
            use_cache: Whether to use cached results
            
        Returns:
            Similarity score
        """
        if item1_id == item2_id:
            return 1.0
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(item1_id, item2_id)
            if cache_key in self.similarity_cache:
                # Move to end (LRU)
                self.similarity_cache.move_to_end(cache_key)
                self.cache_stats['hits'] += 1
                return self.similarity_cache[cache_key]
        
        # Compute similarity
        with self.lock:
            if item1_id not in self.item_features or item2_id not in self.item_features:
                return 0.0
            
            features1 = self.item_features[item1_id]
            features2 = self.item_features[item2_id]
            
            similarity = self.similarity_calculator.cosine_similarity(features1, features2)
        
        # Cache result
        if use_cache:
            self._cache_similarity(item1_id, item2_id, similarity)
            self.cache_stats['misses'] += 1
        
        return similarity
    
    def find_similar_items(self, query_item_id, k=10, use_lsh=True):
        """
        Find k most similar items to query item
        
        Args:
            query_item_id: Query item ID
            k: Number of similar items to return
            use_lsh: Whether to use LSH for candidate generation
            
        Returns:
            List of (item_id, similarity) tuples
        """
        if query_item_id not in self.item_features:
            return []
        
        query_features = self.item_features[query_item_id]
        
        if use_lsh and self.lsh_index:
            # Use LSH for candidate generation
            candidates = self.lsh_index.find_candidates(query_features)
            
            # Limit candidates for efficiency
            if len(candidates) > k * 10:
                candidates = candidates[:k * 10]
        else:
            # Use all items as candidates
            candidates = [item_id for item_id in self.item_features.keys() 
                         if item_id != query_item_id]
        
        # Compute exact similarities for candidates
        similarities = []
        for candidate_id in candidates:
            similarity = self.compute_similarity(query_item_id, candidate_id)
            similarities.append((candidate_id, similarity))
        
        # Return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def batch_similarity_computation(self, item_pairs):
        """
        Compute similarities for multiple item pairs efficiently
        
        Args:
            item_pairs: List of (item1_id, item2_id) tuples
            
        Returns:
            List of similarity scores
        """
        similarities = []
        
        # Group by feature availability
        valid_pairs = []
        for item1_id, item2_id in item_pairs:
            if item1_id in self.item_features and item2_id in self.item_features:
                valid_pairs.append((item1_id, item2_id))
            else:
                similarities.append(0.0)
        
        if not valid_pairs:
            return similarities
        
        # Batch compute similarities
        with self.lock:
            for item1_id, item2_id in valid_pairs:
                # Check cache first
                cache_key = self._get_cache_key(item1_id, item2_id)
                if cache_key in self.similarity_cache:
                    similarity = self.similarity_cache[cache_key]
                    self.similarity_cache.move_to_end(cache_key)
                else:
                    # Compute similarity
                    features1 = self.item_features[item1_id]
                    features2 = self.item_features[item2_id]
                    similarity = self.similarity_calculator.cosine_similarity(features1, features2)
                    
                    # Cache result
                    self._cache_similarity(item1_id, item2_id, similarity)
                
                similarities.append(similarity)
        
        return similarities
    
    def update_lsh_index(self):
        """Update LSH index with current items"""
        if not self.item_features:
            return
        
        # Determine feature dimension
        sample_features = next(iter(self.item_features.values()))
        dimension = len(sample_features)
        
        # Create new LSH index
        self.lsh_index = LSHIndex(dimension=dimension, num_hashes=20, num_bands=10)
        
        # Add all items
        with self.lock:
            for item_id, features in self.item_features.items():
                self.lsh_index.add_item(item_id, features)
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'cache_size': len(self.similarity_cache)
        }
    
    def start_background_updates(self):
        """Start background thread for periodic updates"""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread = threading.Thread(target=self._background_update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
    
    def stop_background_updates(self):
        """Stop background updates"""
        self.stop_updates = True
        if self.update_thread:
            self.update_thread.join()
    
    def _background_update_loop(self):
        """Background thread for periodic maintenance"""
        while not self.stop_updates:
            try:
                # Update LSH index periodically
                self.update_lsh_index()
                
                # Clean old cache entries
                self._cleanup_cache()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Background update error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _get_cache_key(self, item1_id, item2_id):
        """Generate cache key for item pair"""
        # Ensure consistent ordering
        if item1_id < item2_id:
            return f"{item1_id}_{item2_id}"
        else:
            return f"{item2_id}_{item1_id}"
    
    def _cache_similarity(self, item1_id, item2_id, similarity):
        """Cache similarity result with LRU eviction"""
        cache_key = self._get_cache_key(item1_id, item2_id)
        
        # Add to cache
        self.similarity_cache[cache_key] = similarity
        
        # LRU eviction
        if len(self.similarity_cache) > self.cache_size:
            # Remove oldest entry
            self.similarity_cache.popitem(last=False)
    
    def _invalidate_cache_for_item(self, item_id):
        """Remove cached similarities involving specific item"""
        keys_to_remove = []
        
        for cache_key in self.similarity_cache:
            if item_id in cache_key.split('_'):
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            del self.similarity_cache[key]
    
    def _cleanup_cache(self):
        """Clean up old or invalid cache entries"""
        current_time = time.time()
        cache_ttl = 3600  # 1 hour TTL
        
        # Remove entries older than TTL (simplified - would track timestamps)
        if len(self.similarity_cache) > self.cache_size * 0.8:
            # Remove 20% of oldest entries
            remove_count = int(len(self.similarity_cache) * 0.2)
            for _ in range(remove_count):
                if self.similarity_cache:
                    self.similarity_cache.popitem(last=False)
```

## 5. Study Questions

### Beginner Level

1. What are the advantages and disadvantages of different similarity metrics?
2. How does Locality-Sensitive Hashing approximate similarity search?
3. What is the difference between MinHash and standard LSH?
4. How can graph-based similarity complement feature-based similarity?
5. Why is caching important in real-time similarity systems?

### Intermediate Level

6. Implement a similarity system that combines multiple metrics with learned weights.
7. How would you choose the optimal LSH parameters for different data distributions?
8. Design a graph-based similarity system for a content recommendation scenario.
9. What are the trade-offs between accuracy and speed in similarity search?
10. How would you handle dynamic updates in an LSH-based similarity system?

### Advanced Level

11. Implement a multi-modal similarity system that can handle text, images, and numerical features.
12. Design a distributed similarity computation system for large-scale content collections.
13. How would you adapt similarity metrics based on user feedback and recommendation performance?
14. Implement a similarity system that can learn from implicit user interactions.
15. Design a similarity search system that can handle streaming content updates.

### Tricky Questions

16. How would you detect and handle similarity metric degradation in a production system?
17. Design a similarity system that can work across different content domains while maintaining accuracy.
18. How would you implement approximate similarity search with guaranteed error bounds?
19. Design a similarity system that can adapt to changing user preferences and content distributions over time.
20. How would you implement fair similarity computation that avoids bias toward popular or certain types of content?

## Key Takeaways

1. **Multiple similarity metrics** serve different purposes and data types
2. **LSH techniques** enable scalable approximate similarity search
3. **Graph-based methods** capture complex content relationships
4. **Multi-modal approaches** handle diverse content types effectively
5. **Caching and optimization** are crucial for real-time performance
6. **Adaptive systems** can learn and improve similarity computation
7. **Trade-offs** exist between accuracy, speed, and resource consumption

## Next Session Preview

In Day 4.5, we'll explore **Knowledge-Based Recommendation Systems**, covering:
- Ontology-based content representation
- Rule-based recommendation engines
- Constraint satisfaction in recommendations
- Expert system approaches
- Knowledge graph integration