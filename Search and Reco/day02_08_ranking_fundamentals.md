# Day 2: Relevance Ranking Fundamentals

## Table of Contents
1. [Introduction to Relevance Ranking](#introduction)
2. [Probability Ranking Principle](#probability-ranking-principle)
3. [Vector Space Model](#vector-space-model)
4. [Language Models for IR](#language-models)
5. [Learning to Rank](#learning-to-rank)
6. [Query-Document Matching](#query-document-matching)
7. [Study Questions](#study-questions)
8. [Code Examples](#code-examples)

---

## Introduction to Relevance Ranking

Relevance ranking determines the order of search results, transforming information retrieval from exact matching to ranked retrieval based on relevance scores.

### The Ranking Problem

#### **From Boolean to Ranked Retrieval**
- **Boolean Search**: Binary relevance (match/no match)
- **Ranked Retrieval**: Continuous relevance scores
- **User Expectation**: Most relevant results first
- **System Challenge**: Define and compute relevance

#### **What Makes a Document Relevant?**
```python
# Factors influencing relevance:
relevance_factors = {
    'query_term_frequency': 'How often query terms appear in document',
    'term_discrimination': 'How unique/rare the query terms are',
    'document_length': 'Length normalization considerations',
    'term_proximity': 'How close query terms appear together',
    'document_quality': 'Authority, freshness, popularity',
    'user_context': 'Location, search history, preferences'
}
```

### Theoretical Foundations

#### **Information Need vs Query**
- **Information Need**: User's underlying problem/question
- **Query**: Formal expression of information need
- **Relevance**: How well document satisfies information need
- **Topical Relevance**: Document discusses query topic
- **User Relevance**: Document useful to specific user

#### **Relevance Judgments**
```python
class RelevanceScale:
    HIGHLY_RELEVANT = 3    # Perfect match, exactly what user wants
    RELEVANT = 2           # Good match, useful information
    PARTIALLY_RELEVANT = 1 # Some useful information
    NOT_RELEVANT = 0       # No useful information
```

---

## Probability Ranking Principle

Foundation of modern ranking algorithms based on probability theory.

### Robertson's PRP

#### **Core Principle**
"If a reference retrieval system's response to each request is a ranking of the documents in the collection in order of decreasing probability of relevance to the user who submitted the request, where the probabilities are estimated as accurately as possible on the basis of whatever data have been made available to the system for this purpose, the overall effectiveness of the system to its user will be the best that is obtainable on the basis of those data."

#### **Mathematical Formulation**
For documents d₁, d₂, ..., dₙ and query q:
```
Rank documents by P(relevant|q, dᵢ) in decreasing order
```

### Binary Independence Model

#### **Assumptions**
1. **Binary representation**: Documents and queries as binary vectors
2. **Term independence**: Terms occur independently
3. **Relevance independence**: Documents are relevant independently

#### **Scoring Formula**
```python
def binary_independence_score(query_terms, document_terms, collection_stats):
    """Calculate BIM score for document given query"""
    score = 0
    
    for term in query_terms:
        if term in document_terms:
            # Term present in document
            p_term_given_relevant = collection_stats.get_p_term_relevant(term)
            p_term_given_not_relevant = collection_stats.get_p_term_not_relevant(term)
            
            if p_term_given_not_relevant > 0:
                score += math.log(
                    (p_term_given_relevant * (1 - p_term_given_not_relevant)) /
                    (p_term_given_not_relevant * (1 - p_term_given_relevant))
                )
        else:
            # Term absent from document
            p_term_given_relevant = collection_stats.get_p_term_relevant(term)
            p_term_given_not_relevant = collection_stats.get_p_term_not_relevant(term)
            
            if p_term_given_relevant < 1:
                score += math.log(
                    ((1 - p_term_given_relevant) * p_term_given_not_relevant) /
                    (p_term_given_relevant * (1 - p_term_given_not_relevant))
                )
    
    return score
```

### Robertson-Sparck Jones Weight

#### **RSJ Weight Derivation**
The RSJ weight for term t is:
```
w(t) = log((r + 0.5)/(R - r + 0.5)) - log((n - r + 0.5)/(N - n - R + r + 0.5))

where:
r = number of relevant documents containing term t
R = total number of relevant documents  
n = number of documents containing term t
N = total number of documents
```

#### **Implementation**
```python
class RSJWeighting:
    def __init__(self, collection_size):
        self.N = collection_size
        self.term_doc_freq = {}
        self.relevant_docs = set()
        self.relevant_term_freq = {}
    
    def add_relevance_judgments(self, relevant_doc_ids, document_terms):
        """Add relevance judgments for RSJ weight calculation"""
        self.relevant_docs.update(relevant_doc_ids)
        R = len(self.relevant_docs)
        
        # Count relevant documents containing each term
        for doc_id in relevant_doc_ids:
            if doc_id in document_terms:
                for term in document_terms[doc_id]:
                    if term not in self.relevant_term_freq:
                        self.relevant_term_freq[term] = 0
                    self.relevant_term_freq[term] += 1
    
    def calculate_rsj_weight(self, term):
        """Calculate RSJ weight for a term"""
        n = self.term_doc_freq.get(term, 0)  # docs containing term
        R = len(self.relevant_docs)          # total relevant docs
        r = self.relevant_term_freq.get(term, 0)  # relevant docs containing term
        
        # Add smoothing to avoid division by zero
        numerator = (r + 0.5) / (R - r + 0.5)
        denominator = (n - r + 0.5) / (self.N - n - R + r + 0.5)
        
        if denominator > 0:
            return math.log(numerator / denominator)
        else:
            return 0.0
```

---

## Vector Space Model

Geometric approach to information retrieval using vector algebra.

### Mathematical Foundation

#### **Document and Query Vectors**
```python
# Documents and queries as vectors in term space
# d⃗ = (w₁,d, w₂,d, w₃,d, ..., wₙ,d)
# q⃗ = (w₁,q, w₂,q, w₃,q, ..., wₙ,q)
# 
# where wᵢ,d = weight of term i in document d
#       wᵢ,q = weight of term i in query q
```

#### **Similarity Calculation**
```python
def cosine_similarity(doc_vector, query_vector):
    """Calculate cosine similarity between document and query vectors"""
    
    # Dot product
    dot_product = sum(d * q for d, q in zip(doc_vector, query_vector))
    
    # Vector magnitudes
    doc_magnitude = math.sqrt(sum(d * d for d in doc_vector))
    query_magnitude = math.sqrt(sum(q * q for q in query_vector))
    
    # Cosine similarity
    if doc_magnitude == 0 or query_magnitude == 0:
        return 0.0
    
    return dot_product / (doc_magnitude * query_magnitude)
```

### Term Weighting Schemes

#### **TF-IDF Variants**
```python
class TermWeightingSchemes:
    @staticmethod
    def tf_raw(term_freq):
        """Raw term frequency"""
        return term_freq
    
    @staticmethod
    def tf_log(term_freq):
        """Logarithmic term frequency"""
        return 1 + math.log(term_freq) if term_freq > 0 else 0
    
    @staticmethod
    def tf_augmented(term_freq, max_tf):
        """Augmented term frequency"""
        return 0.5 + 0.5 * (term_freq / max_tf) if max_tf > 0 else 0
    
    @staticmethod
    def tf_boolean(term_freq):
        """Boolean term frequency"""
        return 1 if term_freq > 0 else 0
    
    @staticmethod
    def idf_standard(doc_freq, total_docs):
        """Standard inverse document frequency"""
        return math.log(total_docs / doc_freq) if doc_freq > 0 else 0
    
    @staticmethod
    def idf_smooth(doc_freq, total_docs):
        """Smooth IDF"""
        return math.log(total_docs / (1 + doc_freq))
    
    @staticmethod
    def idf_probabilistic(doc_freq, total_docs):
        """Probabilistic IDF"""
        return math.log((total_docs - doc_freq) / doc_freq) if doc_freq > 0 else 0

class VectorSpaceModel:
    def __init__(self, tf_scheme='log', idf_scheme='standard', normalize=True):
        self.tf_scheme = tf_scheme
        self.idf_scheme = idf_scheme
        self.normalize = normalize
        self.weighting = TermWeightingSchemes()
    
    def calculate_term_weight(self, tf, df, total_docs, max_tf=None):
        """Calculate term weight using specified schemes"""
        
        # Calculate TF component
        if self.tf_scheme == 'raw':
            tf_weight = self.weighting.tf_raw(tf)
        elif self.tf_scheme == 'log':
            tf_weight = self.weighting.tf_log(tf)
        elif self.tf_scheme == 'augmented':
            tf_weight = self.weighting.tf_augmented(tf, max_tf)
        elif self.tf_scheme == 'boolean':
            tf_weight = self.weighting.tf_boolean(tf)
        else:
            tf_weight = tf
        
        # Calculate IDF component
        if self.idf_scheme == 'standard':
            idf_weight = self.weighting.idf_standard(df, total_docs)
        elif self.idf_scheme == 'smooth':
            idf_weight = self.weighting.idf_smooth(df, total_docs)
        elif self.idf_scheme == 'probabilistic':
            idf_weight = self.weighting.idf_probabilistic(df, total_docs)
        else:
            idf_weight = 1.0
        
        return tf_weight * idf_weight
```

### Vector Normalization

#### **L2 Normalization**
```python
def l2_normalize_vector(vector):
    """Normalize vector using L2 norm"""
    magnitude = math.sqrt(sum(w * w for w in vector))
    
    if magnitude == 0:
        return vector
    
    return [w / magnitude for w in vector]

def cosine_normalized_similarity(doc_vector, query_vector):
    """Cosine similarity with pre-normalized vectors"""
    # If vectors are already L2 normalized, cosine similarity is just dot product
    return sum(d * q for d, q in zip(doc_vector, query_vector))
```

---

## Language Models for IR

Probabilistic approach using language modeling techniques.

### Query Likelihood Model

#### **Basic Concept**
Rank documents by probability of generating the query:
```
P(q|d) = ∏ P(t|d)^c(t,q)
```
where c(t,q) is count of term t in query q.

#### **Implementation**
```python
class QueryLikelihoodModel:
    def __init__(self, smoothing='dirichlet', mu=2000):
        self.smoothing = smoothing
        self.mu = mu  # Dirichlet prior parameter
    
    def calculate_term_probability(self, term, document, collection_model):
        """Calculate P(term|document) with smoothing"""
        
        tf_term_doc = document.get_term_frequency(term)
        doc_length = document.get_length()
        
        if self.smoothing == 'dirichlet':
            # Dirichlet smoothing
            p_term_collection = collection_model.get_term_probability(term)
            
            numerator = tf_term_doc + self.mu * p_term_collection
            denominator = doc_length + self.mu
            
            return numerator / denominator
        
        elif self.smoothing == 'jelinek_mercer':
            # Jelinek-Mercer smoothing
            lambda_param = 0.7  # mixing parameter
            p_term_doc_ml = tf_term_doc / doc_length if doc_length > 0 else 0
            p_term_collection = collection_model.get_term_probability(term)
            
            return lambda_param * p_term_doc_ml + (1 - lambda_param) * p_term_collection
        
        else:
            # Maximum likelihood (no smoothing)
            return tf_term_doc / doc_length if doc_length > 0 else 0
    
    def score_document(self, query_terms, document, collection_model):
        """Score document using query likelihood"""
        log_prob = 0.0
        
        for term in query_terms:
            p_term_doc = self.calculate_term_probability(term, document, collection_model)
            
            if p_term_doc > 0:
                log_prob += math.log(p_term_doc)
            else:
                # Handle zero probability (should not happen with smoothing)
                log_prob += float('-inf')
        
        return log_prob

class CollectionLanguageModel:
    def __init__(self, documents):
        self.term_counts = defaultdict(int)
        self.total_terms = 0
        
        # Build collection statistics
        for doc in documents:
            for term, count in doc.get_term_counts().items():
                self.term_counts[term] += count
                self.total_terms += count
    
    def get_term_probability(self, term):
        """Get maximum likelihood probability of term in collection"""
        return self.term_counts[term] / self.total_terms if self.total_terms > 0 else 0
```

### KL Divergence Model

#### **Divergence-Based Ranking**
```python
def kl_divergence_score(query_model, document_model):
    """Calculate KL divergence between query and document models"""
    kl_div = 0.0
    
    for term, q_prob in query_model.items():
        if q_prob > 0:
            d_prob = document_model.get(term, 1e-10)  # Small smoothing
            kl_div += q_prob * math.log(q_prob / d_prob)
    
    return -kl_div  # Negative because lower divergence = higher similarity
```

---

## Learning to Rank

Machine learning approaches to relevance ranking using training data.

### LTR Problem Formulation

#### **Training Data Structure**
```python
class LTRTrainingExample:
    def __init__(self, query_id, doc_id, features, relevance_label):
        self.query_id = query_id
        self.doc_id = doc_id
        self.features = features  # Feature vector
        self.relevance_label = relevance_label  # 0, 1, 2, 3 (not relevant to highly relevant)

class LTRDataset:
    def __init__(self):
        self.examples = []
        self.queries = defaultdict(list)  # query_id -> list of examples
    
    def add_example(self, example):
        self.examples.append(example)
        self.queries[example.query_id].append(example)
    
    def get_query_examples(self, query_id):
        return self.queries[query_id]
```

### Feature Engineering

#### **Common LTR Features**
```python
class LTRFeatureExtractor:
    def __init__(self, collection_stats):
        self.collection_stats = collection_stats
    
    def extract_features(self, query, document):
        """Extract comprehensive feature set for LTR"""
        features = {}
        
        # Basic TF-IDF features
        features['tf_idf_sum'] = self.calculate_tf_idf_sum(query, document)
        features['tf_idf_max'] = self.calculate_tf_idf_max(query, document)
        features['tf_idf_mean'] = self.calculate_tf_idf_mean(query, document)
        
        # BM25 score
        features['bm25_score'] = self.calculate_bm25(query, document)
        
        # Query-document matching features
        features['query_coverage'] = self.calculate_query_coverage(query, document)
        features['exact_match_count'] = self.count_exact_matches(query, document)
        
        # Document quality features
        features['doc_length'] = document.get_length()
        features['doc_length_normalized'] = document.get_length() / self.collection_stats.avg_doc_length
        
        # Term proximity features
        features['min_term_distance'] = self.calculate_min_term_distance(query, document)
        features['avg_term_distance'] = self.calculate_avg_term_distance(query, document)
        
        # Frequency-based features
        features['term_freq_sum'] = sum(document.get_term_frequency(term) for term in query.terms)
        features['term_freq_max'] = max((document.get_term_frequency(term) for term in query.terms), default=0)
        
        return list(features.values())  # Return as feature vector
    
    def calculate_query_coverage(self, query, document):
        """Fraction of query terms found in document"""
        matched_terms = sum(1 for term in query.terms if document.contains_term(term))
        return matched_terms / len(query.terms) if query.terms else 0
```

### Pointwise Learning to Rank

#### **Regression Approach**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

class PointwiseLTR:
    def __init__(self, model_type='random_forest'):
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, training_examples):
        """Train pointwise LTR model"""
        X = []  # Feature vectors
        y = []  # Relevance labels
        
        for example in training_examples:
            X.append(example.features)
            y.append(example.relevance_label)
        
        self.model.fit(X, y)
    
    def predict(self, features):
        """Predict relevance score for feature vector"""
        return self.model.predict([features])[0]
    
    def rank_documents(self, query_doc_features):
        """Rank documents by predicted relevance scores"""
        scored_docs = []
        
        for doc_id, features in query_doc_features.items():
            score = self.predict(features)
            scored_docs.append((doc_id, score))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs
```

### Pairwise Learning to Rank

#### **RankNet Approach**
```python
import torch
import torch.nn as nn

class RankNet(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(RankNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class PairwiseLTR:
    def __init__(self, feature_size):
        self.model = RankNet(feature_size)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def create_pairs(self, query_examples):
        """Create pairwise training examples"""
        pairs = []
        
        for i in range(len(query_examples)):
            for j in range(i + 1, len(query_examples)):
                doc1 = query_examples[i]
                doc2 = query_examples[j]
                
                if doc1.relevance_label != doc2.relevance_label:
                    # Create pair where first doc should rank higher
                    if doc1.relevance_label > doc2.relevance_label:
                        pairs.append((doc1.features, doc2.features, 1.0))
                    else:
                        pairs.append((doc2.features, doc1.features, 1.0))
        
        return pairs
    
    def train_epoch(self, training_queries):
        """Train one epoch using pairwise examples"""
        total_loss = 0.0
        
        for query_id, query_examples in training_queries.items():
            pairs = self.create_pairs(query_examples)
            
            for features1, features2, label in pairs:
                # Convert to tensors
                x1 = torch.FloatTensor(features1)
                x2 = torch.FloatTensor(features2)
                target = torch.FloatTensor([label])
                
                # Forward pass
                score1 = self.model(x1)
                score2 = self.model(x2)
                diff = score1 - score2
                
                # Calculate loss
                loss = self.criterion(diff, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss
```

---

## Query-Document Matching

Advanced techniques for matching queries to relevant documents.

### Semantic Matching

#### **Word Embedding Similarities**
```python
import numpy as np

class SemanticMatcher:
    def __init__(self, word_embeddings):
        self.embeddings = word_embeddings  # Dict: word -> vector
        self.embedding_dim = len(next(iter(word_embeddings.values())))
    
    def get_word_vector(self, word):
        """Get embedding vector for word"""
        return self.embeddings.get(word, np.zeros(self.embedding_dim))
    
    def document_embedding(self, document_terms):
        """Compute document embedding as average of word embeddings"""
        if not document_terms:
            return np.zeros(self.embedding_dim)
        
        doc_vector = np.zeros(self.embedding_dim)
        valid_words = 0
        
        for term in document_terms:
            word_vec = self.get_word_vector(term)
            if np.any(word_vec):  # Non-zero vector
                doc_vector += word_vec
                valid_words += 1
        
        if valid_words > 0:
            doc_vector /= valid_words
        
        return doc_vector
    
    def semantic_similarity(self, query_terms, document_terms):
        """Calculate semantic similarity using embeddings"""
        query_vec = self.document_embedding(query_terms)
        doc_vec = self.document_embedding(document_terms)
        
        # Cosine similarity
        query_norm = np.linalg.norm(query_vec)
        doc_norm = np.linalg.norm(doc_vec)
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        
        return np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
```

### Query Expansion

#### **Relevance Feedback**
```python
class RelevanceFeedback:
    def __init__(self, alpha=1.0, beta=0.75, gamma=0.15):
        self.alpha = alpha  # Original query weight
        self.beta = beta    # Relevant docs weight
        self.gamma = gamma  # Non-relevant docs weight
    
    def expand_query(self, original_query, relevant_docs, non_relevant_docs):
        """Rocchio algorithm for query expansion"""
        
        # Initialize expanded query with original
        expanded_query = {term: self.alpha * weight 
                         for term, weight in original_query.items()}
        
        # Add terms from relevant documents
        if relevant_docs:
            for doc in relevant_docs:
                doc_weight = self.beta / len(relevant_docs)
                for term, tf in doc.get_term_frequencies().items():
                    if term not in expanded_query:
                        expanded_query[term] = 0
                    expanded_query[term] += doc_weight * tf
        
        # Subtract terms from non-relevant documents
        if non_relevant_docs:
            for doc in non_relevant_docs:
                doc_weight = self.gamma / len(non_relevant_docs)
                for term, tf in doc.get_term_frequencies().items():
                    if term in expanded_query:
                        expanded_query[term] -= doc_weight * tf
                        # Ensure non-negative weights
                        expanded_query[term] = max(0, expanded_query[term])
        
        # Remove terms with zero weight
        expanded_query = {term: weight for term, weight in expanded_query.items() 
                         if weight > 0}
        
        return expanded_query
```

---

## Study Questions

### Beginner Level
1. What is the Probability Ranking Principle and why is it important?
2. How does the Vector Space Model represent documents and queries?
3. What are the main differences between TF-IDF and language models for ranking?
4. What is the purpose of query expansion in information retrieval?

### Intermediate Level
1. Compare pointwise, pairwise, and listwise approaches to Learning to Rank.
2. How does Dirichlet smoothing work in language models for IR?
3. What are the advantages and disadvantages of cosine similarity for document ranking?
4. How does the Binary Independence Model derive its ranking formula?

### Advanced Level
1. Design a hybrid ranking system that combines multiple relevance signals.
2. How would you handle the vocabulary mismatch problem in traditional ranking models?
3. Analyze the theoretical connections between different ranking models (VSM, LM, BIM).
4. Design a learning to rank system that can adapt to changing user preferences.

### Tricky Questions
1. **Ranking Paradox**: Why might a theoretically superior ranking model perform worse in practice?
2. **Feature Engineering**: How do you avoid overfitting when designing features for learning to rank?
3. **Evaluation Challenge**: How do you evaluate ranking quality when relevance judgments are subjective?
4. **Scalability Issue**: How do you maintain ranking quality while ensuring sub-second response times?

---

## Code Examples

### Complete Ranking System
```python
import math
import numpy as np
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestRegressor

class ComprehensiveRankingSystem:
    def __init__(self):
        self.collection_stats = None
        self.tf_idf_ranker = TFIDFRanker()
        self.bm25_ranker = BM25Ranker()
        self.lm_ranker = LanguageModelRanker()
        self.ltr_ranker = None
        
        # Ensemble weights
        self.weights = {
            'tf_idf': 0.3,
            'bm25': 0.4,
            'language_model': 0.2,
            'ltr': 0.1
        }
    
    def initialize(self, documents):
        """Initialize ranking system with document collection"""
        self.collection_stats = CollectionStatistics(documents)
        
        # Initialize individual rankers
        self.tf_idf_ranker.initialize(self.collection_stats)
        self.bm25_ranker.initialize(self.collection_stats)
        self.lm_ranker.initialize(documents)
    
    def train_ltr_model(self, training_data):
        """Train learning-to-rank model"""
        self.ltr_ranker = LearningToRankSystem()
        self.ltr_ranker.train(training_data, self.collection_stats)
    
    def rank_documents(self, query, candidate_docs):
        """Rank documents using ensemble approach"""
        query_terms = self.tokenize(query)
        
        # Get scores from each ranker
        tf_idf_scores = self.tf_idf_ranker.score_documents(query_terms, candidate_docs)
        bm25_scores = self.bm25_ranker.score_documents(query_terms, candidate_docs)
        lm_scores = self.lm_ranker.score_documents(query_terms, candidate_docs)
        
        # Combine scores
        final_scores = []
        
        for doc in candidate_docs:
            doc_id = doc.doc_id
            
            # Get individual scores
            tf_idf_score = tf_idf_scores.get(doc_id, 0.0)
            bm25_score = bm25_scores.get(doc_id, 0.0)
            lm_score = lm_scores.get(doc_id, 0.0)
            
            # Normalize scores (simple min-max normalization)
            tf_idf_norm = self.normalize_score(tf_idf_score, tf_idf_scores.values())
            bm25_norm = self.normalize_score(bm25_score, bm25_scores.values())
            lm_norm = self.normalize_score(lm_score, lm_scores.values())
            
            # Ensemble score
            ensemble_score = (
                self.weights['tf_idf'] * tf_idf_norm +
                self.weights['bm25'] * bm25_norm +
                self.weights['language_model'] * lm_norm
            )
            
            # Add LTR score if available
            if self.ltr_ranker:
                ltr_score = self.ltr_ranker.score_document(query_terms, doc)
                ltr_norm = self.normalize_score(ltr_score, [ltr_score])  # Simple normalization
                ensemble_score += self.weights['ltr'] * ltr_norm
            
            final_scores.append((doc_id, ensemble_score, {
                'tf_idf': tf_idf_score,
                'bm25': bm25_score,
                'language_model': lm_score,
                'ensemble': ensemble_score
            }))
        
        # Sort by ensemble score
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores
    
    def normalize_score(self, score, all_scores):
        """Min-max normalization"""
        all_scores = list(all_scores)
        if not all_scores:
            return 0.0
        
        min_score = min(all_scores)
        max_score = max(all_scores)
        
        if max_score == min_score:
            return 0.5  # All scores are the same
        
        return (score - min_score) / (max_score - min_score)
    
    def tokenize(self, text):
        """Simple tokenization"""
        import re
        return re.findall(r'\b\w+\b', text.lower())

class TFIDFRanker:
    def __init__(self):
        self.collection_stats = None
    
    def initialize(self, collection_stats):
        self.collection_stats = collection_stats
    
    def score_documents(self, query_terms, documents):
        """Score documents using TF-IDF"""
        scores = {}
        
        for doc in documents:
            score = 0.0
            
            for term in query_terms:
                tf = doc.get_term_frequency(term)
                if tf > 0:
                    # TF component (log normalization)
                    tf_component = 1 + math.log(tf)
                    
                    # IDF component
                    df = self.collection_stats.get_document_frequency(term)
                    total_docs = self.collection_stats.total_documents
                    idf_component = math.log(total_docs / df) if df > 0 else 0
                    
                    score += tf_component * idf_component
            
            scores[doc.doc_id] = score
        
        return scores

class BM25Ranker:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.collection_stats = None
    
    def initialize(self, collection_stats):
        self.collection_stats = collection_stats
    
    def score_documents(self, query_terms, documents):
        """Score documents using BM25"""
        scores = {}
        
        for doc in documents:
            score = 0.0
            doc_length = doc.get_length()
            avg_doc_length = self.collection_stats.average_document_length
            
            for term in query_terms:
                tf = doc.get_term_frequency(term)
                if tf > 0:
                    # IDF component
                    df = self.collection_stats.get_document_frequency(term)
                    total_docs = self.collection_stats.total_documents
                    idf = math.log((total_docs - df + 0.5) / (df + 0.5))
                    
                    # TF component with saturation and length normalization
                    tf_component = (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * doc_length / avg_doc_length)
                    )
                    
                    score += idf * tf_component
            
            scores[doc.doc_id] = score
        
        return scores

class LanguageModelRanker:
    def __init__(self, smoothing='dirichlet', mu=2000):
        self.smoothing = smoothing
        self.mu = mu
        self.collection_model = None
    
    def initialize(self, documents):
        self.collection_model = self.build_collection_model(documents)
    
    def build_collection_model(self, documents):
        """Build collection language model"""
        term_counts = defaultdict(int)
        total_terms = 0
        
        for doc in documents:
            for term, count in doc.get_term_frequencies().items():
                term_counts[term] += count
                total_terms += count
        
        # Convert to probabilities
        collection_model = {}
        for term, count in term_counts.items():
            collection_model[term] = count / total_terms
        
        return collection_model
    
    def score_documents(self, query_terms, documents):
        """Score documents using query likelihood with Dirichlet smoothing"""
        scores = {}
        
        for doc in documents:
            log_likelihood = 0.0
            doc_length = doc.get_length()
            
            for term in query_terms:
                tf = doc.get_term_frequency(term)
                p_term_collection = self.collection_model.get(term, 1e-10)
                
                # Dirichlet smoothing
                p_term_doc = (tf + self.mu * p_term_collection) / (doc_length + self.mu)
                
                if p_term_doc > 0:
                    log_likelihood += math.log(p_term_doc)
            
            scores[doc.doc_id] = log_likelihood
        
        return scores

class LearningToRankSystem:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
    
    def train(self, training_data, collection_stats):
        """Train LTR model"""
        X = []
        y = []
        
        self.feature_extractor.initialize(collection_stats)
        
        for example in training_data:
            features = self.feature_extractor.extract_features(
                example.query_terms, example.document
            )
            X.append(features)
            y.append(example.relevance_label)
        
        self.model.fit(X, y)
        self.is_trained = True
    
    def score_document(self, query_terms, document):
        """Score document using trained LTR model"""
        if not self.is_trained:
            return 0.0
        
        features = self.feature_extractor.extract_features(query_terms, document)
        return self.model.predict([features])[0]

class FeatureExtractor:
    def __init__(self):
        self.collection_stats = None
    
    def initialize(self, collection_stats):
        self.collection_stats = collection_stats
    
    def extract_features(self, query_terms, document):
        """Extract features for LTR"""
        features = []
        
        # Basic matching features
        features.append(self.query_coverage(query_terms, document))
        features.append(self.exact_match_count(query_terms, document))
        
        # TF-IDF features
        features.append(self.tf_idf_sum(query_terms, document))
        features.append(self.tf_idf_max(query_terms, document))
        
        # Document features
        features.append(document.get_length())
        features.append(document.get_length() / self.collection_stats.average_document_length)
        
        return features
    
    def query_coverage(self, query_terms, document):
        """Fraction of query terms in document"""
        matched = sum(1 for term in query_terms if document.get_term_frequency(term) > 0)
        return matched / len(query_terms) if query_terms else 0
    
    def exact_match_count(self, query_terms, document):
        """Number of exact term matches"""
        return sum(1 for term in query_terms if document.get_term_frequency(term) > 0)
    
    def tf_idf_sum(self, query_terms, document):
        """Sum of TF-IDF scores for query terms"""
        total = 0.0
        for term in query_terms:
            tf = document.get_term_frequency(term)
            if tf > 0:
                tf_component = 1 + math.log(tf)
                df = self.collection_stats.get_document_frequency(term)
                idf_component = math.log(self.collection_stats.total_documents / df) if df > 0 else 0
                total += tf_component * idf_component
        return total
    
    def tf_idf_max(self, query_terms, document):
        """Maximum TF-IDF score among query terms"""
        max_score = 0.0
        for term in query_terms:
            tf = document.get_term_frequency(term)
            if tf > 0:
                tf_component = 1 + math.log(tf)
                df = self.collection_stats.get_document_frequency(term)
                idf_component = math.log(self.collection_stats.total_documents / df) if df > 0 else 0
                max_score = max(max_score, tf_component * idf_component)
        return max_score

class CollectionStatistics:
    def __init__(self, documents):
        self.total_documents = len(documents)
        self.total_terms = 0
        self.document_frequencies = defaultdict(int)
        self.average_document_length = 0
        
        # Compute statistics
        total_length = 0
        for doc in documents:
            doc_length = doc.get_length()
            total_length += doc_length
            
            for term in doc.get_unique_terms():
                self.document_frequencies[term] += 1
        
        self.average_document_length = total_length / self.total_documents if self.total_documents > 0 else 0
    
    def get_document_frequency(self, term):
        return self.document_frequencies.get(term, 0)

# Example usage
if __name__ == "__main__":
    # This would be used with actual document and query data
    ranking_system = ComprehensiveRankingSystem()
    
    # Initialize with document collection
    # ranking_system.initialize(documents)
    
    # Train LTR model if training data available
    # ranking_system.train_ltr_model(training_data)
    
    # Rank documents for a query
    # results = ranking_system.rank_documents("machine learning", candidate_documents)
    
    print("Ranking system initialized successfully")
```

---

## Key Takeaways
1. **Theoretical Foundation**: Probability Ranking Principle provides theoretical basis for relevance ranking
2. **Multiple Approaches**: Different ranking models (VSM, LM, LTR) capture different aspects of relevance
3. **Feature Engineering**: Success of learning to rank depends heavily on good feature design
4. **Ensemble Methods**: Combining multiple ranking signals often outperforms individual approaches
5. **Context Dependency**: Optimal ranking approach depends on collection characteristics and user needs

---

**Next**: In day2_ranking_evaluation.md, we'll explore evaluation metrics and methodologies for assessing the effectiveness of information retrieval systems.