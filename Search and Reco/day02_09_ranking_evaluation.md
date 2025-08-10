# Day 2 - Part 09: IR Evaluation Metrics

## Table of Contents
1. [Introduction to IR Evaluation](#introduction)
2. [Binary Classification Metrics](#binary-classification-metrics)
3. [Ranking-Based Metrics](#ranking-based-metrics)
4. [User-Centered Evaluation](#user-centered-evaluation)
5. [Statistical Significance Testing](#statistical-significance)
6. [Evaluation Methodologies](#evaluation-methodologies)
7. [Study Questions](#study-questions)
8. [Code Examples](#code-examples)

---

## Introduction to IR Evaluation

Information Retrieval evaluation measures how well search systems satisfy user information needs through systematic assessment of retrieval effectiveness.

### Why Evaluation Matters

#### **System Improvement**
- **Performance Measurement**: Quantify system effectiveness
- **Component Analysis**: Identify strengths and weaknesses
- **Progress Tracking**: Monitor improvements over time
- **A/B Testing**: Compare different approaches

#### **Research and Development**
- **Algorithm Comparison**: Compare different retrieval methods
- **Feature Impact**: Measure contribution of individual features
- **Parameter Tuning**: Optimize system parameters
- **Generalization**: Assess performance across different collections

### Evaluation Challenges

#### **Relevance Subjectivity**
```python
# Relevance judgments can vary between assessors
relevance_assessments = {
    'assessor_1': {'doc_1': 3, 'doc_2': 1, 'doc_3': 2},
    'assessor_2': {'doc_1': 2, 'doc_2': 2, 'doc_3': 1},
    'assessor_3': {'doc_1': 3, 'doc_2': 1, 'doc_3': 3}
}

# Inter-assessor agreement analysis needed
def calculate_inter_assessor_agreement(assessments):
    # Krippendorff's alpha, Cohen's kappa, etc.
    pass
```

#### **Scale and Cost**
- **Manual Assessment**: Expensive and time-consuming
- **Collection Size**: Modern collections have millions of documents
- **Query Diversity**: Need representative query samples
- **Dynamic Collections**: Web collections change constantly

#### **Evaluation Bias**
- **System Bias**: Evaluating on system's training data
- **Collection Bias**: Limited to specific domains/languages
- **Judgment Bias**: Assessor preferences and expertise
- **Temporal Bias**: Historical relevance vs current needs

---

## Binary Classification Metrics

Traditional metrics treating retrieval as binary classification (relevant/not relevant).

### Precision and Recall

#### **Basic Definitions**
```python
# Confusion Matrix for IR
class IRConfusionMatrix:
    def __init__(self, retrieved_docs, relevant_docs, total_docs):
        self.retrieved = set(retrieved_docs)
        self.relevant = set(relevant_docs)
        self.total = total_docs
        
        # Calculate confusion matrix components
        self.true_positives = len(self.retrieved & self.relevant)
        self.false_positives = len(self.retrieved - self.relevant)
        self.false_negatives = len(self.relevant - self.retrieved)
        self.true_negatives = self.total - self.true_positives - self.false_positives - self.false_negatives
    
    def precision(self):
        """Precision = TP / (TP + FP)"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    def recall(self):
        """Recall = TP / (TP + FN)"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    def f1_score(self):
        """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
```

#### **Precision at K (P@K)**
```python
def precision_at_k(retrieved_ranked_list, relevant_docs, k):
    """Calculate precision at rank k"""
    if k <= 0 or len(retrieved_ranked_list) == 0:
        return 0.0
    
    # Take top k documents
    top_k = retrieved_ranked_list[:k]
    
    # Count relevant documents in top k
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
    
    return relevant_in_top_k / k

def precision_at_k_curve(retrieved_ranked_list, relevant_docs, max_k=None):
    """Calculate P@K for all K values"""
    if max_k is None:
        max_k = len(retrieved_ranked_list)
    
    precision_values = []
    for k in range(1, min(max_k + 1, len(retrieved_ranked_list) + 1)):
        p_at_k = precision_at_k(retrieved_ranked_list, relevant_docs, k)
        precision_values.append((k, p_at_k))
    
    return precision_values
```

#### **Recall at K (R@K)**
```python
def recall_at_k(retrieved_ranked_list, relevant_docs, k):
    """Calculate recall at rank k"""
    if k <= 0 or len(relevant_docs) == 0:
        return 0.0
    
    # Take top k documents
    top_k = retrieved_ranked_list[:k]
    
    # Count relevant documents found in top k
    relevant_found = sum(1 for doc in top_k if doc in relevant_docs)
    
    return relevant_found / len(relevant_docs)

def recall_precision_curve(retrieved_ranked_list, relevant_docs):
    """Generate recall-precision curve points"""
    if not relevant_docs:
        return []
    
    curve_points = []
    relevant_found = 0
    
    for i, doc in enumerate(retrieved_ranked_list):
        if doc in relevant_docs:
            relevant_found += 1
        
        # Calculate precision and recall at this point
        precision = relevant_found / (i + 1)
        recall = relevant_found / len(relevant_docs)
        
        curve_points.append((recall, precision))
    
    return curve_points
```

### Average Precision

#### **Single Query Average Precision**
```python
def average_precision(retrieved_ranked_list, relevant_docs):
    """Calculate Average Precision for a single query"""
    if not relevant_docs:
        return 0.0
    
    precision_sum = 0.0
    relevant_found = 0
    
    for i, doc in enumerate(retrieved_ranked_list):
        if doc in relevant_docs:
            relevant_found += 1
            precision_at_i = relevant_found / (i + 1)
            precision_sum += precision_at_i
    
    return precision_sum / len(relevant_docs)

def interpolated_precision(recall_precision_points):
    """Calculate 11-point interpolated precision"""
    # Standard recall levels
    recall_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    interpolated_precisions = []
    
    for target_recall in recall_levels:
        # Find maximum precision at or above this recall level
        max_precision = 0.0
        for recall, precision in recall_precision_points:
            if recall >= target_recall:
                max_precision = max(max_precision, precision)
        
        interpolated_precisions.append(max_precision)
    
    return list(zip(recall_levels, interpolated_precisions))
```

### Mean Average Precision (MAP)

#### **Multi-Query MAP**
```python
def mean_average_precision(query_results):
    """Calculate MAP across multiple queries"""
    if not query_results:
        return 0.0
    
    ap_scores = []
    
    for query_id, (retrieved_docs, relevant_docs) in query_results.items():
        ap = average_precision(retrieved_docs, relevant_docs)
        ap_scores.append(ap)
    
    return sum(ap_scores) / len(ap_scores)

class MAPEvaluator:
    def __init__(self):
        self.query_results = {}
    
    def add_query_result(self, query_id, retrieved_docs, relevant_docs):
        """Add results for a single query"""
        self.query_results[query_id] = (retrieved_docs, relevant_docs)
    
    def calculate_map(self):
        """Calculate overall MAP"""
        return mean_average_precision(self.query_results)
    
    def calculate_per_query_ap(self):
        """Calculate AP for each query"""
        per_query_ap = {}
        
        for query_id, (retrieved_docs, relevant_docs) in self.query_results.items():
            ap = average_precision(retrieved_docs, relevant_docs)
            per_query_ap[query_id] = ap
        
        return per_query_ap
```

---

## Ranking-Based Metrics

Metrics that consider the ranking order of retrieved documents.

### Discounted Cumulative Gain (DCG)

#### **DCG Calculation**
```python
import math

def dcg_at_k(relevance_scores, k):
    """Calculate DCG at rank k"""
    if k <= 0:
        return 0.0
    
    dcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        relevance = relevance_scores[i]
        # DCG formula: rel_i / log2(i + 2)
        dcg += relevance / math.log2(i + 2)
    
    return dcg

def ideal_dcg_at_k(relevance_scores, k):
    """Calculate Ideal DCG (IDCG) at rank k"""
    # Sort relevance scores in descending order for ideal ranking
    sorted_relevance = sorted(relevance_scores, reverse=True)
    return dcg_at_k(sorted_relevance, k)

def ndcg_at_k(relevance_scores, k):
    """Calculate Normalized DCG at rank k"""
    dcg = dcg_at_k(relevance_scores, k)
    idcg = ideal_dcg_at_k(relevance_scores, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

# Alternative DCG formulation (for higher relevance scores)
def dcg_at_k_exponential(relevance_scores, k):
    """Calculate DCG with exponential gain"""
    if k <= 0:
        return 0.0
    
    dcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        relevance = relevance_scores[i]
        # Exponential DCG: (2^rel_i - 1) / log2(i + 2)
        gain = 2**relevance - 1
        discount = math.log2(i + 2)
        dcg += gain / discount
    
    return dcg
```

#### **NDCG Implementation**
```python
class NDCGEvaluator:
    def __init__(self, max_relevance=3):
        self.max_relevance = max_relevance
        self.query_results = {}
    
    def add_query_result(self, query_id, ranked_docs, relevance_judgments):
        """Add ranked results with relevance judgments"""
        # Create relevance score list based on ranking
        relevance_scores = []
        for doc in ranked_docs:
            relevance = relevance_judgments.get(doc, 0)
            relevance_scores.append(relevance)
        
        self.query_results[query_id] = relevance_scores
    
    def calculate_ndcg_at_k(self, k):
        """Calculate NDCG@k across all queries"""
        ndcg_scores = []
        
        for query_id, relevance_scores in self.query_results.items():
            ndcg = ndcg_at_k(relevance_scores, k)
            ndcg_scores.append(ndcg)
        
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    
    def calculate_ndcg_at_multiple_k(self, k_values):
        """Calculate NDCG at multiple k values"""
        results = {}
        
        for k in k_values:
            results[f'NDCG@{k}'] = self.calculate_ndcg_at_k(k)
        
        return results
```

### Rank-Based Metrics

#### **Mean Reciprocal Rank (MRR)**
```python
def reciprocal_rank(ranked_docs, relevant_docs):
    """Calculate reciprocal rank for a single query"""
    for i, doc in enumerate(ranked_docs):
        if doc in relevant_docs:
            return 1.0 / (i + 1)  # Rank is 1-indexed
    
    return 0.0  # No relevant document found

def mean_reciprocal_rank(query_results):
    """Calculate MRR across multiple queries"""
    rr_scores = []
    
    for query_id, (ranked_docs, relevant_docs) in query_results.items():
        rr = reciprocal_rank(ranked_docs, relevant_docs)
        rr_scores.append(rr)
    
    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0

class MRREvaluator:
    def __init__(self):
        self.query_results = {}
    
    def add_query_result(self, query_id, ranked_docs, relevant_docs):
        self.query_results[query_id] = (ranked_docs, relevant_docs)
    
    def calculate_mrr(self):
        return mean_reciprocal_rank(self.query_results)
    
    def calculate_success_at_k(self, k):
        """Calculate Success@k (fraction of queries with relevant doc in top k)"""
        successful_queries = 0
        
        for query_id, (ranked_docs, relevant_docs) in self.query_results.items():
            top_k = ranked_docs[:k]
            if any(doc in relevant_docs for doc in top_k):
                successful_queries += 1
        
        return successful_queries / len(self.query_results) if self.query_results else 0.0
```

#### **Rank-Biased Precision (RBP)**
```python
def rank_biased_precision(ranked_docs, relevant_docs, p=0.8):
    """Calculate Rank-Biased Precision"""
    rbp = 0.0
    
    for i, doc in enumerate(ranked_docs):
        if doc in relevant_docs:
            # RBP formula: (1-p) * p^i
            rbp += (1 - p) * (p ** i)
    
    return rbp

def expected_reciprocal_rank(ranked_docs, relevance_probs, max_rank=10):
    """Calculate Expected Reciprocal Rank"""
    err = 0.0
    prob_not_satisfied = 1.0
    
    for i in range(min(len(ranked_docs), max_rank)):
        doc = ranked_docs[i]
        relevance_prob = relevance_probs.get(doc, 0.0)
        
        # ERR contribution at rank i+1
        rank = i + 1
        err += prob_not_satisfied * relevance_prob / rank
        
        # Update probability of not being satisfied
        prob_not_satisfied *= (1 - relevance_prob)
    
    return err
```

---

## User-Centered Evaluation

Metrics focusing on user experience and satisfaction.

### Click-Through Metrics

#### **Click-Through Rate Analysis**
```python
class ClickThroughAnalyzer:
    def __init__(self):
        self.click_data = []
    
    def add_query_session(self, query_id, ranked_docs, clicked_docs, click_positions):
        """Add click data for a query session"""
        session_data = {
            'query_id': query_id,
            'ranked_docs': ranked_docs,
            'clicked_docs': clicked_docs,
            'click_positions': click_positions
        }
        self.click_data.append(session_data)
    
    def calculate_ctr_at_k(self, k):
        """Calculate Click-Through Rate at position k"""
        if not self.click_data:
            return 0.0
        
        clicks_at_k = 0
        impressions_at_k = 0
        
        for session in self.click_data:
            if len(session['ranked_docs']) >= k:
                impressions_at_k += 1
                if k - 1 in session['click_positions']:  # k-1 because positions are 0-indexed
                    clicks_at_k += 1
        
        return clicks_at_k / impressions_at_k if impressions_at_k > 0 else 0.0
    
    def calculate_mean_reciprocal_rank_clicks(self):
        """Calculate MRR based on click positions"""
        rr_scores = []
        
        for session in self.click_data:
            if session['click_positions']:
                first_click_position = min(session['click_positions'])
                rr = 1.0 / (first_click_position + 1)  # Convert to 1-indexed
            else:
                rr = 0.0
            
            rr_scores.append(rr)
        
        return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0
```

### Session-Based Metrics

#### **Abandonment and Success**
```python
class SessionMetrics:
    def __init__(self):
        self.sessions = []
    
    def add_session(self, session_data):
        """Add session data"""
        required_fields = ['query_id', 'results_shown', 'clicks', 'dwell_times', 'session_outcome']
        if all(field in session_data for field in required_fields):
            self.sessions.append(session_data)
    
    def calculate_abandonment_rate(self):
        """Calculate query abandonment rate"""
        if not self.sessions:
            return 0.0
        
        abandoned_sessions = sum(1 for s in self.sessions if len(s['clicks']) == 0)
        return abandoned_sessions / len(self.sessions)
    
    def calculate_success_rate(self):
        """Calculate session success rate"""
        if not self.sessions:
            return 0.0
        
        successful_sessions = sum(1 for s in self.sessions 
                                if s['session_outcome'] == 'successful')
        return successful_sessions / len(self.sessions)
    
    def calculate_average_dwell_time(self):
        """Calculate average dwell time on clicked results"""
        all_dwell_times = []
        
        for session in self.sessions:
            all_dwell_times.extend(session['dwell_times'])
        
        return sum(all_dwell_times) / len(all_dwell_times) if all_dwell_times else 0.0
    
    def calculate_depth_metrics(self):
        """Calculate various depth metrics"""
        if not self.sessions:
            return {}
        
        click_depths = []
        scroll_depths = []
        
        for session in self.sessions:
            if session['clicks']:
                max_click_position = max(session['clicks'])
                click_depths.append(max_click_position + 1)  # Convert to 1-indexed
            
            scroll_depth = session.get('max_scroll_position', 0)
            scroll_depths.append(scroll_depth)
        
        return {
            'avg_click_depth': sum(click_depths) / len(click_depths) if click_depths else 0,
            'avg_scroll_depth': sum(scroll_depths) / len(scroll_depths) if scroll_depths else 0,
            'sessions_with_clicks': len(click_depths),
            'sessions_without_clicks': len(self.sessions) - len(click_depths)
        }
```

### User Satisfaction Modeling

#### **Cascade Model for Click Prediction**
```python
class CascadeClickModel:
    def __init__(self):
        self.relevance_probs = {}  # P(relevant | query, doc)
        self.examination_probs = {}  # P(examined | position)
    
    def fit(self, click_data):
        """Fit cascade model to click data"""
        # Simplified implementation - real model would use EM algorithm
        
        # Estimate examination probabilities by position
        position_stats = defaultdict(lambda: {'examined': 0, 'total': 0})
        
        for session in click_data:
            for pos, doc in enumerate(session['ranked_docs']):
                position_stats[pos]['total'] += 1
                
                # In cascade model, if there's a click at or before this position,
                # this position was examined
                if any(click_pos <= pos for click_pos in session['click_positions']):
                    position_stats[pos]['examined'] += 1
        
        # Calculate examination probabilities
        for pos, stats in position_stats.items():
            if stats['total'] > 0:
                self.examination_probs[pos] = stats['examined'] / stats['total']
    
    def predict_click_probability(self, query, doc, position):
        """Predict click probability for query-doc pair at position"""
        relevance_prob = self.relevance_probs.get((query, doc), 0.1)  # Default relevance
        examination_prob = self.examination_probs.get(position, 0.5)  # Default examination
        
        # In cascade model: P(click) = P(examination) * P(relevant)
        return examination_prob * relevance_prob
```

---

## Statistical Significance Testing

Methods to determine if observed differences between systems are statistically significant.

### Paired Statistical Tests

#### **Paired t-Test for AP Scores**
```python
import scipy.stats as stats
import numpy as np

def paired_t_test(system1_scores, system2_scores, alpha=0.05):
    """Perform paired t-test on Average Precision scores"""
    if len(system1_scores) != len(system2_scores):
        raise ValueError("Score lists must have equal length")
    
    # Calculate differences
    differences = np.array(system1_scores) - np.array(system2_scores)
    
    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(system1_scores, system2_scores)
    
    # Determine significance
    is_significant = p_value < alpha
    
    # Calculate effect size (Cohen's d)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    return {
        't_statistic': t_statistic,
        'p_value': p_value,
        'is_significant': is_significant,
        'mean_difference': mean_diff,
        'cohens_d': cohens_d,
        'confidence_interval': stats.t.interval(1 - alpha, len(differences) - 1,
                                               loc=mean_diff,
                                               scale=stats.sem(differences))
    }

def wilcoxon_signed_rank_test(system1_scores, system2_scores, alpha=0.05):
    """Non-parametric alternative to paired t-test"""
    if len(system1_scores) != len(system2_scores):
        raise ValueError("Score lists must have equal length")
    
    # Perform Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(system1_scores, system2_scores)
    
    is_significant = p_value < alpha
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'is_significant': is_significant
    }
```

#### **Bootstrap Confidence Intervals**
```python
def bootstrap_confidence_interval(scores, metric_func, n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence interval for a metric"""
    np.random.seed(42)  # For reproducibility
    
    bootstrap_scores = []
    n_queries = len(scores)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(scores, size=n_queries, replace=True)
        bootstrap_metric = metric_func(bootstrap_sample)
        bootstrap_scores.append(bootstrap_metric)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)
    
    return {
        'original_metric': metric_func(scores),
        'bootstrap_mean': np.mean(bootstrap_scores),
        'confidence_interval': (ci_lower, ci_upper),
        'bootstrap_std': np.std(bootstrap_scores)
    }

def bootstrap_significance_test(system1_scores, system2_scores, n_bootstrap=1000):
    """Bootstrap test for difference in means"""
    np.random.seed(42)
    
    # Calculate observed difference
    observed_diff = np.mean(system1_scores) - np.mean(system2_scores)
    
    # Pool the data under null hypothesis (no difference)
    pooled_data = np.concatenate([system1_scores, system2_scores])
    n1, n2 = len(system1_scores), len(system2_scores)
    
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        # Resample from pooled data
        bootstrap_sample = np.random.choice(pooled_data, size=n1 + n2, replace=True)
        
        bootstrap_system1 = bootstrap_sample[:n1]
        bootstrap_system2 = bootstrap_sample[n1:]
        
        bootstrap_diff = np.mean(bootstrap_system1) - np.mean(bootstrap_system2)
        bootstrap_diffs.append(bootstrap_diff)
    
    # Calculate p-value (two-tailed)
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
    
    return {
        'observed_difference': observed_diff,
        'p_value': p_value,
        'is_significant': p_value < 0.05
    }
```

---

## Evaluation Methodologies

Comprehensive evaluation frameworks and best practices.

### Cranfield Methodology

#### **Test Collection Components**
```python
class CranfieldEvaluation:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.documents = {}  # doc_id -> document
        self.queries = {}    # query_id -> query
        self.relevance_judgments = {}  # (query_id, doc_id) -> relevance_level
        self.system_results = {}  # system_name -> {query_id -> [ranked_doc_ids]}
    
    def load_documents(self, documents):
        """Load document collection"""
        self.documents = documents
    
    def load_queries(self, queries):
        """Load query set"""
        self.queries = queries
    
    def load_relevance_judgments(self, judgments):
        """Load relevance judgments"""
        self.relevance_judgments = judgments
    
    def add_system_results(self, system_name, results):
        """Add results from a retrieval system"""
        self.system_results[system_name] = results
    
    def evaluate_system(self, system_name, metrics=['MAP', 'P@10', 'NDCG@10']):
        """Evaluate a system using specified metrics"""
        if system_name not in self.system_results:
            raise ValueError(f"System {system_name} not found")
        
        system_results = self.system_results[system_name]
        evaluation_results = {}
        
        # Per-query metrics
        per_query_metrics = defaultdict(dict)
        
        for query_id in self.queries:
            if query_id not in system_results:
                continue
            
            ranked_docs = system_results[query_id]
            relevant_docs = self.get_relevant_docs(query_id)
            relevance_scores = self.get_relevance_scores(query_id, ranked_docs)
            
            # Calculate metrics
            if 'MAP' in metrics:
                ap = average_precision(ranked_docs, relevant_docs)
                per_query_metrics[query_id]['AP'] = ap
            
            if 'P@10' in metrics:
                p_at_10 = precision_at_k(ranked_docs, relevant_docs, 10)
                per_query_metrics[query_id]['P@10'] = p_at_10
            
            if 'NDCG@10' in metrics:
                ndcg_10 = ndcg_at_k(relevance_scores, 10)
                per_query_metrics[query_id]['NDCG@10'] = ndcg_10
        
        # Calculate aggregate metrics
        if 'MAP' in metrics:
            ap_scores = [per_query_metrics[qid]['AP'] for qid in per_query_metrics]
            evaluation_results['MAP'] = np.mean(ap_scores)
        
        if 'P@10' in metrics:
            p10_scores = [per_query_metrics[qid]['P@10'] for qid in per_query_metrics]
            evaluation_results['P@10'] = np.mean(p10_scores)
        
        if 'NDCG@10' in metrics:
            ndcg10_scores = [per_query_metrics[qid]['NDCG@10'] for qid in per_query_metrics]
            evaluation_results['NDCG@10'] = np.mean(ndcg10_scores)
        
        evaluation_results['per_query'] = dict(per_query_metrics)
        evaluation_results['num_queries'] = len(per_query_metrics)
        
        return evaluation_results
    
    def get_relevant_docs(self, query_id):
        """Get relevant documents for a query"""
        relevant_docs = set()
        for (qid, doc_id), relevance in self.relevance_judgments.items():
            if qid == query_id and relevance > 0:
                relevant_docs.add(doc_id)
        return relevant_docs
    
    def get_relevance_scores(self, query_id, ranked_docs):
        """Get relevance scores for ranked documents"""
        relevance_scores = []
        for doc_id in ranked_docs:
            relevance = self.relevance_judgments.get((query_id, doc_id), 0)
            relevance_scores.append(relevance)
        return relevance_scores
```

### Cross-Validation for IR

#### **Query-Level Cross-Validation**
```python
def query_level_cross_validation(queries, relevance_judgments, system_func, k_folds=5):
    """Perform k-fold cross-validation at query level"""
    from sklearn.model_selection import KFold
    
    query_ids = list(queries.keys())
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(query_ids)):
        # Split queries into train and test
        train_queries = {query_ids[i]: queries[query_ids[i]] for i in train_indices}
        test_queries = {query_ids[i]: queries[query_ids[i]] for i in test_indices}
        
        # Train system on training queries
        trained_system = system_func(train_queries, relevance_judgments)
        
        # Evaluate on test queries
        fold_metrics = {}
        ap_scores = []
        
        for query_id in test_queries:
            # Get system results
            results = trained_system.retrieve(test_queries[query_id])
            
            # Get relevant documents
            relevant_docs = set()
            for (qid, doc_id), rel in relevance_judgments.items():
                if qid == query_id and rel > 0:
                    relevant_docs.add(doc_id)
            
            # Calculate AP
            ap = average_precision(results, relevant_docs)
            ap_scores.append(ap)
        
        fold_metrics['MAP'] = np.mean(ap_scores)
        fold_metrics['num_test_queries'] = len(test_queries)
        fold_results.append(fold_metrics)
    
    # Aggregate results across folds
    overall_map = np.mean([fold['MAP'] for fold in fold_results])
    map_std = np.std([fold['MAP'] for fold in fold_results])
    
    return {
        'cross_val_MAP': overall_map,
        'MAP_std': map_std,
        'fold_results': fold_results
    }
```

---

## Study Questions

### Beginner Level
1. What is the difference between precision and recall in information retrieval?
2. How does MAP (Mean Average Precision) differ from simple precision?
3. What does NDCG measure and why is it useful for ranking evaluation?
4. Why is statistical significance testing important in IR evaluation?

### Intermediate Level
1. Compare the advantages and disadvantages of DCG vs MAP for ranking evaluation.
2. How do user-centered metrics like click-through rate differ from traditional IR metrics?
3. What are the challenges in creating relevance judgments for large-scale collections?
4. How would you design an evaluation methodology for a new search domain?

### Advanced Level
1. Design a comprehensive evaluation framework for a multi-modal search system.
2. How do you handle the incomplete relevance judgment problem in large-scale evaluation?
3. Analyze the trade-offs between offline evaluation metrics and online user satisfaction.
4. How would you evaluate the fairness and bias of a search ranking system?

### Tricky Questions
1. **Evaluation Paradox**: Why might a system with better offline metrics perform worse for real users?
2. **Metric Gaming**: How can optimizing for specific metrics lead to worse user experience?
3. **Temporal Bias**: How do you evaluate search systems when relevance changes over time?
4. **Scale Challenge**: How do you maintain evaluation quality when scaling to billions of queries?

---

## Code Examples

### Complete Evaluation Framework
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats as stats

class ComprehensiveIREvaluator:
    def __init__(self):
        self.systems = {}  # system_name -> {query_id -> [(doc_id, score)]}
        self.relevance_judgments = {}  # (query_id, doc_id) -> relevance_score
        self.queries = {}  # query_id -> query_text
        self.evaluation_results = {}
    
    def add_system_results(self, system_name, results):
        """Add results from a retrieval system"""
        self.systems[system_name] = results
    
    def add_relevance_judgments(self, judgments):
        """Add relevance judgments"""
        self.relevance_judgments.update(judgments)
    
    def add_queries(self, queries):
        """Add query set"""
        self.queries.update(queries)
    
    def evaluate_all_systems(self, metrics=None):
        """Evaluate all systems with comprehensive metrics"""
        if metrics is None:
            metrics = ['MAP', 'P@5', 'P@10', 'NDCG@5', 'NDCG@10', 'MRR']
        
        for system_name in self.systems:
            self.evaluation_results[system_name] = self.evaluate_system(system_name, metrics)
        
        return self.evaluation_results
    
    def evaluate_system(self, system_name, metrics):
        """Evaluate single system"""
        if system_name not in self.systems:
            raise ValueError(f"System {system_name} not found")
        
        system_results = self.systems[system_name]
        per_query_metrics = defaultdict(dict)
        
        # Calculate per-query metrics
        for query_id, ranked_results in system_results.items():
            if query_id not in self.queries:
                continue
            
            # Extract ranked document list
            ranked_docs = [doc_id for doc_id, score in ranked_results]
            
            # Get relevant documents and relevance scores
            relevant_docs = self.get_relevant_docs(query_id)
            relevance_scores = self.get_relevance_scores(query_id, ranked_docs)
            
            # Calculate metrics
            per_query_metrics[query_id] = self.calculate_query_metrics(
                ranked_docs, relevant_docs, relevance_scores, metrics
            )
        
        # Aggregate metrics across queries
        aggregate_metrics = self.aggregate_metrics(per_query_metrics, metrics)
        
        return {
            'aggregate': aggregate_metrics,
            'per_query': dict(per_query_metrics),
            'num_queries': len(per_query_metrics)
        }
    
    def calculate_query_metrics(self, ranked_docs, relevant_docs, relevance_scores, metrics):
        """Calculate all metrics for a single query"""
        query_metrics = {}
        
        if 'MAP' in metrics or 'AP' in metrics:
            query_metrics['AP'] = average_precision(ranked_docs, relevant_docs)
        
        for k in [5, 10, 20]:
            if f'P@{k}' in metrics:
                query_metrics[f'P@{k}'] = precision_at_k(ranked_docs, relevant_docs, k)
            
            if f'R@{k}' in metrics:
                query_metrics[f'R@{k}'] = recall_at_k(ranked_docs, relevant_docs, k)
            
            if f'NDCG@{k}' in metrics:
                query_metrics[f'NDCG@{k}'] = ndcg_at_k(relevance_scores, k)
        
        if 'MRR' in metrics or 'RR' in metrics:
            query_metrics['RR'] = reciprocal_rank(ranked_docs, relevant_docs)
        
        return query_metrics
    
    def aggregate_metrics(self, per_query_metrics, metrics):
        """Aggregate per-query metrics"""
        aggregate = {}
        
        for metric in metrics:
            if metric == 'MAP':
                ap_scores = [pq['AP'] for pq in per_query_metrics.values() if 'AP' in pq]
                aggregate['MAP'] = np.mean(ap_scores) if ap_scores else 0.0
            
            elif metric == 'MRR':
                rr_scores = [pq['RR'] for pq in per_query_metrics.values() if 'RR' in pq]
                aggregate['MRR'] = np.mean(rr_scores) if rr_scores else 0.0
            
            elif metric.startswith(('P@', 'R@', 'NDCG@')):
                metric_scores = [pq[metric] for pq in per_query_metrics.values() if metric in pq]
                aggregate[metric] = np.mean(metric_scores) if metric_scores else 0.0
        
        return aggregate
    
    def get_relevant_docs(self, query_id):
        """Get relevant documents for query"""
        relevant_docs = set()
        for (qid, doc_id), relevance in self.relevance_judgments.items():
            if qid == query_id and relevance > 0:
                relevant_docs.add(doc_id)
        return relevant_docs
    
    def get_relevance_scores(self, query_id, ranked_docs):
        """Get relevance scores for ranked documents"""
        return [self.relevance_judgments.get((query_id, doc_id), 0) for doc_id in ranked_docs]
    
    def compare_systems(self, system1, system2, metric='MAP', statistical_test='paired_t'):
        """Compare two systems statistically"""
        if system1 not in self.evaluation_results or system2 not in self.evaluation_results:
            raise ValueError("Both systems must be evaluated first")
        
        # Extract per-query scores
        system1_scores = []
        system2_scores = []
        
        queries_1 = set(self.evaluation_results[system1]['per_query'].keys())
        queries_2 = set(self.evaluation_results[system2]['per_query'].keys())
        common_queries = queries_1 & queries_2
        
        for query_id in common_queries:
            if metric == 'MAP':
                score1 = self.evaluation_results[system1]['per_query'][query_id].get('AP', 0)
                score2 = self.evaluation_results[system2]['per_query'][query_id].get('AP', 0)
            else:
                score1 = self.evaluation_results[system1]['per_query'][query_id].get(metric, 0)
                score2 = self.evaluation_results[system2]['per_query'][query_id].get(metric, 0)
            
            system1_scores.append(score1)
            system2_scores.append(score2)
        
        # Perform statistical test
        if statistical_test == 'paired_t':
            test_result = paired_t_test(system1_scores, system2_scores)
        elif statistical_test == 'wilcoxon':
            test_result = wilcoxon_signed_rank_test(system1_scores, system2_scores)
        else:
            raise ValueError(f"Unknown statistical test: {statistical_test}")
        
        # Add comparison info
        test_result['system1'] = system1
        test_result['system2'] = system2
        test_result['metric'] = metric
        test_result['num_queries'] = len(common_queries)
        test_result['system1_mean'] = np.mean(system1_scores)
        test_result['system2_mean'] = np.mean(system2_scores)
        
        return test_result
    
    def generate_evaluation_report(self, output_file=None):
        """Generate comprehensive evaluation report"""
        report = []
        report.append("=== Information Retrieval Evaluation Report ===\n")
        
        # System overview
        report.append(f"Number of systems evaluated: {len(self.systems)}")
        report.append(f"Number of queries: {len(self.queries)}")
        report.append(f"Number of relevance judgments: {len(self.relevance_judgments)}\n")
        
        # Results table
        report.append("System Performance Summary:")
        report.append("-" * 80)
        
        header = f"{'System':<20} {'MAP':<8} {'P@5':<8} {'P@10':<8} {'NDCG@5':<8} {'NDCG@10':<8} {'MRR':<8}"
        report.append(header)
        report.append("-" * 80)
        
        for system_name, results in self.evaluation_results.items():
            aggregate = results['aggregate']
            row = f"{system_name:<20} "
            row += f"{aggregate.get('MAP', 0):<8.4f} "
            row += f"{aggregate.get('P@5', 0):<8.4f} "
            row += f"{aggregate.get('P@10', 0):<8.4f} "
            row += f"{aggregate.get('NDCG@5', 0):<8.4f} "
            row += f"{aggregate.get('NDCG@10', 0):<8.4f} "
            row += f"{aggregate.get('MRR', 0):<8.4f}"
            report.append(row)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def plot_system_comparison(self, metric='MAP', save_path=None):
        """Plot system comparison"""
        systems = list(self.evaluation_results.keys())
        scores = [self.evaluation_results[sys]['aggregate'].get(metric, 0) for sys in systems]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(systems, scores)
        plt.title(f'System Comparison - {metric}')
        plt.ylabel(metric)
        plt.xlabel('Systems')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Example usage
def demo_evaluation_framework():
    """Demonstrate the evaluation framework"""
    
    # Create evaluator
    evaluator = ComprehensiveIREvaluator()
    
    # Add sample queries
    queries = {
        'Q1': 'machine learning algorithms',
        'Q2': 'information retrieval evaluation',
        'Q3': 'natural language processing'
    }
    evaluator.add_queries(queries)
    
    # Add sample relevance judgments
    relevance_judgments = {
        ('Q1', 'D1'): 3, ('Q1', 'D2'): 2, ('Q1', 'D3'): 1,
        ('Q2', 'D1'): 1, ('Q2', 'D4'): 3, ('Q2', 'D5'): 2,
        ('Q3', 'D2'): 2, ('Q3', 'D3'): 3, ('Q3', 'D6'): 1
    }
    evaluator.add_relevance_judgments(relevance_judgments)
    
    # Add sample system results
    system1_results = {
        'Q1': [('D1', 0.9), ('D2', 0.8), ('D3', 0.7), ('D4', 0.6)],
        'Q2': [('D4', 0.95), ('D5', 0.85), ('D1', 0.75), ('D2', 0.65)],
        'Q3': [('D3', 0.92), ('D2', 0.88), ('D6', 0.82), ('D1', 0.72)]
    }
    
    system2_results = {
        'Q1': [('D2', 0.88), ('D1', 0.85), ('D4', 0.78), ('D3', 0.72)],
        'Q2': [('D5', 0.91), ('D4', 0.87), ('D2', 0.81), ('D1', 0.75)],
        'Q3': [('D2', 0.89), ('D3', 0.86), ('D1', 0.83), ('D6', 0.79)]
    }
    
    evaluator.add_system_results('System1', system1_results)
    evaluator.add_system_results('System2', system2_results)
    
    # Evaluate all systems
    results = evaluator.evaluate_all_systems()
    
    # Generate report
    report = evaluator.generate_evaluation_report()
    print(report)
    
    # Compare systems
    comparison = evaluator.compare_systems('System1', 'System2', metric='MAP')
    print(f"\nStatistical Comparison (MAP):")
    print(f"System1 mean: {comparison['system1_mean']:.4f}")
    print(f"System2 mean: {comparison['system2_mean']:.4f}")
    print(f"P-value: {comparison['p_value']:.4f}")
    print(f"Significant: {comparison['is_significant']}")

if __name__ == "__main__":
    demo_evaluation_framework()
```

---

## Key Takeaways
1. **Multi-Metric Evaluation**: Use multiple complementary metrics to assess different aspects of retrieval performance
2. **Statistical Rigor**: Always test statistical significance when comparing systems
3. **User-Centered Focus**: Balance traditional metrics with user experience measures
4. **Evaluation Bias**: Be aware of collection bias, judgment bias, and evaluation methodology limitations
5. **Practical Considerations**: Consider computational cost, scalability, and real-world applicability of evaluation approaches

---

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "create_day2_09_ranking_evaluation", "content": "Create day2_09_ranking_evaluation.md: IR Evaluation Metrics", "status": "completed", "priority": "high"}]