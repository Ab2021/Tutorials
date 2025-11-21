# Day 44: Recommendation System - Core Concepts

## Overview
Design a scalable recommendation system that can serve personalized content to millions of users in real-time, similar to Netflix, YouTube, or Amazon.

## Key Requirements

### Functional Requirements
- **Personalized Recommendations**: Tailored to individual user preferences
- **Real-time Updates**: Reflect recent user behavior
- **Diverse Content**: Avoid filter bubbles, show variety
- **Cold Start Handling**: Recommendations for new users/items
- **Multiple Algorithms**: Collaborative filtering, content-based, hybrid

### Non-Functional Requirements
- **Low Latency**: < 100ms for recommendation retrieval
- **High Throughput**: 1M+ requests per second
- **Scalability**: Handle billions of users and items
- **Freshness**: Incorporate recent interactions quickly
- **Accuracy**: High relevance (CTR, engagement metrics)

## System Architecture

### High-Level Design

```
┌──────────────┐
│    Users     │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│   API Gateway    │
│  (Load Balancer) │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────────┐
│ Online  │ │   Offline    │
│ Serving │ │  Training    │
└────┬────┘ └──────┬───────┘
     │             │
     ▼             ▼
┌─────────────────────────┐
│  Feature Store (Redis)  │
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ Model Store (S3/GCS)    │
└─────────────────────────┘
```

### Data Pipeline

```
User Events → Kafka → Stream Processing → Feature Store
                           ↓
                    Batch Processing → Model Training → Model Store
```

## Core Components

### 1. Data Collection

**User Interaction Events**:
```python
class UserEvent:
    user_id: str
    item_id: str
    event_type: str  # view, click, purchase, rating
    timestamp: int
    context: dict    # device, location, time_of_day
    
class EventCollector:
    def track_event(self, event: UserEvent):
        # Validate event
        if not self.validate(event):
            raise InvalidEventError()
        
        # Publish to Kafka
        self.kafka.produce(
            topic='user_events',
            key=event.user_id,
            value=json.dumps(event.__dict__)
        )
        
        # Update real-time features
        self.update_realtime_features(event)
```

### 2. Feature Engineering

**User Features**:
- Demographics (age, gender, location)
- Behavioral (watch history, search queries)
- Engagement (avg session time, click-through rate)
- Temporal (time of day, day of week preferences)

**Item Features**:
- Content attributes (genre, director, cast)
- Popularity metrics (view count, rating)
- Temporal (release date, trending score)

**Contextual Features**:
- Device type
- Time of day
- Location
- Season

```python
class FeatureStore:
    def get_user_features(self, user_id):
        """Retrieve user features from Redis"""
        key = f"user_features:{user_id}"
        features = self.redis.hgetall(key)
        
        return {
            'user_id': user_id,
            'age': int(features.get('age', 0)),
            'gender': features.get('gender', 'unknown'),
            'avg_session_time': float(features.get('avg_session_time', 0)),
            'favorite_genres': json.loads(features.get('favorite_genres', '[]')),
            'recent_items': json.loads(features.get('recent_items', '[]'))
        }
    
    def update_user_features(self, user_id, features):
        """Update user features incrementally"""
        key = f"user_features:{user_id}"
        pipeline = self.redis.pipeline()
        
        for feature_name, value in features.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            pipeline.hset(key, feature_name, value)
        
        pipeline.execute()
```

### 3. Recommendation Algorithms

#### A. Collaborative Filtering

**User-Based CF**:
```python
class UserBasedCF:
    def find_similar_users(self, user_id, k=10):
        """Find K most similar users using cosine similarity"""
        user_vector = self.get_user_vector(user_id)
        
        # Use approximate nearest neighbors (Annoy, FAISS)
        similar_users = self.ann_index.get_nns_by_vector(
            user_vector, 
            k, 
            include_distances=True
        )
        
        return similar_users
    
    def recommend(self, user_id, n=10):
        """Recommend items liked by similar users"""
        similar_users = self.find_similar_users(user_id)
        
        # Aggregate items from similar users
        candidate_items = defaultdict(float)
        for similar_user_id, similarity in similar_users:
            items = self.get_user_items(similar_user_id)
            for item_id, rating in items:
                candidate_items[item_id] += similarity * rating
        
        # Filter out already consumed items
        user_items = set(self.get_user_items(user_id))
        candidates = [
            (item_id, score) 
            for item_id, score in candidate_items.items()
            if item_id not in user_items
        ]
        
        # Sort by score and return top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in candidates[:n]]
```

**Item-Based CF** (More scalable):
```python
class ItemBasedCF:
    def precompute_item_similarity(self):
        """Offline: Compute item-item similarity matrix"""
        # For each item pair, compute cosine similarity
        # Based on users who interacted with both items
        
        for item_i in self.all_items:
            for item_j in self.all_items:
                if item_i == item_j:
                    continue
                
                users_i = set(self.get_users_for_item(item_i))
                users_j = set(self.get_users_for_item(item_j))
                
                # Jaccard similarity
                similarity = len(users_i & users_j) / len(users_i | users_j)
                
                # Store in Redis sorted set
                self.redis.zadd(
                    f"item_similarity:{item_i}",
                    {item_j: similarity}
                )
    
    def recommend(self, user_id, n=10):
        """Recommend items similar to user's history"""
        user_items = self.get_user_items(user_id)
        
        candidate_items = defaultdict(float)
        for item_id, rating in user_items:
            # Get similar items
            similar_items = self.redis.zrevrange(
                f"item_similarity:{item_id}",
                0, 50,
                withscores=True
            )
            
            for similar_item, similarity in similar_items:
                candidate_items[similar_item] += similarity * rating
        
        # Filter and sort
        candidates = [
            (item_id, score)
            for item_id, score in candidate_items.items()
            if item_id not in user_items
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [item_id for item_id, _ in candidates[:n]]
```

#### B. Content-Based Filtering

```python
class ContentBasedRecommender:
    def __init__(self):
        # TF-IDF vectorizer for text features
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
    def build_item_profiles(self):
        """Create feature vectors for all items"""
        item_texts = []
        item_ids = []
        
        for item in self.all_items:
            # Combine text features
            text = f"{item.title} {item.description} {item.genre} {item.tags}"
            item_texts.append(text)
            item_ids.append(item.item_id)
        
        # Create TF-IDF matrix
        self.item_vectors = self.vectorizer.fit_transform(item_texts)
        self.item_id_to_index = {
            item_id: idx for idx, item_id in enumerate(item_ids)
        }
    
    def build_user_profile(self, user_id):
        """Build user profile from interaction history"""
        user_items = self.get_user_items(user_id)
        
        # Weighted average of item vectors
        user_vector = np.zeros(self.item_vectors.shape[1])
        total_weight = 0
        
        for item_id, rating in user_items:
            idx = self.item_id_to_index[item_id]
            item_vector = self.item_vectors[idx].toarray()[0]
            user_vector += rating * item_vector
            total_weight += rating
        
        if total_weight > 0:
            user_vector /= total_weight
        
        return user_vector
    
    def recommend(self, user_id, n=10):
        """Recommend items similar to user profile"""
        user_vector = self.build_user_profile(user_id)
        
        # Compute cosine similarity with all items
        similarities = cosine_similarity(
            user_vector.reshape(1, -1),
            self.item_vectors
        )[0]
        
        # Get top N items
        top_indices = np.argsort(similarities)[::-1][:n]
        
        return [self.index_to_item_id[idx] for idx in top_indices]
```

#### C. Matrix Factorization (SVD)

```python
class MatrixFactorization:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        
    def train(self, interactions):
        """Train using Alternating Least Squares (ALS)"""
        # Build user-item matrix
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        
        R = np.zeros((n_users, n_items))
        for user_id, item_id, rating in interactions:
            user_idx = self.user_id_to_index[user_id]
            item_idx = self.item_id_to_index[item_id]
            R[user_idx, item_idx] = rating
        
        # Perform SVD
        U, sigma, Vt = np.linalg.svd(R, full_matrices=False)
        
        # Keep top K factors
        self.user_factors = U[:, :self.n_factors]
        self.item_factors = Vt[:self.n_factors, :].T
        self.sigma = np.diag(sigma[:self.n_factors])
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        user_idx = self.user_id_to_index[user_id]
        item_idx = self.item_id_to_index[item_id]
        
        user_vec = self.user_factors[user_idx]
        item_vec = self.item_factors[item_idx]
        
        return np.dot(user_vec, np.dot(self.sigma, item_vec))
    
    def recommend(self, user_id, n=10):
        """Recommend top N items for user"""
        user_idx = self.user_id_to_index[user_id]
        user_vec = self.user_factors[user_idx]
        
        # Compute scores for all items
        scores = np.dot(
            user_vec,
            np.dot(self.sigma, self.item_factors.T)
        )
        
        # Get top N
        top_indices = np.argsort(scores)[::-1][:n]
        
        return [self.index_to_item_id[idx] for idx in top_indices]
```

### 4. Hybrid Approach

```python
class HybridRecommender:
    def __init__(self):
        self.cf_recommender = ItemBasedCF()
        self.content_recommender = ContentBasedRecommender()
        self.mf_recommender = MatrixFactorization()
    
    def recommend(self, user_id, n=10):
        """Combine multiple algorithms"""
        # Get recommendations from each algorithm
        cf_recs = self.cf_recommender.recommend(user_id, n=50)
        content_recs = self.content_recommender.recommend(user_id, n=50)
        mf_recs = self.mf_recommender.recommend(user_id, n=50)
        
        # Weighted scoring
        scores = defaultdict(float)
        
        for idx, item_id in enumerate(cf_recs):
            scores[item_id] += 0.4 * (50 - idx) / 50
        
        for idx, item_id in enumerate(content_recs):
            scores[item_id] += 0.3 * (50 - idx) / 50
        
        for idx, item_id in enumerate(mf_recs):
            scores[item_id] += 0.3 * (50 - idx) / 50
        
        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [item_id for item_id, _ in ranked[:n]]
```

### 5. Serving Layer

```python
class RecommendationService:
    def __init__(self):
        self.cache = Redis()
        self.recommender = HybridRecommender()
    
    def get_recommendations(self, user_id, n=10):
        """Serve recommendations with caching"""
        cache_key = f"recs:{user_id}"
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Generate recommendations
        recommendations = self.recommender.recommend(user_id, n)
        
        # Enrich with item metadata
        enriched = []
        for item_id in recommendations:
            item_data = self.get_item_metadata(item_id)
            enriched.append(item_data)
        
        # Cache for 1 hour
        self.cache.setex(cache_key, 3600, json.dumps(enriched))
        
        return enriched
```

## Cold Start Problem

### New User Cold Start

```python
class ColdStartHandler:
    def handle_new_user(self, user_id):
        """Recommendations for users with no history"""
        # Strategy 1: Popular items
        popular = self.get_trending_items(limit=20)
        
        # Strategy 2: Demographic-based
        user_demo = self.get_user_demographics(user_id)
        demo_recs = self.get_popular_for_demographic(user_demo, limit=20)
        
        # Strategy 3: Onboarding quiz
        preferences = self.get_onboarding_preferences(user_id)
        if preferences:
            pref_recs = self.get_items_by_preferences(preferences, limit=20)
        else:
            pref_recs = []
        
        # Combine strategies
        recommendations = self.merge_and_diversify(
            [popular, demo_recs, pref_recs]
        )
        
        return recommendations[:10]
```

### New Item Cold Start

```python
def handle_new_item(self, item_id):
    """Get initial exposure for new items"""
    # Strategy 1: Content-based matching
    item_features = self.get_item_features(item_id)
    similar_items = self.find_similar_items_by_content(item_features)
    
    # Get users who liked similar items
    candidate_users = set()
    for similar_item in similar_items:
        users = self.get_users_for_item(similar_item)
        candidate_users.update(users)
    
    # Strategy 2: Exploration (show to random sample)
    random_users = self.sample_active_users(n=1000)
    candidate_users.update(random_users)
    
    return list(candidate_users)
```

## Evaluation Metrics

```python
class RecommenderEvaluator:
    def evaluate(self, recommendations, ground_truth):
        """Evaluate recommendation quality"""
        metrics = {}
        
        # Precision@K
        metrics['precision@10'] = self.precision_at_k(
            recommendations, ground_truth, k=10
        )
        
        # Recall@K
        metrics['recall@10'] = self.recall_at_k(
            recommendations, ground_truth, k=10
        )
        
        # NDCG (Normalized Discounted Cumulative Gain)
        metrics['ndcg@10'] = self.ndcg_at_k(
            recommendations, ground_truth, k=10
        )
        
        # Coverage (% of catalog recommended)
        metrics['coverage'] = len(set(recommendations)) / self.catalog_size
        
        # Diversity (avg pairwise distance)
        metrics['diversity'] = self.calculate_diversity(recommendations)
        
        return metrics
```

## Key Takeaways

1. **Hybrid approaches** outperform single-algorithm systems
2. **Feature engineering** is critical for model performance
3. **Real-time updates** improve recommendation freshness
4. **Cold start** requires special handling strategies
5. **Caching** is essential for low-latency serving
6. **A/B testing** validates algorithm improvements
7. **Diversity** prevents filter bubbles and improves engagement

## References
- Netflix Recommendations: https://netflixtechblog.com/
- Collaborative Filtering: https://en.wikipedia.org/wiki/Collaborative_filtering
- Matrix Factorization: https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf
