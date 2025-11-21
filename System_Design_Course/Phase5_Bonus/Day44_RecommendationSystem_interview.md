# Day 44: Recommendation System - Interview Prep

## Common Interview Questions

### Q1: Design a recommendation system like Netflix

**Approach**:

1. **Clarify Requirements**
   - Scale: 200M users, 10K movies
   - Latency: < 100ms for recommendations
   - Personalization: Yes, based on watch history
   - Cold start: Handle new users and new content

2. **High-Level Architecture**
   ```
   User → API Gateway → Recommendation Service → Feature Store (Redis)
                                               → Model Store (S3)
                                               → Cassandra (User history)
   
   Offline: Kafka → Spark → Model Training → Model Store
   ```

3. **Algorithm Selection**
   - **Collaborative Filtering**: User-item matrix factorization
   - **Content-Based**: Match user preferences to movie features
   - **Hybrid**: Combine both approaches
   - **Deep Learning**: Neural collaborative filtering for better accuracy

4. **Data Model**
   ```python
   # User-Item Interactions
   {
     "user_id": "user123",
     "item_id": "movie456",
     "rating": 4.5,
     "timestamp": 1234567890,
     "watch_duration": 7200  # seconds
   }
   
   # User Features
   {
     "user_id": "user123",
     "favorite_genres": ["action", "sci-fi"],
     "avg_rating": 4.2,
     "watch_history": ["movie1", "movie2", ...]
   }
   ```

5. **Serving Strategy**
   - **Offline**: Precompute recommendations daily
   - **Online**: Real-time adjustments based on recent activity
   - **Caching**: Cache top recommendations per user (TTL: 1 hour)

**Follow-ups**:
- **Q**: How do you handle new users?
  - **A**: Show popular/trending items, ask for preferences during onboarding, use demographic-based recommendations
  
- **Q**: How do you evaluate recommendation quality?
  - **A**: Precision@K, Recall@K, NDCG, A/B testing (CTR, watch time)

---

### Q2: How would you implement collaborative filtering at scale?

**Answer**:

1. **Matrix Factorization (SVD)**
   ```python
   # User-item matrix R ≈ U × Σ × V^T
   # U: user factors, V: item factors
   
   def train_matrix_factorization(interactions, n_factors=50):
       # Build sparse matrix
       R = build_user_item_matrix(interactions)
       
       # Perform SVD
       U, sigma, Vt = scipy.sparse.linalg.svds(R, k=n_factors)
       
       return U, sigma, Vt
   
   def recommend(user_id, U, sigma, Vt, k=10):
       user_idx = user_id_to_index[user_id]
       user_vec = U[user_idx]
       
       # Predict scores for all items
       scores = np.dot(user_vec, np.dot(np.diag(sigma), Vt))
       
       # Get top K
       top_k_indices = np.argsort(scores)[::-1][:k]
       return [index_to_item_id[idx] for idx in top_k_indices]
   ```

2. **Scalability Challenges**
   - **Problem**: 100M users × 1M items = 100 trillion entries
   - **Solution**: Use sparse matrix representation, only store non-zero entries

3. **Distributed Training**
   - Use Spark MLlib for distributed ALS (Alternating Least Squares)
   ```python
   from pyspark.ml.recommendation import ALS
   
   als = ALS(
       maxIter=10,
       regParam=0.01,
       userCol="user_id",
       itemCol="item_id",
       ratingCol="rating",
       coldStartStrategy="drop"
   )
   
   model = als.fit(training_data)
   ```

4. **Serving at Scale**
   - **Precompute**: Generate recommendations offline for all users
   - **Store in Redis**: Hash map per user
   ```python
   # Store recommendations
   redis.hset(f"recs:{user_id}", mapping={
       "items": json.dumps(recommended_items),
       "timestamp": time.time()
   })
   ```

**Trade-offs**:
- **Offline precomputation**: Fast serving, but stale recommendations
- **Online computation**: Fresh recommendations, but higher latency

---

### Q3: How do you handle the cold start problem?

**Answer**:

**1. New User Cold Start**

**Strategy A: Popular Items**
```python
def recommend_for_new_user(user_id):
    # Show trending/popular items
    popular = redis.zrevrange("trending_items", 0, 19, withscores=True)
    return [item_id for item_id, score in popular]
```

**Strategy B: Onboarding Quiz**
```python
def recommend_from_preferences(preferences):
    # User selects favorite genres during signup
    candidate_items = []
    for genre in preferences['genres']:
        items = get_top_items_by_genre(genre, limit=10)
        candidate_items.extend(items)
    
    # Diversify and return
    return diversify(candidate_items, k=20)
```

**Strategy C: Demographic-Based**
```python
def recommend_by_demographics(user_demographics):
    # Find similar users by age, gender, location
    similar_users = find_users_with_demographics(user_demographics)
    
    # Aggregate their favorite items
    items = aggregate_popular_items(similar_users)
    return items[:20]
```

**2. New Item Cold Start**

**Strategy A: Content-Based Matching**
```python
def recommend_new_item(item_id):
    # Find similar items by content features
    item_features = get_item_features(item_id)
    similar_items = find_similar_items_by_content(item_features)
    
    # Get users who liked similar items
    candidate_users = []
    for similar_item in similar_items:
        users = get_users_who_liked(similar_item)
        candidate_users.extend(users)
    
    return candidate_users
```

**Strategy B: Exploration (Epsilon-Greedy)**
```python
def recommend_with_exploration(user_id, epsilon=0.1):
    if random.random() < epsilon:
        # Explore: show random items (including new ones)
        return sample_random_items(k=10)
    else:
        # Exploit: show personalized recommendations
        return get_personalized_recommendations(user_id, k=10)
```

---

### Q4: How would you design a real-time recommendation system?

**Answer**:

**Architecture**:
```
User Event → Kafka → Stream Processor → Feature Store (Redis)
                                      → Recommendation Service
```

**1. Real-Time Feature Updates**
```python
class RealtimeFeatureUpdater:
    def process_event(self, event):
        user_id = event['user_id']
        item_id = event['item_id']
        
        # Update recent items (sliding window)
        redis.lpush(f"user:{user_id}:recent", item_id)
        redis.ltrim(f"user:{user_id}:recent", 0, 49)  # Keep last 50
        
        # Update genre preferences (exponential moving average)
        genre = get_item_genre(item_id)
        current_score = float(redis.hget(f"user:{user_id}:genres", genre) or 0)
        new_score = 0.9 * current_score + 0.1 * 1.0
        redis.hset(f"user:{user_id}:genres", genre, new_score)
        
        # Invalidate recommendation cache
        redis.delete(f"recs:{user_id}")
```

**2. Fast Serving with Approximate Nearest Neighbors**
```python
class FastRecommender:
    def __init__(self):
        # Build FAISS index for item embeddings
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(item_embeddings)
    
    def recommend(self, user_id, k=10):
        # Get user embedding
        user_emb = self.get_user_embedding(user_id)
        
        # Fast ANN search
        distances, indices = self.index.search(
            user_emb.reshape(1, -1), 
            k=100
        )
        
        # Re-rank top 100 with full model
        candidates = [self.item_ids[idx] for idx in indices[0]]
        scores = self.full_model.score(user_id, candidates)
        
        # Return top K
        top_k = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k]
        return [item_id for item_id, _ in top_k]
```

**3. Latency Optimization**
- **L1 Cache**: In-memory cache (local to service)
- **L2 Cache**: Redis cache (shared)
- **Async Updates**: Don't block on cache invalidation

**Target Latency**: < 50ms P95

---

### Q5: How do you evaluate recommendation quality?

**Answer**:

**1. Offline Metrics**

**Precision@K**:
```python
def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    
    hits = len([item for item in recommended_k if item in relevant_set])
    return hits / k
```

**Recall@K**:
```python
def recall_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    
    hits = len([item for item in recommended_k if item in relevant_set])
    return hits / len(relevant_set) if relevant_set else 0
```

**NDCG (Normalized Discounted Cumulative Gain)**:
```python
def ndcg_at_k(recommended, relevant_scores, k=10):
    dcg = sum(
        (2**relevant_scores.get(item, 0) - 1) / np.log2(idx + 2)
        for idx, item in enumerate(recommended[:k])
    )
    
    # Ideal DCG
    ideal_scores = sorted(relevant_scores.values(), reverse=True)
    idcg = sum(
        (2**score - 1) / np.log2(idx + 2)
        for idx, score in enumerate(ideal_scores[:k])
    )
    
    return dcg / idcg if idcg > 0 else 0
```

**2. Online Metrics (A/B Testing)**

- **Click-Through Rate (CTR)**: % of recommendations clicked
- **Conversion Rate**: % of clicks that lead to purchase/watch
- **Engagement Time**: Total time spent on recommended items
- **Diversity**: Variety of recommended items
- **Coverage**: % of catalog recommended

**3. Business Metrics**
- **Revenue**: Total revenue from recommended items
- **User Retention**: % of users who return
- **Session Length**: Time spent on platform

---

### Q6: How would you implement diversity in recommendations?

**Answer**:

**1. Maximal Marginal Relevance (MMR)**
```python
def mmr_diversify(candidates, k=10, lambda_param=0.5):
    selected = []
    remaining = candidates.copy()
    
    # Select first item (highest relevance)
    selected.append(remaining.pop(0))
    
    while len(selected) < k and remaining:
        best_item = None
        best_score = -float('inf')
        
        for item in remaining:
            # Relevance score
            relevance = item['score']
            
            # Max similarity to selected items
            max_sim = max(
                similarity(item, selected_item)
                for selected_item in selected
            )
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_item = item
        
        selected.append(best_item)
        remaining.remove(best_item)
    
    return selected
```

**2. Genre Diversity**
```python
def ensure_genre_diversity(recommendations):
    """Ensure at least 3 different genres in top 10"""
    genres_seen = set()
    diversified = []
    
    for item in recommendations:
        if len(diversified) >= 10:
            break
        
        item_genre = item['genre']
        
        # Add if genre not over-represented
        if genres_seen.count(item_genre) < 4:
            diversified.append(item)
            genres_seen.add(item_genre)
    
    return diversified
```

**3. Temporal Diversity**
```python
def temporal_diversity(recommendations):
    """Mix old and new content"""
    old_items = [item for item in recommendations if item['age_days'] > 365]
    new_items = [item for item in recommendations if item['age_days'] <= 365]
    
    # 70% new, 30% old
    return new_items[:7] + old_items[:3]
```

---

### Q7: How do you handle scalability for billions of users?

**Answer**:

**1. Sharding Strategy**
```python
def get_shard_id(user_id, num_shards=100):
    # Consistent hashing
    hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return hash_value % num_shards

def get_recommendations(user_id):
    shard_id = get_shard_id(user_id)
    redis_client = redis_pool[shard_id]
    
    # Fetch from appropriate shard
    recs = redis_client.hget(f"recs:{user_id}", "items")
    return json.loads(recs) if recs else []
```

**2. Tiered Storage**
- **Hot tier** (Redis): Active users (last 30 days) - 100M users
- **Warm tier** (Cassandra): Inactive users - 1B users
- **Cold tier** (S3): Historical data

**3. Batch Precomputation**
```python
# Spark job runs daily
def precompute_recommendations():
    # Load user-item interactions
    interactions = spark.read.parquet("s3://data/interactions/")
    
    # Train model
    als = ALS(maxIter=10, regParam=0.01)
    model = als.fit(interactions)
    
    # Generate recommendations for all users
    user_recs = model.recommendForAllUsers(20)
    
    # Write to Redis
    for row in user_recs.collect():
        user_id = row['user_id']
        items = [rec['item_id'] for rec in row['recommendations']]
        
        redis.hset(f"recs:{user_id}", "items", json.dumps(items))
```

**4. Caching Strategy**
- **User-level cache**: Recommendations per user (TTL: 1 hour)
- **Popular items cache**: Trending items (TTL: 15 minutes)
- **Model cache**: Loaded models in memory

---

## Code Snippets

### Complete Recommendation Service

```python
class RecommendationService:
    def __init__(self):
        self.redis = Redis()
        self.model = load_model('s3://models/recommender.pkl')
        self.feature_store = FeatureStore()
    
    def get_recommendations(self, user_id, k=10):
        # Check cache
        cache_key = f"recs:{user_id}"
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Get user features
        user_features = self.feature_store.get_user_features(user_id)
        
        # Generate recommendations
        candidates = self.model.predict(user_features, k=100)
        
        # Re-rank and diversify
        reranked = self.rerank(user_id, candidates)
        diversified = self.diversify(reranked, k=k)
        
        # Cache result
        self.redis.setex(cache_key, 3600, json.dumps(diversified))
        
        return diversified
    
    def rerank(self, user_id, candidates):
        # Apply business rules, boost new content, etc.
        return sorted(candidates, key=lambda x: x['score'], reverse=True)
    
    def diversify(self, items, k=10):
        # MMR diversification
        return mmr_diversify(items, k=k)
```

---

## Key Metrics to Monitor

1. **Latency**: P50, P95, P99 (target: < 100ms)
2. **Cache Hit Rate**: > 90%
3. **CTR**: Click-through rate
4. **Diversity**: Avg pairwise distance
5. **Coverage**: % of catalog recommended
6. **Model Freshness**: Time since last training

---

## Common Pitfalls

1. **Ignoring cold start** → Poor experience for new users
2. **No diversity** → Filter bubbles, user fatigue
3. **Stale recommendations** → Not reflecting recent behavior
4. **Over-optimization for CTR** → Clickbait, poor long-term engagement
5. **No A/B testing** → Can't validate improvements
6. **Ignoring scalability** → System breaks at scale

---

## Interview Tips

1. **Start broad, then dive deep** - Don't jump into algorithms immediately
2. **Discuss trade-offs** - Accuracy vs. latency, diversity vs. relevance
3. **Consider scale** - What works for 1K users won't work for 1B
4. **Mention evaluation** - How do you know if recommendations are good?
5. **Think about production** - Caching, monitoring, A/B testing
6. **Use concrete examples** - "Like Netflix" or "Like Amazon"

---

## Further Reading

- Netflix Recommendations: https://netflixtechblog.com/
- Matrix Factorization: https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf
- Deep Learning for Recommendations: https://arxiv.org/abs/1708.05031
- Collaborative Filtering: https://en.wikipedia.org/wiki/Collaborative_filtering
