# Day 44: Recommendation System - Deep Dive

## Advanced Machine Learning Approaches

### 1. Deep Learning for Recommendations

#### Neural Collaborative Filtering (NCF)

```python
import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=50, hidden_layers=[128, 64, 32]):
        super(NCF, self).__init__()
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        
        # Pass through MLP
        output = self.mlp(x)
        
        return output

# Training
model = NCF(n_users=100000, n_items=50000)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(10):
    for batch in train_loader:
        user_ids, item_ids, labels = batch
        
        # Forward pass
        predictions = model(user_ids, item_ids)
        loss = criterion(predictions, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### Two-Tower Model (YouTube DNN)

```python
class TwoTowerModel(nn.Module):
    def __init__(self, user_feature_dim, item_feature_dim, embedding_dim=128):
        super(TwoTowerModel, self).__init__()
        
        # User tower
        self.user_tower = nn.Sequential(
            nn.Linear(user_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
        # Item tower
        self.item_tower = nn.Sequential(
            nn.Linear(item_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
    
    def forward(self, user_features, item_features):
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        
        # Cosine similarity
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        
        score = torch.sum(user_emb * item_emb, dim=1)
        
        return score
    
    def get_user_embedding(self, user_features):
        return self.user_tower(user_features)
    
    def get_item_embedding(self, item_features):
        return self.item_tower(item_features)
```

**Serving with Two-Tower**:
```python
class TwoTowerServing:
    def __init__(self, model):
        self.model = model
        self.item_embeddings = None
        self.item_ids = None
        
    def precompute_item_embeddings(self, all_items):
        """Offline: Compute embeddings for all items"""
        self.model.eval()
        with torch.no_grad():
            item_features = torch.tensor([item.features for item in all_items])
            self.item_embeddings = self.model.get_item_embedding(item_features)
            self.item_ids = [item.id for item in all_items]
        
        # Build FAISS index for fast retrieval
        self.index = faiss.IndexFlatIP(self.item_embeddings.shape[1])
        self.index.add(self.item_embeddings.numpy())
    
    def recommend(self, user_features, k=10):
        """Online: Fast retrieval using precomputed embeddings"""
        with torch.no_grad():
            user_emb = self.model.get_user_embedding(
                torch.tensor(user_features).unsqueeze(0)
            )
        
        # Search nearest neighbors
        distances, indices = self.index.search(user_emb.numpy(), k)
        
        return [self.item_ids[idx] for idx in indices[0]]
```

### 2. Contextual Bandits for Exploration-Exploitation

```python
class ContextualBandit:
    """Thompson Sampling for recommendation exploration"""
    
    def __init__(self, n_items):
        self.n_items = n_items
        # Beta distribution parameters for each item
        self.alpha = np.ones(n_items)  # Successes
        self.beta = np.ones(n_items)   # Failures
    
    def select_item(self, context=None):
        """Select item to recommend"""
        # Sample from Beta distribution for each item
        samples = np.random.beta(self.alpha, self.beta)
        
        # Select item with highest sample
        return np.argmax(samples)
    
    def update(self, item_id, reward):
        """Update based on user feedback"""
        if reward > 0:
            self.alpha[item_id] += reward
        else:
            self.beta[item_id] += 1
    
    def get_expected_reward(self, item_id):
        """Get expected reward for item"""
        return self.alpha[item_id] / (self.alpha[item_id] + self.beta[item_id])
```

**LinUCB (Linear Upper Confidence Bound)**:
```python
class LinUCB:
    def __init__(self, n_features, alpha=1.0):
        self.alpha = alpha
        self.n_features = n_features
        
        # Initialize parameters
        self.A = np.identity(n_features)  # Design matrix
        self.b = np.zeros(n_features)     # Response vector
    
    def select_item(self, context, candidate_items):
        """Select item with highest UCB"""
        A_inv = np.linalg.inv(self.A)
        theta = A_inv.dot(self.b)
        
        best_item = None
        best_ucb = -float('inf')
        
        for item in candidate_items:
            item_features = self.get_item_features(item, context)
            
            # Expected reward
            expected_reward = theta.dot(item_features)
            
            # Confidence bound
            confidence = self.alpha * np.sqrt(
                item_features.dot(A_inv).dot(item_features)
            )
            
            ucb = expected_reward + confidence
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_item = item
        
        return best_item
    
    def update(self, item_features, reward):
        """Update model with observed reward"""
        self.A += np.outer(item_features, item_features)
        self.b += reward * item_features
```

### 3. Sequential Recommendations (Session-Based)

#### RNN-based Recommender

```python
class GRU4Rec(nn.Module):
    """GRU for session-based recommendations"""
    
    def __init__(self, n_items, embedding_dim=100, hidden_dim=100):
        super(GRU4Rec, self).__init__()
        
        self.embedding = nn.Embedding(n_items, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_items)
    
    def forward(self, session_items):
        # session_items: [batch_size, seq_len]
        embedded = self.embedding(session_items)
        
        # GRU forward pass
        output, hidden = self.gru(embedded)
        
        # Use last hidden state for prediction
        logits = self.fc(output[:, -1, :])
        
        return logits

# Training
model = GRU4Rec(n_items=50000)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for batch in train_loader:
    session_items, target_item = batch
    
    logits = model(session_items)
    loss = criterion(logits, target_item)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

#### Transformer for Recommendations (BERT4Rec)

```python
class BERT4Rec(nn.Module):
    def __init__(self, n_items, max_len=50, n_layers=2, n_heads=2):
        super(BERT4Rec, self).__init__()
        
        self.item_embedding = nn.Embedding(n_items + 1, 64)  # +1 for mask token
        self.position_embedding = nn.Embedding(max_len, 64)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, 
            nhead=n_heads,
            dim_feedforward=256
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.fc = nn.Linear(64, n_items)
    
    def forward(self, item_seq):
        batch_size, seq_len = item_seq.shape
        
        # Item embeddings
        item_emb = self.item_embedding(item_seq)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        
        # Combine
        x = item_emb + pos_emb
        
        # Transformer
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        
        # Predict next item
        logits = self.fc(x)
        
        return logits
```

### 4. Multi-Task Learning

```python
class MultiTaskRecommender(nn.Module):
    """Jointly predict clicks, likes, and purchases"""
    
    def __init__(self, n_users, n_items, embedding_dim=64):
        super(MultiTaskRecommender, self).__init__()
        
        # Shared embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task-specific heads
        self.click_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.like_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.purchase_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        x = torch.cat([user_emb, item_emb], dim=1)
        shared_repr = self.shared(x)
        
        click_pred = self.click_head(shared_repr)
        like_pred = self.like_head(shared_repr)
        purchase_pred = self.purchase_head(shared_repr)
        
        return click_pred, like_pred, purchase_pred

# Training with multi-task loss
def train_multitask(model, batch):
    user_ids, item_ids, click_labels, like_labels, purchase_labels = batch
    
    click_pred, like_pred, purchase_pred = model(user_ids, item_ids)
    
    # Weighted loss
    click_loss = F.binary_cross_entropy(click_pred, click_labels)
    like_loss = F.binary_cross_entropy(like_pred, like_labels)
    purchase_loss = F.binary_cross_entropy(purchase_pred, purchase_labels)
    
    total_loss = 0.5 * click_loss + 0.3 * like_loss + 0.2 * purchase_loss
    
    return total_loss
```

### 5. Real-Time Feature Engineering

```python
class RealtimeFeatureEngine:
    def __init__(self):
        self.redis = Redis()
        self.kafka_consumer = KafkaConsumer('user_events')
    
    def process_event_stream(self):
        """Process events in real-time"""
        for message in self.kafka_consumer:
            event = json.loads(message.value)
            
            # Update user features
            self.update_user_features(event)
            
            # Update item features
            self.update_item_features(event)
            
            # Update session features
            self.update_session_features(event)
    
    def update_user_features(self, event):
        """Incrementally update user features"""
        user_id = event['user_id']
        
        pipeline = self.redis.pipeline()
        
        # Update recent items (sliding window)
        recent_key = f"user:{user_id}:recent_items"
        pipeline.lpush(recent_key, event['item_id'])
        pipeline.ltrim(recent_key, 0, 49)  # Keep last 50
        
        # Update genre preferences (exponential moving average)
        genre = event['item_genre']
        genre_key = f"user:{user_id}:genre_scores"
        current_score = float(self.redis.hget(genre_key, genre) or 0)
        new_score = 0.9 * current_score + 0.1 * 1.0  # EMA with alpha=0.1
        pipeline.hset(genre_key, genre, new_score)
        
        # Update activity metrics
        pipeline.hincrby(f"user:{user_id}:stats", 'total_interactions', 1)
        pipeline.hset(f"user:{user_id}:stats", 'last_active', time.time())
        
        pipeline.execute()
    
    def get_realtime_features(self, user_id):
        """Fetch latest features for serving"""
        pipeline = self.redis.pipeline()
        
        # Recent items
        pipeline.lrange(f"user:{user_id}:recent_items", 0, 49)
        
        # Genre preferences
        pipeline.hgetall(f"user:{user_id}:genre_scores")
        
        # Activity stats
        pipeline.hgetall(f"user:{user_id}:stats")
        
        results = pipeline.execute()
        
        return {
            'recent_items': results[0],
            'genre_scores': results[1],
            'stats': results[2]
        }
```

### 6. Diversity and Serendipity

```python
class DiversityOptimizer:
    def diversify_recommendations(self, candidates, k=10):
        """Maximize diversity while maintaining relevance"""
        selected = []
        remaining = candidates.copy()
        
        # Select first item (highest score)
        selected.append(remaining.pop(0))
        
        while len(selected) < k and remaining:
            best_item = None
            best_score = -float('inf')
            
            for item in remaining:
                # Relevance score
                relevance = item['score']
                
                # Diversity score (min distance to selected items)
                diversity = min(
                    self.item_distance(item, selected_item)
                    for selected_item in selected
                )
                
                # Combined score (trade-off parameter lambda)
                combined = 0.7 * relevance + 0.3 * diversity
                
                if combined > best_score:
                    best_score = combined
                    best_item = item
            
            selected.append(best_item)
            remaining.remove(best_item)
        
        return selected
    
    def item_distance(self, item1, item2):
        """Compute distance between items"""
        # Jaccard distance on genres
        genres1 = set(item1['genres'])
        genres2 = set(item2['genres'])
        
        intersection = len(genres1 & genres2)
        union = len(genres1 | genres2)
        
        return 1 - (intersection / union) if union > 0 else 1
```

**Determinantal Point Process (DPP) for Diversity**:
```python
class DPPDiversifier:
    def select_diverse_items(self, items, k=10):
        """Use DPP to select diverse subset"""
        n = len(items)
        
        # Build kernel matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Quality * Diversity
                quality = items[i]['score'] * items[j]['score']
                similarity = self.cosine_similarity(
                    items[i]['features'],
                    items[j]['features']
                )
                K[i, j] = quality * similarity
        
        # Sample from DPP (approximate)
        selected_indices = self.sample_dpp(K, k)
        
        return [items[i] for i in selected_indices]
```

### 7. A/B Testing Framework

```python
class RecommenderABTest:
    def __init__(self):
        self.experiments = {}
        self.metrics_store = MetricsStore()
    
    def assign_variant(self, user_id, experiment_id):
        """Assign user to control or treatment"""
        # Consistent hashing for stable assignment
        hash_value = int(hashlib.md5(
            f"{user_id}:{experiment_id}".encode()
        ).hexdigest(), 16)
        
        variant = 'control' if hash_value % 2 == 0 else 'treatment'
        
        return variant
    
    def get_recommendations(self, user_id, experiment_id):
        """Serve recommendations based on experiment variant"""
        variant = self.assign_variant(user_id, experiment_id)
        
        if variant == 'control':
            recs = self.baseline_recommender.recommend(user_id)
        else:
            recs = self.new_recommender.recommend(user_id)
        
        # Log exposure
        self.log_exposure(user_id, experiment_id, variant)
        
        return recs
    
    def track_outcome(self, user_id, experiment_id, metric, value):
        """Track experiment metrics"""
        variant = self.assign_variant(user_id, experiment_id)
        
        self.metrics_store.record(
            experiment_id=experiment_id,
            variant=variant,
            metric=metric,
            value=value,
            timestamp=time.time()
        )
    
    def analyze_results(self, experiment_id):
        """Statistical significance testing"""
        control_metrics = self.metrics_store.get_metrics(
            experiment_id, 'control'
        )
        treatment_metrics = self.metrics_store.get_metrics(
            experiment_id, 'treatment'
        )
        
        # T-test for significance
        t_stat, p_value = stats.ttest_ind(
            control_metrics['ctr'],
            treatment_metrics['ctr']
        )
        
        return {
            'control_mean': np.mean(control_metrics['ctr']),
            'treatment_mean': np.mean(treatment_metrics['ctr']),
            'lift': (np.mean(treatment_metrics['ctr']) - 
                    np.mean(control_metrics['ctr'])) / 
                    np.mean(control_metrics['ctr']),
            'p_value': p_value,
            'significant': p_value < 0.05
        }
```

### 8. Production Monitoring

```python
class RecommenderMonitoring:
    def __init__(self):
        self.statsd = StatsD()
        self.logger = logging.getLogger(__name__)
    
    def track_recommendation_request(self, user_id, latency, num_recs):
        """Track serving metrics"""
        self.statsd.timing('recommender.latency', latency)
        self.statsd.increment('recommender.requests')
        self.statsd.histogram('recommender.num_recs', num_recs)
    
    def track_recommendation_quality(self, user_id, recommendations):
        """Track quality metrics"""
        # Diversity
        diversity = self.calculate_diversity(recommendations)
        self.statsd.gauge('recommender.diversity', diversity)
        
        # Coverage
        coverage = len(set(recommendations)) / self.catalog_size
        self.statsd.gauge('recommender.coverage', coverage)
        
        # Freshness (avg age of recommended items)
        avg_age = np.mean([
            (time.time() - item.created_at) / 86400
            for item in recommendations
        ])
        self.statsd.gauge('recommender.avg_item_age_days', avg_age)
    
    def detect_anomalies(self):
        """Detect recommendation quality issues"""
        # Check if diversity dropped suddenly
        recent_diversity = self.get_recent_metric('diversity', window=3600)
        if np.mean(recent_diversity) < 0.3:
            self.alert('Low diversity detected')
        
        # Check if latency spiked
        recent_latency = self.get_recent_metric('latency', window=300)
        if np.percentile(recent_latency, 95) > 200:
            self.alert('High latency detected')
```

## Key Takeaways

1. **Deep learning** models (NCF, Two-Tower) outperform traditional methods
2. **Contextual bandits** balance exploration and exploitation
3. **Sequential models** (RNN, Transformer) capture temporal patterns
4. **Multi-task learning** improves overall performance
5. **Real-time features** enhance recommendation freshness
6. **Diversity** is critical for user engagement
7. **A/B testing** validates improvements rigorously
8. **Monitoring** ensures production quality
