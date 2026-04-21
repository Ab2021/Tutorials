# 🏢 FAANG: Advanced ML System Design Coding
> **Focus:** ML Pipeline Engineering, Feature Engineering Algorithms, Production ML

---

## 🏢 PROBLEM 1: Online Feature Engineering — Rolling Window Features

### Theory
In production ML (fraud, recommendation), features must be computed in real-time from streaming events. Common pattern: rolling window aggregations (last N minutes/transactions).

**Design requirements:**
- Sub-millisecond feature computation
- Time-correct features (no future leakage)
- Efficient updates (O(1) per new event)

### Implementation
```python
from collections import deque
from typing import Dict, List, Optional
import numpy as np

class RollingWindowFeatures:
    """
    Compute rolling window statistics over streaming entity events.
    
    Features:
    - Transaction count in window
    - Transaction sum/mean in window
    - Fraud rate in window
    - Velocity (transactions per hour)
    
    Time: O(1) amortized per event
    Space: O(window_size)
    """
    
    def __init__(self, window_secs: int = 3600):
        self.window_secs = window_secs
        # Store (timestamp, amount, is_fraud) tuples
        self.events: deque = deque()
        self._running_sum = 0.0
        self._running_fraud_count = 0
    
    def _evict_old(self, current_time: float) -> None:
        """Remove events outside the window."""
        cutoff = current_time - self.window_secs
        while self.events and self.events[0][0] < cutoff:
            ts, amt, is_fraud = self.events.popleft()
            self._running_sum -= amt
            self._running_fraud_count -= int(is_fraud)
    
    def add_event(self, timestamp: float, amount: float, is_fraud: bool = False) -> Dict:
        """
        Add event and return current feature vector.
        Called with CURRENT event BEFORE recording (to avoid leakage).
        """
        self._evict_old(timestamp)
        
        # Compute features BEFORE adding current event (proper temporal ordering)
        features = self._compute_features(timestamp)
        
        # Now add current event to state
        self.events.append((timestamp, amount, is_fraud))
        self._running_sum += amount
        self._running_fraud_count += int(is_fraud)
        
        return features
    
    def _compute_features(self, current_time: float) -> Dict:
        n = len(self.events)
        if n == 0:
            return {
                'tx_count': 0,
                'tx_sum': 0.0,
                'tx_mean': 0.0,
                'tx_std': 0.0,
                'fraud_rate': 0.0,
                'velocity_per_hour': 0.0,
                'max_amount': 0.0,
                'time_since_last_tx': self.window_secs,
            }
        
        amounts = [e[1] for e in self.events]
        window_duration = max(current_time - self.events[0][0], 1)
        
        return {
            'tx_count': n,
            'tx_sum': self._running_sum,
            'tx_mean': self._running_sum / n,
            'tx_std': float(np.std(amounts)) if n > 1 else 0.0,
            'fraud_rate': self._running_fraud_count / n,
            'velocity_per_hour': n / (window_duration / 3600),
            'max_amount': max(amounts),
            'time_since_last_tx': current_time - self.events[-1][0],
        }

class MultiEntityFeatureStore:
    """
    Feature store for multiple entities (users/cards) simultaneously.
    Uses a dict of RollingWindowFeatures instances.
    """
    
    def __init__(self, window_secs: int = 3600):
        self.window_secs = window_secs
        self.entity_windows: Dict[str, RollingWindowFeatures] = {}
    
    def get_features(self, entity_id: str, timestamp: float, 
                     amount: float, is_fraud: bool = False) -> Dict:
        """Get features for entity, creating window if first seen."""
        if entity_id not in self.entity_windows:
            self.entity_windows[entity_id] = RollingWindowFeatures(self.window_secs)
        return self.entity_windows[entity_id].add_event(timestamp, amount, is_fraud)

# Test
store = MultiEntityFeatureStore(window_secs=600)  # 10-minute window

# Simulate transaction stream
transactions = [
    ('user_001', 1000.0, 25.0,  False),
    ('user_001', 1010.0, 150.0, False),
    ('user_001', 1020.0, 5000.0, True),   # Suspicious large amount
    ('user_002', 1005.0, 30.0,  False),
    ('user_001', 1025.0, 4800.0, True),   # Another large
]

print("Real-time features:")
for user, ts, amt, fraud in transactions:
    features = store.get_features(user, ts, amt, fraud)
    print(f"\n{user} @ t={ts}: amount=${amt}")
    print(f"  tx_count={features['tx_count']}, "
          f"fraud_rate={features['fraud_rate']:.2f}, "
          f"velocity={features['velocity_per_hour']:.1f}/hr")
```

---

## 🏢 PROBLEM 2: Mini Batch Generator with Time-Based Split

### Theory
**Critical for fraud/temporal ML:** Standard random train/test split leaks future information.

**Point-in-time correct split:** All training features must be computed using only data available BEFORE the prediction timestamp.

**Stratified batching:** Maintain class ratio within each mini-batch for stable gradients.

### Implementation
```python
class TemporalTrainTestSplit:
    """
    Time-based train/test split that prevents temporal data leakage.
    
    Unlike random split, this uses time as the split dimension:
    - Training: events before cutoff_date
    - Testing: events after cutoff_date
    - Gap: optional buffer to avoid information bleed (e.g., 7-day gap)
    """
    
    def __init__(self, test_ratio: float = 0.2, gap_days: int = 0):
        self.test_ratio = test_ratio
        self.gap_secs = gap_days * 86400
    
    def split(self, X: np.ndarray, y: np.ndarray, 
              timestamps: np.ndarray) -> tuple:
        """
        Split data temporally.
        
        X: (n, features)
        y: (n,) labels
        timestamps: (n,) unix timestamps
        
        Returns: (X_train, X_test, y_train, y_test)
        """
        # Sort by time
        sort_idx = np.argsort(timestamps)
        X = X[sort_idx]
        y = y[sort_idx]
        timestamps = timestamps[sort_idx]
        
        # Find cutoff point
        n = len(X)
        train_end_idx = int(n * (1 - self.test_ratio))
        cutoff_time = timestamps[train_end_idx]
        
        # Training: before cutoff
        train_mask = timestamps < cutoff_time
        # Test: after cutoff + gap
        test_mask = timestamps >= (cutoff_time + self.gap_secs)
        
        return (X[train_mask], X[test_mask], 
                y[train_mask], y[test_mask])

class StratifiedMiniBatchGenerator:
    """
    Mini-batch generator that maintains class ratio per batch.
    Critical for imbalanced fraud datasets.
    """
    
    def __init__(self, batch_size: int = 256, fraud_ratio: float = 0.1,
                 shuffle: bool = True, seed: int = 42):
        """
        fraud_ratio: desired fraction of positive (fraud) samples per batch.
        """
        self.batch_size = batch_size
        self.fraud_ratio = fraud_ratio
        self.shuffle = shuffle
        self.seed = seed
    
    def generate(self, X: np.ndarray, y: np.ndarray):
        """Yield (X_batch, y_batch) batches with controlled class ratio."""
        rng = np.random.RandomState(self.seed)
        
        pos_idx = np.where(y == 1)[0]  # Fraud indices
        neg_idx = np.where(y == 0)[0]  # Normal indices
        
        n_pos_per_batch = max(1, int(self.batch_size * self.fraud_ratio))
        n_neg_per_batch = self.batch_size - n_pos_per_batch
        
        n_batches = len(neg_idx) // n_neg_per_batch
        
        if self.shuffle:
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
        
        for i in range(n_batches):
            # Sample negatives sequentially
            neg_batch = neg_idx[i*n_neg_per_batch : (i+1)*n_neg_per_batch]
            
            # Sample positives (cycle through if needed)
            pos_batch_idx = rng.choice(pos_idx, n_pos_per_batch, replace=True)
            
            batch_idx = np.concatenate([neg_batch, pos_batch_idx])
            if self.shuffle:
                rng.shuffle(batch_idx)
            
            yield X[batch_idx], y[batch_idx]

# Test
np.random.seed(42)
n = 10000
X_all = np.random.randn(n, 20)
y_all = (np.random.rand(n) < 0.02).astype(int)  # 2% fraud
ts_all = np.sort(np.random.randint(0, 1000000, n))  # Random timestamps

# Temporal split
splitter = TemporalTrainTestSplit(test_ratio=0.2, gap_days=7)
X_tr, X_te, y_tr, y_te = splitter.split(X_all, y_all, ts_all)
print(f"Train: {len(X_tr):,} samples, {y_tr.mean():.3%} fraud")
print(f"Test:  {len(X_te):,} samples, {y_te.mean():.3%} fraud")

# Stratified batching
gen = StratifiedMiniBatchGenerator(batch_size=128, fraud_ratio=0.3)
for i, (Xb, yb) in enumerate(gen.generate(X_tr, y_tr)):
    if i < 3:
        print(f"Batch {i}: size={len(Xb)}, fraud_ratio={yb.mean():.3f}")
```

---

## 🏢 PROBLEM 3: Feature Importance — Permutation Importance

### Theory
**Permutation importance** measures feature importance by measuring how much model performance drops when a feature is randomly shuffled (breaking its relationship with target).

$$\text{Importance}(f) = \text{Score}(\text{original}) - \text{Score}(\text{after shuffling } f)$$

**Advantages over tree-based importance:**
- Model-agnostic
- Measures true predictive power, not just usage in splits
- Handles correlated features better

### Implementation
```python
def permutation_importance(model, X: np.ndarray, y: np.ndarray, 
                            metric_fn, n_repeats: int = 10,
                            random_state: int = 42) -> dict:
    """
    Model-agnostic permutation feature importance.
    
    model: any model with .predict() method
    metric_fn: function(y_true, y_pred) → score (higher = better)
    n_repeats: number of shuffles per feature (for stability)
    
    Returns: dict of {feature_idx: (mean_importance, std_importance)}
    """
    rng = np.random.RandomState(random_state)
    
    # Baseline score
    baseline_score = metric_fn(y, model.predict(X))
    
    importances = {}
    
    for feat_idx in range(X.shape[1]):
        scores_feat = []
        
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            # Shuffle only this feature (break relationship with target)
            rng.shuffle(X_shuffled[:, feat_idx])
            
            shuffled_score = metric_fn(y, model.predict(X_shuffled))
            scores_feat.append(baseline_score - shuffled_score)
        
        importances[feat_idx] = {
            'mean': np.mean(scores_feat),
            'std': np.std(scores_feat),
            'scores': scores_feat
        }
    
    return {
        'baseline_score': baseline_score,
        'importances': importances,
        'feature_ranking': sorted(
            importances.keys(), 
            key=lambda k: importances[k]['mean'], 
            reverse=True
        )
    }

# Test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

np.random.seed(42)
n, d = 500, 10
X = np.random.randn(n, d)
# Feature 0 and 2 are predictive; rest are noise
y = ((X[:, 0] + X[:, 2]) > 0).astype(int)

model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
metric = lambda yt, yp: roc_auc_score(yt, yp)

result = permutation_importance(model, X, y, metric, n_repeats=5)
print(f"Baseline AUC: {result['baseline_score']:.4f}")
print(f"Feature ranking: {result['feature_ranking'][:5]}")
# Features 0 and 2 should be ranked 1st and 2nd ✓
```

---

## 📊 FAANG CODING INTERVIEW — QUICK REFERENCE

```python
# ===== PATTERN TEMPLATES =====

# 1. Sliding Window (max/min in window)
from collections import deque
def sliding_window_max(nums, k):
    dq = deque()
    result = []
    for i, v in enumerate(nums):
        while dq and nums[dq[-1]] <= v: dq.pop()
        dq.append(i)
        if dq[0] == i - k: dq.popleft()
        if i >= k - 1: result.append(nums[dq[0]])
    return result

# 2. Top-K (heap)
import heapq
def top_k(data, k):
    heap = []
    for x in data:
        heapq.heappush(heap, x)
        if len(heap) > k: heapq.heappop(heap)
    return sorted(heap, reverse=True)

# 3. Union-Find (one-liner class)
class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x): 
        if self.p[x] != x: self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, x, y): self.p[self.find(x)] = self.find(y)
    def connected(self, x, y): return self.find(x) == self.find(y)

# 4. Binary Search Template
def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target: return mid
        elif nums[mid] < target: lo = mid + 1
        else: hi = mid - 1
    return -1  # or lo for "first position >= target"

# 5. DFS on adjacency list
def dfs(graph, start, visited=None):
    if visited is None: visited = set()
    visited.add(start)
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

# 6. BFS for shortest path
from collections import deque
def bfs_shortest(graph, start, end):
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if node == end: return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None  # No path found
```
