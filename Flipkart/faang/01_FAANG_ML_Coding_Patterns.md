# 🏢 FAANG ML Coding Questions — System Design + Advanced Algorithms
> **Difficulty:** Hard | **Focus:** Google, Meta, Amazon, Apple, Microsoft ML Interviews

---

## FAANG vs PRODUCT COMPANY DIFFERENCES

| Dimension | FAANG (Google/Meta/Amazon) | Flipkart |
|---|---|---|
| Coding depth | More algorithmic (DP, Graphs, Heaps) | More ML-specific coding |
| Scale | Billions of users → distributed ML | 500M users → production ML |
| GenAI expectation | LLM system design | RAG, agents, fraud detection |
| Math depth | Derivation from scratch | Application + implementation |
| Favorite topics | Sliding windows, Union-Find, Top-K | NDCG, attention, backprop |

---

## 🏢 PROBLEM 1: Sliding Window — Real-Time Fraud Transaction Rate

### Theory
**Leetcode pattern: Sliding Window / Deque**

In fraud detection: calculate rolling rate of fraudulent transactions in a time window.

**Fixed window:** Group by time bucket (easy, but misses cross-boundary fraud bursts)

**Sliding window:** For each incoming transaction, maintain a deque of timestamps within the window.

### Things to Focus On
- ✅ Deque-based O(1) average amortized per event (each event added/removed once)
- ✅ Fixed vs sliding: fixed misses boundary spikes; sliding is exact
- ✅ Extension: concurrent sliding windows across users (use dict of deques)
- ✅ Rate limiting (same pattern): API rate limiter using sliding window counter

### Implementation
```python
from collections import deque
from typing import List, Tuple

class SlidingWindowFraudRate:
    """
    Real-time sliding window for fraud transaction rate.
    
    For each transaction, compute: # fraud transactions in last W seconds / W
    """
    
    def __init__(self, window_seconds: int = 3600):
        self.window = window_seconds  # Window size in seconds
        self.fraud_times = deque()    # Timestamps of fraud events
        self.all_times = deque()      # Timestamps of all events
    
    def add_transaction(self, timestamp: float, is_fraud: bool) -> float:
        """
        Add new transaction and return fraud rate in last W seconds.
        
        Time: O(1) amortized
        """
        # Add to queues
        self.all_times.append(timestamp)
        if is_fraud:
            self.fraud_times.append(timestamp)
        
        cutoff = timestamp - self.window
        
        # Remove expired events from left of deques
        while self.all_times and self.all_times[0] < cutoff:
            self.all_times.popleft()
        while self.fraud_times and self.fraud_times[0] < cutoff:
            self.fraud_times.popleft()
        
        # Fraud rate in window
        total = len(self.all_times)
        fraud = len(self.fraud_times)
        return fraud / total if total > 0 else 0.0
    
    def get_rolling_rates(self, events: List[Tuple[float, bool]]) -> List[float]:
        """Process a list of (timestamp, is_fraud) events."""
        return [self.add_transaction(ts, fraud) for ts, fraud in events]

# Test — simulate transaction stream
import numpy as np
np.random.seed(42)
events = sorted([(i * 10, np.random.rand() < 0.05) for i in range(100)])  # 5% fraud rate

detector = SlidingWindowFraudRate(window_seconds=300)  # 5-minute window
rates = detector.get_rolling_rates(events)
print(f"Average fraud rate: {np.mean(rates):.3f}")  # ~0.05
print(f"Max fraud rate in window: {max(rates):.3f}")
```

---

## 🏢 PROBLEM 2: Top-K Frequent Elements (Heap Pattern)

### Theory
**Heap (Priority Queue) pattern** — O(n log k) time, O(n+k) space.

Used in: ML feature importance ranking, top-K similar items (embeddings), K-nearest neighbors, online learning (reservoir sampling).

**Three approaches:**
1. Sort: O(n log n)
2. Min-heap of size k: O(n log k)
3. QuickSelect: O(n) average, O(n²) worst

### Things to Focus On
- ✅ MinHeap of size k: maintain k largest elements seen so far
- ✅ heapq in Python is a min-heap; negate for max-heap
- ✅ For streaming (can't hold all data): reservoir sampling for uniform random k, min-heap for top-k
- ✅ Application: Top-K feature selection by mutual information

### Implementation
```python
import heapq
from collections import Counter
from typing import List, Any

def top_k_frequent(elements: List[Any], k: int) -> List[Any]:
    """
    Return k most frequent elements.
    Time: O(n log k), Space: O(n + k)
    """
    # Count frequencies
    freq = Counter(elements)  # O(n)
    
    # Use min-heap of size k: (frequency, element)
    heap = []
    for element, count in freq.items():
        heapq.heappush(heap, (count, element))
        if len(heap) > k:
            heapq.heappop(heap)  # Remove smallest frequency
    
    # Extract top-k (sorted by descending frequency)
    return [element for _, element in sorted(heap, reverse=True)]

def top_k_similar_embeddings(query: np.ndarray, corpus: np.ndarray, 
                               item_ids: List[int], k: int) -> List[int]:
    """
    Find top-k most similar embeddings to query using cosine similarity.
    Uses min-heap to maintain top-k candidates.
    
    Time: O(n*d + n log k), much better than sorting all n items
    """
    # Normalize for cosine similarity
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    
    heap = []  # Min-heap: (similarity, item_id)
    
    for i, (emb, item_id) in enumerate(zip(corpus_norm, item_ids)):
        sim = float(query_norm @ emb)
        heapq.heappush(heap, (sim, item_id))
        if len(heap) > k:
            heapq.heappop(heap)  # Remove lowest similarity
    
    # Return top-k item IDs (highest similarity first)
    return [item_id for _, item_id in sorted(heap, reverse=True)]

class StreamingTopK:
    """
    Maintain top-k items in a streaming setting (can't store all data).
    Uses min-heap of size k.
    """
    
    def __init__(self, k: int):
        self.k = k
        self.heap = []  # (score, item)
    
    def add(self, score: float, item: Any) -> None:
        """Add item with score. O(log k)."""
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (score, str(item)))  # str for comparison
        elif score > self.heap[0][0]:
            heapq.heapreplace(self.heap, (score, str(item)))
    
    def get_top_k(self) -> List[tuple]:
        """Return top-k items sorted by score descending. O(k log k)."""
        return sorted(self.heap, reverse=True)

# Test
items = ['apple', 'banana', 'apple', 'cherry', 'apple', 'banana', 'date']
print(top_k_frequent(items, k=2))  # ['apple', 'banana']

# Embedding similarity
np.random.seed(42)
query = np.random.randn(64)
corpus = np.random.randn(1000, 64)
ids = list(range(1000))
top5 = top_k_similar_embeddings(query, corpus, ids, k=5)
print(f"Top-5 similar item ids: {top5}")
```

---

## 🏢 PROBLEM 3: Connected Components — Fraud Ring Detection (Union-Find)

### Theory
**Union-Find (Disjoint Set Union)** — O(α(n)) ≈ O(1) per operation (inverse Ackermann).

**Fraud ring detection:** If person A shared a phone with B, and B shared a device with C, then A, B, C form a fraud ring (connected component).

Applications:
- Fraud ring detection (entity linking)
- Graph connectivity in social networks
- Clustering by shared attributes
- Kruskal's MST algorithm

### Things to Focus On
- ✅ Path compression + union by rank = near O(1) amortized
- ✅ find() with path compression: flatten tree to reduce future lookups
- ✅ union(): always attach smaller tree under larger (by rank/size)
- ✅ For fraud: nodes = entities (users, devices, IPs); edges = shared attribute

### Implementation
```python
class UnionFind:
    """
    Union-Find (Disjoint Set Union) with path compression + union by rank.
    
    Applications in ML:
    - Fraud ring detection
    - Entity resolution (finding duplicate records)
    - Graph-based clustering
    """
    
    def __init__(self, n: int):
        self.parent = list(range(n))  # parent[i] = i (each node is its own root)
        self.rank   = [0] * n         # Tree height estimate
        self.size   = [1] * n         # Cluster size
        self.n_components = n
    
    def find(self, x: int) -> int:
        """
        Find root of x with path compression.
        Path compression: makes all nodes on path point directly to root.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        Unite sets containing x and y.
        Returns True if they were in different sets (new merge).
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already connected
        
        # Union by rank: attach smaller tree under larger
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x  # root_y's tree goes under root_x
        self.size[root_x] += self.size[root_y]
        
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        self.n_components -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same component."""
        return self.find(x) == self.find(y)
    
    def get_component_size(self, x: int) -> int:
        """Return size of x's component."""
        return self.size[self.find(x)]
    
    def get_all_components(self) -> dict:
        """Return {root: [members]} for all components."""
        components = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)
        return components

def detect_fraud_rings(entities: List[str], shared_attributes: List[tuple]) -> dict:
    """
    Detect fraud rings: entities that share attributes (phone, device, IP).
    
    entities: list of entity IDs (users, accounts)
    shared_attributes: list of (entity1, entity2) pairs that share an attribute
    
    Returns: dict of {component_id: [entity_list]} for suspicious groups
    """
    n = len(entities)
    entity_to_idx = {e: i for i, e in enumerate(entities)}
    
    uf = UnionFind(n)
    
    # Connect entities sharing attributes
    for e1, e2 in shared_attributes:
        if e1 in entity_to_idx and e2 in entity_to_idx:
            uf.union(entity_to_idx[e1], entity_to_idx[e2])
    
    # Get all components
    components = uf.get_all_components()
    idx_to_entity = {i: e for e, i in entity_to_idx.items()}
    
    # Return suspicious rings (size >= 3)
    suspicious = {
        root: [idx_to_entity[i] for i in members]
        for root, members in components.items()
        if len(members) >= 3
    }
    return suspicious

# Test — fraud ring detection
entities = ['user_A', 'user_B', 'user_C', 'user_D', 'user_E', 'user_F']

# These users shared devices/IPs — potential fraud ring
shared = [
    ('user_A', 'user_B'),  # Same device
    ('user_B', 'user_C'),  # Same IP
    ('user_D', 'user_E'),  # Same phone
    # user_F is standalone
]

rings = detect_fraud_rings(entities, shared)
print("Suspected Fraud Rings:")
for ring_id, members in rings.items():
    print(f"  Ring: {members}")
# Expected: [user_A, user_B, user_C] form a ring
```

---

## 🏢 PROBLEM 4: LRU Cache (ML Feature Store)

### Theory
**LRU (Least Recently Used)** cache eviction — used in online feature stores.

Implementation: **Doubly Linked List + HashMap** → O(1) get and put.

**ML Application:** Feature store cache (avoid recomputing expensive features; evict least recently accessed).

### Implementation
```python
class LRUCache:
    """
    LRU Cache using doubly linked list + hashmap.
    O(1) get and put.
    
    ML Application: Feature store cache, embedding cache, prediction cache
    """
    
    class Node:
        def __init__(self, key=None, val=None):
            self.key = key
            self.val = val
            self.prev = None
            self.next = None
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}   # key → Node
        
        # Sentinel nodes (never removed)
        self.head = self.Node()  # LRU end
        self.tail = self.Node()  # MRU end
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node: 'LRUCache.Node') -> None:
        """Remove node from linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_tail(self, node: 'LRUCache.Node') -> None:
        """Add node before tail (most recently used position)."""
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node
    
    def get(self, key) -> Any:
        """Get value and mark as most recently used. O(1)."""
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)        # Remove from current position
        self._add_to_tail(node)   # Move to MRU position
        return node.val
    
    def put(self, key, value) -> None:
        """Insert or update. Evict LRU if at capacity. O(1)."""
        if key in self.cache:
            self._remove(self.cache[key])
            del self.cache[key]
        
        if len(self.cache) >= self.capacity:
            # Evict LRU node (first after head)
            lru_node = self.head.next
            self._remove(lru_node)
            del self.cache[lru_node.key]
        
        # Insert new node
        node = self.Node(key, value)
        self._add_to_tail(node)
        self.cache[key] = node

# Test
cache = LRUCache(3)
cache.put('user_1', {'feature_a': 0.5, 'feature_b': 1.2})
cache.put('user_2', {'feature_a': 0.3, 'feature_b': 0.8})
cache.put('user_3', {'feature_a': 0.9, 'feature_b': 2.1})

print(cache.get('user_1'))  # Access user_1 (makes it MRU)
cache.put('user_4', {'feature_a': 0.7, 'feature_b': 1.5})  # Evicts user_2

print(cache.get('user_2'))  # -1 (evicted)
print(cache.get('user_3'))  # Still there
```

---

## 🏢 PROBLEM 5: ML Similarity Search — KD-Tree vs Approximate NN

### Theory

**Exact KNN:** O(n·d) per query — too slow for large n.

**KD-Tree:** O(d × log n) average for low-d spaces, but degrades to O(n) in high dimensions (curse of dimensionality).

**FAISS / HNSW:** Approximate nearest neighbor (ANN) — trade exact results for O(log n) query time. Used in RAG, recommendation.

**Implementation from scratch:** Focus on brute-force ball-tree partitioning.

### Implementation
```python
class BruteForceKNN:
    """
    Exact KNN with numpy vectorization.
    O(n*d) per query but efficient with vectorized numpy.
    For small-medium scale or as reference implementation.
    """
    
    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric
        self.X_fit = None
        self.labels = None
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'BruteForceKNN':
        self.X_fit = X
        self.labels = y
        return self
    
    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        """Compute distances from x to all training points."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_fit - x) ** 2, axis=1))
        elif self.metric == 'cosine':
            x_norm = x / (np.linalg.norm(x) + 1e-8)
            X_norm = self.X_fit / (np.linalg.norm(self.X_fit, axis=1, keepdims=True) + 1e-8)
            return 1 - X_norm @ x_norm  # Distance = 1 - similarity
    
    def kneighbors(self, X_query: np.ndarray, k: int) -> tuple:
        """Return (distances, indices) of k nearest neighbors for each query."""
        all_distances = []
        all_indices = []
        
        for x in X_query:
            dists = self._compute_distances(x)
            idx = np.argpartition(dists, k)[:k]  # O(n) vs O(n log n) sort
            idx = idx[np.argsort(dists[idx])]    # Sort the k candidates
            all_distances.append(dists[idx])
            all_indices.append(idx)
        
        return np.array(all_distances), np.array(all_indices)
    
    def predict(self, X_query: np.ndarray, k: int = 5) -> np.ndarray:
        """KNN classification: majority vote among k neighbors."""
        _, indices = self.kneighbors(X_query, k)
        predictions = []
        for neighbors in indices:
            neighbor_labels = self.labels[neighbors]
            vote = Counter(neighbor_labels.tolist()).most_common(1)[0][0]
            predictions.append(vote)
        return np.array(predictions)

# Quick Interview Template for any KNN-style question
def cosine_top_k(query: np.ndarray, corpus: np.ndarray, k: int) -> np.ndarray:
    """Vectorized cosine similarity top-k. O(n*d)."""
    q_norm = query / (np.linalg.norm(query) + 1e-8)
    c_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    scores = c_norm @ q_norm
    return np.argpartition(-scores, k)[:k]  # Top-k indices

# Test
from collections import Counter
np.random.seed(42)
X_tr = np.random.randn(200, 10)
y_tr = (X_tr[:, 0] > 0).astype(int)
X_te = np.random.randn(20, 10)

knn = BruteForceKNN(metric='euclidean').fit(X_tr, y_tr)
preds = knn.predict(X_te, k=5)
print(f"KNN predictions: {preds}")
```

---

## 📋 FAANG CODING PATTERNS FOR ML INTERVIEWS

| Pattern | Data Structure | ML Application | Example |
|---|---|---|---|
| **Sliding Window** | Deque | Real-time metrics, rate limiting | Fraud rate per window |
| **Two Pointers** | Sorted array | Finding threshold, binary search | Best classification threshold |
| **Top-K** | Min-Heap | Recommendation, feature selection | Top-K similar items |
| **Union-Find** | Array + rank | Fraud rings, entity resolution | Connected accounts |
| **LRU Cache** | DLL + HashMap | Feature store, embedding cache | Online prediction cache |
| **BFS/DFS** | Queue/Stack | Graph neural networks, fraud paths | Shortest fraud path |
| **Dynamic Programming** | 2D array | Sequence alignment, edit distance | Levenshtein distance |
| **Reservoir Sampling** | Array of size k | Streaming data, random subset | Random sample from log stream |
| **Trie** | Tree | Autocomplete, spell check | Product search suggestions |
| **Monotonic Deque** | Deque | Rolling min/max, feature windows | Rolling Sharpe ratio |

---

## 🎯 FAANG INTERVIEW FOLLOW-UP QUESTIONS

1. **"How would you scale fraud ring detection to 100M users?"** → Distributed Union-Find using consistent hashing. Group entities by shared device hash → process locally. Merge cross-partition components via a coordinator. Or use Spark's connected components (GraphX).
2. **"Your embedding cache has 99% hit rate but queries still time out. Why?"** → Hot keys: a few embeddings are extremely popular → one cache shard is overwhelmed. Solution: replicate hot embeddings across shards or use consistent hashing with virtual nodes.
3. **"For ANN search with 1B vectors, what would you use?"** → FAISS with HNSW or IVF index. HNSW: graph-based, O(log n) query, ~95% recall. IVF: inverted file, product quantization reduces memory 16-32x. Choose based on recall/latency tradeoff.
4. **"Difference between BFS and DFS for fraud graph traversal?"** → BFS explores all 1-hop neighbors before 2-hop (level-by-level) → better for shortest fraud path. DFS goes deep first → better for cycle detection (finding fraud rings), less memory for sparse graphs.
